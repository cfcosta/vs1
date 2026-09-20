use std::{
    collections::BTreeMap,
    fs,
    net::TcpStream,
    path::PathBuf,
    thread,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, bail, ensure};
use base64::{Engine, engine::general_purpose::STANDARD};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tungstenite::{Message, WebSocket, stream::MaybeTlsStream};

pub const SNAPSHOT: &str = include_str!("../assets/snapshot.js");

#[derive(Debug)]
pub struct Stale(pub &'static str);
impl std::fmt::Display for Stale {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}
impl std::error::Error for Stale {}

pub struct Browser {
    socket: WebSocket<MaybeTlsStream<TcpStream>>,
    next_id: u64,
    pub target: String,
    session: String,
    pub counts: BTreeMap<String, usize>,
    after_input: Option<Value>,
    recording: Option<PathBuf>,
    frames: Vec<Value>,
    version: Value,
}

impl Browser {
    pub fn connect(endpoint: &str, url: &str) -> Result<Self> {
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(15))
            .build()?;
        let version: Value = client
            .get(format!("{}/json/version", endpoint.trim_end_matches('/')))
            .send()?
            .error_for_status()?
            .json()?;
        let ws = version["webSocketDebuggerUrl"]
            .as_str()
            .context("Chrome did not advertise a debugger socket")?;
        let (mut socket, _) = tungstenite::connect(ws)?;
        if let MaybeTlsStream::Plain(stream) = socket.get_mut() {
            stream.set_read_timeout(Some(Duration::from_secs(30)))?;
            stream.set_write_timeout(Some(Duration::from_secs(30)))?;
        }
        let mut browser = Self {
            socket,
            next_id: 0,
            target: String::new(),
            session: String::new(),
            counts: BTreeMap::new(),
            after_input: None,
            recording: None,
            frames: vec![],
            version,
        };
        let target = browser.call(
            "Target.createTarget",
            json!({"url":"about:blank", "background":true}),
        )?;
        browser.target = target["targetId"]
            .as_str()
            .context("missing target ID")?
            .to_owned();
        let attached = browser.call(
            "Target.attachToTarget",
            json!({"targetId":browser.target,"flatten":true}),
        )?;
        browser.session = attached["sessionId"]
            .as_str()
            .context("missing session ID")?
            .to_owned();
        browser.call("Emulation.setDeviceMetricsOverride", json!({"width":1120,"height":780,"deviceScaleFactor":1,"mobile":false}))?;
        browser.call(
            "Emulation.setFocusEmulationEnabled",
            json!({"enabled":true}),
        )?;
        browser.call("Page.navigate", json!({"url":url}))?;
        let deadline = Instant::now() + Duration::from_secs(15);
        loop {
            match browser.evaluate("document.readyState") {
                Ok(v) if v == "complete" => break,
                Err(e) if e.downcast_ref::<Stale>().is_none() => return Err(e),
                _ => (),
            }
            ensure!(Instant::now() < deadline, "initial navigation timed out");
            thread::sleep(Duration::from_millis(20));
        }
        Ok(browser)
    }

    fn send(&mut self, method: &str, params: Value) -> Result<u64> {
        self.next_id += 1;
        let mut request =
            json!({"id":self.next_id,"method":method,"params":params});
        if !self.session.is_empty()
            && !method.starts_with("Target.")
            && !method.starts_with("Browser.")
        {
            request["sessionId"] = json!(self.session);
        }
        *self.counts.entry(method.to_owned()).or_default() += 1;
        self.socket
            .send(Message::Text(request.to_string().into()))?;
        Ok(self.next_id)
    }

    pub fn call(&mut self, method: &str, params: Value) -> Result<Value> {
        let id = self.send(method, params)?;
        loop {
            let message = self.socket.read().context("CDP connection interrupted; browser mutation must not be retried")?;
            let Message::Text(text) = message else {
                if matches!(message, Message::Close(_)) {
                    bail!("Chrome closed the debugging connection");
                }
                continue;
            };
            let response: Value = serde_json::from_str(&text)?;
            if response["method"] == "Page.screencastFrame"
                && response["sessionId"] == self.session
            {
                let p = &response["params"];
                if let Some(folder) = &self.recording {
                    let name = format!("{:06}.jpg", self.frames.len());
                    fs::write(
                        folder.join(&name),
                        STANDARD.decode(
                            p["data"].as_str().context("invalid frame")?,
                        )?,
                    )?;
                    self.frames.push(json!({"file":name,"timestamp":p["metadata"]["timestamp"]}));
                }
                // Ack without recursively waiting; its response is skipped below.
                self.send(
                    "Page.screencastFrameAck",
                    json!({"sessionId":p["sessionId"]}),
                )?;
            }
            if response["id"].as_u64() != Some(id) {
                continue;
            }
            if let Some(error) = response.get("error") {
                bail!("CDP {method}: {error}");
            }
            return Ok(response["result"].clone());
        }
    }

    pub fn evaluate(&mut self, expression: &str) -> Result<Value> {
        let result = self.call("Runtime.evaluate", json!({"expression":expression,"returnByValue":true,"awaitPromise":true}))?;
        if result.get("exceptionDetails").is_some() {
            return Err(Stale("document changed during a read").into());
        }
        Ok(result["result"]["value"].clone())
    }

    pub fn observe(&mut self) -> Result<Value> {
        if let Some(action) = self.after_input.take() {
            let wait = include_str!("../assets/settle.js")
                .replace("__ACTION__", &action.to_string());
            // A settle read cannot undo an already logged execution.
            let _ = self.evaluate(&wait);
        }
        for attempt in 0..10 {
            match self.evaluate(SNAPSHOT) {
                Ok(mut page) if page.is_object() => {
                    let content = json!([
                        page["url"],
                        page["text"],
                        page["actions"],
                        page["scroll"]
                    ]);
                    page["fingerprint"] = json!(format!(
                        "{:x}",
                        Sha256::digest(content.to_string().as_bytes())
                    ));
                    return Ok(page);
                }
                Err(e) if e.downcast_ref::<Stale>().is_none() => return Err(e),
                _ => (),
            }
            if attempt < 9 {
                thread::sleep(Duration::from_millis(20));
            }
        }
        Err(Stale("page did not settle").into())
    }

    pub fn fresh(
        &mut self,
        page: &Value,
        action: Option<&Value>,
    ) -> Result<bool> {
        if let Some(action) =
            action.filter(|a| a["kind"] == "click" || a["kind"] == "select")
        {
            let node =
                action["node"].as_u64().context("invalid observed node")?;
            let script = format!(
                "(() => {{ const c=window.__jevFast; return c ? [c.pageKey(),c.guard(c.nodes.get({node}))] : null; }})()"
            );
            return Ok(self.evaluate(&script)?
                == json!([
                    page["page_key"],
                    page["guards"][node.to_string()]
                ]));
        }
        let script = format!(
            "(() => {{const state={SNAPSHOT}; return state?.marker ?? null;}})()"
        );
        Ok(self.evaluate(&script)? == page["marker"])
    }

    pub fn act(
        &mut self,
        action: &Value,
        page: &Value,
        text: Option<&str>,
    ) -> Result<()> {
        if !self.fresh(page, Some(action))? {
            return Err(Stale("page changed before input").into());
        }
        let kind = action["kind"].as_str().context("missing action kind")?;
        match kind {
            "wait" => thread::sleep(Duration::from_millis(100)),
            "scroll" => {
                self.call("Input.dispatchMouseEvent", json!({"type":"mouseWheel","x":550,"y":650,"deltaX":0,"deltaY":action["delta"]}))?;
            }
            "click" | "fill" | "select" => {
                ensure!(
                    action["node"].as_u64().is_some(),
                    "invalid observed node"
                );
                if kind == "fill" {
                    ensure!(text.is_some(), "missing field text");
                }
                let script = include_str!("../assets/target.js")
                    .replace("__ACTION__", &action.to_string());
                let result = self.call(
                    "Runtime.evaluate",
                    json!({"expression":script,"returnByValue":true}),
                )?;
                // SELECT mutates in this evaluation. An interrupted result is never a stale retry.
                if kind == "select" {
                    ensure!(
                        result.get("exceptionDetails").is_none()
                            && result["result"]["value"].is_object(),
                        "dropdown execution not confirmed; inspect before retrying"
                    );
                } else if result.get("exceptionDetails").is_some()
                    || !result["result"]["value"].is_object()
                {
                    return Err(Stale("target changed or is covered").into());
                }
                if kind != "select" {
                    let p = &result["result"]["value"];
                    for event in ["mousePressed", "mouseReleased"] {
                        self.call("Input.dispatchMouseEvent", json!({"type":event,"x":p["x"],"y":p["y"],"button":"left","clickCount":1}))?;
                    }
                    if kind == "fill" {
                        for event in ["keyDown", "keyUp"] {
                            self.call("Input.dispatchKeyEvent", json!({"type":event,"key":"a","code":"KeyA","modifiers":if cfg!(target_os="macos") {4} else {2},"commands":["selectAll"]}))?;
                        }
                        self.call("Input.insertText", json!({"text":text}))?;
                    }
                }
            }
            _ => bail!("unsupported action kind {kind}"),
        }
        if kind != "wait" {
            self.after_input = Some(action.clone());
        }
        Ok(())
    }

    pub fn screenshot(&mut self, path: &std::path::Path) -> Result<()> {
        let result = self.call(
            "Page.captureScreenshot",
            json!({"format":"jpeg","quality":85}),
        )?;
        fs::write(
            path,
            STANDARD.decode(
                result["data"].as_str().context("missing screenshot")?,
            )?,
        )?;
        Ok(())
    }

    pub fn start_recording(&mut self, folder: PathBuf) -> Result<()> {
        fs::create_dir_all(&folder)?;
        self.screenshot(&folder.join("initial.jpg"))?;
        self.recording = Some(folder);
        self.call("Page.startScreencast", json!({"format":"jpeg","quality":80,"maxWidth":1120,"maxHeight":780,"everyNthFrame":2}))?;
        Ok(())
    }

    pub fn stop_recording(&mut self) -> Result<()> {
        if self.recording.is_some() {
            self.call("Page.stopScreencast", json!({}))?;
            fs::write(
                self.recording.as_ref().unwrap().join("frames.json"),
                serde_json::to_vec_pretty(&self.frames)?,
            )?;
            self.recording = None;
        }
        Ok(())
    }

    pub fn version(&self) -> &Value {
        &self.version["Browser"]
    }
}

impl Drop for Browser {
    fn drop(&mut self) {
        let _ = self.stop_recording();
        if !self.target.is_empty() {
            let _ = self
                .call("Target.closeTarget", json!({"targetId":self.target}));
        }
    }
}

#[cfg(test)]
mod tests {
    use std::net::TcpListener;

    use super::*;

    fn fake(
        responses: Vec<Value>,
    ) -> (Browser, std::thread::JoinHandle<Vec<String>>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let worker = std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let mut socket = tungstenite::accept(stream).unwrap();
            let mut methods = vec![];
            for result in responses {
                let request: Value = serde_json::from_str(
                    socket.read().unwrap().to_text().unwrap(),
                )
                .unwrap();
                methods.push(request["method"].as_str().unwrap().to_owned());
                let response = if result.get("protocol_error").is_some() {
                    json!({"id":request["id"],"error":{"message":"injected mutation interruption"}})
                } else {
                    json!({"id":request["id"],"result":result})
                };
                socket
                    .send(Message::Text(response.to_string().into()))
                    .unwrap();
            }
            methods
        });
        let (socket, _) =
            tungstenite::connect(format!("ws://{address}")).unwrap();
        (
            Browser {
                socket,
                next_id: 0,
                target: String::new(),
                session: "test".into(),
                counts: BTreeMap::new(),
                after_input: None,
                recording: None,
                frames: vec![],
                version: Value::Null,
            },
            worker,
        )
    }
    fn fresh_result() -> Value {
        json!({"result":{"value":[["page"],["guard"]]}})
    }
    fn page() -> Value {
        json!({"page_key":["page"],"guards":{"1":["guard"]}})
    }

    #[test]
    fn uncertain_select_is_not_a_retryable_stale_read() {
        let (mut browser, worker) = fake(vec![
            fresh_result(),
            json!({"exceptionDetails":{"text":"navigation"}}),
        ]);
        let action =
            json!({"id":"e1","kind":"select","node":1,"value":"Design"});
        let error = browser.act(&action, &page(), None).unwrap_err();
        assert!(error.downcast_ref::<Stale>().is_none());
        assert!(error.to_string().contains("not confirmed"));
        assert_eq!(
            worker.join().unwrap(),
            ["Runtime.evaluate", "Runtime.evaluate"]
        );
    }
    #[test]
    fn interrupted_mouse_release_is_not_retried() {
        let (mut browser, worker) = fake(vec![
            fresh_result(),
            json!({"result":{"value":{"x":5,"y":5}}}),
            json!({}),
            json!({"protocol_error":true}),
        ]);
        let action = json!({"id":"e1","kind":"click","node":1});
        let error = browser.act(&action, &page(), None).unwrap_err();
        assert!(error.downcast_ref::<Stale>().is_none());
        assert_eq!(
            worker.join().unwrap(),
            [
                "Runtime.evaluate",
                "Runtime.evaluate",
                "Input.dispatchMouseEvent",
                "Input.dispatchMouseEvent"
            ]
        );
    }
    #[test]
    fn missing_click_target_is_rejected_before_input() {
        let (mut browser, worker) =
            fake(vec![fresh_result(), json!({"result":{"value":null}})]);
        let action = json!({"id":"e1","kind":"click","node":1});
        assert!(
            browser
                .act(&action, &page(), None)
                .unwrap_err()
                .downcast_ref::<Stale>()
                .is_some()
        );
        assert_eq!(
            worker.join().unwrap(),
            ["Runtime.evaluate", "Runtime.evaluate"]
        );
    }
}
