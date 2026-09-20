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
            // Timers continue to progress even when background rAF is throttled.
            thread::sleep(Duration::from_millis(if action["kind"] == "fill" {
                200
            } else {
                100
            }));
            return self.settle_after_input(&action, Duration::from_secs(5));
        }
        for attempt in 0..10 {
            match self.evaluate(SNAPSHOT) {
                Ok(mut page) if page.is_object() => {
                    fingerprint(&mut page);
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

    pub fn observe_effect(&mut self, previous: &Value) -> Result<Value> {
        let mut page = self.observe()?;
        let deadline = Instant::now() + Duration::from_millis(1200);
        while page["fingerprint"] == previous["fingerprint"]
            && Instant::now() < deadline
        {
            thread::sleep(Duration::from_millis(100));
            page = self.observe()?;
        }
        Ok(page)
    }

    fn settle_after_input(
        &mut self,
        action: &Value,
        timeout: Duration,
    ) -> Result<Value> {
        let script = include_str!("../assets/settle.js")
            .replace("__SNAPSHOT__", SNAPSHOT.trim().trim_end_matches(';'))
            .replace("__ACTION__", &action.to_string());
        let deadline = Instant::now() + timeout;
        loop {
            let waiting_for_menu = match self.evaluate(&script) {
                Ok(mut result) => {
                    if result["ready"] == true && result["page"].is_object() {
                        let mut page = result["page"].take();
                        fingerprint(&mut page);
                        return Ok(page);
                    }
                    result["waiting_for_menu"] == true
                }
                // Navigation may destroy the old context. Reobserve, never replay input.
                Err(e) if e.downcast_ref::<Stale>().is_some() => false,
                Err(e) => return Err(e),
            };
            ensure!(
                Instant::now() < deadline,
                "{} did not become ready after input within {} ms; action was not retried",
                if waiting_for_menu { "menu" } else { "page" },
                timeout.as_millis()
            );
            thread::sleep(
                Duration::from_millis(50)
                    .min(deadline.saturating_duration_since(Instant::now())),
            );
        }
    }

    pub fn fresh(
        &mut self,
        page: &Value,
        action: Option<&Value>,
    ) -> Result<bool> {
        if let Some(action) = action.filter(|a| {
            a["kind"] == "click" || a["kind"] == "select" || a["kind"] == "key"
        }) {
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
            "click" | "fill" | "select" | "key" => {
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
                if kind == "key" {
                    let key = action["key"].as_str().context("missing key")?;
                    let code = match key {
                        "ArrowLeft" => 37,
                        "ArrowRight" => 39,
                        "Home" => 36,
                        "End" => 35,
                        "Enter" => 13,
                        _ => bail!("unsupported key"),
                    };
                    for event in ["keyDown", "keyUp"] {
                        self.call("Input.dispatchKeyEvent", json!({"type":event,"key":key,"code":key,"windowsVirtualKeyCode":code}))?;
                    }
                } else if kind != "select" {
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
            let mut pending = action.clone();
            pending["time_origin"] = page["marker"][0].clone();
            self.after_input = Some(pending);
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

fn fingerprint(page: &mut Value) {
    // Progress is semantic. DOM identity and geometry remain in freshness guards,
    // but replacing an identical node or animating it is not task progress.
    let mut actions = page["actions"].clone();
    if let Some(actions) = actions.as_array_mut() {
        for action in actions {
            if let Some(action) = action.as_object_mut() {
                for key in ["node", "id", "rect"] {
                    action.remove(key);
                }
            }
        }
    }
    let content = json!([
        page["url"],
        page["text"],
        actions,
        page["scroll"],
        page["graphics"]
    ]);
    page["fingerprint"] = json!(format!(
        "{:x}",
        Sha256::digest(content.to_string().as_bytes())
    ));
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
    fn progress_ignores_geometry_and_replacement_but_tracks_graph_labels() {
        let mut a = json!({"text":"Chart", "actions":[{"node":1,"id":"e1","rect":{"x":1},"label":"Next","kind":"click"}],"graphics":[{"text":"October"}]});
        fingerprint(&mut a);
        let mut b = a.clone();
        b["actions"][0]["node"] = json!(2);
        b["actions"][0]["rect"]["x"] = json!(300);
        fingerprint(&mut b);
        assert_eq!(a["fingerprint"], b["fingerprint"]);
        b["graphics"][0]["text"] = json!("November");
        fingerprint(&mut b);
        assert_ne!(a["fingerprint"], b["fingerprint"]);
    }

    #[test]
    fn post_input_wait_reobserves_without_replaying_mutations() {
        let (mut browser, worker) = fake(vec![
            json!({"result":{"value":{"ready":false,"waiting_for_menu":true,"page":{"text":"old page"}}}}),
            json!({"exceptionDetails":{"text":"navigation destroyed context"}}),
            json!({"result":{"value":{"ready":true,"page":{"url":"https://example.test","text":"One way","actions":[],"scroll":{}}}}}),
        ]);
        let page = browser
            .settle_after_input(
                &json!({"kind":"click"}),
                Duration::from_secs(1),
            )
            .unwrap();
        assert_eq!(page["text"], "One way");
        assert!(page["fingerprint"].is_string());
        assert_eq!(
            worker.join().unwrap(),
            ["Runtime.evaluate", "Runtime.evaluate", "Runtime.evaluate"]
        );
    }

    #[test]
    fn readiness_timeout_is_terminal_not_a_retryable_mutation() {
        let (mut browser, worker) = fake(vec![
            json!({"result":{"value":{"ready":false,"waiting_for_menu":true}}}),
        ]);
        let error = browser
            .settle_after_input(&json!({"kind":"click"}), Duration::ZERO)
            .unwrap_err();
        assert!(error.downcast_ref::<Stale>().is_none());
        assert!(error.to_string().contains("menu did not become ready"));
        assert_eq!(worker.join().unwrap(), ["Runtime.evaluate"]);
    }

    #[test]
    #[ignore = "requires Chrome CDP on port 9222"]
    fn delayed_transparent_menu_is_waited_for_and_input_executes_once() {
        let html = r#"<title>Delayed menu</title><style>button,[role=option]{display:block;width:200px;height:50px}</style>
        <main id="main"><button role="combobox" aria-haspopup="listbox" aria-controls="placeholder" aria-expanded="false"
        onclick="window.clicks=(window.clicks||0)+1;this.setAttribute('aria-expanded','true');document.querySelector('#main').setAttribute('aria-hidden','true');document.querySelector('#menu').style.display='block';setTimeout(()=>document.querySelector('#menu').style.opacity='1',600)">Trip type</button></main>
        <span id="placeholder" role="listbox" hidden></span>
        <div id="menu" role="listbox" style="display:none;opacity:0"><button role="option" onclick="window.selected=true">One way</button></div>"#;
        let url = format!("data:text/html;base64,{}", STANDARD.encode(html));
        let mut browser =
            Browser::connect("http://127.0.0.1:9222", &url).unwrap();
        let page = browser.observe().unwrap();
        let action = page["actions"]
            .as_array()
            .unwrap()
            .iter()
            .find(|a| a["label"] == "Trip type")
            .unwrap()
            .clone();
        let started = Instant::now();
        browser.act(&action, &page, None).unwrap();
        let ready = browser.observe().unwrap();
        assert!(started.elapsed() >= Duration::from_millis(500));
        assert_eq!(browser.evaluate("window.clicks").unwrap(), 1);
        let option = ready["actions"]
            .as_array()
            .unwrap()
            .iter()
            .find(|a| a["label"] == "One way")
            .unwrap();
        browser.act(option, &ready, None).unwrap();
        assert_eq!(browser.evaluate("window.selected").unwrap(), true);
    }

    #[test]
    #[ignore = "requires Chrome CDP on port 9222"]
    fn keyboard_graph_exposes_labels_and_inspects_without_clicking() {
        let html = r#"<title>Chart</title><style>[role=region]{width:400px;height:200px}svg{width:300px;height:100px}</style>
        <div role="region" tabindex="0" aria-label="Fare chart" onclick="window.clicks=(window.clicks||0)+1"
          onkeydown="if(event.key==='ArrowRight'){document.querySelector('#value').textContent='October 21: R$3775';window.keys=(window.keys||0)+1}">
          <svg aria-hidden="true"><text x="10" y="30">October</text><text x="10" y="60" opacity="0">SECRET</text></svg>
          <p id="value">October 20: R$2720</p>
        </div><div aria-hidden="true"><svg><text x="10" y="30">HIDDEN</text></svg></div>"#;
        let url = format!("data:text/html;base64,{}", STANDARD.encode(html));
        let mut browser =
            Browser::connect("http://127.0.0.1:9222", &url).unwrap();
        let page = browser.observe().unwrap();
        assert_eq!(page["graphics"][0]["text"], "October");
        assert_eq!(page["graphics"].as_array().unwrap().len(), 1);
        let action = page["actions"]
            .as_array()
            .unwrap()
            .iter()
            .find(|a| a["key"] == "ArrowRight")
            .unwrap();
        browser.act(action, &page, None).unwrap();
        let next = browser.observe_effect(&page).unwrap();
        assert!(
            next["text"]
                .as_str()
                .unwrap()
                .contains("October 21: R$3775")
        );
        assert_ne!(next["fingerprint"], page["fingerprint"]);
        assert_eq!(browser.evaluate("window.keys").unwrap(), 1);
        assert_eq!(browser.evaluate("window.clicks || 0").unwrap(), 0);
        browser.evaluate("document.body.insertAdjacentHTML('beforeend', '<div id=cover style=\"position:fixed;inset:0;background:white;z-index:9999\"></div>')").unwrap();
        assert!(
            !browser.observe().unwrap()["actions"]
                .as_array()
                .unwrap()
                .iter()
                .any(|a| a["kind"] == "key")
        );
        assert!(
            browser
                .act(action, &page, None)
                .unwrap_err()
                .downcast_ref::<Stale>()
                .is_some()
        );
        assert_eq!(browser.evaluate("window.keys").unwrap(), 1);
        browser
            .evaluate("document.querySelector('#cover').remove()")
            .unwrap();
        browser.evaluate("document.querySelector('[role=region]').setAttribute('aria-disabled','true')").unwrap();
        assert!(
            !browser.observe().unwrap()["actions"]
                .as_array()
                .unwrap()
                .iter()
                .any(|a| a["kind"] == "key")
        );
    }

    #[test]
    fn interrupted_key_release_is_not_retried() {
        let (mut browser, worker) = fake(vec![
            fresh_result(),
            json!({"result":{"value":{"x":5,"y":5}}}),
            json!({}),
            json!({"protocol_error":true}),
        ]);
        let action =
            json!({"id":"e1","kind":"key","node":1,"key":"ArrowRight"});
        let error = browser.act(&action, &page(), None).unwrap_err();
        assert!(error.downcast_ref::<Stale>().is_none());
        assert_eq!(
            worker.join().unwrap(),
            [
                "Runtime.evaluate",
                "Runtime.evaluate",
                "Input.dispatchKeyEvent",
                "Input.dispatchKeyEvent"
            ]
        );
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
