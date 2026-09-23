//! Hosted Jev with explicit credentials, model selection and observable HTTP calls.
//! Responses are returned as supplied by the provider; probabilities and choices
//! are not silently normalized. No local fallback or weight download occurs.
use std::{
    sync::{
        OnceLock,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use crate::{Result, SystemOneError, SystemOneRequest, SystemOneResponse};

/// Cumulative counters. Snapshots taken during concurrent calls are approximate.
#[derive(Debug, Default, Clone, Copy, serde::Serialize)]
pub struct JevStats {
    /// Logical requests started (not batches).
    pub calls: usize,
    /// Actual HTTP attempts, including retries and failures.
    pub attempts: usize,
    pub retries: usize,
    pub successes: usize,
    /// Questions in logical requests, counted once regardless of retries.
    pub questions: usize,
}
/// Blocking hosted backend. The caller supplies the key and model explicitly.
/// Default: 16 concurrent requests, 30s timeout, up to three attempts for
/// HTTP 429/503/529. Authentication, connection and parse failures are not retried.
pub struct JevClient {
    client: reqwest::blocking::Client,
    authorization: reqwest::header::HeaderValue,
    endpoint: reqwest::Url,
    model: String,
    concurrency: usize,
    context_tokens: usize,
    tokenizer: OnceLock<Result<tokenizers::Tokenizer>>,
    calls: AtomicUsize,
    attempts: AtomicUsize,
    retries: AtomicUsize,
    successes: AtomicUsize,
    questions: AtomicUsize,
}
impl JevClient {
    pub fn new(
        key: impl Into<String>,
        model: impl Into<String>,
    ) -> Result<Self> {
        let key = key.into();
        let model = model.into();
        if key.trim().is_empty() || model.trim().is_empty() {
            return Err(SystemOneError::Config(
                "Jev requires a nonempty API key and model".into(),
            ));
        }
        let mut authorization =
            reqwest::header::HeaderValue::from_str(&format!("Bearer {key}"))
                .map_err(|_| {
                    SystemOneError::Config("invalid Jev API key".into())
                })?;
        authorization.set_sensitive(true);
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(30))
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|_| {
                SystemOneError::Remote("cannot create HTTP client".into())
            })?;
        Ok(Self {
            client,
            authorization,
            endpoint: reqwest::Url::parse(
                "https://api.typesafe.ai/v1/systemone",
            )
            .expect("static URL"),
            model,
            concurrency: 16,
            context_tokens: 8192,
            tokenizer: OnceLock::new(),
            calls: AtomicUsize::new(0),
            attempts: AtomicUsize::new(0),
            retries: AtomicUsize::new(0),
            successes: AtomicUsize::new(0),
            questions: AtomicUsize::new(0),
        })
    }
    /// Override the complete endpoint, for a caller-controlled gateway or testing.
    /// Redirects are disabled so credentials cannot be forwarded by a redirect.
    pub fn with_endpoint(mut self, endpoint: &str) -> Result<Self> {
        let url = reqwest::Url::parse(endpoint).map_err(|_| {
            SystemOneError::Config("invalid Jev endpoint".into())
        })?;
        if !["http", "https"].contains(&url.scheme())
            || !url.username().is_empty()
            || url.password().is_some()
        {
            return Err(SystemOneError::Config(
                "Jev endpoint must be HTTP(S) without embedded credentials"
                    .into(),
            ));
        }
        self.endpoint = url;
        Ok(self)
    }
    pub fn with_concurrency(mut self, n: usize) -> Result<Self> {
        if n == 0 {
            return Err(SystemOneError::Config(
                "Jev concurrency must be positive".into(),
            ));
        }
        self.concurrency = n;
        Ok(self)
    }
    pub fn model_name(&self) -> &str {
        &self.model
    }
    /// Sets vs1's prompt budget (default 8192), not a documented provider limit.
    pub fn with_context_tokens(mut self, n: usize) -> Result<Self> {
        if n == 0 {
            return Err(SystemOneError::Config(
                "Jev context tokens must be positive".into(),
            ));
        }
        self.context_tokens = n;
        Ok(self)
    }
    /// vs1's declared prompt budget, not a documented provider limit.
    pub fn context_tokens(&self) -> usize {
        self.context_tokens
    }
    /// Approximates each question's size with OpenJev's tokenizer, counting the
    /// state, instructions and candidate descriptions. The hosted tokenizer and
    /// prompt format are unknown. Loads only the tokenizer on the first check
    /// and keeps it on this client; no model weights are loaded.
    pub fn request_fits(&self, request: &SystemOneRequest) -> Result<bool> {
        let tokenizer = self
            .tokenizer
            .get_or_init(crate::openjev::load_tokenizer)
            .as_ref()
            .map_err(|error| SystemOneError::Tokenizer(error.to_string()))?;
        let state_tokens =
            tokenizer.encode(request.state.render(), false)?.len();
        for question in request.questions.values() {
            let mut tokens = state_tokens
                + tokenizer
                    .encode(question.instructions().render(), false)?
                    .len();
            for description in question.render_options() {
                tokens += tokenizer.encode(description, false)?.len();
            }
            if tokens > self.context_tokens {
                return Ok(false);
            }
        }
        Ok(true)
    }
    pub fn stats(&self) -> JevStats {
        JevStats {
            calls: self.calls.load(Ordering::Relaxed),
            attempts: self.attempts.load(Ordering::Relaxed),
            retries: self.retries.load(Ordering::Relaxed),
            successes: self.successes.load(Ordering::Relaxed),
            questions: self.questions.load(Ordering::Relaxed),
        }
    }
    /// The client's selected model overrides the request's optional model field.
    pub fn system_one(
        &self,
        request: &SystemOneRequest,
    ) -> Result<SystemOneResponse> {
        if request.questions.is_empty() {
            return Err(SystemOneError::Config(
                "Jev requires at least one question".into(),
            ));
        }
        let mut body = request.clone();
        body.model = Some(self.model.clone());
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.questions
            .fetch_add(request.questions.len(), Ordering::Relaxed);
        for attempt in 0..3 {
            self.attempts.fetch_add(1, Ordering::Relaxed);
            let response = self
                .client
                .post(self.endpoint.clone())
                .header(
                    reqwest::header::AUTHORIZATION,
                    self.authorization.clone(),
                )
                .json(&body)
                .send()
                .map_err(|_| {
                    SystemOneError::Remote(
                        "HTTP connection or timeout failure".into(),
                    )
                })?;
            let status = response.status().as_u16();
            if [429, 503, 529].contains(&status) && attempt < 2 {
                let delay = response
                    .headers()
                    .get("retry-after")
                    .and_then(|h| h.to_str().ok())
                    .and_then(|s| s.parse::<u64>().ok())
                    .unwrap_or(1 << attempt);
                // Refuse an excessively long backoff instead of retrying before the server allows.
                if delay > 60 {
                    return Err(SystemOneError::HttpStatus(status));
                }
                self.retries.fetch_add(1, Ordering::Relaxed);
                std::thread::sleep(Duration::from_secs(delay));
                continue;
            }
            if !response.status().is_success() {
                return Err(SystemOneError::HttpStatus(status));
            }
            let parsed: SystemOneResponse = response.json().map_err(|_| {
                SystemOneError::Remote("invalid response JSON".into())
            })?;
            if parsed.model.is_empty()
                || parsed.answers.len() != request.questions.len()
                || request.questions.iter().any(|(id, q)| {
                    !matches!(
                        (q, parsed.answers.get(id)),
                        (
                            crate::Question::Choice(_),
                            Some(crate::Answer::Choice(_))
                        ) | (
                            crate::Question::Score(_),
                            Some(crate::Answer::Score(_))
                        ) | (
                            crate::Question::Noul(_),
                            Some(crate::Answer::Noul(_))
                        )
                    )
                })
            {
                return Err(SystemOneError::Remote(
                    "response answers do not match request questions".into(),
                ));
            }
            self.successes.fetch_add(1, Ordering::Relaxed);
            return Ok(parsed);
        }
        unreachable!("last attempt returns")
    }
    /// Bounded concurrency; output order matches input order. No batch endpoint is assumed.
    pub fn system_one_batch(
        &self,
        requests: &[SystemOneRequest],
    ) -> Result<Vec<SystemOneResponse>> {
        let mut output = Vec::with_capacity(requests.len());
        for batch in requests.chunks(self.concurrency) {
            let responses = std::thread::scope(|scope| {
                let handles = batch
                    .iter()
                    .map(|r| scope.spawn(move || self.system_one(r)))
                    .collect::<Vec<_>>();
                handles
                    .into_iter()
                    .map(|h| {
                        h.join().map_err(|_| {
                            SystemOneError::Remote(
                                "HTTP worker panicked".into(),
                            )
                        })?
                    })
                    .collect::<Result<Vec<_>>>()
            })?;
            output.extend(responses);
        }
        Ok(output)
    }
}
#[cfg(test)]
mod tests {
    use std::{
        io::{Read, Write},
        net::TcpListener,
    };

    use serde_json::{Value, json};

    use super::*;
    fn server(
        statuses: Vec<u16>,
    ) -> (String, std::thread::JoinHandle<Vec<Value>>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url =
            format!("http://{}/v1/systemone", listener.local_addr().unwrap());
        let handle = std::thread::spawn(move || {
            let mut bodies = Vec::new();
            for status in statuses {
                let (mut socket, _) = listener.accept().unwrap();
                socket
                    .set_read_timeout(Some(std::time::Duration::from_secs(5)))
                    .unwrap();
                let mut bytes = Vec::new();
                let mut buf = [0; 4096];
                let (header, len) = loop {
                    let n = socket.read(&mut buf).unwrap();
                    assert!(n > 0);
                    bytes.extend_from_slice(&buf[..n]);
                    if let Some(i) =
                        bytes.windows(4).position(|s| s == b"\r\n\r\n")
                    {
                        let head = String::from_utf8_lossy(&bytes[..i])
                            .to_ascii_lowercase();
                        assert!(
                            head.contains("authorization: bearer test-key")
                        );
                        let len = head
                            .lines()
                            .find_map(|l| l.strip_prefix("content-length: "))
                            .unwrap()
                            .parse::<usize>()
                            .unwrap();
                        break (i + 4, len);
                    }
                };
                while bytes.len() < header + len {
                    let n = socket.read(&mut buf).unwrap();
                    assert!(n > 0);
                    bytes.extend_from_slice(&buf[..n]);
                }
                let request: Value =
                    serde_json::from_slice(&bytes[header..header + len])
                        .unwrap();
                let response = if status == 200 {
                    json!({"model":"jev-test","answers":{"q":{"type":"noul","noul":request["state"]["p"],"confidence":0.5}},"usage":{"input_tokens":7,"output_tokens":1}}).to_string()
                } else {
                    "private response body test-key".into()
                };
                bodies.push(request);
                write!(socket,"HTTP/1.1 {status} Test\r\nContent-Length: {}\r\nContent-Type: application/json\r\nRetry-After: 0\r\nConnection: close\r\n\r\n{response}",response.len()).unwrap();
            }
            bodies
        });
        (url, handle)
    }
    fn request(p: f64) -> SystemOneRequest {
        SystemOneRequest::new(json!({"p":p}))
            .question("q", crate::Question::noul("is it true?"))
    }
    #[test]
    fn caller_model_overrides_request_and_retry_counts_actual_calls() {
        let (url, server) = server(vec![429, 200]);
        let client = JevClient::new("test-key", "chosen-model")
            .unwrap()
            .with_endpoint(&url)
            .unwrap();
        let mut r = request(0.7);
        r.model = Some("request-model".into());
        let response = client.system_one(&r).unwrap();
        assert_eq!(response.model, "jev-test");
        let bodies = server.join().unwrap();
        assert_eq!(bodies[0]["model"], "chosen-model");
        assert_eq!(bodies[0]["state"], json!({"p":0.7}));
        assert_eq!(bodies[0], bodies[1]);
        let s = client.stats();
        assert_eq!(
            (s.calls, s.attempts, s.retries, s.successes, s.questions),
            (1, 2, 1, 1, 1)
        );
    }
    #[test]
    fn bounded_batch_preserves_input_order() {
        let (url, server) = server(vec![200; 3]);
        let client = JevClient::new("test-key", "chosen-model")
            .unwrap()
            .with_endpoint(&url)
            .unwrap()
            .with_concurrency(2)
            .unwrap();
        let rs = [request(0.1), request(0.2), request(0.3)];
        let answers = client.system_one_batch(&rs).unwrap();
        for (i, a) in answers.iter().enumerate() {
            let crate::Answer::Noul(a) = &a.answers["q"] else {
                panic!()
            };
            assert!((a.noul - (i + 1) as f32 / 10.0).abs() < 1e-6);
        }
        server.join().unwrap();
        assert_eq!(client.stats().attempts, 3);
        assert!(client.system_one_batch(&[]).unwrap().is_empty());
        assert_eq!(client.stats().attempts, 3);
    }
    #[test]
    fn auth_failure_is_not_retried_or_leaked() {
        let (url, server) = server(vec![401]);
        let client = JevClient::new("test-key", "chosen-model")
            .unwrap()
            .with_endpoint(&url)
            .unwrap();
        let err = client.system_one(&request(0.1)).unwrap_err().to_string();
        assert!(err.contains("401"));
        assert!(!err.contains("test-key"));
        assert!(!err.contains("private"));
        server.join().unwrap();
        assert_eq!(client.stats().attempts, 1);
    }
    #[test]
    fn invalid_config_is_rejected() {
        assert!(JevClient::new("", "model").is_err());
        assert!(JevClient::new("key", "").is_err());
        assert!(
            JevClient::new("key", "model")
                .unwrap()
                .with_concurrency(0)
                .is_err()
        );
        assert!(
            JevClient::new("key", "model")
                .unwrap()
                .with_context_tokens(0)
                .is_err()
        );
    }

    #[test]
    fn request_fits_counts_state_instructions_and_candidates_per_question() {
        use tokenizers::{
            Tokenizer,
            models::wordlevel::WordLevel,
            pre_tokenizers::whitespace::WhitespaceSplit,
        };
        let client = JevClient::new("key", "model").unwrap();
        assert_eq!(client.context_tokens(), 8192);
        assert!(client.tokenizer.get().is_none());
        let client = client.with_context_tokens(4).unwrap();
        let mut tokenizer = Tokenizer::new(
            WordLevel::builder()
                .vocab([("[UNK]".into(), 0)].into_iter().collect())
                .unk_token("[UNK]".into())
                .build()
                .unwrap(),
        );
        tokenizer.with_pre_tokenizer(Some(WhitespaceSplit));
        client.tokenizer.set(Ok(tokenizer)).unwrap();
        let question: crate::Question = serde_json::from_value(json!({
            "type": "choice", "instructions": "Choose", "criteria": ["yes", "no"]
        })).unwrap();
        let request = SystemOneRequest::new("state")
            .question("first", question.clone())
            .question("second", question);
        let model: crate::DecisionModel = client.into();
        assert_eq!(model.context_tokens(), 4);
        assert!(model.request_fits(&request).unwrap());
        let mut long_state = request.clone();
        long_state.state = "two words".into();
        assert!(!model.request_fits(&long_state).unwrap());
        for question in [
            crate::Question::choice(
                "Choose",
                [("a", "many words"), ("b", "x")],
            ),
            crate::Question::score("Rate", ["many words", "good"]),
            crate::Question::noul_with_criteria("Paid?", "many words", "no"),
        ] {
            assert!(
                !model
                    .request_fits(&request.clone().question("long", question))
                    .unwrap()
            );
        }
        let mut long_instructions = request.clone();
        let crate::Question::Choice(question) =
            &mut long_instructions.questions["second"]
        else {
            panic!()
        };
        question.instructions = "Choose carefully".into();
        assert!(!model.request_fits(&long_instructions).unwrap());
    }

    #[test]
    #[ignore = "downloads the OpenJev tokenizer only"]
    fn request_fits_loads_and_keeps_the_openjev_tokenizer() {
        let client = JevClient::new("key", "model")
            .unwrap()
            .with_context_tokens(64)
            .unwrap();
        assert!(client.request_fits(&request(0.7)).unwrap());
        let tokenizer = client.tokenizer.get().unwrap() as *const _;
        let mut long_request = request(0.7);
        long_request.state = "background ".repeat(64).into();
        assert!(!client.request_fits(&long_request).unwrap());
        assert_eq!(tokenizer, client.tokenizer.get().unwrap() as *const _);
        assert_eq!(client.stats().attempts, 0);
    }

    #[test]
    fn shared_backend_delegates_without_loading_local_weights() {
        let (url, server) = server(vec![200]);
        let model: crate::DecisionModel = JevClient::new("test-key", "chosen")
            .unwrap()
            .with_endpoint(&url)
            .unwrap()
            .into();
        assert!(model.local().is_none());
        assert_eq!(model.model_name(), "chosen");
        assert_eq!(model.system_one(&request(0.4)).unwrap().model, "jev-test");
        server.join().unwrap();
    }
    #[test]
    fn retries_are_bounded_and_schema_mismatches_fail() {
        let (url, handle) = server(vec![503; 3]);
        let client = JevClient::new("test-key", "chosen")
            .unwrap()
            .with_endpoint(&url)
            .unwrap();
        assert!(matches!(
            client.system_one(&request(0.4)),
            Err(SystemOneError::HttpStatus(503))
        ));
        handle.join().unwrap();
        assert_eq!(client.stats().attempts, 3);
        assert_eq!(client.stats().retries, 2);
        let (url, handle) = server(vec![200]);
        let client = JevClient::new("test-key", "chosen")
            .unwrap()
            .with_endpoint(&url)
            .unwrap();
        let r = SystemOneRequest::new(serde_json::json!({"p":0.4}))
            .question("different", crate::Question::noul("test"));
        assert!(client.system_one(&r).is_err());
        handle.join().unwrap();
        assert_eq!(client.stats().successes, 0);
    }
}
