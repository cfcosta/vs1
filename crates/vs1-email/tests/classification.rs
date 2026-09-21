use serde_json::json;
use vs1::{State, SystemOneResponse};
use vs1_email::{Config, Email, classification_request, classify};

fn config() -> Config {
    Config::parse(
        r#"
[owner]
capture_venues = ["Nubank"]
[[rules]]
category = "capture"
what = "Periodic statement"
not_for = "Single purchase"
examples = ["Extrato"]
[[rules]]
category = "other"
what = "None of the above"
"#,
    )
    .unwrap()
}

fn email() -> Email {
    Email {
        uid: 42,
        message_id: "<a@example.test>".into(),
        from: "bank@example.test".into(),
        to: "owner@example.test".into(),
        subject: "Extrato".into(),
        date: "Today".into(),
        body: "Balance: 10".into(),
    }
}

fn response() -> SystemOneResponse {
    serde_json::from_value(json!({
        "model": "laya", "usage": {"input_tokens": 123, "output_tokens": 0},
        "answers": {"category": {"type":"choice", "choice":"capture", "confidence":0.6,
        "probabilities":{"capture":0.9,"other":0.1}}}
    })).unwrap()
}

#[test]
fn request_preserves_context_and_all_criteria_in_order() {
    let request = classification_request(&config(), &email());
    let State::Json(state) = &request.state else {
        panic!("expected JSON state")
    };
    assert_eq!(state["owner"]["capture_venues"], json!(["Nubank"]));
    assert_eq!(state["email"]["subject"], "Extrato");
    assert_eq!(state["email"]["body"], "Balance: 10");
    let question = &request.questions["category"];
    let options = question.render_options();
    assert_eq!(options, ["capture", "other"]);
    assert_eq!(state["criteria"]["capture"]["not_for"], "Single purchase");
    assert_eq!(state["criteria"]["capture"]["examples"], json!(["Extrato"]));
    assert_eq!(
        state
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        ["criteria", "owner", "email"]
    );
}

#[test]
fn classification_uses_model_and_reports_identity_and_probabilities() {
    let mut calls = 0;
    let result = classify(
        &config(),
        &email(),
        &mut |request: &vs1::SystemOneRequest| {
            calls += 1;
            assert!(request.questions.contains_key("category"));
            Ok(response())
        },
    )
    .unwrap();
    assert_eq!(calls, 1);
    let value = serde_json::to_value(result).unwrap();
    assert_eq!(value["uid"], 42);
    assert_eq!(value["message_id"], "<a@example.test>");
    assert_eq!(value["category"], "capture");
    assert_eq!(value["model"], "laya");
    assert!(value["probabilities"]["capture"].as_f64().unwrap() > 0.89);
    assert_eq!(value["usage"]["input_tokens"], 123);
}

#[test]
fn rejects_missing_wrong_unknown_or_invalid_model_answers() {
    let base = serde_json::to_value(response()).unwrap();
    let mut cases = Vec::new();
    let mut value = base.clone();
    value["answers"] = json!({});
    cases.push(value);
    let mut value = base.clone();
    value["answers"]["category"] = json!({"type":"noul", "noul":0.8});
    cases.push(value);
    let mut value = base.clone();
    value["answers"]["category"]["choice"] = json!("unknown");
    cases.push(value);
    let mut value = base.clone();
    value["answers"]["category"]["probabilities"] = json!({"capture":1.0});
    cases.push(value);
    let mut value = base.clone();
    value["answers"]["category"]["confidence"] = json!(2.0);
    cases.push(value);
    let mut value = base.clone();
    value["answers"]["category"]["probabilities"] =
        json!({"capture":-0.1,"other":1.1});
    cases.push(value);
    let mut value = base.clone();
    value["answers"]["category"]["probabilities"] =
        json!({"capture":0.1,"other":0.1});
    cases.push(value);
    let mut value = base;
    value["answers"]["category"]["choice"] = json!("other");
    cases.push(value);
    for value in cases {
        let response: SystemOneResponse =
            serde_json::from_value(value).unwrap();
        assert!(
            classify(&config(), &email(), &mut |_| Ok(response.clone()))
                .is_err()
        );
    }
    assert!(
        classify(&config(), &email(), &mut |_| anyhow::bail!(
            "inference failed"
        ))
        .is_err()
    );
}
