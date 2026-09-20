//! The wire format is TypeSafe's `systemone` API, field for field.
//!
//! `tests/fixtures/jev/` holds requests and responses captured from
//! `api.typesafe.ai/v1/systemone` (model `jev-1.13.0`) by
//! jev-ultrafast on a Google Flights run: one small three-option menu
//! and one page with 24 elements and three question heads. They only
//! ever ask `choice` questions, so the score and noul shapes come from
//! the documented examples at <https://docs.typesafe.ai/api>.

use serde_json::{Value, json};
use vs1::{
    Action,
    Answer,
    ChoiceAnswer,
    Description,
    NoulAnswer,
    Question,
    ScoreAnswer,
    SystemOneRequest,
    SystemOneResponse,
};

fn fixture(name: &str) -> Value {
    let path =
        format!("{}/tests/fixtures/jev/{name}", env!("CARGO_MANIFEST_DIR"));
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn sorted_keys(value: &Value) -> Vec<&str> {
    let mut keys: Vec<&str> = value
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    keys.sort_unstable();
    keys
}

/// Same keys at every level, same nesting, and numbers equal to within
/// f32 precision (TypeSafe prints f64, this crate stores f32). Key
/// order is not compared: TypeSafe returns `probabilities` in no
/// particular order.
fn assert_same_shape(actual: &Value, expected: &Value, path: &str) {
    match (actual, expected) {
        (Value::Object(a), Value::Object(e)) => {
            assert_eq!(
                sorted_keys(actual),
                sorted_keys(expected),
                "keys differ at {path}"
            );
            for (key, value) in e {
                assert_same_shape(&a[key], value, &format!("{path}/{key}"));
            }
        }
        (Value::Array(a), Value::Array(e)) => {
            assert_eq!(a.len(), e.len(), "length differs at {path}");
            for (i, (x, y)) in a.iter().zip(e).enumerate() {
                assert_same_shape(x, y, &format!("{path}[{i}]"));
            }
        }
        (Value::Number(a), Value::Number(e)) => {
            let (a, e) = (a.as_f64().unwrap(), e.as_f64().unwrap());
            assert!(
                (a - e).abs() <= 1e-6 * e.abs().max(1.0),
                "{path}: {a} != {e}"
            );
        }
        _ => assert_eq!(actual, expected, "value differs at {path}"),
    }
}

#[test]
fn captured_requests_parse_and_serialise_back_unchanged() {
    for name in ["call3_request.json", "call5_request.json"] {
        let raw = fixture(name);
        let request: SystemOneRequest =
            serde_json::from_value(raw.clone()).unwrap();
        assert_eq!(request.model.as_deref(), Some("jev-latest"));
        let mut ids: Vec<&str> =
            request.questions.keys().map(String::as_str).collect();
        ids.sort_unstable();
        assert_eq!(ids, sorted_keys(&raw["questions"]), "{name}: question ids");
        for (id, question) in &request.questions {
            let Question::Choice(q) = question else {
                panic!("{name}: {id} is a choice question in the capture");
            };
            let mut labels = q.criteria.labels();
            labels.sort_unstable();
            assert_eq!(
                labels,
                sorted_keys(&raw["questions"][id]["criteria"]),
                "{name}: {id} options"
            );
        }
        let back = serde_json::to_value(&request).unwrap();
        assert_same_shape(&back, &raw, name);
    }
}

#[test]
fn captured_responses_parse_and_serialise_back_unchanged() {
    for (request, response) in [
        ("call3_request.json", "call3_response.json"),
        ("call5_request.json", "call5_response.json"),
    ] {
        let raw = fixture(response);
        let parsed: SystemOneResponse =
            serde_json::from_value(raw.clone()).unwrap();
        let request: SystemOneRequest =
            serde_json::from_value(fixture(request)).unwrap();
        assert_eq!(parsed.model, "jev-1.13.0");
        assert!(parsed.usage.output_tokens > 0);

        // One answer per question, under the same ids, over exactly
        // the question's options.
        let mut want: Vec<&str> =
            request.questions.keys().map(String::as_str).collect();
        want.sort_unstable();
        let mut got: Vec<&str> =
            parsed.answers.keys().map(String::as_str).collect();
        got.sort_unstable();
        assert_eq!(got, want, "{response}: answer ids");
        for (id, answer) in &parsed.answers {
            let Answer::Choice(a) = answer else {
                panic!("{response}: {id} is a choice answer in the capture");
            };
            assert!(a.action.is_none(), "action is not on the wire");
            let Question::Choice(q) = &request.questions[id] else {
                unreachable!()
            };
            let mut labels = q.criteria.labels();
            labels.sort_unstable();
            let mut keys: Vec<&str> =
                a.probabilities.keys().map(String::as_str).collect();
            keys.sort_unstable();
            assert_eq!(keys, labels, "{response}: {id} probabilities");
            assert!(labels.contains(&a.choice.as_str()));
        }

        let back = serde_json::to_value(&parsed).unwrap();
        assert_same_shape(&back, &raw, response);
    }
}

#[test]
fn documented_score_and_noul_answers_parse_and_serialise_back_unchanged() {
    // The response examples from https://docs.typesafe.ai/api plus the
    // structured-legend example from /primitives/score.
    let documented = json!({
        "model": "jev-latest",
        "answers": {
            "is_urgent": { "type": "noul", "noul": 0.92 },
            "department": {
                "type": "choice",
                "choice": "technical",
                "probabilities": { "billing": 0.08, "technical": 0.85, "sales": 0.07 },
                "confidence": 0.82
            },
            "frustration": {
                "type": "score",
                "score": 1.6,
                "legend": { "0": "Calm", "1": "Frustrated", "2": "Very angry" },
                "probabilities": { "0": 0.05, "1": 0.3, "2": 0.65 },
                "confidence": 0.78
            },
            "severity": {
                "type": "score",
                "score": 1.06,
                "confidence": 0.91,
                "legend": {
                    "0": {
                        "what": "Cosmetic; no impact to functionality",
                        "examples": ["typo in a label", "misaligned icon"]
                    },
                    "1": {
                        "what": "Broken or degraded feature, but workaround exists",
                        "examples": ["export fails in one browser"]
                    }
                },
                "probabilities": { "0": 0.94, "1": 0.06 }
            }
        },
        "usage": { "input_tokens": 312, "output_tokens": 48 }
    });
    let response: SystemOneResponse =
        serde_json::from_value(documented.clone()).unwrap();
    let Answer::Noul(noul) = &response.answers["is_urgent"] else {
        panic!("noul")
    };
    assert_eq!(noul.noul, 0.92);
    let Answer::Score(severity) = &response.answers["severity"] else {
        panic!("score")
    };
    assert!(
        matches!(severity.legend["0"], Description::Json(_)),
        "structured levels stay structured"
    );
    let back = serde_json::to_value(&response).unwrap();
    assert_same_shape(&back, &documented, "documented");
}

#[test]
fn answers_carry_exactly_the_documented_keys() {
    let action = Some(Action {
        act_probability: 0.5,
    });
    let choice = Answer::Choice(ChoiceAnswer {
        choice: "a".into(),
        probabilities: [("a".to_string(), 0.9), ("b".to_string(), 0.1)]
            .into_iter()
            .collect(),
        confidence: 0.6,
        action,
    });
    let score = Answer::Score(ScoreAnswer {
        score: 0.0,
        legend: [("0".to_string(), Description::from("low"))]
            .into_iter()
            .collect(),
        probabilities: [("0".to_string(), 1.0)].into_iter().collect(),
        confidence: 1.0,
        action,
    });
    let noul = Answer::Noul(NoulAnswer { noul: 0.3, action });
    let keys = |answer: &Answer| {
        sorted_keys(&serde_json::to_value(answer).unwrap())
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>()
    };
    assert_eq!(
        keys(&choice),
        ["choice", "confidence", "probabilities", "type"]
    );
    assert_eq!(
        keys(&score),
        ["confidence", "legend", "probabilities", "score", "type"]
    );
    assert_eq!(keys(&noul), ["noul", "type"]);
}

#[test]
fn requests_may_leave_instructions_out() {
    // TypeSafe accepts `instructions` as a string, object, array, or
    // null, and its SDK defaults it to null.
    let request: SystemOneRequest = serde_json::from_value(json!({
        "model": "jev-latest",
        "state": "x",
        "questions": {
            "a": { "type": "noul" },
            "b": { "type": "choice", "instructions": null, "criteria": { "x": null, "y": null } },
            "c": { "type": "score", "criteria": ["lo", "hi"] }
        }
    }))
    .unwrap();
    assert_eq!(request.questions.len(), 3);
    for question in request.questions.values() {
        assert_eq!(question.instructions().render(), "null");
    }
}
