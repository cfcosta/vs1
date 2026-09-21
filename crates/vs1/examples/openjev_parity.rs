//! Compare native OpenJev predictions with the independent GLiClass reference.
//! cargo run --release -p vs1 --example openjev_parity -- CHECKPOINT REFERENCE cpu
use std::{collections::HashMap, time::Instant};

use anyhow::{Context, ensure};
use candle_core::{DType, Device};
use serde::Deserialize;
use vs1::{OpenJev, Question, SystemOneRequest, openjev::ABSTENTION_ID};

// Score/noul are continuous outputs. A rubric argmax is diagnostic, not the
// public score. Choice and abstention remain exact categorical checks.
fn semantic_error(
    q: &Question,
    expected: &HashMap<String, f32>,
    actual: &indexmap::IndexMap<String, f32>,
    selected: &str,
    actual_selected: &str,
) -> anyhow::Result<f32> {
    ensure!(
        (selected == ABSTENTION_ID) == (actual_selected == ABSTENTION_ID),
        "abstention changed"
    );
    if selected == ABSTENTION_ID {
        return Ok(0.);
    }
    let mass = |probs: &HashMap<String, f32>| {
        probs
            .iter()
            .filter(|(id, _)| id.as_str() != ABSTENTION_ID)
            .map(|(_, p)| p)
            .sum::<f32>()
    };
    let actual: HashMap<_, _> =
        actual.iter().map(|(k, v)| (k.clone(), *v)).collect();
    let (a, b) = (mass(expected), mass(&actual));
    Ok(match q {
        Question::Choice(_) => {
            ensure!(selected == actual_selected, "choice changed");
            0.
        }
        Question::Noul(_) => (expected["true"] / a - actual["true"] / b).abs(),
        Question::Score(q) => {
            let score = |p: &HashMap<String, f32>, total: f32| {
                (0..q.criteria.len())
                    .map(|i| i as f32 * p[&i.to_string()] / total)
                    .sum::<f32>()
            };
            (score(expected, a) - score(&actual, b)).abs()
                / (q.criteria.len() - 1) as f32
        }
    })
}

#[derive(Deserialize)]
struct Case {
    request: SystemOneRequest,
    prompt: String,
    ids: Vec<u32>,
    markers: Vec<usize>,
    logits: Vec<f32>,
    probabilities: HashMap<String, f32>,
    selected: String,
    temperature: f32,
}
fn main() -> anyhow::Result<()> {
    let args: Vec<_> = std::env::args().collect();
    ensure!(
        (4..=5).contains(&args.len()),
        "expected CHECKPOINT REFERENCE cpu|cuda [f32|bf16]"
    );
    let cases: Vec<Case> = serde_json::from_slice(&std::fs::read(&args[2])?)?;
    let device = match args[3].as_str() {
        "cpu" => Device::Cpu,
        #[cfg(feature = "cuda")]
        "cuda" => Device::new_cuda(0)?,
        _ => anyhow::bail!("unsupported device"),
    };
    let dtype = match args.get(4).map(String::as_str).unwrap_or("f32") {
        "f32" => DType::F32,
        "bf16" => DType::BF16,
        _ => anyhow::bail!("unsupported dtype"),
    };
    let model: OpenJev = OpenJev::from(&args[1])
        .with_device(device)
        .with_dtype(dtype)
        .try_into()?;
    let inputs = cases
        .iter()
        .map(|case| {
            let q = case
                .request
                .questions
                .get("decision")
                .context("missing question")?;
            let input = model.build_input(
                &case.request.state.render(),
                "decision",
                q,
            )?;
            ensure!(input.prompt == case.prompt, "prompt mismatch");
            ensure!(input.ids == case.ids, "token IDs mismatch");
            ensure!(input.markers == case.markers, "markers mismatch");
            Ok(input)
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    ensure!(model.predict(&[])?.is_empty(), "empty native batch");
    ensure!(
        model.system_one_batch(&[])?.is_empty(),
        "empty request batch"
    );
    let oversized = vs1::Question::choice(
        "Choose",
        [("a", "label ".repeat(600)), ("b", "other".into())],
    );
    ensure!(
        model.build_input("state", "oversized", &oversized).is_err(),
        "truncated label header accepted"
    );
    let mut invalid = inputs[0].clone();
    invalid.ids[0] = u32::MAX;
    ensure!(
        model.predict(&[invalid]).is_err(),
        "out-of-range token accepted"
    );
    // Compare mixed-length/cardinality batching and singleton execution.
    let batch = model.predict(&inputs)?;
    let mut max_logit_error = 0f32;
    let mut max_probability_error = 0f32;
    let mut max_batch_error = 0f32;
    let mut agreement = 0;
    let mut batch_agreement = 0;
    let mut details = Vec::new();
    let mut max_semantic_error = 0f32;
    for ((input, actual), expected) in inputs.iter().zip(&batch).zip(&cases) {
        let single = model.predict(std::slice::from_ref(input))?.remove(0);
        details.push(serde_json::json!({"request":expected.request,"reference":expected.probabilities,
            "batch":actual,"single":single}));
        let q = &expected.request.questions["decision"];
        for output in [actual, &single] {
            max_semantic_error = max_semantic_error.max(semantic_error(
                q,
                &expected.probabilities,
                &output.probabilities,
                &expected.selected,
                &output.selected,
            )?);
            for (id, p) in &output.probabilities {
                max_probability_error = max_probability_error
                    .max((p - expected.probabilities[id]).abs());
            }
        }
        let batch_probs = actual
            .probabilities
            .iter()
            .map(|(id, p)| (id.clone(), *p))
            .collect();
        max_semantic_error = max_semantic_error.max(semantic_error(
            q,
            &batch_probs,
            &single.probabilities,
            &actual.selected,
            &single.selected,
        )?);
        ensure!(
            actual.temperature == expected.temperature,
            "temperature mismatch"
        );
        if actual.selected == expected.selected {
            agreement += 1;
        }
        for (x, y) in actual.logits.iter().zip(&expected.logits) {
            max_logit_error = max_logit_error.max((x - y).abs());
        }
        for (id, p) in &actual.probabilities {
            max_probability_error = max_probability_error
                .max((p - expected.probabilities[id]).abs());
            max_batch_error =
                max_batch_error.max((p - single.probabilities[id]).abs());
        }
        if actual.selected == single.selected {
            batch_agreement += 1;
        } else {
            eprintln!(
                "batch decision: {} vs singleton {} for {}",
                actual.selected, single.selected, input.prompt
            );
        }
    }
    let requests: Vec<_> = cases.iter().map(|x| x.request.clone()).collect();
    let responses = model.system_one_batch(&requests)?;
    for (response, native) in responses.iter().zip(&batch) {
        let answer = &response.answers["decision"];
        ensure!(
            answer.abstention().is_some() == native.abstained,
            "abstention adapter mismatch"
        );
        ensure!(
            answer.action().is_none(),
            "OpenJev must not invent a Laya action"
        );
    }
    // Warm latency for the first (short, two-option) case, excluding loading.
    let mut timings = Vec::new();
    for _ in 0..20 {
        let start = Instant::now();
        model.predict(&inputs[..1])?;
        timings.push(start.elapsed().as_secs_f64() * 1000.);
    }
    timings.sort_by(f64::total_cmp);
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "device":args[3], "dtype":format!("{dtype:?}"), "cases":cases.len(), "decisions_agree":agreement,
            "max_logit_error":max_logit_error, "max_probability_error":max_probability_error,
            "max_batch_probability_error":max_batch_error, "short_warm_median_ms":timings[10],
            "batch_decisions_agree":batch_agreement,
            "short_tokens":inputs[0].ids.len(),
            "details":details,
            "max_normalized_score_or_noul_error":max_semantic_error,
        }))?
    );
    if dtype == DType::F32 {
        ensure!(agreement == cases.len(), "decision mismatch");
        ensure!(batch_agreement == cases.len(), "batch changes decision");
    }
    let tolerance = if dtype == DType::F32 { 0.0002 } else { 0.02 };
    let batch_tolerance = if dtype == DType::F32 { 0.0002 } else { 0.03 };
    let semantic_tolerance = if dtype == DType::F32 { 0.0002 } else { 0.01 };
    ensure!(
        max_semantic_error < semantic_tolerance,
        "numeric answer error exceeds {semantic_tolerance}"
    );
    ensure!(
        max_probability_error < tolerance,
        "probability error exceeds {tolerance}"
    );
    ensure!(
        max_batch_error < batch_tolerance,
        "batch probability error exceeds {batch_tolerance}"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn score_argmax_flip_is_not_a_categorical_answer_change() {
        let expected = HashMap::from([
            ("0".into(), 0.31),
            ("1".into(), 0.30),
            ("2".into(), 0.19),
            (ABSTENTION_ID.into(), 0.20),
        ]);
        let actual = indexmap::IndexMap::from([
            ("0".into(), 0.30),
            ("1".into(), 0.31),
            ("2".into(), 0.19),
            (ABSTENTION_ID.into(), 0.20),
        ]);
        let score = Question::score("rate", ["low", "medium", "high"]);
        assert!(
            semantic_error(&score, &expected, &actual, "0", "1").unwrap()
                < 0.01
        );
        let choice = Question::choice(
            "pick",
            [("0", "low"), ("1", "medium"), ("2", "high")],
        );
        assert!(semantic_error(&choice, &expected, &actual, "0", "1").is_err());
        assert!(
            semantic_error(&score, &expected, &actual, "0", ABSTENTION_ID)
                .is_err()
        );
    }
}
