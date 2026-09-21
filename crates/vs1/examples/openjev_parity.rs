//! Compare native OpenJev predictions with the independent GLiClass reference.
//! cargo run --release -p vs1 --example openjev_parity -- CHECKPOINT REFERENCE cpu
use std::{collections::HashMap, time::Instant};

use anyhow::{Context, ensure};
use candle_core::{DType, Device};
use serde::Deserialize;
use vs1::{OpenJev, SystemOneRequest};

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
    ensure!(args.len() == 4, "expected CHECKPOINT REFERENCE cpu|cuda");
    let cases: Vec<Case> = serde_json::from_slice(&std::fs::read(&args[2])?)?;
    let device = match args[3].as_str() {
        "cpu" => Device::Cpu,
        #[cfg(feature = "cuda")]
        "cuda" => Device::new_cuda(0)?,
        _ => anyhow::bail!("unsupported device"),
    };
    let dtype = DType::F32;
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
    for ((input, actual), expected) in inputs.iter().zip(&batch).zip(&cases) {
        let single = model.predict(std::slice::from_ref(input))?.remove(0);
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
            "device":args[3], "dtype":"f32", "cases":cases.len(), "decisions_agree":agreement,
            "max_logit_error":max_logit_error, "max_probability_error":max_probability_error,
            "max_batch_probability_error":max_batch_error, "short_warm_median_ms":timings[10],
            "batch_decisions_agree":batch_agreement,
            "short_tokens":inputs[0].ids.len(),
        }))?
    );
    ensure!(agreement == cases.len(), "decision mismatch");
    ensure!(batch_agreement == cases.len(), "batch changes decision");
    let tolerance = 0.0002;
    ensure!(
        max_probability_error < tolerance,
        "probability error exceeds {tolerance}"
    );
    ensure!(
        max_batch_error < tolerance,
        "batch probability error exceeds {tolerance}"
    );
    Ok(())
}
