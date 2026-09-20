//! Repeatable end-to-end CUDA timing and exact-output regression corpus.
//! cargo run --release -p vs1 --features flash-attn --example cuda_bench -- OUT.json [ITERATIONS] [BASELINE.json]
use std::{hint::black_box, time::Instant};

use anyhow::{Result, ensure};
use candle_core::{DType, Device};
use serde_json::{Value, json};
use vs1::{Question, SystemOne, SystemOneRequest};

fn snapshot(responses: &[vs1::SystemOneResponse]) -> Result<Value> {
    Ok(
        json!({"json":responses,"action_probabilities":responses.iter().map(|r|
        r.answers.iter().map(|(id,a)| (id.clone(), a.action().map(|a| a.act_probability))).collect::<Vec<_>>()
    ).collect::<Vec<_>>()}),
    )
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    ensure!(args.len() >= 2, "output path required");
    let iterations: usize =
        args.get(2).map(|s| s.parse()).transpose()?.unwrap_or(30);
    ensure!(iterations >= 2, "at least two iterations required");
    #[cfg(feature = "cuda")]
    let device = Device::new_cuda(0)?;
    #[cfg(not(feature = "cuda"))]
    let device = Device::Cpu;
    let dtype = match std::env::var("VS1_BENCH_DTYPE").as_deref() {
        Ok("f32") => DType::F32,
        _ if device.is_cuda() => DType::BF16,
        _ => DType::F32,
    };
    let model: SystemOne = SystemOne::from(vs1::DEFAULT_REPO_ID)
        .with_device(device.clone())
        .with_dtype(dtype)
        .try_into()?;
    let paragraph = "The indexing pipeline compares content hashes and updates changed documents. Unchanged files are skipped. ";
    let candidate = |i: usize, repeats: usize| {
        SystemOneRequest::new(format!(
            "Document {i}. {}",
            paragraph.repeat(repeats)
        ))
        .question(
            "relevant",
            Question::noul("Does this explain how changed files are selected?"),
        )
    };
    let triage = SystemOneRequest::new(paragraph.repeat(5))
        .question("relevance", Question::noul("Does this discuss indexing?"))
        .question(
            "topic",
            Question::choice(
                "Choose the topic",
                [
                    ("index", "indexing"),
                    ("travel", "travel"),
                    ("food", "food"),
                ],
            ),
        )
        .question(
            "detail",
            Question::score("How detailed?", ["brief", "medium", "detailed"]),
        )
        .question("pdf", Question::noul("Does it mention PDF files?"))
        .question("hash", Question::noul("Does it mention hashes?"));
    let call3 = serde_json::from_str(include_str!(
        "../tests/fixtures/jev/call3_request.json"
    ))?;
    let call5 = serde_json::from_str(include_str!(
        "../tests/fixtures/jev/call5_request.json"
    ))?;
    let mut cases = vec![
        ("one", vec![candidate(0, 5)]),
        ("five", vec![triage]),
        ("thirty", (0..30).map(|i| candidate(i, 5)).collect()),
        (
            "mixed_lengths",
            (0..30)
                .map(|i| candidate(i, [1, 3, 9, 20][i % 4]))
                .collect(),
        ),
        ("browser_call3", vec![call3]),
        ("browser_call5", vec![call5]),
    ];
    if std::env::var_os("VS1_BENCH_LARGE").is_some() {
        for (name, n) in [("eight", 8), ("32", 32), ("64", 64), ("128", 128)] {
            cases.push((name, (0..n).map(|i| candidate(i, 5)).collect()));
        }
        cases.push((
            "mixed128",
            (0..128)
                .map(|i| candidate(i, [1, 3, 9, 20][i % 4]))
                .collect(),
        ));
        let mut shared = SystemOneRequest::new(paragraph.repeat(5));
        for i in 0..128 {
            shared = shared.question(format!("q{i}"), Question::noul(
                format!("Question {i}: Does this explain how changed files are selected?")));
        }
        cases.push(("shared128", vec![shared]));
    }
    let mut results = vec![];
    for (name, requests) in &cases {
        let questions: usize = requests.iter().map(|r| r.questions.len()).sum();
        let mut lengths = vec![];
        for request in requests {
            let state = model.encode_state(&request.state)?;
            for (id, question) in &request.questions {
                lengths.push(
                    model.build_sequence(&state, id, question)?.ids.len(),
                );
            }
        }
        let first = Instant::now();
        black_box(model.system_one_batch(requests)?);
        device.synchronize()?;
        let first_call_ms = first.elapsed().as_secs_f64() * 1000.0;
        for _ in 0..5 {
            black_box(model.system_one_batch(requests)?);
        }
        let expected = snapshot(&model.system_one_batch(requests)?)?;
        let mut times = vec![];
        for _ in 0..iterations {
            device.synchronize()?;
            let started = Instant::now();
            let actual = model.system_one_batch(black_box(requests))?;
            device.synchronize()?;
            times.push(started.elapsed().as_secs_f64() * 1000.0);
            ensure!(
                snapshot(&actual)? == expected,
                "nondeterministic output for {name}"
            );
        }
        times.sort_by(f64::total_cmp);
        let p50 = (times[(iterations - 1) / 2] + times[iterations / 2]) / 2.0;
        let p95 = times[((iterations as f64 * 0.95).ceil() as usize - 1)
            .min(iterations - 1)];
        eprintln!(
            "{name}: p50={p50:.3}ms p95={p95:.3}ms questions/s={:.1}",
            questions as f64 * 1000.0 / p50
        );
        let changed: Vec<SystemOneRequest> = requests
            .iter()
            .map(|r| {
                serde_json::from_str(
                    &serde_json::to_string(r)
                        .unwrap()
                        .replace("changed", "removed")
                        .replace("One way", "Two way"),
                )
            })
            .collect::<std::result::Result<_, _>>()?;
        let alternate_outputs = snapshot(&model.system_one_batch(&changed)?)?;
        for _ in 0..3 {
            ensure!(
                snapshot(&model.system_one_batch(requests)?)? == expected,
                "original changed after replay"
            );
            ensure!(
                snapshot(&model.system_one_batch(&changed)?)?
                    == alternate_outputs,
                "changed input replay drift"
            );
        }
        results.push(json!({"name":name,"first_call_ms":first_call_ms,"alternate_outputs":alternate_outputs,"questions":questions,"lengths":lengths,"p50_ms":p50,"p95_ms":p95,
            "questions_per_second":questions as f64 * 1000.0 / p50,"latencies_ms":times,"outputs":expected}));
        // Keep completed cases if a later case fails (e.g. capture support).
        std::fs::write(
            &args[1],
            serde_json::to_vec_pretty(&json!({"dtype":format!("{dtype:?}"),
            "flash_attn":cfg!(feature="flash-attn"),"iterations":iterations,"cases":results}))?,
        )?;
    }
    let mut churn = vec![];
    for _ in 0..5 {
        let started = Instant::now();
        for (i, (_, reqs)) in cases.iter().enumerate() {
            ensure!(
                snapshot(&model.system_one_batch(reqs)?)?
                    == results[i]["outputs"],
                "shape-change output drift"
            );
        }
        device.synchronize()?;
        churn.push(started.elapsed().as_secs_f64() * 1000.0);
    }
    let mut output: Value = serde_json::from_slice(&std::fs::read(&args[1])?)?;
    output["shape_churn_ms"] = json!(churn);
    std::fs::write(&args[1], serde_json::to_vec_pretty(&output)?)?;
    if let Some(reference) = args.get(3) {
        let previous: Value =
            serde_json::from_slice(&std::fs::read(reference)?)?;
        let current: Value = serde_json::from_slice(&std::fs::read(&args[1])?)?;
        ensure!(
            previous["dtype"] == current["dtype"]
                && previous["flash_attn"] == current["flash_attn"],
            "benchmark configuration changed"
        );
        ensure!(
            previous["cases"].as_array().unwrap().len()
                == current["cases"].as_array().unwrap().len(),
            "case count changed"
        );
        for (a, b) in previous["cases"]
            .as_array()
            .unwrap()
            .iter()
            .zip(current["cases"].as_array().unwrap())
        {
            ensure!(
                a["name"] == b["name"]
                    && a["lengths"] == b["lengths"]
                    && a["outputs"] == b["outputs"]
                    && a["alternate_outputs"] == b["alternate_outputs"],
                "output drift: {}",
                a["name"]
            );
        }
    }
    Ok(())
}
