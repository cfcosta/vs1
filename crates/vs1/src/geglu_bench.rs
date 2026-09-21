//! Opt-in paired model benchmark; run alone with --test-threads=1.
//! cargo test --release -p vs1 --features flash-attn paired_model_latency -- --ignored --nocapture --test-threads=1
use std::{
    sync::atomic::{AtomicBool, Ordering},
    time::Instant,
};

use candle_core::{DType, Device};
use serde_json::{Value, json};

use crate::{Question, SystemOne, SystemOneRequest, geglu_cuda::REFERENCE_MLP};

fn snapshot(responses: &[crate::SystemOneResponse]) -> Value {
    json!({"json": responses, "actions": responses.iter().map(|r| {
        r.answers.values().map(|a| a.action().map(|a| a.act_probability)).collect::<Vec<_>>()
    }).collect::<Vec<_>>()})
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    (values[(values.len() - 1) / 2] + values[values.len() / 2]) / 2.0
}

#[test]
#[ignore = "requires CUDA, the Laya checkpoint, and exclusive benchmark execution"]
fn paired_model_latency() -> anyhow::Result<()> {
    run_paired(&REFERENCE_MLP)
}

pub(crate) fn run_paired(reference: &'static AtomicBool) -> anyhow::Result<()> {
    run_paired_many(&[reference])
}

#[test]
#[ignore = "requires CUDA and checkpoint; run alone"]
fn paired_round3_latency() -> anyhow::Result<()> {
    run_paired_many(&[
        &crate::geglu_cuda::REFERENCE_VECTOR,
        &crate::rope_cuda::REFERENCE_ROPE,
        &crate::residual_norm_cuda::REFERENCE_NORM,
    ])
}

fn run_paired_many(references: &[&'static AtomicBool]) -> anyhow::Result<()> {
    struct Reset<'a>(&'a [&'static AtomicBool]);
    impl Drop for Reset<'_> {
        fn drop(&mut self) {
            for reference in self.0 {
                reference.store(false, Ordering::Relaxed);
            }
        }
    }
    let _reset = Reset(references);
    let select_reference = |value| {
        for reference in references {
            reference.store(value, Ordering::Relaxed);
        }
    };
    let device = Device::new_cuda(0)?;
    let model: SystemOne = SystemOne::from(crate::DEFAULT_REPO_ID)
        .with_device(device.clone())
        .with_dtype(DType::BF16)
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
    let cases = [
        ("one", vec![candidate(0, 5)]),
        ("thirty", (0..30).map(|i| candidate(i, 5)).collect()),
        (
            "mixed_lengths",
            (0..30)
                .map(|i| candidate(i, [1, 3, 9, 20][i % 4]))
                .collect(),
        ),
        (
            "browser_call3",
            vec![serde_json::from_str(include_str!(
                "../tests/fixtures/jev/call3_request.json"
            ))?],
        ),
        (
            "browser_call5",
            vec![serde_json::from_str(include_str!(
                "../tests/fixtures/jev/call5_request.json"
            ))?],
        ),
    ];
    let mut report = vec![];
    for (name, requests) in cases {
        select_reference(true);
        let expected = snapshot(&model.system_one_batch(&requests)?);
        for warmup in 0..10 {
            select_reference(warmup % 2 == 0);
            assert_eq!(snapshot(&model.system_one_batch(&requests)?), expected);
        }
        let (mut baseline, mut fused, mut ratios) = (vec![], vec![], vec![]);
        let mut faster = 0;
        for iteration in 0..60 {
            let mut pair = [0.0; 2];
            let order = if iteration % 2 == 0 { [0, 1] } else { [1, 0] };
            for index in order {
                select_reference(index == 0);
                device.synchronize()?;
                let started = Instant::now();
                let responses = model.system_one_batch(&requests)?;
                device.synchronize()?;
                pair[index] = started.elapsed().as_secs_f64() * 1000.0;
                assert_eq!(
                    snapshot(&responses),
                    expected,
                    "{name}, iteration {iteration}, variant {index}"
                );
            }
            faster += usize::from(pair[1] < pair[0]);
            baseline.push(pair[0]);
            fused.push(pair[1]);
            ratios.push(pair[1] / pair[0]);
        }
        report.push(json!({"name": name, "baseline_p50_ms": median(&mut baseline),
            "fused_p50_ms": median(&mut fused), "paired_change_percent": 100.0 * (median(&mut ratios) - 1.0),
            "faster_pairs": faster, "pairs": 60, "outputs_exact": true}));
        eprintln!("{}", report.last().unwrap());
    }
    eprintln!("PAIRED_REPORT={}", serde_json::to_string(&report)?);
    Ok(())
}

#[test]
#[ignore = "requires CUDA, checkpoint, isolated GPU"]
fn paired_deferred() -> anyhow::Result<()> {
    run_paired(&crate::modernbert::REFERENCE_DEFERRED)
}
