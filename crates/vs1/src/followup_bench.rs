//! Opt-in experiments that must run alone on the GPU.
use std::time::Instant;

use serde_json::{Value, json};

use super::*;

fn model(device: Device) -> anyhow::Result<SystemOne> {
    Ok(SystemOne::from(crate::DEFAULT_REPO_ID)
        .with_device(device)
        .with_dtype(DType::BF16)
        .try_into()?)
}
fn snapshot(responses: &[SystemOneResponse]) -> Value {
    json!({"responses":responses,"actions":responses.iter().map(|r| r.answers.values().map(|a| a.action().map(|a|a.act_probability)).collect::<Vec<_>>()).collect::<Vec<_>>()})
}
fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    (v[(v.len() - 1) / 2] + v[v.len() / 2]) / 2.0
}
fn path(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../artifacts/inference-followups")
        .join(name)
}
#[test]
#[ignore = "requires CUDA, checkpoint and isolated timing"]
fn paired_batch_streams() -> anyhow::Result<()> {
    use std::sync::atomic::Ordering;
    let model: SystemOne = SystemOne::from(crate::DEFAULT_REPO_ID)
        .with_device(Device::new_cuda(0)?)
        .with_dtype(DType::BF16)
        .with_parallel_cuda_batches(true)
        .try_into()?;
    let devices = [
        model.device.clone(),
        model.batch_worker.as_ref().unwrap().model.device.clone(),
    ];
    let run = |parallel: bool, requests: &[SystemOneRequest]| {
        REFERENCE_BATCHES.store(!parallel, Ordering::Relaxed);
        model.system_one_batch(requests)
    };
    let cases: Vec<_> = super::batch_bench::cases()
        .into_iter()
        .filter(|(n, _)| {
            ["32", "64", "128", "mixed128", "shared128"].contains(&n.as_str())
        })
        .collect();
    let mut report = vec![];
    for (name, requests) in cases {
        let expected = snapshot(&run(false, &requests)?);
        for _ in 0..3 {
            assert_eq!(
                snapshot(&run(true, &requests)?),
                expected,
                "warmup {name}"
            );
        }
        let mut times = [vec![], vec![]];
        let mut ratios = vec![];
        let mut faster = 0;
        for iteration in 0..30 {
            let mut pair = [0.0; 2];
            for variant in if iteration % 2 == 0 { [0, 1] } else { [1, 0] } {
                for d in &devices {
                    d.synchronize()?;
                }
                let start = Instant::now();
                let output = if variant == 0 {
                    run(false, &requests)?
                } else {
                    run(true, &requests)?
                };
                for d in &devices {
                    d.synchronize()?;
                }
                pair[variant] = start.elapsed().as_secs_f64() * 1000.0;
                assert_eq!(
                    snapshot(&output),
                    expected,
                    "{name} iteration {iteration} variant {variant}"
                );
            }
            faster += usize::from(pair[1] < pair[0]);
            ratios.push(pair[1] / pair[0]);
            for i in 0..2 {
                times[i].push(pair[i]);
            }
        }
        let changed: Vec<_> = requests
            .iter()
            .map(|r| {
                serde_json::from_str(
                    &serde_json::to_string(r)
                        .unwrap()
                        .replace("changed", "removed"),
                )
                .unwrap()
            })
            .collect();
        let alt = snapshot(&run(false, &changed)?);
        for _ in 0..3 {
            assert_eq!(snapshot(&run(true, &changed)?), alt);
            assert_eq!(snapshot(&run(true, &requests)?), expected);
        }
        report.push(json!({"name":name,"baseline_p50_ms":median(&mut times[0]),"candidate_p50_ms":median(&mut times[1]),"paired_change_percent":100.0*(median(&mut ratios)-1.0),"faster_pairs":faster,"pairs":30,"outputs_exact":true,"alternates_exact":true,"samples_ms":times}));
        std::fs::write(
            path("02-batch-final.json"),
            serde_json::to_vec_pretty(&report)?,
        )?;
        eprintln!("{}", report.last().unwrap());
    }
    Ok(())
}

#[test]
#[ignore = "requires CUDA and checkpoint"]
fn batch_workers_preserve_default_outputs() -> anyhow::Result<()> {
    let mut cases = super::batch_bench::cases();
    let source = cases.iter().find(|(n, _)| n == "128").unwrap().1.clone();
    for n in [0, 31, 33, 63, 65, 97] {
        cases.push((format!("partial-{n}"), source[..n].to_vec()));
    }
    let invalid = vec![
        SystemOneRequest::new("state")
            .question("bad", Question::choice("invalid", [("one", "single")])),
    ];
    let original = model(Device::new_cuda(0)?)?;
    let expected: Vec<_> = cases
        .iter()
        .map(|(_, r)| snapshot(&original.system_one_batch(r).unwrap()))
        .collect();
    let error = original.system_one_batch(&invalid).unwrap_err().to_string();
    drop(original);
    let parallel: SystemOne = SystemOne::from(crate::DEFAULT_REPO_ID)
        .with_device(Device::new_cuda(0)?)
        .with_dtype(DType::BF16)
        .with_parallel_cuda_batches(true)
        .try_into()?;
    for ((name, requests), expected) in cases.iter().zip(&expected) {
        assert_eq!(
            &snapshot(&parallel.system_one_batch(requests)?),
            expected,
            "{name}"
        );
    }
    assert_eq!(
        parallel.system_one_batch(&invalid).unwrap_err().to_string(),
        error
    );
    // Concurrent callers must not race the worker streams or swap outputs.
    std::thread::scope(|scope| {
        let jobs: Vec<_> = cases
            .iter()
            .zip(&expected)
            .skip(2)
            .take(4)
            .map(|((_, r), e)| {
                let model = &parallel;
                scope.spawn(move || {
                    assert_eq!(
                        &snapshot(&model.system_one_batch(r).unwrap()),
                        e
                    )
                })
            })
            .collect();
        for job in jobs {
            job.join().unwrap();
        }
    });
    Ok(())
}
