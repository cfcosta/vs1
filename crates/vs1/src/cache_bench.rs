//! Opt-in exact-cache benchmarks. Run alone with one test thread.
use std::{sync::atomic::Ordering, time::Instant};

use serde_json::{Value, json};

use super::*;
fn model() -> anyhow::Result<SystemOne> {
    Ok(SystemOne::from(crate::DEFAULT_REPO_ID)
        .with_device(Device::new_cuda(0)?)
        .with_dtype(DType::BF16)
        .with_result_cache_capacity(32)
        .try_into()?)
}
fn path(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../artifacts/inference-followups")
        .join(name)
}
fn clear(model: &SystemOne) {
    model.result_cache.as_ref().unwrap().lock().unwrap().clear();
}
fn snapshot(r: Result<Vec<SystemOneResponse>>) -> Value {
    match r {
        Ok(r) => {
            json!({"responses":r,"actions":r.iter().map(|r|r.answers.values().map(|a|a.action().map(|a|a.act_probability)).collect::<Vec<_>>()).collect::<Vec<_>>()})
        }
        Err(e) => json!({"error":e.to_string()}),
    }
}
fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    (v[(v.len() - 1) / 2] + v[v.len() / 2]) / 2.0
}
#[test]
#[ignore = "requires CUDA, checkpoint and saved browser trace corpus"]
fn paired_cache_replay() -> anyhow::Result<()> {
    let model = model()?;
    let corpus: Vec<Value> =
        serde_json::from_slice(&std::fs::read(path("replay-inputs.json"))?)?;
    let sessions: Vec<Vec<SystemOneRequest>> = corpus
        .iter()
        .map(|s| serde_json::from_value(s["requests"].clone()).unwrap())
        .collect();
    let run = |enabled: bool| -> anyhow::Result<(f64, Vec<Value>, usize)> {
        REFERENCE_CACHE.store(!enabled, Ordering::Relaxed);
        let mut elapsed = 0.0;
        let mut hits = 0;
        let mut outputs = vec![];
        for session in &sessions {
            clear(&model);
            for request in session {
                model.device.synchronize()?;
                let start = Instant::now();
                let r = model.system_one_batch(std::slice::from_ref(request));
                model.device.synchronize()?;
                elapsed += start.elapsed().as_secs_f64() * 1000.0;
                outputs.push(snapshot(r));
            }
            hits += model.result_cache.as_ref().unwrap().lock().unwrap().hits;
        }
        Ok((elapsed, outputs, hits))
    };
    let (_, expected, _) = run(false)?;
    let invalid = expected.iter().filter(|v| v.get("error").is_some()).count();
    let mut pairs = vec![];
    let mut candidate_hits = vec![];
    for i in 0..5 {
        let mut times = [0.0; 2];
        for variant in if i % 2 == 0 { [0, 1] } else { [1, 0] } {
            let (elapsed, outputs, hits) = run(variant == 1)?;
            if variant == 1 {
                candidate_hits.push(hits);
            }
            assert_eq!(outputs, expected, "pair {i} variant {variant}");
            times[variant] = elapsed;
        }
        pairs.push(times);
        eprintln!("REPLAY_PAIR={times:?}");
    }
    let mut ratios: Vec<_> = pairs.iter().map(|p| p[1] / p[0]).collect();
    let report = json!({"sessions":sessions.len(),"requests":expected.len(),"invalid_requests":invalid,"cache_hits_per_replay":candidate_hits,"cold_cache_per_session":true,"outputs_and_errors_exact":true,"pairs_ms":pairs,"paired_change_percent":100.0*(median(&mut ratios)-1.0)});
    std::fs::write(
        path("04-cache-replay.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    eprintln!("CACHE_REPLAY={report}");
    REFERENCE_CACHE.store(false, Ordering::Relaxed);
    Ok(())
}
#[test]
#[ignore = "requires CUDA and checkpoint"]
fn paired_cache_misses() -> anyhow::Result<()> {
    let model = model()?;
    let mut report = vec![];
    for (name, requests) in batch_bench::cases()
        .into_iter()
        .filter(|(n, _)| ["1", "8", "browser_call5"].contains(&n.as_str()))
    {
        for i in 0..4 {
            REFERENCE_CACHE.store(i % 2 == 0, Ordering::Relaxed);
            clear(&model);
            model.system_one_batch(&requests)?;
        }
        clear(&model);
        let mut pairs = vec![];
        for iteration in 0..40 {
            // Preserve valid types and order while changing actual state text.
            let mut changed = serde_json::to_value(&requests)?;
            for request in changed.as_array_mut().unwrap() {
                let original = request["state"].clone();
                request["state"] =
                    json!(format!("Unique replay {iteration}. {original}"));
            }
            let requests: Vec<SystemOneRequest> =
                serde_json::from_value(changed)?;
            let mut times = [0.0; 2];
            let mut outputs = [Value::Null, Value::Null];
            for variant in if iteration % 2 == 0 { [0, 1] } else { [1, 0] } {
                REFERENCE_CACHE.store(variant == 0, Ordering::Relaxed);
                model.device.synchronize()?;
                let start = Instant::now();
                let r = model.system_one_batch(&requests);
                model.device.synchronize()?;
                times[variant] = start.elapsed().as_secs_f64() * 1000.0;
                outputs[variant] = snapshot(r);
            }
            assert_eq!(outputs[0], outputs[1]);
            assert!(outputs[0].get("error").is_none());
            pairs.push(times);
        }
        let mut ratios: Vec<_> = pairs.iter().map(|p| p[1] / p[0]).collect();
        assert_eq!(
            model.result_cache.as_ref().unwrap().lock().unwrap().hits,
            0,
            "must measure misses"
        );
        report.push(json!({"name":name,"paired_change_percent":100.0*(median(&mut ratios)-1.0),"pairs_ms":pairs,"outputs_exact":true}));
    }
    std::fs::write(
        path("04-cache-misses.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    eprintln!("CACHE_MISSES={}", serde_json::to_string(&report)?);
    REFERENCE_CACHE.store(false, Ordering::Relaxed);
    Ok(())
}

#[test]
#[ignore = "requires CUDA and checkpoint"]
fn cache_preserves_labels_and_batch_context() -> anyhow::Result<()> {
    let model = model()?;
    let original = SystemOneRequest::new("Billing refund").question(
        "department",
        Question::choice(
            "Who should handle this?",
            [("billing", "payments"), ("support", "technical support")],
        ),
    );
    let renamed = SystemOneRequest::new("Billing refund").question(
        "team",
        Question::choice(
            "Who should handle this?",
            [("accounts", "payments"), ("help", "technical support")],
        ),
    );
    assert_ne!(
        model.prepare_items(std::slice::from_ref(&original))?[0].2,
        model.prepare_items(std::slice::from_ref(&renamed))?[0].2
    );
    let alias = SystemOneRequest::new("Billing refund")
        .question("new_id", original.questions["department"].clone());
    assert_eq!(
        model.prepare_items(std::slice::from_ref(&original))?[0].2,
        model.prepare_items(std::slice::from_ref(&alias))?[0].2
    );
    let mut cases = vec![
        vec![original.clone()],
        vec![renamed],
        vec![alias],
        vec![original],
    ];
    cases.extend(batch_bench::cases().into_iter().map(|(_, r)| r));
    REFERENCE_CACHE.store(true, Ordering::Relaxed);
    let expected: Vec<_> = cases
        .iter()
        .map(|r| snapshot(model.system_one_batch(r)))
        .collect();
    REFERENCE_CACHE.store(false, Ordering::Relaxed);
    clear(&model);
    for _ in 0..3 {
        for (requests, expected) in cases.iter().zip(&expected) {
            assert_eq!(&snapshot(model.system_one_batch(requests)), expected);
        }
    }
    assert!(model.result_cache.as_ref().unwrap().lock().unwrap().hits > 0);
    drop(model);
    let parallel: SystemOne = SystemOne::from(crate::DEFAULT_REPO_ID)
        .with_device(Device::new_cuda(0)?)
        .with_dtype(DType::BF16)
        .with_parallel_cuda_batches(true)
        .with_result_cache_capacity(32)
        .try_into()?;
    for _ in 0..2 {
        for (requests, expected) in cases.iter().zip(&expected) {
            assert_eq!(
                &snapshot(parallel.system_one_batch(requests)),
                expected
            );
        }
    }
    assert!(parallel.result_cache.as_ref().unwrap().lock().unwrap().hits > 0);
    Ok(())
}

#[test]
#[ignore = "requires checkpoint and saved local browser traces"]
fn audit_exact_replay_keys() -> anyhow::Result<()> {
    let model = model()?;
    let corpus: Vec<Value> =
        serde_json::from_slice(&std::fs::read(path("replay-inputs.json"))?)?;
    let mut sessions = vec![];
    for (index, session) in corpus.iter().enumerate() {
        let requests: Vec<SystemOneRequest> =
            serde_json::from_value(session["requests"].clone())?;
        let mut entries: std::collections::VecDeque<Vec<EncodedItem>> =
            std::collections::VecDeque::new();
        let (mut hits, mut calls, mut invalid, mut bytes) = (0, 0, 0, 0usize);
        for request in &requests {
            let items = match model.prepare_items(std::slice::from_ref(request))
            {
                Ok(items) => items,
                Err(_) => {
                    invalid += 1;
                    continue;
                }
            };
            let mut order: Vec<usize> = (0..items.len()).collect();
            order.sort_by_key(|&i| std::cmp::Reverse(items[i].2.ids.len()));
            for chunk in order.chunks(model.batch_size) {
                calls += 1;
                let key: Vec<_> =
                    chunk.iter().map(|&i| items[i].2.clone()).collect();
                if let Some(position) = entries.iter().position(|k| *k == key) {
                    hits += 1;
                    let key = entries.remove(position).unwrap();
                    entries.push_back(key);
                } else {
                    let size = |k: &Vec<EncodedItem>| {
                        k.iter()
                            .map(|i| i.ids.len() * 4 + i.markers.len() * 8 + 32)
                            .sum::<usize>()
                    };
                    let new_bytes = size(&key);
                    if new_bytes <= 4 * 1024 * 1024 {
                        while !entries.is_empty()
                            && (entries.len() >= 32
                                || bytes + new_bytes > 4 * 1024 * 1024)
                        {
                            bytes -= size(&entries.pop_front().unwrap());
                        }
                        bytes += new_bytes;
                        entries.push_back(key);
                    }
                }
            }
        }
        sessions.push(json!({"session":index,"requests":requests.len(),"batches":calls,"hits":hits,"invalid_requests":invalid,"retained_key_bytes":bytes}));
    }
    std::fs::write(
        path("04-replay-audit.json"),
        serde_json::to_vec_pretty(&sessions)?,
    )?;
    eprintln!("REPLAY_AUDIT={}", serde_json::to_string(&sessions)?);
    Ok(())
}
