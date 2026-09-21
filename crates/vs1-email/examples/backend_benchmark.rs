//! Read-only three-backend benchmark. Output files contain private mail.
use std::{collections::BTreeMap, fs, path::Path, time::Instant};

use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use vs1::{SystemOneRequest, SystemOneResponse};
use vs1_email::{Config, Email, classification_request};

fn openjev_settings(backend: &str) -> (candle_core::DType, usize) {
    if backend == "openjev-bf16" {
        (candle_core::DType::BF16, 16)
    } else {
        (candle_core::DType::F32, 4)
    }
}

fn score(
    predicted: &[Option<&str>],
    reference: &[Option<&str>],
) -> (usize, usize, usize) {
    assert_eq!(predicted.len(), reference.len());
    let mut result = (0, 0, 0);
    for (p, r) in predicted.iter().zip(reference) {
        if let Some(r) = r {
            result.1 += 1;
            result.0 += usize::from(*p == Some(*r));
            result.2 += usize::from(p.is_none());
        }
    }
    result
}
fn pool(rows: &[(usize, Value)]) -> Value {
    let total = rows.iter().map(|(n, _)| (*n).max(1)).sum::<usize>() as f64;
    let mut sums = serde_json::Map::new();
    for (n, probs) in rows {
        for (id, p) in probs.as_object().expect("probabilities object") {
            let old = sums.get(id).and_then(Value::as_f64).unwrap_or(0.0);
            sums.insert(
                id.clone(),
                json!(old + p.as_f64().unwrap() * (*n).max(1) as f64 / total),
            );
        }
    }
    Value::Object(sums)
}
fn winner(probs: &Value) -> Option<String> {
    let mut best: Option<(&str, f64)> = None;
    for (id, p) in probs.as_object().unwrap() {
        let p = p.as_f64().unwrap();
        if best.is_none_or(|(_, old)| p > old) {
            best = Some((id, p));
        }
    }
    best.filter(|(id, _)| *id != vs1::openjev::ABSTENTION_ID)
        .map(|(id, _)| id.to_owned())
}
fn direct(config: &Config, email: &Email) -> SystemOneRequest {
    let grouped = classification_request(config, email);
    let mut criteria = serde_json::Map::new();
    let mut instructions = Value::Null;
    for q in grouped.questions.values() {
        let q = serde_json::to_value(q).unwrap();
        instructions = q["instructions"].clone();
        criteria.extend(q["criteria"].as_object().unwrap().clone());
    }
    let mut request = SystemOneRequest::new(grouped.state);
    request.questions.insert("category".into(), serde_json::from_value(json!({"type":"choice","instructions":instructions,"criteria":criteria})).unwrap());
    request
}
#[derive(Default, serde::Serialize)]
struct Metrics {
    calls: usize,
    questions: usize,
    batch_calls: usize,
    call_wall_seconds: f64,
    // Sum of individual hosted request latencies; overlaps under concurrency.
    hosted_request_seconds_sum: f64,
    batch_seconds: Vec<f64>,
}
impl Metrics {
    fn measure(
        &mut self,
        requests: &[SystemOneRequest],
        f: impl FnOnce() -> Result<Vec<SystemOneResponse>>,
    ) -> Result<Vec<SystemOneResponse>> {
        self.calls += requests.len();
        self.questions +=
            requests.iter().map(|r| r.questions.len()).sum::<usize>();
        self.batch_calls += 1;
        let start = Instant::now();
        let result = f();
        let seconds = start.elapsed().as_secs_f64();
        self.call_wall_seconds += seconds;
        self.batch_seconds.push(seconds);
        result
    }
}
fn write_new(path: &Path, value: &impl serde::Serialize) -> Result<()> {
    use std::io::Write;
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    file.write_all(&serde_json::to_vec_pretty(value)?)?;
    Ok(())
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    ensure!(
        args.len() == 4,
        "ROOT export|laya|jev|openjev|openjev-bf16 RUN_NAME"
    );
    let root = Path::new(&args[1]);
    let backend = args[2].as_str();
    let total = Instant::now();
    let config = Config::parse(&fs::read_to_string(root.join("email.toml"))?)?;
    let mailbox = vs1_email::read_maildir(&root.join("sample"), 200)?;
    ensure!(
        mailbox.emails.len() == 200 && mailbox.failures.is_empty(),
        "expected 200 parsed emails"
    );
    if backend == "export" {
        return write_new(&root.join("emails.json"), &mailbox.emails);
    }
    let labels: BTreeMap<String, Option<String>> =
        serde_json::from_slice(&fs::read(root.join("labels.json"))?)?;
    ensure!(labels.len() == 200, "expected 200 reference entries");
    let reference = mailbox
        .emails
        .iter()
        .map(|e| {
            labels
                .get(e.path.file_name().unwrap().to_str().unwrap())
                .context("missing label")
                .map(|s| s.as_deref())
        })
        .collect::<Result<Vec<_>>>()?;
    let setup = total.elapsed().as_secs_f64();
    let load = Instant::now();
    let mut metrics = Metrics::default();
    let mut predictions = BTreeMap::<String, Option<String>>::new();
    let mut chunks = 0usize;
    let mut chunk_abstentions = 0usize;
    let mut http = json!({"calls":0,"attempts":0,"retries":0,"successes":0});
    let load_seconds;
    let run_seconds;
    let mut records = Vec::<Value>::new();
    match backend {
        "laya" => {
            let model: vs1::SystemOne =
                vs1::SystemOne::from(vs1::DEFAULT_REPO_ID)
                    .with_subfolder("typed-decisions")
                    .with_device(candle_core::Device::new_cuda(0)?)
                    .with_dtype(candle_core::DType::BF16)
                    .with_batch_size(16)
                    .try_into()?;
            eprintln!(
                "{} {:?} {:?}",
                model.model_name(),
                model.device(),
                model.dtype()
            );
            load_seconds = load.elapsed().as_secs_f64();
            let run = Instant::now();
            let report = vs1_email::dry_run(
                &config,
                &mailbox,
                16,
                &mut |r| vs1_email::request_fits(&model, r),
                &mut |rs| {
                    metrics.measure(rs, || Ok(model.system_one_batch(rs)?))
                },
            )?;
            run_seconds = run.elapsed().as_secs_f64();
            ensure!(report.failures.is_empty(), "laya mailbox failures");
            for c in &report.classifications {
                chunks += c.chunks.len();
                predictions.insert(
                    c.path.file_name().unwrap().to_str().unwrap().into(),
                    Some(c.category.clone()),
                );
            }
            records.push(serde_json::to_value(report)?);
        }
        "jev" => {
            let client = vs1::JevClient::new(
                std::env::var("TYPESAFE_API_KEY")?,
                "jev-1.13.0",
            )?;
            load_seconds = load.elapsed().as_secs_f64();
            let run = Instant::now();
            let report = vs1_email::dry_run_single_choice_with_progress(
                &config,
                &mailbox,
                16,
                &mut |rs| {
                    let mut sum = 0.0;
                    let result = metrics.measure(rs, || {
                        std::thread::scope(|scope| {
                            let handles = rs
                                .iter()
                                .map(|r| {
                                    let client = &client;
                                    scope.spawn(move || {
                                        let start = Instant::now();
                                        let result = client.system_one(r);
                                        (result, start.elapsed().as_secs_f64())
                                    })
                                })
                                .collect::<Vec<_>>();
                            let mut results = Vec::new();
                            for handle in handles {
                                let (result, seconds) = handle
                                    .join()
                                    .expect("request worker panicked");
                                sum += seconds;
                                results.push(result);
                            }
                            results
                                .into_iter()
                                .map(|r| r.map_err(Into::into))
                                .collect()
                        })
                    });
                    metrics.hosted_request_seconds_sum += sum;
                    result
                },
                &mut |_| Ok(()),
            )?;
            run_seconds = run.elapsed().as_secs_f64();
            http = serde_json::to_value(client.stats())?;
            ensure!(report.failures.is_empty(), "Jev mailbox failures");
            for c in &report.classifications {
                chunks += c.chunks.len();
                predictions.insert(
                    c.path.file_name().unwrap().to_str().unwrap().into(),
                    Some(c.category.clone()),
                );
            }
            records.push(serde_json::to_value(report)?);
        }
        "openjev" | "openjev-bf16" => {
            let (dtype, batch_size) = openjev_settings(backend);
            // One extra token makes truncation detectable: only accept <=1024.
            let model: vs1::OpenJev =
                vs1::OpenJev::from("artifacts/openjev/checkpoint")
                    .with_device(candle_core::Device::new_cuda(0)?)
                    .with_dtype(dtype)
                    .with_max_len(1025)
                    .with_batch_size(batch_size)
                    .try_into()?;
            eprintln!(
                "{} {:?} {:?}; effective budget 1024",
                model.model_name(),
                model.device(),
                model.dtype()
            );
            load_seconds = load.elapsed().as_secs_f64();
            let run = Instant::now();
            let mut inputs = Vec::new();
            let mut jobs = Vec::new();
            for email in &mailbox.emails {
                let mut part = email.clone();
                let bodies = vs1_email::split_body(&email.body, &mut |body| {
                    part.body = body.into();
                    let r = direct(&config, &part);
                    Ok(model
                        .build_input(
                            &r.state.render(),
                            "category",
                            &r.questions["category"],
                        )?
                        .ids
                        .len()
                        <= 1024)
                })?;
                let offset = inputs.len();
                let mut lengths = Vec::new();
                for body in &bodies {
                    part.body = (*body).into();
                    let r = direct(&config, &part);
                    let input = model.build_input(
                        &r.state.render(),
                        "category",
                        &r.questions["category"],
                    )?;
                    ensure!(
                        input.ids.len() <= 1024,
                        "OpenJev input would truncate"
                    );
                    inputs.push(input);
                    lengths.push(body.chars().count());
                }
                let id = email
                    .path
                    .file_name()
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .to_owned();
                jobs.push((id, offset, lengths));
            }
            let mut outputs = Vec::new();
            for batch in inputs.chunks(batch_size) {
                metrics.calls += batch.len();
                metrics.questions += batch.len();
                metrics.batch_calls += 1;
                let start = Instant::now();
                let results = model.predict(batch)?;
                let seconds = start.elapsed().as_secs_f64();
                metrics.call_wall_seconds += seconds;
                metrics.batch_seconds.push(seconds);
                outputs.extend(results);
                if outputs.len().is_multiple_of(100) {
                    eprintln!(
                        "OpenJev {}/{} chunks",
                        outputs.len(),
                        inputs.len()
                    );
                }
            }
            chunks = outputs.len();
            chunk_abstentions = outputs.iter().filter(|p| p.abstained).count();
            for (id, offset, lengths) in jobs {
                let outputs = &outputs[offset..offset + lengths.len()];
                let probs = pool(
                    &lengths
                        .iter()
                        .zip(outputs)
                        .map(|(n, p)| {
                            (
                                *n,
                                serde_json::to_value(&p.probabilities).unwrap(),
                            )
                        })
                        .collect::<Vec<_>>(),
                );
                predictions.insert(id.clone(), winner(&probs));
                records.push(
                    json!({"id":id,"probabilities":probs,"chunks":outputs}),
                );
            }
            run_seconds = run.elapsed().as_secs_f64();
        }
        _ => anyhow::bail!("unknown backend"),
    }
    ensure!(predictions.len() == 200, "missing predictions");
    let predicted = labels
        .keys()
        .map(|id| predictions[id].as_deref())
        .collect::<Vec<_>>();
    let (correct, labeled, labeled_abstentions) = score(&predicted, &reference);
    write_new(
        &root.join(format!("{}.results.json", args[3])),
        &json!({"predictions":predictions,"records":records}),
    )?;
    let summary = json!({"backend":backend,"messages":200,"chunks":chunks,"chunk_abstentions":chunk_abstentions,"abstentions":predicted.iter().filter(|p|p.is_none()).count(),"correct":correct,"labeled":labeled,"labeled_abstentions":labeled_abstentions,"accuracy":correct as f64/labeled as f64,"metrics":metrics,"http":http,"setup_seconds":setup,"load_seconds":load_seconds,"run_seconds":run_seconds,"total_seconds":total.elapsed().as_secs_f64(),"timing":"cold inference; total includes setup, load, run and result serialization; call wall excludes tokenization for native OpenJev only"});
    write_new(&root.join(format!("{}.summary.json", args[3])), &summary)?;
    eprintln!("{summary}");
    Ok(())
}
#[test]
fn metrics_keep_abstentions_in_denominator() {
    assert_eq!(
        score(
            &[Some("bulk"), Some("ops"), None],
            &[Some("bulk"), None, Some("ops")]
        ),
        (1, 2, 1)
    );
}
#[test]
fn pooling_preserves_abstention_mass() {
    let result = pool(&[
        (1, json!({"bulk":0.9,"__insufficient_evidence__":0.1})),
        (3, json!({"bulk":0.1,"__insufficient_evidence__":0.9})),
    ]);
    assert!((result["bulk"].as_f64().unwrap() - 0.3).abs() < 1e-6);
    assert_eq!(winner(&result), None);
}

#[test]
fn precision_is_explicit_for_openjev_comparison() {
    assert_eq!(openjev_settings("openjev"), (candle_core::DType::F32, 4));
    assert_eq!(
        openjev_settings("openjev-bf16"),
        (candle_core::DType::BF16, 16)
    );
}
