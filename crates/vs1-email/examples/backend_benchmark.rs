//! Read-only backend benchmark. Output files contain private mail.
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
fn compact_description(category: &str) -> Result<&'static str> {
    Ok(match category {
        "capture" => "a periodic financial statement",
        "bills" => "an unpaid bill requiring payment",
        "fiscal" => "tax or accounting paperwork",
        "income" => "money received or an incoming payment",
        "receipts" => "a purchase receipt or order confirmation",
        "careers" => "a job application or recruitment interview",
        "clients" => "client work or project coordination",
        "papers" => "a signed legal document",
        "equity" => "shares, stock options or ownership",
        "household" => "family coordination with a spouse",
        "identity" => "identity verification or immigration paperwork",
        "security" => "a login alert, password reset or verification code",
        "ops" => "a developer infrastructure or repository notification",
        "health" => "a medical, fitness or veterinary appointment",
        "travel" => "a travel reservation or itinerary",
        "bulk" => "a newsletter, advertisement or general announcement",
        "other" => "unrelated to any listed category",
        _ => anyhow::bail!("unknown compact-audit category: {category}"),
    })
}
fn audit_budget(mode: &str) -> Result<usize> {
    match mode {
        "original" | "compact1024" => Ok(1024),
        "compact512" | "plain512" => Ok(512),
        _ => anyhow::bail!("unknown audit mode"),
    }
}
fn default_audit(backend: &str) -> &'static str {
    if backend.starts_with("openjev") {
        "compact512"
    } else {
        "original"
    }
}
fn resolve_audit_budget(backend: &str, mode: &str) -> Result<Option<usize>> {
    if backend == "cua-s1" {
        // Budgeted audit modes are OpenJev-only.
        ensure!(
            mode == "original",
            "cua-s1 supports only original audit mode"
        );
        return Ok(None);
    }
    let budget = audit_budget(mode)?;
    ensure!(
        mode == "original" || backend.starts_with("openjev"),
        "audit modes are OpenJev-only"
    );
    Ok(Some(budget))
}
fn audit_request(
    config: &Config,
    email: &Email,
    mode: &str,
) -> Result<SystemOneRequest> {
    let mut request = direct(config, email);
    if mode != "original" {
        let mut q = serde_json::to_value(&request.questions["category"])?;
        for (id, value) in q["criteria"].as_object_mut().unwrap() {
            *value = json!(compact_description(id)?);
        }
        request
            .questions
            .insert("category".into(), serde_json::from_value(q)?);
    }
    if mode == "plain512" {
        request.state = vs1::State::from(format!(
            "Subject: {}\nFrom: {}\nDate: {}\nBody:\n{}\nOwner: {}",
            email.subject,
            email.from,
            email.date,
            email.body,
            config.owner()
        ));
    }
    Ok(request)
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
struct EmailChunks {
    offset: usize,
    lengths: Vec<usize>,
}
fn split_requests(
    emails: &[Email],
    request_for: impl Fn(&Email) -> Result<SystemOneRequest>,
    mut fits: impl FnMut(&SystemOneRequest) -> Result<bool>,
) -> Result<(Vec<SystemOneRequest>, Vec<EmailChunks>)> {
    let mut requests = Vec::new();
    let mut jobs = Vec::new();
    for email in emails {
        let mut part = email.clone();
        let bodies = vs1_email::split_body(&email.body, &mut |body| {
            part.body = body.into();
            fits(&request_for(&part)?)
        })
        .with_context(|| format!("cannot chunk {}", email.path.display()))?;
        let offset = requests.len();
        let mut lengths = Vec::new();
        for body in bodies {
            part.body = body.into();
            requests.push(request_for(&part)?);
            lengths.push(body.chars().count());
        }
        jobs.push(EmailChunks { offset, lengths });
    }
    Ok((requests, jobs))
}
fn calculate_confidence(probs: &Value) -> f32 {
    let probs = probs.as_object().unwrap();
    let entropy: f64 = probs
        .values()
        .map(|p| p.as_f64().unwrap())
        .filter(|p| *p > 0.0)
        .map(|p| -p * p.ln())
        .sum();
    (1.0 - entropy / (probs.len() as f64).ln()).clamp(0.0, 1.0) as f32
}
fn record_jev_classification(
    config: &Config,
    email: &Email,
    lengths: &[usize],
    outputs: &[Value],
    probabilities: &[Value],
    pooled: &Value,
) -> Result<Value> {
    let mut usage = vs1::Usage::default();
    let mut evidence = Vec::new();
    let model = &outputs.first().context("no chunks")?["model"];
    for ((chars, output), probs) in
        lengths.iter().zip(outputs).zip(probabilities)
    {
        ensure!(&output["model"] == model, "mixed models in one email");
        let chunk_usage: vs1::Usage =
            serde_json::from_value(output["usage"].clone())?;
        usage.input_tokens += chunk_usage.input_tokens;
        usage.output_tokens += chunk_usage.output_tokens;
        let decisions = json!({"provider":output,"evaluated":[{"candidates":config.rules().iter().map(|r| &r.category).collect::<Vec<_>>(),"answers":output["answers"]}]});
        evidence.push(json!({"decisions":decisions,"body_chars":chars,"category":winner(probs),"confidence":calculate_confidence(probs),"probabilities":probs,"usage":chunk_usage}));
    }
    Ok(
        json!({"path":email.path,"message_id":email.message_id,"subject":email.subject,"category":winner(pooled),"confidence":calculate_confidence(pooled),"probabilities":pooled,"model":model,"usage":usage,"chunks":evidence,"decisions":null}),
    )
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
        (4..=6).contains(&args.len()),
        "ROOT export|laya|jev|openjev|openjev-bf16|cua-s1 RUN_NAME [original|compact1024|compact512|plain512] [none|matched|full|text|labels|labels-native|prepare]\nAll backends split complete requests at their model context limit and pool chunks by body character count; Laya keeps its per-chunk tournament. OpenJev audit modes configure max_len (original/compact1024: 1024, compact512/plain512: 512). Retrieval fit checks use exactly the context sent. cua-s1: original only (default), full descriptions, at most 26 candidates; prepare is OpenJev-only. Summary context_tokens records the model limit; budget retains the audit setting."
    );
    let root = Path::new(&args[1]);
    let backend = args[2].as_str();
    let audit_mode = args
        .get(4)
        .map(String::as_str)
        .unwrap_or(default_audit(backend));
    let retrieval_mode = args.get(5).map(String::as_str).unwrap_or("none");
    ensure!(
        [
            "none",
            "matched",
            "full",
            "text",
            "labels",
            "labels-native",
            "prepare"
        ]
        .contains(&retrieval_mode),
        "unknown retrieval mode"
    );
    ensure!(
        backend != "jev" || retrieval_mode == "none",
        "Jev baseline only"
    );
    ensure!(
        retrieval_mode != "prepare" || backend.starts_with("openjev"),
        "context preparation requires OpenJev"
    );
    let examples: Value = if retrieval_mode == "none" {
        json!({})
    } else {
        serde_json::from_slice(&fs::read(root.join("examples.json"))?)?
    };
    let count = if root.join("benchmark.json").exists() {
        serde_json::from_slice::<Value>(&fs::read(
            root.join("benchmark.json"),
        )?)?["messages"]
            .as_u64()
            .context("messages must be positive integer")? as usize
    } else {
        200
    };
    ensure!(count > 0, "empty benchmark");
    let budget = resolve_audit_budget(backend, audit_mode)?;
    let total = Instant::now();
    let config = Config::parse(&fs::read_to_string(root.join("email.toml"))?)?;
    let mailbox = vs1_email::read_maildir(&root.join("sample"), count)?;
    ensure!(
        mailbox.emails.len() == count && mailbox.failures.is_empty(),
        "unexpected parsed email count"
    );
    if backend == "export" {
        return write_new(&root.join("emails.json"), &mailbox.emails);
    }
    let labels: BTreeMap<String, Option<String>> =
        serde_json::from_slice(&fs::read(root.join("labels.json"))?)?;
    ensure!(labels.len() == count, "unexpected reference count");
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
    let context_tokens;
    let mut records = Vec::<Value>::new();
    let mut chunk_states = Vec::<Value>::new();
    match backend {
        "laya" => {
            let model: vs1::SystemOne =
                vs1::SystemOne::from(vs1::DEFAULT_REPO_ID)
                    .with_subfolder("typed-decisions")
                    .with_device(candle_core::Device::new_cuda(0)?)
                    .with_dtype(candle_core::DType::BF16)
                    .with_batch_size(16)
                    .try_into()?;
            context_tokens = model.context_tokens();
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
                &mut |r| {
                    Ok(model.request_fits(&retrieval_request(
                        r,
                        &examples,
                        retrieval_mode,
                    )?)?)
                },
                &mut |rs| {
                    for r in rs.iter().filter(|r| r.questions.len() > 1) {
                        chunk_states.push(serde_json::to_value(&r.state)?);
                    }
                    let augmented = rs
                        .iter()
                        .map(|r| {
                            retrieval_request(r, &examples, retrieval_mode)
                        })
                        .collect::<Result<Vec<_>>>()?;
                    metrics.measure(&augmented, || {
                        Ok(model.system_one_batch(&augmented)?)
                    })
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
        "jev" | "openjev" | "openjev-bf16" | "cua-s1" => {
            let batch_size = if backend.starts_with("openjev") {
                openjev_settings(backend).1
            } else {
                16
            };
            let model: vs1::DecisionModel = match backend {
                "jev" => {
                    ensure!(
                        config.rules().len() <= 255,
                        "Jev supports at most 255 categories"
                    );
                    vs1::JevClient::new(
                        std::env::var("TYPESAFE_API_KEY")?,
                        "jev-1.13.0",
                    )?
                    .into()
                }
                "openjev" | "openjev-bf16" => {
                    let budget =
                        budget.context("OpenJev requires a token budget")?;
                    let model: vs1::OpenJev =
                        vs1::OpenJev::from("artifacts/openjev/checkpoint")
                            .with_device(candle_core::Device::new_cuda(0)?)
                            .with_dtype(openjev_settings(backend).0)
                            .with_max_len(budget)
                            .with_batch_size(batch_size)
                            .try_into()?;
                    eprintln!(
                        "{} {:?} {:?}; effective budget {budget}",
                        model.model_name(),
                        model.device(),
                        model.dtype()
                    );
                    model.into()
                }
                "cua-s1" => {
                    ensure!(
                        config.rules().len() <= 26,
                        "cua-s1 supports at most 26 categories"
                    );
                    let model: vs1::CuaS1 =
                        vs1::CuaS1::from(vs1::cua_s1::DEFAULT_REPO_ID)
                            .with_device(candle_core::Device::new_cuda(0)?)
                            .with_dtype(candle_core::DType::BF16)
                            .with_max_len(vs1::cua_s1::DEFAULT_MAX_LEN)
                            .try_into()?;
                    eprintln!(
                        "{} {:?} {:?}",
                        model.model_name(),
                        model.device(),
                        model.dtype()
                    );
                    model.into()
                }
                _ => unreachable!(),
            };
            context_tokens = model.context_tokens();
            load_seconds = load.elapsed().as_secs_f64();
            let run = Instant::now();
            if retrieval_mode == "prepare" {
                let mut fitted = examples.clone();
                for email in &mailbox.emails {
                    let key = format!("{}\n{}", email.subject, email.date);
                    let mut part = email.clone();
                    part.body.clear();
                    let base = audit_request(&config, &part, audit_mode)?;
                    ensure!(
                        model.request_fits(&base)?,
                        "metadata exceeds budget"
                    );
                    let fitted_row = fit_example_text(
                        examples.get(&key).context("missing examples")?.clone(),
                        |rows| {
                            let map = json!({key.clone():rows});
                            let r = retrieval_request(&base, &map, "full")?;
                            Ok(model.request_fits(&r)?)
                        },
                    )?;
                    fitted[&key] = fitted_row;
                }
                write_new(&root.join("examples-fitted.json"), &fitted)?;
                return Ok(());
            }
            let (requests, jobs) = split_requests(
                &mailbox.emails,
                |email| {
                    let r = audit_request(&config, email, audit_mode)?;
                    retrieval_request(&r, &examples, retrieval_mode)
                },
                |r| Ok(model.request_fits(r)?),
            )?;
            for r in &requests {
                let mut state = serde_json::to_value(&r.state)?;
                if let Some(o) = state.as_object_mut() {
                    o.remove("labeled_examples");
                }
                chunk_states.push(state);
            }
            let mut outputs = Vec::new();
            let mut probabilities = Vec::new();
            if let vs1::DecisionModel::OpenJev(model) = &model {
                let inputs = requests
                    .iter()
                    .map(|r| {
                        model.build_input(
                            &r.state.render(),
                            "category",
                            &r.questions["category"],
                        )
                    })
                    .collect::<vs1::Result<Vec<_>>>()?;
                for batch in inputs.chunks(batch_size) {
                    metrics.calls += batch.len();
                    metrics.questions += batch.len();
                    metrics.batch_calls += 1;
                    let start = Instant::now();
                    let results = model.predict(batch)?;
                    let seconds = start.elapsed().as_secs_f64();
                    metrics.call_wall_seconds += seconds;
                    metrics.batch_seconds.push(seconds);
                    for result in results {
                        chunk_abstentions += usize::from(result.abstained);
                        probabilities
                            .push(serde_json::to_value(&result.probabilities)?);
                        outputs.push(serde_json::to_value(result)?);
                    }
                    if outputs.len().is_multiple_of(100) {
                        eprintln!(
                            "OpenJev {}/{} chunks",
                            outputs.len(),
                            inputs.len()
                        );
                    }
                }
                write_new(
                    &root.join(format!("{}.parity.json", args[3])),
                    &inputs
                        .iter()
                        .zip(&outputs)
                        .take(16)
                        .map(|(i, p)| json!({"input":i,"prediction":p}))
                        .collect::<Vec<_>>(),
                )?;
            } else {
                for batch in requests.chunks(batch_size) {
                    let mut sum = 0.0;
                    let responses = metrics.measure(batch, || {
                        if let vs1::DecisionModel::Jev(client) = &model {
                            std::thread::scope(|scope| {
                                let handles = batch
                                    .iter()
                                    .map(|r| {
                                        scope.spawn(move || {
                                            let start = Instant::now();
                                            let result = client.system_one(r);
                                            (
                                                result,
                                                start.elapsed().as_secs_f64(),
                                            )
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
                        } else {
                            Ok(model.system_one_batch(batch)?)
                        }
                    });
                    metrics.hosted_request_seconds_sum += sum;
                    let responses = responses?;
                    ensure!(
                        responses.len() == batch.len(),
                        "unexpected {backend} response count"
                    );
                    for response in responses {
                        let answer = response
                            .answers
                            .get("category")
                            .context("missing category answer")?;
                        let probs = match answer {
                            vs1::Answer::Choice(a) => &a.probabilities,
                            vs1::Answer::Abstain(a) => &a.probabilities,
                            _ => {
                                anyhow::bail!("expected category choice answer")
                            }
                        };
                        chunk_abstentions += usize::from(matches!(
                            answer,
                            vs1::Answer::Abstain(_)
                        ));
                        probabilities.push(serde_json::to_value(probs)?);
                        outputs.push(serde_json::to_value(response)?);
                    }
                }
            }
            ensure!(
                outputs.len() == requests.len(),
                "unexpected {backend} response count"
            );
            chunks = outputs.len();
            let mut classifications = Vec::new();
            for (email, EmailChunks { offset, lengths }) in
                mailbox.emails.iter().zip(jobs)
            {
                let outputs = &outputs[offset..offset + lengths.len()];
                let probabilities =
                    &probabilities[offset..offset + lengths.len()];
                let probs = pool(
                    &lengths
                        .iter()
                        .copied()
                        .zip(probabilities.iter().cloned())
                        .collect::<Vec<_>>(),
                );
                let id = email
                    .path
                    .file_name()
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .to_owned();
                predictions.insert(id.clone(), winner(&probs));
                if backend == "jev" {
                    classifications.push(record_jev_classification(
                        &config,
                        email,
                        &lengths,
                        outputs,
                        probabilities,
                        &probs,
                    )?);
                } else {
                    records.push(
                        json!({"id":id,"probabilities":probs,"chunks":outputs}),
                    );
                }
            }
            if let vs1::DecisionModel::Jev(client) = &model {
                http = serde_json::to_value(client.stats())?;
                records.push(json!({"dry_run":true,"mailbox":mailbox.path,"failures":mailbox.failures,"classifications":classifications}));
            }
            run_seconds = run.elapsed().as_secs_f64();
        }
        _ => anyhow::bail!("unknown backend"),
    }
    ensure!(predictions.len() == count, "missing predictions");
    write_new(
        &root.join(format!("{}.chunks.json", args[3])),
        &chunk_states,
    )?;
    let predicted = labels
        .keys()
        .map(|id| predictions[id].as_deref())
        .collect::<Vec<_>>();
    let (correct, labeled, labeled_abstentions) = score(&predicted, &reference);
    write_new(
        &root.join(format!("{}.results.json", args[3])),
        &json!({"predictions":predictions,"records":records}),
    )?;
    let summary = json!({"backend":backend,"audit_mode":audit_mode,"budget":budget,"context_tokens":context_tokens,"messages":count,"retrieval_mode":retrieval_mode,"chunks":chunks,"chunk_abstentions":chunk_abstentions,"abstentions":predicted.iter().filter(|p|p.is_none()).count(),"correct":correct,"labeled":labeled,"labeled_abstentions":labeled_abstentions,"accuracy":correct as f64/labeled as f64,"metrics":metrics,"http":http,"setup_seconds":setup,"load_seconds":load_seconds,"run_seconds":run_seconds,"total_seconds":total.elapsed().as_secs_f64(),"timing":"cold inference; total includes setup, load, run and result serialization; call wall excludes tokenization for native OpenJev only"});
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

#[test]
fn compact_audit_preserves_category_ids_and_rejects_unknown_rules() {
    assert_eq!(
        compact_description("receipts").unwrap(),
        "a purchase receipt or order confirmation"
    );
    assert!(compact_description("unconfigured").is_err());
    assert_eq!(audit_budget("compact512").unwrap(), 512);
    assert_eq!(audit_budget("compact1024").unwrap(), 1024);
    assert!(audit_budget("typo").is_err());
    assert_eq!(default_audit("openjev"), "compact512");
    assert_eq!(default_audit("openjev-bf16"), "compact512");
    assert_eq!(default_audit("laya"), "original");
    assert_eq!(default_audit("jev"), "original");
}

#[test]
fn cua_s1_defaults_to_original_without_an_audit_budget() {
    assert_eq!(default_audit("cua-s1"), "original");
    assert_eq!(
        resolve_audit_budget("cua-s1", default_audit("cua-s1")).unwrap(),
        None
    );
    assert_eq!(resolve_audit_budget("cua-s1", "original").unwrap(), None);
    for mode in ["compact1024", "compact512", "plain512", "typo"] {
        assert!(resolve_audit_budget("cua-s1", mode).is_err(), "{mode}");
    }
}

#[test]
fn audit_modes_preserve_existing_backend_budgets() {
    for backend in ["laya", "jev", "export", "openjev", "openjev-bf16"] {
        assert_eq!(
            resolve_audit_budget(backend, "original").unwrap(),
            Some(1024)
        );
        for mode in ["compact1024", "compact512", "plain512"] {
            let budget = resolve_audit_budget(backend, mode);
            if backend.starts_with("openjev") {
                assert_eq!(budget.unwrap(), Some(audit_budget(mode).unwrap()));
            } else {
                assert!(budget.is_err(), "{backend} {mode}");
            }
        }
        assert!(resolve_audit_budget(backend, "typo").is_err());
    }
}

fn retrieval_request(
    request: &SystemOneRequest,
    examples: &Value,
    mode: &str,
) -> Result<SystemOneRequest> {
    let mode = if mode == "labels-native" {
        "labels"
    } else {
        mode
    };
    if matches!(mode, "none" | "matched") {
        return Ok(request.clone());
    }
    ensure!(
        matches!(mode, "full" | "text" | "labels"),
        "unknown retrieval mode"
    );
    let mut request = request.clone();
    let vs1::State::Json(ref mut state) = request.state else {
        anyhow::bail!("JSON required")
    };
    let key = format!(
        "{}\n{}",
        state["email"]["subject"]
            .as_str()
            .context("subject missing")?,
        state["email"]["date"].as_str().context("date missing")?
    );
    let mut selected = examples
        .get(&key)
        .context("missing examples")?
        .as_array()
        .context("examples must be an array")?
        .clone();
    for row in &mut selected {
        let row = row.as_object_mut().context("example must be object")?;
        match mode {
            "text" => {
                row.remove("category");
            }
            "labels" => {
                row.remove("subject");
                row.remove("body");
            }
            _ => {}
        }
    }
    if selected.is_empty() {
        return Ok(request);
    }
    state["labeled_examples"] = json!(selected);
    Ok(request)
}

#[test]
fn retrieval_ablations_preserve_target_and_remove_only_named_fields() {
    let r = SystemOneRequest::new(
        json!({"email":{"subject":"target","date":"now","body":"target text"}}),
    );
    let examples = json!({"target\nnow":[{"subject":"example","body":"example text","category":"bills"}]});
    for (mode, expected) in [
        (
            "full",
            json!([{"subject":"example","body":"example text","category":"bills"}]),
        ),
        ("text", json!([{"subject":"example","body":"example text"}])),
        ("labels", json!([{"category":"bills"}])),
    ] {
        let changed = serde_json::to_value(
            retrieval_request(&r, &examples, mode).unwrap(),
        )
        .unwrap();
        assert_eq!(changed["state"]["email"]["body"], "target text");
        assert_eq!(changed["state"]["labeled_examples"], expected);
        assert_eq!(changed["questions"], json!({}));
    }
    for mode in ["none", "matched"] {
        assert_eq!(
            serde_json::to_value(
                retrieval_request(&r, &examples, mode).unwrap()
            )
            .unwrap(),
            serde_json::to_value(&r).unwrap()
        );
    }
    assert!(retrieval_request(&r, &examples, "invalid").is_err());
    assert!(retrieval_request(&r, &json!({}), "full").is_err());
}

fn fit_example_text(
    mut examples: Value,
    mut fits: impl FnMut(&Value) -> Result<bool>,
) -> Result<Value> {
    loop {
        if fits(&examples)? {
            return Ok(examples);
        }
        let rows = examples.as_array_mut().context("examples must be array")?;
        let mut longest = None;
        for (i, row) in rows.iter().enumerate() {
            for field in ["subject", "body"] {
                let n = row[field].as_str().unwrap_or("").chars().count();
                if n > 0 && longest.is_none_or(|(_, _, len)| n > len) {
                    longest = Some((i, field, n));
                }
            }
        }
        if let Some((i, field, n)) = longest {
            rows[i][field] = json!(
                rows[i][field]
                    .as_str()
                    .unwrap()
                    .chars()
                    .take(n / 2)
                    .collect::<String>()
            );
        } else {
            ensure!(!rows.is_empty(), "target metadata cannot fit");
            rows.pop();
        }
    }
}
#[test]
fn fitting_context_shrinks_text_before_dropping_labels() {
    let examples =
        json!([{"subject":"longsubject","body":"longbody","category":"bills"}]);
    let fitted = fit_example_text(examples, |v| {
        Ok(v[0]["subject"].as_str().unwrap().is_empty()
            && v[0]["body"].as_str().unwrap().is_empty())
    })
    .unwrap();
    assert_eq!(fitted, json!([{"subject":"","body":"","category":"bills"}]));
    let empty = fit_example_text(json!([{"category":"bills"}]), |v| {
        Ok(v.as_array().unwrap().is_empty())
    })
    .unwrap();
    assert_eq!(empty, json!([]));
    assert!(fit_example_text(json!([]), |_| Ok(false)).is_err());
}

#[test]
fn native_labels_send_the_same_context_as_labels() {
    let r = SystemOneRequest::new(
        json!({"email":{"subject":"target","date":"now","body":"all target text"}}),
    );
    let examples = json!({"target\nnow":[{"subject":"example","body":"long example text","category":"bills"}]});
    let native = retrieval_request(&r, &examples, "labels-native").unwrap();
    let labels = retrieval_request(&r, &examples, "labels").unwrap();
    assert_eq!(
        serde_json::to_value(native).unwrap(),
        serde_json::to_value(labels).unwrap()
    );
}

#[cfg(test)]
fn make_email(body: &str) -> Email {
    Email {
        path: "message".into(),
        message_id: "id".into(),
        from: "sender".into(),
        to: "owner".into(),
        subject: "target".into(),
        date: "now".into(),
        body: body.into(),
    }
}

#[cfg(test)]
fn parse_category_config() -> Config {
    Config::parse("[[rules]]\ncategory = 'bulk'\nwhat = 'announcements'\n[[rules]]\ncategory = 'ops'\nwhat = 'developer notifications'\n").unwrap()
}

#[test]
fn splitting_checks_the_exact_audit_and_retrieval_requests() {
    let config = parse_category_config();
    let email = make_email(&"á😀日 ".repeat(20));
    let examples = json!({"target\nnow":[{"subject":"example","body":"long example text","category":"bulk"}]});
    for audit_mode in ["original", "compact1024", "compact512", "plain512"] {
        for retrieval_mode in
            ["none", "matched", "full", "text", "labels", "labels-native"]
        {
            if audit_mode == "plain512"
                && !matches!(retrieval_mode, "none" | "matched")
            {
                continue;
            }
            let request_for = |email: &Email| {
                retrieval_request(
                    &audit_request(&config, email, audit_mode)?,
                    &examples,
                    retrieval_mode,
                )
            };
            let empty = make_email("");
            let limit = serde_json::to_string(&request_for(&empty).unwrap())
                .unwrap()
                .chars()
                .count()
                + 12;
            let mut checked = Vec::new();
            let (requests, jobs) = split_requests(
                std::slice::from_ref(&email),
                request_for,
                |r| {
                    let serialized = serde_json::to_string(r)?;
                    let fits = serialized.chars().count() <= limit;
                    if fits {
                        checked.push(serialized);
                    }
                    Ok(fits)
                },
            )
            .unwrap();
            assert!(requests.len() > 1, "{audit_mode} {retrieval_mode}");
            assert_eq!(jobs[0].offset, 0);
            assert_eq!(
                jobs[0].lengths.iter().sum::<usize>(),
                email.body.chars().count()
            );
            let mut restored = String::new();
            for (request, chars) in requests.iter().zip(&jobs[0].lengths) {
                assert!(
                    checked.contains(&serde_json::to_string(request).unwrap())
                );
                let state = serde_json::to_value(&request.state).unwrap();
                let body = if audit_mode == "plain512" {
                    state
                        .as_str()
                        .unwrap()
                        .split_once("\nBody:\n")
                        .unwrap()
                        .1
                        .rsplit_once("\nOwner: ")
                        .unwrap()
                        .0
                } else {
                    state["email"]["body"].as_str().unwrap()
                };
                assert_eq!(body.chars().count(), *chars);
                restored.push_str(body);
            }
            assert_eq!(restored, email.body);
        }
    }
}

#[test]
fn splitting_preserves_short_and_empty_emails_and_rejects_oversized_metadata() {
    let config = parse_category_config();
    let emails = [make_email("short"), make_email("")];
    let (requests, jobs) = split_requests(
        &emails,
        |email| Ok(direct(&config, email)),
        |_| Ok(true),
    )
    .unwrap();
    assert_eq!(requests.len(), 2);
    assert_eq!(jobs[0].offset, 0);
    assert_eq!(jobs[0].lengths, [5]);
    assert_eq!(jobs[1].offset, 1);
    assert_eq!(jobs[1].lengths, [0]);
    assert!(
        split_requests(
            &emails,
            |email| Ok(direct(&config, email)),
            |_| Ok(false)
        )
        .is_err()
    );
}

#[test]
fn pooling_weights_empty_chunks_and_keeps_the_first_tied_candidate() {
    let result = pool(&[
        (0, json!({"ops":0.75,"bulk":0.25})),
        (1, json!({"ops":0.25,"bulk":0.75})),
    ]);
    assert_eq!(result, json!({"ops":0.5,"bulk":0.5}));
    assert_eq!(winner(&result).as_deref(), Some("ops"));
}

#[test]
fn jev_records_preserve_chunk_evidence_usage_and_pooled_abstentions() {
    let config = parse_category_config();
    let email = make_email("abcd");
    let lengths = [1, 3];
    let probabilities = [
        json!({"bulk":0.8,"ops":0.1,"__insufficient_evidence__":0.1}),
        json!({"bulk":0.1,"ops":0.1,"__insufficient_evidence__":0.8}),
    ];
    let outputs = vec![
        json!({"model":"jev","answers":{"category":{"type":"choice","choice":"bulk","confidence":0.4,"probabilities":probabilities[0]}},"usage":{"input_tokens":10,"output_tokens":0}}),
        json!({"model":"jev","answers":{"category":{"type":"abstain","question_type":"choice","reason":"insufficient evidence","probabilities":probabilities[1]}},"usage":{"input_tokens":20,"output_tokens":0}}),
    ];
    let pooled = pool(
        &lengths
            .iter()
            .copied()
            .zip(probabilities.iter().cloned())
            .collect::<Vec<_>>(),
    );
    let record = record_jev_classification(
        &config,
        &email,
        &lengths,
        &outputs,
        &probabilities,
        &pooled,
    )
    .unwrap();
    assert_eq!(record["category"], Value::Null);
    assert_eq!(record["probabilities"], pooled);
    assert_eq!(record["usage"]["input_tokens"], 30);
    assert_eq!(record["chunks"][0]["category"], "bulk");
    assert_eq!(record["chunks"][1]["category"], Value::Null);
    for (i, output) in outputs.iter().enumerate() {
        assert_eq!(&record["chunks"][i]["decisions"]["provider"], output);
        assert_eq!(record["chunks"][i]["body_chars"], lengths[i]);
        assert_eq!(record["chunks"][i]["probabilities"], probabilities[i]);
    }
}
