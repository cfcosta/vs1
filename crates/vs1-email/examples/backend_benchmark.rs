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
        "ROOT export|laya|jev|openjev|openjev-bf16 RUN_NAME [original|compact1024|compact512|plain512] [none|matched|full|text|labels|labels-native|prepare]"
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
    let budget = audit_budget(audit_mode)?;
    ensure!(
        audit_mode == "original" || backend.starts_with("openjev"),
        "audit modes are OpenJev-only"
    );
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
                    let mode = retrieval_fit_mode(retrieval_mode);
                    vs1_email::request_fits(
                        &model,
                        &retrieval_request(r, &examples, mode)?,
                    )
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
            // One extra token makes truncation detectable.
            let model: vs1::OpenJev =
                vs1::OpenJev::from("artifacts/openjev/checkpoint")
                    .with_device(candle_core::Device::new_cuda(0)?)
                    .with_dtype(dtype)
                    .with_max_len(budget + 1)
                    .with_batch_size(batch_size)
                    .try_into()?;
            eprintln!(
                "{} {:?} {:?}; effective budget {budget}",
                model.model_name(),
                model.device(),
                model.dtype()
            );
            load_seconds = load.elapsed().as_secs_f64();
            let run = Instant::now();
            if retrieval_mode == "prepare" {
                let mut fitted = examples.clone();
                for email in &mailbox.emails {
                    let key = format!("{}\n{}", email.subject, email.date);
                    let mut part = email.clone();
                    part.body.clear();
                    let base = audit_request(&config, &part, audit_mode)?;
                    let base_len = model
                        .build_input(
                            &base.state.render(),
                            "category",
                            &base.questions["category"],
                        )?
                        .ids
                        .len();
                    ensure!(base_len <= budget, "metadata exceeds budget");
                    let reserve = 64.min((budget - base_len) / 2);
                    let fitted_row = fit_example_text(
                        examples.get(&key).context("missing examples")?.clone(),
                        |rows| {
                            let map = json!({key.clone():rows});
                            let r = retrieval_request(&base, &map, "full")?;
                            Ok(model
                                .build_input(
                                    &r.state.render(),
                                    "category",
                                    &r.questions["category"],
                                )?
                                .ids
                                .len()
                                <= budget - reserve)
                        },
                    )?;
                    fitted[&key] = fitted_row;
                }
                write_new(&root.join("examples-fitted.json"), &fitted)?;
                return Ok(());
            }
            let mut inputs = Vec::new();
            let mut jobs = Vec::new();
            for email in &mailbox.emails {
                let mut part = email.clone();
                let bodies = vs1_email::split_body(&email.body, &mut |body| {
                    part.body = body.into();
                    let r = audit_request(&config, &part, audit_mode)?;
                    let r = retrieval_request(
                        &r,
                        &examples,
                        retrieval_fit_mode(retrieval_mode),
                    )?;
                    Ok(model
                        .build_input(
                            &r.state.render(),
                            "category",
                            &r.questions["category"],
                        )?
                        .ids
                        .len()
                        <= budget)
                })?;
                let offset = inputs.len();
                let mut lengths = Vec::new();
                for body in &bodies {
                    part.body = (*body).into();
                    let r = audit_request(&config, &part, audit_mode)?;
                    let r = retrieval_request(&r, &examples, retrieval_mode)?;
                    let input = model.build_input(
                        &r.state.render(),
                        "category",
                        &r.questions["category"],
                    )?;
                    ensure!(
                        input.ids.len() <= budget,
                        "OpenJev input would truncate"
                    );
                    let mut state = serde_json::to_value(&r.state)?;
                    if let Some(o) = state.as_object_mut() {
                        o.remove("labeled_examples");
                    }
                    chunk_states.push(state);
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
            write_new(
                &root.join(format!("{}.parity.json", args[3])),
                &inputs
                    .iter()
                    .zip(&outputs)
                    .take(16)
                    .map(|(i, p)| json!({"input":i,"prediction":p}))
                    .collect::<Vec<_>>(),
            )?;
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
    let summary = json!({"backend":backend,"audit_mode":audit_mode,"budget":budget,"messages":count,"retrieval_mode":retrieval_mode,"chunks":chunks,"chunk_abstentions":chunk_abstentions,"abstentions":predicted.iter().filter(|p|p.is_none()).count(),"correct":correct,"labeled":labeled,"labeled_abstentions":labeled_abstentions,"accuracy":correct as f64/labeled as f64,"metrics":metrics,"http":http,"setup_seconds":setup,"load_seconds":load_seconds,"run_seconds":run_seconds,"total_seconds":total.elapsed().as_secs_f64(),"timing":"cold inference; total includes setup, load, run and result serialization; call wall excludes tokenization for native OpenJev only"});
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
fn native_labels_reserve_only_the_context_sent_to_inference() {
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
    assert_eq!(retrieval_fit_mode("labels-native"), "labels");
    assert_eq!(retrieval_fit_mode("labels"), "full");
    assert_eq!(retrieval_fit_mode("matched"), "full");
    assert_eq!(retrieval_fit_mode("none"), "none");
}

fn retrieval_fit_mode(mode: &str) -> &str {
    match mode {
        "none" => "none",
        "labels-native" => "labels",
        _ => "full",
    }
}
