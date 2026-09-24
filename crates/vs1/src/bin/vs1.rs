//! Answer typed questions from the command line.
//!
//! ```text
//! vs1 request.json
//! vs1 --device cuda request.json
//! vs1 --dump-ids request.json
//! ```
//!
//! The file holds one request or a JSON array of them, and the output
//! mirrors it: one response, or an array of them, in the JSON shape
//! TypeSafe's `systemone` endpoint returns. `--dump-ids` wraps each
//! response as `{"response", "ids"}`, adding the token sequence and
//! marker positions built for every question, which is what the
//! parity check against laya's Python implementation compares.

use std::{fs, time::Instant};

use anyhow::{Context, bail};
use candle_core::{DType, Device};
use serde_json::json;
use vs1::{DecisionModel, SystemOne, SystemOneRequest};

fn device(name: &str) -> anyhow::Result<Device> {
    match name {
        "cpu" => Ok(Device::Cpu),
        #[cfg(feature = "cuda")]
        "cuda" => Ok(Device::new_cuda(0)?),
        #[cfg(feature = "metal")]
        "metal" => Ok(Device::new_metal(0)?),
        other => bail!("unknown or unsupported device {other:?}"),
    }
}

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let mut path = None;
    let mut dump_ids = false;
    let mut model_id: Option<String> = None;
    let mut backend = "laya".to_string();
    let mut subfolder = String::new();
    let mut device_name = "cpu".to_string();
    let mut dtype: Option<DType> = None;
    let mut batch_size: Option<usize> = None;
    let mut max_len: Option<usize> = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--dump-ids" => dump_ids = true,
            "--backend" => backend = args.next().context(
                "--backend needs laya, openjev, gliner-decide, cua-s1 or jev",
            )?,
            "--model" => {
                model_id = Some(args.next().context("--model needs a value")?)
            }
            "--subfolder" => {
                subfolder = args.next().context("--subfolder needs a value")?
            }
            "--device" => {
                device_name = args.next().context("--device needs a value")?
            }
            "--dtype" => {
                dtype = Some(match args.next().as_deref() {
                    Some("f32") => DType::F32,
                    Some("bf16") => DType::BF16,
                    Some("f16") => DType::F16,
                    other => bail!("unknown dtype {other:?}"),
                })
            }
            "--batch-size" => {
                batch_size = Some(
                    args.next()
                        .context("--batch-size needs a value")?
                        .parse()?,
                )
            }
            "--max-len" => {
                max_len = Some(
                    args.next().context("--max-len needs a value")?.parse()?,
                )
            }
            other if path.is_none() => path = Some(other.to_string()),
            other => bail!("unexpected argument {other:?}"),
        }
    }
    let path = path.context(
        "usage: vs1 [--backend laya|openjev|gliner-decide|cua-s1|jev] [--model MODEL] [--dump-ids] [--device cpu|cuda] [--max-len TOKENS] request.json",
    )?;
    anyhow::ensure!(
        max_len.is_none()
            || matches!(backend.as_str(), "cua-s1" | "gliner-decide"),
        "--max-len is supported only for --backend cua-s1 and gliner-decide"
    );
    let raw = fs::read_to_string(&path)?;
    let (requests, single): (Vec<SystemOneRequest>, bool) =
        match serde_json::from_str(&raw) {
            Ok(list) => (list, false),
            Err(_) => (vec![serde_json::from_str(&raw)?], true),
        };

    let model: DecisionModel = match backend.as_str() {
        "jev" => {
            anyhow::ensure!(
                !dump_ids
                    && subfolder.is_empty()
                    && device_name == "cpu"
                    && dtype.is_none(),
                "--dump-ids, --subfolder, --device and --dtype are local-only"
            );
            #[cfg(feature = "jev")]
            {
                vs1::JevClient::new(
                    std::env::var("TYPESAFE_API_KEY")
                        .context("TYPESAFE_API_KEY is required for Jev")?,
                    model_id.as_deref().unwrap_or("jev-latest"),
                )?
                .with_concurrency(batch_size.unwrap_or(16))?
                .into()
            }
            #[cfg(not(feature = "jev"))]
            {
                bail!("Jev requires building with --features jev")
            }
        }
        "openjev" => {
            anyhow::ensure!(
                subfolder.is_empty(),
                "OpenJev does not use --subfolder"
            );
            let started = Instant::now();
            let mut builder = vs1::OpenJev::from(
                model_id.as_deref().unwrap_or(vs1::openjev::DEFAULT_REPO_ID),
            )
            .with_device(device(&device_name)?);
            if let Some(dtype) = dtype {
                builder = builder.with_dtype(dtype);
            }
            if let Some(size) = batch_size {
                builder = builder.with_batch_size(size);
            }
            let model: vs1::OpenJev = builder.try_into()?;
            eprintln!(
                "loaded {} on {:?} as {:?} in {:.1?}",
                model.model_name(),
                model.device(),
                model.dtype(),
                started.elapsed()
            );
            model.into()
        }
        "gliner-decide" => {
            anyhow::ensure!(
                subfolder.is_empty(),
                "GLiNER2.5-Decide does not use --subfolder"
            );
            let started = Instant::now();
            let mut builder = vs1::GlinerDecide::from(
                model_id
                    .as_deref()
                    .unwrap_or(vs1::gliner_decide::DEFAULT_REPO_ID),
            )
            .with_device(device(&device_name)?);
            if let Some(dtype) = dtype {
                builder = builder.with_dtype(dtype);
            }
            if let Some(size) = batch_size {
                builder = builder.with_batch_size(size);
            }
            if let Some(tokens) = max_len {
                builder = builder.with_max_len(tokens);
            }
            let model: vs1::GlinerDecide = builder.try_into()?;
            eprintln!(
                "loaded {} on {:?} as {:?} in {:.1?}",
                model.model_name(),
                model.device(),
                model.dtype(),
                started.elapsed()
            );
            model.into()
        }
        "cua-s1" => {
            anyhow::ensure!(
                !dump_ids && subfolder.is_empty() && batch_size.is_none(),
                "Cua-S1 does not use --dump-ids, --subfolder or --batch-size"
            );
            let started = Instant::now();
            let mut builder = vs1::CuaS1::from(
                model_id.as_deref().unwrap_or(vs1::cua_s1::DEFAULT_REPO_ID),
            )
            .with_device(device(&device_name)?);
            if let Some(dtype) = dtype {
                builder = builder.with_dtype(dtype);
            }
            if let Some(tokens) = max_len {
                builder = builder.with_max_len(tokens);
            }
            let model: vs1::CuaS1 = builder.try_into()?;
            eprintln!(
                "loaded {} on {:?} as {:?} in {:.1?}",
                model.model_name(),
                model.device(),
                model.dtype(),
                started.elapsed()
            );
            model.into()
        }
        "laya" => {
            let started = Instant::now();
            let mut builder = SystemOne::from(
                model_id.as_deref().unwrap_or(vs1::DEFAULT_REPO_ID),
            )
            .with_subfolder(subfolder)
            .with_device(device(&device_name)?);
            if let Some(dtype) = dtype {
                builder = builder.with_dtype(dtype);
            }
            if let Some(batch_size) = batch_size {
                builder = builder.with_batch_size(batch_size);
            }
            let model: SystemOne = builder.try_into()?;
            eprintln!(
                "loaded {} on {:?} as {:?} in {:.1?}",
                model.model_name(),
                model.device(),
                model.dtype(),
                started.elapsed()
            );
            model.into()
        }
        _ => bail!(
            "--backend must be laya, openjev, gliner-decide, cua-s1 or jev"
        ),
    };

    let started = Instant::now();
    let result = model.system_one_batch(&requests);
    #[cfg(feature = "jev")]
    if let DecisionModel::Jev(client) = &model {
        eprintln!("Jev calls: {}", serde_json::to_string(&client.stats())?);
    }
    let responses = result?;
    let elapsed = started.elapsed();
    let question_count: usize =
        requests.iter().map(|r| r.questions.len()).sum();
    eprintln!(
        "{} requests, {question_count} questions in {elapsed:.1?} ({:.1} ms/question)",
        requests.len(),
        elapsed.as_secs_f64() * 1000.0 / question_count.max(1) as f64
    );

    let mut out = Vec::with_capacity(responses.len());
    for (request, response) in requests.iter().zip(&responses) {
        let mut entry = serde_json::to_value(response)?;
        if dump_ids {
            let mut ids = serde_json::Map::new();
            match &model {
                DecisionModel::GlinerDecide(local) => {
                    let input =
                        serde_json::to_value(local.build_input(request)?)?;
                    ids = input.as_object().context("input object")?.clone();
                }
                DecisionModel::OpenJev(local) => {
                    let state = request.state.render();
                    for (id, q) in &request.questions {
                        ids.insert(
                            id.clone(),
                            serde_json::to_value(
                                local.build_input(&state, id, q)?,
                            )?,
                        );
                    }
                }
                _ => {
                    let local =
                        model.local().context("--dump-ids is local-only")?;
                    let state = local.encode_state(&request.state)?;
                    for (id, q) in &request.questions {
                        let item = local.build_sequence(&state, id, q)?;
                        ids.insert(
                            id.clone(),
                            json!({"ids":item.ids,"markers":item.markers}),
                        );
                    }
                }
            }
            entry = json!({ "response": entry, "ids": ids });
        }
        out.push(entry);
    }
    let printed = if single {
        out.into_iter().next().context("no response")?
    } else {
        serde_json::Value::Array(out)
    };
    println!("{}", serde_json::to_string_pretty(&printed)?);
    Ok(())
}
