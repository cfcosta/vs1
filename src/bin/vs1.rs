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
use vs1::{SystemOne, SystemOneRequest};

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
    let mut model_id = vs1::DEFAULT_REPO_ID.to_string();
    let mut subfolder = String::new();
    let mut device_name = "cpu".to_string();
    let mut dtype: Option<DType> = None;
    let mut batch_size: Option<usize> = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--dump-ids" => dump_ids = true,
            "--model" => {
                model_id = args.next().context("--model needs a value")?
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
            other if path.is_none() => path = Some(other.to_string()),
            other => bail!("unexpected argument {other:?}"),
        }
    }
    let path = path.context(
        "usage: decide [--dump-ids] [--device cpu|cuda] request.json",
    )?;
    let raw = fs::read_to_string(&path)?;
    let (requests, single): (Vec<SystemOneRequest>, bool) =
        match serde_json::from_str(&raw) {
            Ok(list) => (list, false),
            Err(_) => (vec![serde_json::from_str(&raw)?], true),
        };

    let started = Instant::now();
    let mut builder = SystemOne::from(&model_id)
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

    let started = Instant::now();
    let responses = model.system_one_batch(&requests)?;
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
            let state = model.encode_state(&request.state)?;
            let mut ids = serde_json::Map::new();
            for (id, question) in &request.questions {
                let item = model.build_sequence(&state, id, question)?;
                ids.insert(
                    id.clone(),
                    json!({ "ids": item.ids, "markers": item.markers }),
                );
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
