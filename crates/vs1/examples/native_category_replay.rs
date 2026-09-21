//! Replay JSONL typed-decisions rows without exposing reference labels to inference.
use serde_json::{Value, json};
use vs1::SystemOneRequest;

fn request(row: &Value) -> serde_json::Result<SystemOneRequest> {
    let state: Value =
        serde_json::from_str(row["state"].as_str().unwrap_or("null"))?;
    let questions: Value =
        serde_json::from_str(row["questions"].as_str().unwrap_or("null"))?;
    serde_json::from_value(
        json!({"state":state,"questions":{"category":questions["category"]}}),
    )
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::{fs, io::Write, time::Instant};
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(args.len(), 3, "INPUT.jsonl OUTPUT.json");
    let start = Instant::now();
    let rows = fs::read_to_string(&args[1])?
        .lines()
        .map(serde_json::from_str::<Value>)
        .collect::<Result<Vec<_>, _>>()?;
    let requests = rows.iter().map(request).collect::<Result<Vec<_>, _>>()?;
    let model: vs1::SystemOne = vs1::SystemOne::from(vs1::DEFAULT_REPO_ID)
        .with_subfolder("typed-decisions")
        .with_device(candle_core::Device::new_cuda(0)?)
        .with_dtype(candle_core::DType::BF16)
        .with_batch_size(16)
        .try_into()?;
    let empty = model.encode_state(&vs1::State::from(""))?;
    for r in &requests {
        let state = model.encode_state(&r.state)?;
        let head = model.build_sequence(
            &empty,
            "category",
            &r.questions["category"],
        )?;
        assert!(
            state.len() + head.ids.len() <= model.config().max_len,
            "state would be truncated"
        );
    }
    let mut results = Vec::new();
    let mut call_seconds = 0.0;
    for (rs, source) in requests.chunks(16).zip(rows.chunks(16)) {
        let call = Instant::now();
        let answers = model.system_one_batch(rs)?;
        call_seconds += call.elapsed().as_secs_f64();
        assert_eq!(answers.len(), source.len());
        for (row, response) in source.iter().zip(answers) {
            results.push(json!({"id":row["id"],"split":row["split"],"response":response}));
        }
    }
    let output = json!({"results":results,"calls":requests.len(),"batch_calls":requests.len().div_ceil(16),"call_seconds":call_seconds,"total_seconds":start.elapsed().as_secs_f64(),"model":model.model_name()});
    fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[2])?
        .write_all(&serde_json::to_vec_pretty(&output)?)?;
    eprintln!("{} calls in {:.3}s", requests.len(), call_seconds);
    Ok(())
}

#[test]
fn replay_preserves_native_state_and_category_but_excludes_targets() {
    let state = json!({"account":{"tier":"standard"},"thread":[{"role":"customer","text":"Refund please"}]});
    let category = json!({"type":"choice","instructions":"What is this customer conversation primarily about?","criteria":{"billing":"A charge problem.","refund":"Asking for money back."}});
    let row = json!({"state":state.to_string(),"questions":json!({"category":category,"action":{"type":"noul"}}).to_string(),"gold":"SECRET","factors":"SECRET","label_agreement":"SECRET"});
    let actual = serde_json::to_value(request(&row).unwrap()).unwrap();
    assert_eq!(actual["state"], state);
    assert_eq!(actual["questions"], json!({"category":category}));
    assert!(!actual.to_string().contains("SECRET"));
}
