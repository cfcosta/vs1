//! Paired local retrieval experiment. Never writes to the mailbox.
use std::{fs, path::Path, time::Instant};

use anyhow::{Result, ensure};
use vs1::SystemOne;
use vs1_email::{
    Config,
    classification_request,
    dry_run,
    read_maildir,
    request_fits,
};
fn augment(
    request: &vs1::SystemOneRequest,
    examples: &serde_json::Value,
) -> vs1::SystemOneRequest {
    let mut request = request.clone();
    let vs1::State::Json(ref mut state) = request.state else {
        panic!("JSON email state required")
    };
    let key = format!(
        "{}\n{}",
        state["email"]["subject"].as_str().unwrap(),
        state["email"]["date"].as_str().unwrap()
    );
    state["labeled_examples"] = examples[&key].clone();
    request
}
#[test]
fn examples_are_separate_from_target() {
    let request = vs1::SystemOneRequest::new(
        serde_json::json!({"email":{"subject":"target", "date":"date", "body":"body"}}),
    );
    let examples = serde_json::json!({"target\ndate":[{"subject":"example","category":"receipts"}]});
    let changed = augment(&request, &examples);
    let state = serde_json::to_value(changed.state).unwrap();
    assert_eq!(state["email"]["body"], "body");
    assert_eq!(state["labeled_examples"][0]["category"], "receipts");
    assert!(
        serde_json::to_value(request.state)
            .unwrap()
            .get("labeled_examples")
            .is_none()
    );
}
fn main() -> Result<()> {
    let args = std::env::args().collect::<Vec<_>>();
    ensure!(
        args.len() == 6,
        "CONFIG MAILDIR NEW_OUTPUT_DIR SUBFOLDER EXAMPLES"
    );
    let config = Config::parse(&fs::read_to_string(&args[1])?)?;
    let mut mailbox = read_maildir(Path::new(&args[2]), 100)?;
    ensure!(
        mailbox.emails.len() == 100 && mailbox.failures.is_empty(),
        "expected 100 clean sample messages"
    );
    let examples: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&args[5])?)?;
    mailbox.emails.retain(|e| {
        examples.get(format!("{}\n{}", e.subject, e.date)).is_some()
    });
    ensure!(!mailbox.emails.is_empty(), "no selected messages");
    let keys = mailbox
        .emails
        .iter()
        .map(|e| format!("{}\n{}", e.subject, e.date))
        .collect::<std::collections::BTreeSet<_>>();
    ensure!(
        keys.len() == mailbox.emails.len(),
        "duplicate subject/date keys"
    );
    ensure!(
        examples.as_object().is_some_and(|m| m.len() == keys.len()),
        "unmatched example keys"
    );
    for key in keys {
        let entries = examples[&key]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("examples must be arrays"))?;
        ensure!(entries.len() == 2, "expected two examples per message");
        for entry in entries {
            ensure!(
                config
                    .rules()
                    .iter()
                    .any(|r| entry["category"] == r.category),
                "unknown example category"
            );
            ensure!(
                entry["subject"].is_string()
                    && entry["body"]
                        .as_str()
                        .is_some_and(|s| s.chars().count() <= 160),
                "invalid example text"
            );
        }
    }
    let out = Path::new(&args[3]);
    fs::create_dir(out)?;
    fs::write(
        out.join("emails.json"),
        serde_json::to_vec(&mailbox.emails)?,
    )?;
    let model: SystemOne = SystemOne::from(vs1::DEFAULT_REPO_ID)
        .with_subfolder(&args[4])
        .with_device(candle_core::Device::new_cuda(0)?)
        .with_batch_size(16)
        .try_into()?;
    eprintln!(
        "{} {:?} {:?}",
        model.model_name(),
        model.device(),
        model.dtype()
    );
    let mut email = mailbox.emails[0].clone();
    email.body.clear();
    model.system_one(&augment(
        &classification_request(&config, &email),
        &examples,
    ))?;
    for (run, retrieval) in [false, true, true, false].into_iter().enumerate() {
        let transform = |r: &vs1::SystemOneRequest| {
            if retrieval {
                augment(r, &examples)
            } else {
                r.clone()
            }
        };
        let start = Instant::now();
        let mut questions = 0;
        let report = dry_run(
            &config,
            &mailbox,
            16,
            &mut |r| request_fits(&model, &transform(r)),
            &mut |rs| {
                questions +=
                    rs.iter().map(|r| r.questions.len()).sum::<usize>();
                let augmented = rs.iter().map(transform).collect::<Vec<_>>();
                for r in &augmented {
                    ensure!(
                        request_fits(&model, r)?,
                        "augmented request overflow"
                    );
                }
                Ok(model.system_one_batch(&augmented)?)
            },
        )?;
        let seconds = start.elapsed().as_secs_f64();
        let summary = serde_json::json!({"retrieval":retrieval,"seconds":seconds,"questions":questions,"chunks":report.classifications.iter().map(|r|r.chunks.len()).sum::<usize>(),"messages":report.classifications.len(),"failures":report.failures.len()});
        fs::write(
            out.join(format!("results-{run}.json")),
            serde_json::to_vec(&report)?,
        )?;
        fs::write(
            out.join(format!("summary-{run}.json")),
            serde_json::to_vec_pretty(&summary)?,
        )?;
        eprintln!("{run}: {summary}");
    }
    Ok(())
}
