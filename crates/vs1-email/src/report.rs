use std::path::PathBuf;

use anyhow::{Context, Result, ensure};
use serde::Serialize;
use vs1::{SystemOneRequest, SystemOneResponse};

use crate::{
    Classification,
    Config,
    Mailbox,
    MessageFailure,
    classification::classify_batch,
    classification_request,
    split_body,
};

#[derive(Serialize)]
pub struct DryRunReport {
    pub dry_run: bool,
    pub mailbox: PathBuf,
    pub failures: Vec<MessageFailure>,
    pub classifications: Vec<Classification>,
}

pub fn dry_run(
    config: &Config,
    mailbox: &Mailbox,
    batch_size: usize,
    fits: &mut impl FnMut(&SystemOneRequest) -> Result<bool>,
    decide: &mut impl FnMut(&[SystemOneRequest]) -> Result<Vec<SystemOneResponse>>,
) -> Result<DryRunReport> {
    dry_run_with_progress(
        config,
        mailbox,
        batch_size,
        fits,
        decide,
        &mut |_| Ok(()),
    )
}

/// Calls progress after each complete email; earlier records survive later errors.
pub fn dry_run_with_progress(
    config: &Config,
    mailbox: &Mailbox,
    batch_size: usize,
    fits: &mut impl FnMut(&SystemOneRequest) -> Result<bool>,
    decide: &mut impl FnMut(&[SystemOneRequest]) -> Result<Vec<SystemOneResponse>>,
    progress: &mut impl FnMut(&Classification) -> Result<()>,
) -> Result<DryRunReport> {
    ensure!(batch_size > 0, "batch size must be greater than zero");
    let mut classifications = Vec::with_capacity(mailbox.emails.len());
    let mut failures = mailbox.failures.clone();
    for emails in mailbox.emails.chunks(batch_size) {
        let mut jobs = Vec::new();
        for (index, email) in emails.iter().enumerate() {
            let mut part = email.clone();
            let bodies = split_body(&email.body, &mut |text| {
                part.body = text.to_owned();
                fits(&classification_request(config, &part))
            })
            .with_context(|| format!("cannot chunk {}", email.path.display()));
            let bodies = match bodies {
                Ok(bodies) => bodies,
                Err(error) => {
                    failures.push(MessageFailure {
                        path: email.path.clone(),
                        error: format!("{error:#}"),
                    });
                    continue;
                }
            };
            for body in bodies {
                part.body = body.to_owned();
                jobs.push((index, body.chars().count(), part.clone()));
            }
        }
        let mut results =
            (0..emails.len()).map(|_| Vec::new()).collect::<Vec<_>>();
        for batch in jobs.chunks(batch_size) {
            let parts = batch
                .iter()
                .map(|(_, _, part)| part.clone())
                .collect::<Vec<_>>();
            let responses = classify_batch(config, &parts, decide)?;
            for ((index, chars, _), result) in batch.iter().zip(responses) {
                results[*index].push((*chars, result));
            }
        }
        for (email, chunks) in emails.iter().zip(results) {
            if !chunks.is_empty() {
                let classification = aggregate(config, email, chunks)?;
                progress(&classification)?;
                classifications.push(classification);
            }
        }
    }
    Ok(DryRunReport {
        dry_run: true,
        mailbox: mailbox.path.clone(),
        failures,
        classifications,
    })
}

fn aggregate(
    config: &Config,
    email: &crate::Email,
    chunks: Vec<(usize, Classification)>,
) -> Result<Classification> {
    use serde_json::{Map, json};

    use crate::classification::ChunkEvidence;
    let total: usize = chunks.iter().map(|(chars, _)| (*chars).max(1)).sum();
    let mut scores = vec![0.0f64; config.rules().len()];
    let mut usage = vs1::Usage::default();
    let mut evidence = Vec::new();
    let model = chunks.first().context("no chunks")?.1.model.clone();
    for (chars, chunk) in chunks {
        ensure!(chunk.model == model, "mixed models in one email");
        for (i, rule) in config.rules().iter().enumerate() {
            scores[i] += chunk.probabilities[&rule.category]
                .as_f64()
                .context("invalid chunk probability")?
                * chars.max(1) as f64
                / total as f64;
        }
        usage.input_tokens += chunk.usage.input_tokens;
        usage.output_tokens += chunk.usage.output_tokens;
        evidence.push(ChunkEvidence {
            decisions: chunk.decisions,
            body_chars: chars,
            category: chunk.category,
            confidence: chunk.confidence,
            probabilities: chunk.probabilities,
            usage: chunk.usage,
        });
    }
    // Stable ties follow configuration order.
    let best = (0..scores.len())
        .reduce(|a, b| if scores[b] > scores[a] { b } else { a })
        .context("no categories")?;
    let entropy: f64 = scores
        .iter()
        .filter(|p| **p > 0.0)
        .map(|p| -p * p.ln())
        .sum();
    let confidence =
        (1.0 - entropy / (scores.len() as f64).ln()).clamp(0.0, 1.0) as f32;
    let probabilities: Map<String, serde_json::Value> = config
        .rules()
        .iter()
        .zip(scores)
        .map(|(r, p)| (r.category.clone(), json!(p)))
        .collect();
    Ok(Classification {
        path: email.path.clone(),
        message_id: email.message_id.clone(),
        subject: email.subject.clone(),
        category: config.rules()[best].category.clone(),
        confidence,
        probabilities: json!(probabilities),
        model,
        usage,
        chunks: evidence,
        decisions: serde_json::Value::Null,
    })
}
/// One complete email and all categories per request, for hosted models with
/// larger context/choice budgets. Never truncates or splits the body. Oversized
/// requests are rejected by the provider. Raw provider answers are retained.
pub fn dry_run_single_choice_with_progress(
    config: &Config,
    mailbox: &Mailbox,
    batch_size: usize,
    decide: &mut impl FnMut(&[SystemOneRequest]) -> Result<Vec<SystemOneResponse>>,
    progress: &mut impl FnMut(&Classification) -> Result<()>,
) -> Result<DryRunReport> {
    ensure!(batch_size > 0, "batch size must be greater than zero");
    ensure!(
        config.rules().len() <= 255,
        "Jev supports at most 255 categories"
    );
    let mut classifications = Vec::with_capacity(mailbox.emails.len());
    for emails in mailbox.emails.chunks(batch_size) {
        let mut raw = Vec::new();
        let results = crate::classification::classify_batch_all(
            config,
            emails,
            &mut |requests| {
                let mut responses = decide(requests)?;
                raw = responses.clone();
                for response in &mut responses {
                    normalize_hosted_choices(response)?;
                }
                Ok(responses)
            },
        )?;
        for ((email, mut result), original) in
            emails.iter().zip(results).zip(raw)
        {
            result.decisions = serde_json::json!({"provider":original,"evaluated":result.decisions});
            let classification = aggregate(
                config,
                email,
                vec![(email.body.chars().count(), result)],
            )?;
            progress(&classification)?;
            classifications.push(classification);
        }
    }
    Ok(DryRunReport {
        dry_run: true,
        mailbox: mailbox.path.clone(),
        failures: mailbox.failures.clone(),
        classifications,
    })
}
/// The email policy uses probability argmax. Repair only small provider rounding
/// discrepancies; the library's JevClient itself always returns unmodified answers.
fn normalize_hosted_choices(response: &mut SystemOneResponse) -> Result<()> {
    for answer in response.answers.values_mut() {
        let vs1::Answer::Choice(a) = answer else {
            anyhow::bail!("expected hosted category choice")
        };
        ensure!(
            a.probabilities.contains_key(&a.choice),
            "unknown provider choice"
        );
        ensure!(
            a.probabilities
                .values()
                .all(|p| p.is_finite() && (0.0..=1.0).contains(p)),
            "invalid provider probability"
        );
        let sum = a.probabilities.values().sum::<f32>();
        ensure!(
            (sum - 1.0).abs() <= 0.025,
            "provider probabilities do not sum to one within rounding tolerance"
        );
        for p in a.probabilities.values_mut() {
            *p /= sum;
        }
        a.choice = a
            .probabilities
            .iter()
            .reduce(|a, b| if b.1 > a.1 { b } else { a })
            .context("no probabilities")?
            .0
            .clone();
    }
    Ok(())
}
