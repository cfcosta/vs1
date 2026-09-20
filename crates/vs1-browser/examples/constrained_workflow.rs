//! Experimental declarative workflow: observed postconditions, finite target choices.
#[allow(dead_code)]
#[path = "../src/browser.rs"]
mod browser;
#[allow(dead_code)]
#[path = "../src/policy.rs"]
mod policy;
#[allow(dead_code)]
#[path = "../src/verify.rs"]
mod verify;
use std::{fs, path::PathBuf, time::Instant};

use anyhow::{Context, Result, ensure};
use clap::Parser;
use serde::Deserialize;
use serde_json::{Map, Value, json};
use vs1::{SystemOne, SystemOneRequest};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    plan: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    checkpoint: String,
    #[arg(long, default_value="model", value_parser=["model","lexical"])]
    chooser: String,
    #[arg(long, default_value="none", value_parser=["none","overlap"])]
    retrieval: String,
    #[arg(long, default_value = "http://127.0.0.1:9222")]
    cdp: String,
    #[arg(long, default_value_t = 3)]
    repeat: usize,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Plan {
    goal: String,
    steps: Vec<Step>,
    completion: Vec<Check>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Step {
    instruction: String,
    kind: String,
    role: String,
    #[serde(default)]
    text: Option<String>,
    // Planner-supplied search terms for the lexical baseline and optional retrieval.
    terms: String,
    after: Vec<Check>,
}
#[derive(Deserialize)]
#[serde(tag = "check", rename_all = "snake_case", deny_unknown_fields)]
enum Check {
    Text { contains: String },
    Title { equals: String },
    UrlSuffix { suffix: String },
    Value { label: String, equals: String },
    Checked { label: String, equals: bool },
}
fn satisfied(page: &Value, checks: &[Check]) -> bool {
    !checks.is_empty()
        && checks.iter().all(|check| match check {
            Check::Text { contains } => {
                page["text"].as_str().is_some_and(|s| s.contains(contains))
            }
            Check::Title { equals } => page["title"] == *equals,
            Check::UrlSuffix { suffix } => {
                page["url"].as_str().is_some_and(|s| s.ends_with(suffix))
            }
            Check::Value { label, equals } => {
                page["actions"].as_array().is_some_and(|a| {
                    a.iter().any(|a| {
                        let name = a["label"]
                            .as_str()
                            .unwrap_or("")
                            .split(" → ")
                            .next()
                            .unwrap_or("");
                        name == label
                            && a.get("current_value").unwrap_or(&a["value"])
                                == equals
                    })
                })
            }
            Check::Checked { label, equals } => {
                page["actions"].as_array().is_some_and(|a| {
                    a.iter().any(|a| {
                        a["label"] == *label
                            && (a["checked"].as_bool().or_else(|| {
                                a["checked"]
                                    .as_str()
                                    .and_then(|s| s.parse::<bool>().ok())
                            })) == Some(*equals)
                    })
                })
            }
        })
}
fn words(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(str::to_lowercase)
        .collect()
}
fn overlaps(label: &str, terms: &str) -> bool {
    let label = words(label);
    words(terms).iter().any(|w| label.contains(w))
}
fn candidates(page: &Value, step: &Step) -> Vec<Value> {
    page["actions"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|a| a["kind"] == step.kind && a["role"] == step.role)
        .cloned()
        .collect()
}
fn lexical(step: &Step, candidates: &[Value]) -> usize {
    let terms: Vec<String> = step
        .terms
        .split_whitespace()
        .map(str::to_lowercase)
        .collect();
    candidates
        .iter()
        .enumerate()
        .max_by_key(|(i, a)| {
            let label = a["label"].as_str().unwrap_or("").to_lowercase();
            (
                terms
                    .iter()
                    .filter(|word| label.contains(word.as_str()))
                    .count(),
                std::cmp::Reverse(*i),
            )
        })
        .map(|(i, _)| i)
        .unwrap_or(0)
}
fn execute(
    browser: &mut browser::Browser,
    plan: &Plan,
    model: Option<&SystemOne>,
    retrieval: &str,
) -> Result<Value> {
    let started = Instant::now();
    let mut trace = vec![];
    let mut page = browser.observe()?;
    let mut model_calls = 0;
    let mut forced = 0;
    let mut error = None;
    if !satisfied(&page, &plan.completion) {
        for (index, step) in plan.steps.iter().enumerate() {
            if satisfied(&page, &step.after) {
                trace.push(json!({"step":index,"status":"already_satisfied"}));
                continue;
            }
            let tick = (|| -> Result<()> {
                let all = candidates(&page, step);
                let offered: Vec<Value> = all
                    .iter()
                    .filter(|a| {
                        retrieval == "none"
                            || overlaps(
                                a["label"].as_str().unwrap_or(""),
                                &step.terms,
                            )
                    })
                    .cloned()
                    .collect();
                ensure!(
                    !offered.is_empty(),
                    "no observed candidates for step {index}"
                );
                let mut criteria = Map::new();
                for a in &offered {
                    criteria.insert(
                        a["id"].as_str().context("missing id")?.into(),
                        json!(a["label"].as_str().unwrap_or("")),
                    );
                }
                let state = json!(format!(
                    "Page: {}\nTask: {}\nCurrent step: {}",
                    page["title"], plan.goal, step.instruction
                ));
                let request = json!({"state":state,"questions":{"target":{"type":"choice","instructions":step.instruction,"criteria":criteria}}});
                let mut response = Value::Null;
                let choice = if offered.len() == 1 {
                    forced += 1;
                    0
                } else if let Some(model) = model {
                    model_calls += 1;
                    let parsed: SystemOneRequest =
                        serde_json::from_value(request.clone())?;
                    response =
                        serde_json::to_value(model.system_one(&parsed)?)?;
                    let id = policy::validate_choice(
                        &response["answers"]["target"],
                        &criteria,
                    )?;
                    offered
                        .iter()
                        .position(|a| a["id"] == id)
                        .context("unknown target")?
                } else {
                    lexical(step, &offered)
                };
                let action = &offered[choice];
                // Existing freshness/visibility checks guard every actual mutation.
                browser.act(action, &page, step.text.as_deref())?;
                let after = browser.observe()?;
                let passed = satisfied(&after, &step.after);
                trace.push(json!({"step":index,"status":if passed {"verified"} else {"wrong_target_or_failed_effect"},
                    "candidate_count":offered.len(),"before_retrieval":all.len(),"action":action,"request":request,"response":response,"before":page,"after":after}));
                page = after;
                ensure!(
                    passed,
                    "postcondition failed at step {index}; no fallback or oracle correction"
                );
                Ok(())
            })();
            if let Err(e) = tick {
                error = Some(format!("{e:#}"));
                break;
            }
        }
    }
    let independent = verify::outcome(&page, Some("hotel"), None, &[]);
    Ok(
        json!({"passed":error.is_none() && satisfied(&page,&plan.completion) && independent["passed"]==true,
        "error":error,"model_calls":model_calls,"forced_singletons":forced,"independent_verification":independent,
        "elapsed_ms":started.elapsed().as_secs_f64()*1000.0,"trace":trace}),
    )
}
fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        args.repeat > 0 && !args.output.exists(),
        "positive repeat and fresh output required"
    );
    let plan: Plan = serde_json::from_slice(&fs::read(&args.plan)?)?;
    ensure!(
        !plan.steps.is_empty() && !plan.completion.is_empty(),
        "empty plan"
    );
    for s in &plan.steps {
        ensure!(
            !s.after.is_empty()
                && matches!(s.kind.as_str(), "fill" | "select" | "click"),
            "invalid step"
        );
        ensure!(
            s.kind != "fill" || s.text.is_some(),
            "fill requires explicit value"
        );
    }
    fs::create_dir_all(&args.output)?;
    fs::write(
        args.output.join("fixture.html"),
        include_str!("../assets/fixture.html"),
    )?;
    let url = url::Url::from_file_path(
        args.output.join("fixture.html").canonicalize()?,
    )
    .unwrap();
    let model: Option<SystemOne> = if args.chooser == "model" {
        Some(SystemOne::from(&args.checkpoint).try_into()?)
    } else {
        None
    };
    let mut results = vec![];
    for variant in [
        "base",
        "distractors",
        "reordered",
        "already_filtered",
        "complete",
    ] {
        for run in 0..args.repeat {
            let mut browser =
                browser::Browser::connect(&args.cdp, url.as_str())?;
            if matches!(variant, "distractors" | "reordered") {
                browser.evaluate("places.push({name:'Casa Azul',city:'Lisbon',category:'Design',free:true,price:110,description:'Another design stay'});travel();document.querySelector('#search').insertAdjacentHTML('afterbegin',`<input type='search' aria-label='Guest name'><button type='button'>Subscribe</button>`);document.querySelector('.filters').insertAdjacentHTML('afterbegin',`<label><input type='checkbox'>Breakfast</label>`)")?;
            }
            if variant == "reordered" {
                browser.evaluate("for(const parent of [document.querySelector('#search'),document.querySelector('.filters'),document.querySelector('#results')]) { for(const child of [...parent.children].reverse()) parent.append(child); }")?;
            }
            if variant == "already_filtered" {
                browser.evaluate("query='Lisbon';searched=true;category='Design';free=true;travel();")?;
            }
            if variant == "complete" {
                browser.evaluate("query='Lisbon';searched=true;category='Design';free=true;detail(places[0]);")?;
            }
            let result =
                execute(&mut browser, &plan, model.as_ref(), &args.retrieval)?;
            browser.call(
                "Target.closeTarget",
                json!({"targetId":browser.target}),
            )?;
            eprintln!(
                "{variant} {run}: passed={} calls={} error={}",
                result["passed"], result["model_calls"], result["error"]
            );
            results.push(json!({"variant":variant,"run":run,"result":result}));
            fs::write(
                args.output.join("results.json"),
                serde_json::to_vec_pretty(
                    &json!({"chooser":args.chooser,"retrieval":args.retrieval,"checkpoint":args.checkpoint,"plan":args.plan,"runs":results}),
                )?,
            )?;
        }
    }
    ensure!(
        results.iter().all(|r| r["result"]["passed"] == true),
        "one or more workflow runs failed independent verification; see results.json"
    );
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn retrieval_keeps_multiple_plausible_targets() {
        assert!(overlaps("View Casa Flora", "View Casa Flora"));
        assert!(overlaps("View Casa Azul", "View Casa Flora"));
        assert!(!overlaps("Find stays", "View Casa Flora"));
        assert!(!overlaps("Destination", "nation"));
    }
    #[test]
    fn missing_controls_are_unknown_not_satisfied() {
        assert!(!satisfied(
            &json!({"actions":[]}),
            &[Check::Value {
                label: "Destination".into(),
                equals: "Lisbon".into()
            }]
        ));
        assert!(!satisfied(&json!({}), &[]));
    }
    #[test]
    fn checkbox_value_on_is_not_checked() {
        let page = json!({"actions":[{"label":"Free cancellation","value":"on","checked":"false"}]});
        assert!(!satisfied(
            &page,
            &[Check::Checked {
                label: "Free cancellation".into(),
                equals: true
            }]
        ));
    }
}
