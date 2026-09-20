//! JSON-defined browser scenarios with inline setup and verification scripts.
use std::{
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use serde_json::{Map, Value, json};

use crate::{browser, model::Backend, policy};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Plan {
    source: Source,
    #[serde(default)]
    scripts: std::collections::BTreeMap<String, Vec<String>>,
    variants: Vec<Variant>,
    verify: Vec<String>,
    goal: String,
    steps: Vec<Step>,
    completion: Vec<Check>,
}
#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum Source {
    File {
        path: PathBuf,
        #[serde(default)]
        query: Option<String>,
    },
    Url {
        url: String,
    },
}
impl Source {
    fn resolve(&self, scenario: &Path) -> Result<url::Url> {
        match self {
            Self::File { path, query } => {
                let path = scenario
                    .parent()
                    .unwrap_or(Path::new("."))
                    .join(path)
                    .canonicalize()?;
                let mut url = url::Url::from_file_path(path)
                    .map_err(|_| anyhow::anyhow!("invalid fixture path"))?;
                url.set_query(query.as_deref());
                Ok(url)
            }
            Self::Url { url } => {
                let url = url::Url::parse(url)?;
                ensure!(
                    matches!(url.scheme(), "http" | "https"),
                    "source URL must be HTTP(S)"
                );
                Ok(url)
            }
        }
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Variant {
    name: String,
    #[serde(default)]
    setup: Vec<String>,
}
fn validate(plan: &Plan) -> Result<()> {
    ensure!(
        !plan.goal.trim().is_empty()
            && !plan.steps.is_empty()
            && !plan.completion.is_empty()
            && !plan.verify.is_empty()
            && !plan.variants.is_empty(),
        "scenario needs goal, steps, completion, verify, and variants"
    );
    let mut names = std::collections::BTreeSet::new();
    for v in &plan.variants {
        ensure!(
            !v.name.is_empty() && names.insert(&v.name),
            "duplicate or empty variant name"
        );
        for script in &v.setup {
            ensure!(
                plan.scripts.get(script).is_some_and(|s| !s.is_empty()),
                "unknown or empty setup script {script}"
            );
        }
    }
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
    Ok(())
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
fn script(browser: &mut browser::Browser, lines: &[String]) -> Result<Value> {
    let result=browser.call("Runtime.evaluate",json!({"expression":lines.join("\n"),"returnByValue":true,"awaitPromise":true}))?;
    if let Some(details) = result.get("exceptionDetails") {
        anyhow::bail!(
            "scenario script failed: {}",
            details["exception"]["description"]
                .as_str()
                .or_else(|| details["text"].as_str())
                .unwrap_or("JavaScript exception")
        );
    }
    Ok(result["result"]["value"].clone())
}
fn execute(
    browser: &mut browser::Browser,
    plan: &Plan,
    model: Option<&Backend>,
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
                    response = model.decide(&request)?;
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
    let verification = script(browser, &plan.verify);
    let independent = match verification {
        Ok(value) if value.is_boolean() => json!({"passed":value}),
        Ok(_) => {
            json!({"passed":false,"error":"verify script must return a boolean"})
        }
        Err(e) => json!({"passed":false,"error":e.to_string()}),
    };
    Ok(
        json!({"passed":error.is_none() && satisfied(&page,&plan.completion) && independent["passed"]==true,
        "error":error,"model_calls":model_calls,"forced_singletons":forced,"independent_verification":independent,
        "elapsed_ms":started.elapsed().as_secs_f64()*1000.0,"trace":trace}),
    )
}
pub fn run(args: &crate::Cli, scenario: &Path) -> Result<()> {
    ensure!(
        args.repeat > 0 && !args.output.exists(),
        "positive repeat and fresh output required"
    );
    let bytes = fs::read(scenario)?;
    let plan: Plan = serde_json::from_slice(&bytes)?;
    validate(&plan)?;
    ensure!(
        plan.steps.len() <= args.max_steps,
        "scenario exceeds max-steps"
    );
    let url = plan.source.resolve(scenario)?;
    let model = if args.chooser == "model" {
        Some(Backend::load(&args.model)?)
    } else {
        None
    };
    fs::create_dir_all(&args.output)?;
    fs::write(args.output.join("scenario.json"), &bytes)?;
    let mut results = vec![];
    for variant in &plan.variants {
        for run in 0..args.repeat {
            let mut browser =
                browser::Browser::connect(&args.cdp, url.as_str())?;
            let attempt = (|| -> Result<Value> {
                for name in &variant.setup {
                    script(&mut browser, &plan.scripts[name])
                        .with_context(|| format!("setup script {name}"))?;
                }
                execute(&mut browser, &plan, model.as_ref(), &args.retrieval)
            })();
            let mut result = attempt.unwrap_or_else(
                |e| json!({"passed":false,"error":format!("{e:#}")}),
            );
            if let Err(e) = browser
                .call("Target.closeTarget", json!({"targetId":browser.target}))
            {
                result["passed"] = json!(false);
                result["cleanup_error"] = json!(e.to_string());
            }
            eprintln!(
                "{} {run}: passed={} calls={} error={}",
                variant.name,
                result["passed"],
                result["model_calls"],
                result["error"]
            );
            results.push(
                json!({"variant":variant.name,"run":run,"result":result}),
            );
            fs::write(
                args.output.join("results.json"),
                serde_json::to_vec_pretty(
                    &json!({"chooser":args.chooser,"retrieval":args.retrieval,"configuration":model.as_ref().map(|m|&m.metadata),"scenario":scenario,"source_url":url.as_str(),"runs":results}),
                )?,
            )?;
        }
    }
    ensure!(
        results.iter().all(|r| r["result"]["passed"] == true),
        "one or more scenario runs failed independent verification; see results.json"
    );
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn scenario_files_validate_and_resolve_relative_fixtures() {
        for name in ["hotel.json", "reading-room.json"] {
            let file = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../examples")
                .join(name);
            let plan: Plan =
                serde_json::from_slice(&fs::read(&file).unwrap()).unwrap();
            validate(&plan).unwrap();
            assert!(
                plan.source
                    .resolve(&file)
                    .unwrap()
                    .to_file_path()
                    .unwrap()
                    .is_file()
            );
        }
    }
    #[test]
    fn invalid_setup_references_and_duplicate_variants_are_rejected() {
        let raw = fs::read(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../examples/hotel.json"),
        )
        .unwrap();
        let mut plan: Plan = serde_json::from_slice(&raw).unwrap();
        plan.variants[0].setup.push("missing".into());
        assert!(validate(&plan).is_err());
        plan.variants[0].setup.clear();
        plan.variants[1].name = plan.variants[0].name.clone();
        assert!(validate(&plan).is_err());
    }
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
