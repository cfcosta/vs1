//! JSON-defined browser scenarios with inline setup and verification scripts.
use std::{
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Map, Value, json};

use crate::{browser, model::Backend, policy};

#[derive(Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum Mode {
    #[default]
    Constrained,
    Agent,
}
#[derive(Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum ExpectedStatus {
    Done,
    Blocked,
}
fn deserialize_expectation<'de, D, T>(
    deserializer: D,
) -> Result<Option<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    // Omission is allowed; an explicit null is not an expectation.
    T::deserialize(deserializer).map(Some)
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Plan {
    #[serde(default)]
    mode: Mode,
    source: Source,
    #[serde(default)]
    scripts: std::collections::BTreeMap<String, Vec<String>>,
    variants: Vec<Variant>,
    verify: Vec<String>,
    goal: String,
    #[serde(default, deserialize_with = "deserialize_expectation")]
    expect_status: Option<ExpectedStatus>,
    #[serde(default, deserialize_with = "deserialize_expectation")]
    max_actions: Option<u64>,
    #[serde(default)]
    steps: Vec<Step>,
    #[serde(default)]
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
            && !plan.verify.is_empty()
            && !plan.variants.is_empty(),
        "scenario needs goal, verify, and variants"
    );
    match plan.mode {
        Mode::Constrained => {
            ensure!(
                plan.expect_status.is_none() && plan.max_actions.is_none(),
                "expect_status and max_actions require agent mode"
            );
            ensure!(
                !plan.steps.is_empty() && !plan.completion.is_empty(),
                "constrained mode needs steps and completion"
            );
        }
        Mode::Agent => ensure!(
            plan.steps.is_empty() && plan.completion.is_empty(),
            "agent mode uses goal and verify, not fixed steps or completion"
        ),
    }
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
pub(crate) fn verification(
    browser: &mut browser::Browser,
    lines: &[String],
    page: &Value,
) -> Value {
    // A fresh observed snapshot is available as `page`; scripts may also inspect the DOM.
    let expression =
        format!("((page) => eval({}))({})", json!(lines.join("\n")), page);
    match script(browser, &[expression]) {
        Ok(value) if value.is_boolean() => json!({"passed":value}),
        Ok(_) => {
            json!({"passed":false,"error":"verify script must return a boolean"})
        }
        Err(e) => json!({"passed":false,"error":e.to_string()}),
    }
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
    let independent = verification(browser, &plan.verify, &page);
    Ok(
        json!({"passed":error.is_none() && satisfied(&page,&plan.completion) && independent["passed"]==true,
        "error":error,"model_calls":model_calls,"forced_singletons":forced,"independent_verification":independent,
        "elapsed_ms":started.elapsed().as_secs_f64()*1000.0,"trace":trace}),
    )
}
fn evaluate_agent_result(plan: &Plan, result: &mut Value) {
    let expected_status =
        plan.expect_status.as_ref().unwrap_or(&ExpectedStatus::Done);
    let mut failures = vec![];
    if result["status"] != json!(expected_status) {
        failures.push("status_mismatch");
    }
    if result["verification"]["passed"] != true {
        failures.push("verifier_failure");
    }
    if plan.max_actions.is_some_and(|limit| {
        result["actions"]
            .as_u64()
            .is_some_and(|actions| actions > limit)
    }) {
        failures.push("too_many_actions");
    }
    result["passed"] = json!(failures.is_empty() && result["error"].is_null());
    result["expectation_failures"] = json!(failures);
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
        plan.mode == Mode::Agent || args.policy == "questions",
        "constrained scenarios support only --policy questions; use an agent-mode scenario for native policy"
    );
    ensure!(args.max_steps > 0, "max-steps must be positive");
    ensure!(
        plan.mode != Mode::Agent || args.chooser == "model",
        "agent mode requires the model chooser"
    );
    ensure!(
        plan.mode == Mode::Agent || (!args.record && !args.screenshots),
        "recording flags require agent mode"
    );
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
    for (variant_index, variant) in plan.variants.iter().enumerate() {
        for run in 0..args.repeat {
            let setup = Instant::now();
            let mut browser =
                browser::Browser::connect(&args.cdp, url.as_str())?;
            let attempt = (|| -> Result<Value> {
                for name in &variant.setup {
                    script(&mut browser, &plan.scripts[name])
                        .with_context(|| format!("setup script {name}"))?;
                }
                match plan.mode {
                    Mode::Constrained => execute(
                        &mut browser,
                        &plan,
                        model.as_ref(),
                        &args.retrieval,
                    ),
                    Mode::Agent => {
                        let folder = args.output.join(format!(
                            "variant-{variant_index:02}-run-{run:02}"
                        ));
                        fs::create_dir(&folder)?;
                        let mut summary = crate::run_agent_page(
                            args,
                            model.as_ref().context("agent needs model")?,
                            &plan.goal,
                            &folder,
                            &mut browser,
                            setup,
                            Some(&plan.verify),
                        )?;
                        summary["model_calls"] = summary["decisions"].clone();
                        Ok(summary)
                    }
                }
            })();
            let mut result = attempt.unwrap_or_else(
                |e| json!({"passed":false,"error":format!("{e:#}")}),
            );
            if plan.mode == Mode::Agent {
                evaluate_agent_result(&plan, &mut result);
            }
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
                    &json!({"chooser":args.chooser,"retrieval":args.retrieval,"policy":args.policy,"configuration":model.as_ref().map(|m|&m.metadata),"scenario":scenario,"source_url":url.as_str(),"runs":results}),
                )?,
            )?;
        }
    }
    ensure!(
        results.iter().all(|r| r["result"]["passed"] == true),
        "one or more scenario runs failed; see results.json"
    );
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    fn parse_agent_scenario() -> Value {
        serde_json::from_str(include_str!(
            "../../../examples/reading-room-agent.json"
        ))
        .unwrap()
    }
    #[test]
    fn agent_expectations_default_to_done_without_an_action_limit() {
        let plan: Plan =
            serde_json::from_value(parse_agent_scenario()).unwrap();
        validate(&plan).unwrap();
        assert!(plan.expect_status.is_none());
        assert!(plan.max_actions.is_none());
        let mut result = json!({"status":"done","error":null,"actions":100,
            "verification":{"passed":true}});
        evaluate_agent_result(&plan, &mut result);
        assert_eq!(result["passed"], true);
        assert_eq!(result["expectation_failures"], json!([]));
    }
    #[test]
    fn agent_expectations_accept_terminal_statuses_and_nonnegative_limits() {
        for status in ["done", "blocked"] {
            for limit in [0, 1, u64::MAX] {
                let mut raw = parse_agent_scenario();
                raw["expect_status"] = json!(status);
                raw["max_actions"] = json!(limit);
                let plan: Plan = serde_json::from_value(raw).unwrap();
                validate(&plan).unwrap();
                assert_eq!(json!(plan.expect_status), status);
                assert_eq!(plan.max_actions, Some(limit));
            }
        }
    }
    #[test]
    fn invalid_agent_expectations_are_rejected() {
        for (field, values) in [
            (
                "expect_status",
                json!(["ready", "error", "DONE", "", 0, true, null, [], {}]),
            ),
            (
                "max_actions",
                json!([
                    -1,
                    1.5,
                    1.0,
                    "0",
                    true,
                    null,
                    [],
                    {},
                    18446744073709551616.0
                ]),
            ),
        ] {
            for value in values.as_array().unwrap() {
                let mut raw = parse_agent_scenario();
                raw[field] = value.clone();
                assert!(
                    serde_json::from_value::<Plan>(raw).is_err(),
                    "{field}={value}"
                );
            }
        }
    }
    #[test]
    fn constrained_mode_rejects_agent_expectations() {
        for (field, value) in [
            ("expect_status", json!("done")),
            ("expect_status", json!("blocked")),
            ("max_actions", json!(0)),
            ("max_actions", json!(1)),
        ] {
            let mut raw: Value = serde_json::from_str(include_str!(
                "../../../examples/hotel.json"
            ))
            .unwrap();
            raw[field] = value;
            let plan: Plan = serde_json::from_value(raw).unwrap();
            assert_eq!(
                validate(&plan).unwrap_err().to_string(),
                "expect_status and max_actions require agent mode"
            );
        }
    }
    #[test]
    fn unknown_scenario_fields_are_rejected() {
        let mut raw = parse_agent_scenario();
        raw["expected_status"] = json!("done");
        assert!(serde_json::from_value::<Plan>(raw).is_err());
    }
    #[test]
    fn agent_results_require_the_expected_terminal_status() {
        for expectation in [None, Some("done"), Some("blocked")] {
            let mut raw = parse_agent_scenario();
            if let Some(status) = expectation {
                raw["expect_status"] = json!(status);
            }
            let plan: Plan = serde_json::from_value(raw).unwrap();
            for status in ["done", "blocked", "error", "ready"] {
                let mut result = json!({"status":status,"error":null,"actions":0,
                    "verification":{"passed":true}});
                evaluate_agent_result(&plan, &mut result);
                let matches_expectation =
                    status == expectation.unwrap_or("done");
                assert_eq!(result["passed"], matches_expectation);
                assert_eq!(
                    result["expectation_failures"],
                    if matches_expectation {
                        json!([])
                    } else {
                        json!(["status_mismatch"])
                    }
                );
            }
        }
    }
    #[test]
    fn agent_action_limits_include_the_boundary_and_allow_zero_actions() {
        for limit in [0, 2] {
            let mut raw = parse_agent_scenario();
            raw["max_actions"] = json!(limit);
            let plan: Plan = serde_json::from_value(raw).unwrap();
            for actions in 0..=limit + 1 {
                let mut result = json!({"status":"done","error":null,"actions":actions,
                    "decisions":10,"verification":{"passed":true}});
                evaluate_agent_result(&plan, &mut result);
                assert_eq!(result["passed"], actions <= limit);
                assert_eq!(
                    result["expectation_failures"],
                    if actions <= limit {
                        json!([])
                    } else {
                        json!(["too_many_actions"])
                    }
                );
            }
        }
    }
    #[test]
    fn agent_results_require_verification_for_both_terminal_statuses() {
        for status in ["done", "blocked"] {
            let mut raw = parse_agent_scenario();
            raw["expect_status"] = json!(status);
            let plan: Plan = serde_json::from_value(raw).unwrap();
            for verification in [
                json!({"passed":false}),
                json!({"passed":false,"error":"verify script failed"}),
            ] {
                let mut result = json!({"status":status,"error":null,"actions":0,
                    "verification":verification});
                evaluate_agent_result(&plan, &mut result);
                assert_eq!(result["passed"], false);
                assert_eq!(
                    result["expectation_failures"],
                    json!(["verifier_failure"])
                );
            }
        }
    }
    #[test]
    fn agent_results_report_all_failed_expectations() {
        let mut raw = parse_agent_scenario();
        raw["expect_status"] = json!("blocked");
        raw["max_actions"] = json!(0);
        let plan: Plan = serde_json::from_value(raw).unwrap();
        let mut result = json!({"status":"done","error":null,"actions":1,
            "verification":{"passed":false}});
        evaluate_agent_result(&plan, &mut result);
        assert_eq!(result["passed"], false);
        assert_eq!(
            result["expectation_failures"],
            json!(["status_mismatch", "verifier_failure", "too_many_actions"])
        );
    }
    #[test]
    fn agent_run_errors_still_fail_when_expectations_pass() {
        let plan: Plan =
            serde_json::from_value(parse_agent_scenario()).unwrap();
        let mut result = json!({"status":"done","error":"run failed","actions":0,
            "verification":{"passed":true}});
        evaluate_agent_result(&plan, &mut result);
        assert_eq!(result["passed"], false);
        assert_eq!(result["expectation_failures"], json!([]));
    }
    #[test]
    fn scenario_files_validate_and_resolve_relative_fixtures() {
        let examples =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples");
        let mut files: Vec<_> = fs::read_dir(examples)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| path.extension().is_some_and(|ext| ext == "json"))
            .collect();
        files.sort();
        assert!(!files.is_empty());
        for file in files {
            let plan: Plan = serde_json::from_slice(&fs::read(&file).unwrap())
                .unwrap_or_else(|error| panic!("{}: {error}", file.display()));
            validate(&plan)
                .unwrap_or_else(|error| panic!("{}: {error}", file.display()));
            let url = plan.source.resolve(&file).unwrap();
            match plan.source {
                Source::File { .. } => {
                    assert!(url.to_file_path().unwrap().is_file())
                }
                Source::Url { .. } => assert_eq!(url.scheme(), "https"),
            }
        }
    }
    #[test]
    fn modes_require_distinct_planning_contracts() {
        let raw = fs::read(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../examples/hotel.json"),
        )
        .unwrap();
        let mut plan: Plan = serde_json::from_slice(&raw).unwrap();
        plan.mode = Mode::Agent;
        assert!(validate(&plan).is_err());
        plan.steps.clear();
        plan.completion.clear();
        validate(&plan).unwrap();
        plan.mode = Mode::Constrained;
        assert!(validate(&plan).is_err());
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
