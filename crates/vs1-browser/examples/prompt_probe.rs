//! Capture controlled fixture states and generate prompt ablations (no policy actions).
#[allow(dead_code)]
#[path = "../src/browser.rs"]
mod browser;
#[allow(dead_code)]
#[path = "../src/policy.rs"]
mod policy;
use std::{fs, path::PathBuf};

use anyhow::Result;
use serde_json::{Value, json};

fn main() -> Result<()> {
    let out = PathBuf::from(
        std::env::args()
            .nth(1)
            .unwrap_or("artifacts/prompt-probe".into()),
    );
    anyhow::ensure!(!out.exists(), "choose a fresh output directory");
    fs::create_dir_all(&out)?;
    fs::write(
        out.join("fixture.html"),
        include_str!("../assets/fixture.html"),
    )?;
    let url =
        url::Url::from_file_path(out.join("fixture.html").canonicalize()?)
            .unwrap();
    let mut browser =
        browser::Browser::connect("http://127.0.0.1:9222", url.as_str())?;
    let goal = "Use the destination search and filters to find Design stays in Lisbon with Free cancellation, then open Casa Flora.";
    // Controlled setup is an oracle for evaluation only, never a model input.
    let cases = vec![
        (
            "initial",
            "query=''; searched=false; category='all'; free=false; travel();",
            false,
            vec!["TYPE_TEXT", "SELECT", "CLICK"],
        ),
        (
            "typed",
            "query='Lisbon'; searched=false; category='all'; free=false; travel();",
            false,
            vec!["CLICK", "SELECT"],
        ),
        (
            "searched",
            "query='Lisbon'; searched=true; category='all'; free=false; travel();",
            false,
            vec!["SELECT", "CLICK"],
        ),
        (
            "design",
            "query='Lisbon'; searched=true; category='Design'; free=false; travel();",
            false,
            vec!["CLICK"],
        ),
        (
            "filtered",
            "query='Lisbon'; searched=true; category='Design'; free=true; travel();",
            false,
            vec!["CLICK"],
        ),
        (
            "complete",
            "query='Lisbon'; searched=true; category='Design'; free=true; detail(places[0]);",
            true,
            vec!["DONE"],
        ),
        (
            "wrong_filters",
            "query=''; searched=false; category='all'; free=false; detail(places[0]);",
            false,
            vec!["CLICK"],
        ),
        (
            "wrong_property",
            "query='Lisbon'; searched=true; category='Design'; free=true; detail(places[1]);",
            false,
            vec!["CLICK"],
        ),
    ];
    let mut cases: Vec<_> = cases
        .into_iter()
        .map(|(name, setup, complete, valid)| {
            (name, setup.to_string(), complete, valid, goal)
        })
        .collect();
    let form_goal = "Enter Ada in Name, accept Terms, submit the form, and stop at Confirmation.";
    for (name, value, checked, complete, valid) in [
        ("form_empty", "", "", false, vec!["TYPE_TEXT", "CLICK"]),
        ("form_named", "Ada", "", false, vec!["CLICK"]),
        ("form_ready", "Ada", "checked", false, vec!["CLICK"]),
        ("form_complete", "Ada", "checked", true, vec!["DONE"]),
    ] {
        let html = if complete {
            "<h1>Confirmation</h1><p>Submitted Name: Ada. Terms accepted.</p><button>Start another form</button>".to_string()
        } else {
            format!(
                "<h1>Registration</h1><label>Name <input aria-label='Name' value='{value}'></label><label><input type='checkbox' {checked}>Terms</label><button>Submit</button>"
            )
        };
        let setup = format!(
            "document.title={}; document.body.innerHTML={};",
            json!(if complete {
                "Confirmation"
            } else {
                "Registration"
            }),
            json!(html)
        );
        cases.push((name, setup, complete, valid, form_goal));
    }
    let mut requests = vec![];
    let mut manifest = vec![];
    let mut captures = vec![];
    for (name, setup, complete, valid_operations, goal) in cases {
        browser.evaluate(&setup)?;
        let page = browser.observe()?;
        captures.push(json!({"case":name,"goal":goal,"page":page}));
        for variant in [
            "upstream",
            "compact",
            "short_instruction",
            "short_labels",
            "observations",
            "next_requirement",
            "facts",
            "flat_actions",
        ] {
            let (mut body, _) =
                policy::request(&page, goal, &[], variant != "upstream")?;
            let mut op = body["questions"]["operation"].clone();
            if !matches!(variant, "upstream" | "compact") {
                op["instructions"] = json!(
                    "Choose the next browser operation to accomplish the goal. DONE means the entire goal is already satisfied."
                );
            }
            if matches!(
                variant,
                "short_labels" | "observations" | "next_requirement"
            ) {
                let labels = json!({"CLICK":"Click a visible control", "TYPE_TEXT":"Type into a field", "SELECT":"Change a dropdown", "SCROLL_DOWN":"Scroll down", "WAIT":"Wait for loading", "DONE":"Goal fully achieved", "BLOCKED":"Cannot proceed"});
                for (k, v) in op["criteria"].as_object_mut().unwrap() {
                    *v = labels[k].clone();
                }
            }
            if matches!(variant, "observations" | "next_requirement") {
                let controls: Vec<Value> = page["actions"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .filter(|a| {
                        matches!(
                            a["kind"].as_str(),
                            Some("fill" | "select" | "click")
                        )
                    })
                    .cloned()
                    .collect();
                body["state"] = json!(format!(
                    "Goal: {goal}\nPage title: {}\nVisible page: {}\nControls: {}",
                    page["title"],
                    page["text"],
                    serde_json::to_string(&controls)?
                ));
            }
            if variant == "next_requirement" {
                op["instructions"] = json!(format!(
                    "Requirements: {goal} Compare each requirement with the observed page. Choose an operation for an unmet requirement. DONE only if every requirement is satisfied."
                ));
            }
            if matches!(variant, "facts" | "flat_actions") {
                let space = policy::action_space(&page)?;
                let controls: Vec<String> = space
                    .elements
                    .iter()
                    .map(|e| {
                        format!(
                            "{}: {} value={} checked={}",
                            e["role"],
                            e["label"],
                            e["value"],
                            e.get("checked").unwrap_or(&Value::Null)
                        )
                    })
                    .collect();
                body["state"] = json!(format!(
                    "Goal: {goal}\nPage title: {}\nCurrent controls:\n{}\nVisible text:\n{}",
                    page["title"],
                    controls.join("\n"),
                    page["text"].as_str().unwrap_or("")
                ));
                op["instructions"] = json!(
                    "What should the browser do next? Use current values. Stop only after completing the entire goal."
                );
                let labels = json!({"CLICK":"Click", "TYPE_TEXT":"Type", "SELECT":"Select dropdown", "SCROLL_DOWN":"Scroll", "WAIT":"Wait", "DONE":"Task complete", "BLOCKED":"Cannot proceed"});
                for (k, v) in op["criteria"].as_object_mut().unwrap() {
                    *v = labels[k].clone();
                }
                if variant == "flat_actions" {
                    let mut criteria = serde_json::Map::new();
                    for a in page["actions"].as_array().unwrap() {
                        let label = format!(
                            "{} {}",
                            a["kind"].as_str().unwrap(),
                            a["label"].as_str().unwrap()
                        );
                        criteria.insert(
                            a["id"].as_str().unwrap().to_string(),
                            json!(label),
                        );
                    }
                    criteria.insert("DONE".into(), json!("Task complete"));
                    op["criteria"] = json!(criteria);
                }
            }
            // Same observation for independent completion probes; no labels in input.
            body["questions"] = json!({"operation":op,
                "complete":{"type":"noul","instructions":format!("Has this entire task already been completed? {goal}")},
                "completion_choice":{"type":"choice","instructions":format!("Is the entire task completed? {goal}"),"criteria":{"unfinished":"One or more requirements remain unfinished","complete":"All requirements are already satisfied"}}
            });
            requests.push(body);
            manifest.push(json!({"case":name,"variant":variant,"complete":complete,"valid_operations":valid_operations}));
        }
    }
    browser.call("Target.closeTarget", json!({"targetId":browser.target}))?;
    for (name, value) in [
        ("requests.json", json!(requests)),
        ("manifest.json", json!(manifest)),
        ("captures.json", json!(captures)),
    ] {
        fs::write(out.join(name), serde_json::to_vec_pretty(&value)?)?;
    }
    Ok(())
}
