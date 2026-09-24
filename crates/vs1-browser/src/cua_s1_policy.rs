use std::collections::HashSet;

use anyhow::{Context, Result, ensure};
use serde::Serialize;
use serde_json::{Map, Value, json};
use vs1::{CuaS1Option, CuaS1OptionPrediction};

use crate::policy;

#[derive(Serialize)]
pub struct Request {
    pub app: String,
    pub task_family: String,
    pub goal: String,
    pub ax_tree: String,
    pub removed_visible_text_lines: usize,
    pub options: Vec<(CuaS1Option, Value)>,
    #[serde(skip)]
    space: policy::Space,
}

fn trim_visible_text(state: &str) -> (String, usize) {
    let Some((before_visible, visible)) = state.split_once("\nVisible text: ")
    else {
        return (state.into(), 0);
    };
    let Some((visible, after_visible)) = visible.rsplit_once("\nGraphics: ")
    else {
        return (state.into(), 0);
    };
    let Some(controls) = before_visible
        .split_once("\nControls:\n")
        .and_then(|(_, controls)| controls.rsplit_once("\nRecent actions: "))
        .map(|(controls, _)| controls)
    else {
        return (state.into(), 0);
    };
    // Control labels can span lines; only the complete label is a match.
    let labels: HashSet<_> = controls
        .split("\n[")
        .filter_map(|control| {
            let (_, control) = control.split_once("] ")?;
            let (_, control) = control.split_once(' ')?;
            let (label, _) = control.split_once(" value=")?;
            Some(label.trim())
        })
        .collect();
    let mut removed_visible_text_lines = 0;
    let visible = visible
        .split('\n')
        .filter(|line| {
            if labels.contains(line.trim()) {
                removed_visible_text_lines += 1;
                false
            } else {
                true
            }
        })
        .collect::<Vec<_>>()
        .join("\n");
    (
        format!(
            "{before_visible}\nVisible text: {visible}\nGraphics: {after_visible}"
        ),
        removed_visible_text_lines,
    )
}

pub fn build_request(
    page: &Value,
    goal: &str,
    history: &[Value],
    _compact: bool,
) -> Result<Request> {
    // Cua-S1 always needs textual state, including for upstream policy prompts.
    let (body, space) = policy::request(page, goal, history, true)?;
    let (ax_tree, removed_visible_text_lines) = trim_visible_text(
        body["state"].as_str().context("missing textual state")?,
    );
    let title = page["title"].as_str().unwrap_or("");
    let app = if title.is_empty() {
        let url = page["url"]
            .as_str()
            .filter(|url| !url.is_empty())
            .unwrap_or("web page");
        url::Url::parse(url)
            .ok()
            .and_then(|url| url.host_str().map(str::to_owned))
            .unwrap_or_else(|| url.to_owned())
    } else {
        title.to_owned()
    };
    let mut options = vec![];
    for (operation, candidates) in &space.targets {
        for (target, action) in candidates.as_object().unwrap() {
            let index = target.split(':').next().unwrap().parse::<usize>()?;
            let element = &space.elements[index - 1];
            let role = element["role"].as_str().unwrap_or("");
            let label = element["label"].as_str().unwrap_or("");
            let label = match operation.as_str() {
                "SELECT" => action["label"].as_str().unwrap_or("").to_owned(),
                "PRESS_KEY" => format!(
                    "{label} (key {})",
                    action["key"].as_str().context("missing key")?
                ),
                _ => label.to_owned(),
            };
            let verb = match operation.as_str() {
                "TYPE_TEXT" => "fill",
                "CLICK" if matches!(role, "checkbox" | "radio" | "switch") => {
                    "check"
                }
                _ => "click",
            };
            options.push((
                CuaS1Option {
                    element_id: action["id"]
                        .as_str()
                        .context("missing ID")?
                        .into(),
                    role: role.into(),
                    label,
                    action: verb.into(),
                    entity_id: None,
                },
                action.clone(),
            ));
        }
    }
    for action in space.controls.values() {
        options.push((
            CuaS1Option {
                element_id: action["id"].as_str().context("missing ID")?.into(),
                role: "page".into(),
                label: action["label"].as_str().unwrap_or("").into(),
                action: "click".into(),
                entity_id: None,
            },
            action.clone(),
        ));
    }
    for (id, label) in [
        ("DONE", "Finish: every requirement is visibly satisfied"),
        ("BLOCKED", "Give up: no supported operation can progress"),
    ] {
        options.push((
            CuaS1Option {
                element_id: id.into(),
                role: "task".into(),
                label: label.into(),
                action: "click".into(),
                entity_id: None,
            },
            json!({"id":id,"kind":"terminal","label":id}),
        ));
    }
    let mut ids = HashSet::new();
    ensure!(
        options
            .iter()
            .all(|(option, _)| ids.insert(&option.element_id)),
        "duplicate action IDs"
    );
    Ok(Request {
        app,
        task_family: "web_navigation".into(),
        goal: goal.into(),
        ax_tree,
        removed_visible_text_lines,
        options,
        space,
    })
}

pub fn resolve(
    request: &Request,
    predictions: &[CuaS1OptionPrediction],
) -> Result<Value> {
    ensure!(
        predictions.len() == request.options.len()
            && predictions.iter().zip(&request.options).all(
                |(prediction, (option, _))| {
                    prediction.option.element_id == option.element_id
                }
            ),
        "native predictions do not match observed candidates"
    );
    ensure!(
        predictions.iter().filter(|p| p.is_selected).count() == 1,
        "native predictions must select exactly one option"
    );
    let probabilities: Map<String, Value> = predictions
        .iter()
        .map(|p| (p.option.element_id.clone(), json!(p.probability)))
        .collect();
    let probs: Vec<_> = predictions.iter().map(|p| p.probability).collect();
    ensure!(
        probs
            .iter()
            .all(|p| p.is_finite() && (0.0..=1.0).contains(p))
            && (probs.iter().sum::<f32>() - 1.0).abs() < 0.02,
        "invalid native probability distribution"
    );
    // Tournament winners need not maximize the hierarchical probability.
    let selected = predictions.iter().position(|p| p.is_selected).unwrap();
    let action = &request.options[selected].1;
    let mut operation =
        action["id"].as_str().context("missing ID")?.to_uppercase();
    let mut target = Value::Null;
    for (name, candidates) in &request.space.targets {
        if let Some((index, _)) = candidates
            .as_object()
            .unwrap()
            .iter()
            .find(|(_, a)| a["id"] == action["id"])
        {
            operation = name.clone();
            target = json!(index);
            break;
        }
    }
    Ok(
        json!({"choice":action["id"],"operation":operation,"target":target,"action":action,
        "confidence":vs1::head::confidence_from_probs(&probs, probs.len()),"probabilities":probabilities,
        "options":predictions,"model":vs1::cua_s1::MODEL_NAME,
        "usage":{"input_tokens":null,"output_tokens":0,"forward_passes":predictions[selected].forward_passes}}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn removes_visible_text_lines_matching_trimmed_control_labels() {
        let state = "Goal: Find a stay\nPage: Stays\nControls:\n[1] button   Search  value=\"\"\n[2] link Log in value=null\nRecent actions: []\nVisible text: \tSearch \nIntro\n Log in\t\nSearch\n\nEnd\nGraphics: []\nRecovery: {}";
        let (trimmed, removed) = trim_visible_text(state);
        assert_eq!(removed, 3);
        assert_eq!(
            trimmed,
            state.replace(
                "\tSearch \nIntro\n Log in\t\nSearch\n\nEnd",
                "Intro\n\nEnd"
            )
        );
    }

    #[test]
    fn keeps_near_matches_and_multiline_control_fragments() {
        let state = "Goal: Find a stay\nPage: Stays\nControls:\n[1] button Search value=\"\"\n[2] link Log in value=null\n[3] link 1 \n\t Formal systems value=\"\"\nRecent actions: []\nVisible text: search\nSearch results\nLog  in\nFormal systems\n1\nGraphics: []\nRecovery: {}";
        assert_eq!(trim_visible_text(state), (state.into(), 0));
    }

    #[test]
    fn preserves_other_sections_and_the_visible_text_header() {
        let state = "Goal: Search\nPage: Search\nControls:\n[1] button Search value=\"\"\nRecent actions: [\"Search\"]\nVisible text: Search\nGraphics: [\"Search\"]\nRecovery: Search\n";
        let (trimmed, removed) = trim_visible_text(state);
        assert_eq!(removed, 1);
        assert_eq!(
            trimmed,
            state.replace("Visible text: Search", "Visible text: ")
        );
        for state in ["", "Goal: Search\nVisible text: Search"] {
            assert_eq!(trim_visible_text(state), (state.into(), 0));
        }
    }

    fn page() -> Value {
        json!({"url":"https://travel.example.test/stays","title":"Stays","text":"Find a stay","fingerprint":"same","actions":[
            {"id":"e1","kind":"fill","label":"Destination","role":"searchbox","value":"","node":10},
            {"id":"e2","kind":"click","label":"Open Destination","role":"searchbox","node":10},
            {"id":"e3","kind":"click","label":"Search","role":"button","node":20},
            {"id":"e4","kind":"click","label":"Breakfast","role":"checkbox","checked":"false","node":30},
            {"id":"e5","kind":"click","label":"One way","role":"radio","node":40},
            {"id":"e6","kind":"click","label":"Alerts","role":"switch","node":50},
            {"id":"e7","kind":"select","label":"Room → Suite","role":"combobox","value":"suite","current_value":"Standard","node":60},
            {"id":"e8","kind":"select","label":"Room → Double","role":"combobox","value":"double","current_value":"Standard","node":60},
            {"id":"e9","kind":"key","label":"Calendar → ArrowLeft","role":"region","key":"ArrowLeft","node":70},
            {"id":"e10","kind":"key","label":"Calendar → ArrowRight","role":"region","key":"ArrowRight","node":70},
            {"id":"scroll_down","kind":"scroll","label":"Scroll down","delta":560},
            {"id":"wait","kind":"wait","label":"Wait for the page to update"}
        ]})
    }

    #[test]
    fn maps_each_action_kind_and_keeps_terminals_last() {
        let page = page();
        let request = build_request(&page, "Find a stay", &[], true).unwrap();
        let descriptions: Vec<_> = request
            .options
            .iter()
            .map(|(option, _)| {
                (
                    option.element_id.as_str(),
                    option.role.as_str(),
                    option.label.as_str(),
                    option.action.as_str(),
                )
            })
            .collect();
        assert_eq!(
            descriptions,
            vec![
                ("e1", "searchbox", "Destination", "fill"),
                ("e2", "searchbox", "Destination", "click"),
                ("e3", "button", "Search", "click"),
                ("e4", "checkbox", "Breakfast", "check"),
                ("e5", "radio", "One way", "check"),
                ("e6", "switch", "Alerts", "check"),
                ("e7", "combobox", "Room → Suite", "click"),
                ("e8", "combobox", "Room → Double", "click"),
                ("e9", "region", "Calendar (key ArrowLeft)", "click"),
                ("e10", "region", "Calendar (key ArrowRight)", "click"),
                ("scroll_down", "page", "Scroll down", "click"),
                ("wait", "page", "Wait for the page to update", "click"),
                (
                    "DONE",
                    "task",
                    "Finish: every requirement is visibly satisfied",
                    "click"
                ),
                (
                    "BLOCKED",
                    "task",
                    "Give up: no supported operation can progress",
                    "click"
                ),
            ]
        );
        let mut ids = HashSet::new();
        for (option, action) in &request.options {
            assert!(ids.insert(&option.element_id));
            assert!(option.entity_id.is_none());
            assert_eq!(option.element_id, action["id"]);
            if action["kind"] == "terminal" {
                assert_eq!(
                    *action,
                    json!({"id":option.element_id,"kind":"terminal","label":option.element_id})
                );
            } else {
                assert_eq!(
                    Some(action),
                    page["actions"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .find(|a| a["id"] == option.element_id)
                );
            }
        }
    }

    #[test]
    fn reuses_compact_state_and_preserves_prompt_fields() {
        let page = page();
        let history = vec![
            json!({"action":"Search","kind":"click","page_changed":false,"before_fingerprint":"same","input":page["actions"][2]}),
        ];
        let (body, _) =
            policy::request(&page, "Find a stay in Lisbon", &history, true)
                .unwrap();
        for compact in [false, true] {
            let request = build_request(
                &page,
                "Find a stay in Lisbon",
                &history,
                compact,
            )
            .unwrap();
            assert_eq!(request.app, "Stays");
            assert_eq!(request.task_family, "web_navigation");
            assert_eq!(request.goal, "Find a stay in Lisbon");
            assert_eq!(request.ax_tree, body["state"]);
        }
    }

    #[test]
    fn trims_only_native_state_and_serializes_removed_line_count() {
        let mut page = page();
        page["text"] = json!("Search\nFind a stay\nSearch");
        let (body, _) =
            policy::request(&page, "Find a stay", &[], true).unwrap();
        assert!(
            body["state"].as_str().unwrap().contains(
                "Visible text: Search\nFind a stay\nSearch\nGraphics: "
            )
        );
        for compact in [false, true] {
            let request =
                build_request(&page, "Find a stay", &[], compact).unwrap();
            assert_eq!(
                request.ax_tree,
                body["state"].as_str().unwrap().replace(
                    "Visible text: Search\nFind a stay\nSearch",
                    "Visible text: Find a stay"
                )
            );
            assert_eq!(request.removed_visible_text_lines, 2);
            assert_eq!(
                serde_json::to_value(&request).unwrap()["removed_visible_text_lines"],
                2
            );
        }
    }

    #[test]
    fn uses_url_host_when_title_is_empty() {
        let mut page = page();
        page["title"] = json!("");
        page["url"] = json!("https://travel.example.test:8443/stays?q=Lisbon");
        let request = build_request(&page, "Find a stay", &[], true).unwrap();
        assert_eq!(request.app, "travel.example.test");
    }

    #[test]
    fn uses_full_url_for_untitled_file_page() {
        let mut page = page();
        page["title"] = json!("");
        page["url"] = json!("file:///tmp/hotel.html");
        let request = build_request(&page, "Find a stay", &[], true).unwrap();
        assert_eq!(request.app, "file:///tmp/hotel.html");
    }

    #[test]
    fn suppresses_ineffective_targets_only_in_the_same_state() {
        let mut page = page();
        let history: Vec<_> = page["actions"].as_array().unwrap().iter().map(|action| json!({
            "before_fingerprint":"same","page_changed":false,"input":action
        })).collect();
        for compact in [false, true] {
            let request =
                build_request(&page, "Find a stay", &history, compact).unwrap();
            let ids: Vec<_> = request
                .options
                .iter()
                .map(|(option, _)| option.element_id.as_str())
                .collect();
            assert_eq!(ids, ["e1", "scroll_down", "wait", "DONE", "BLOCKED"]);
        }
        page["fingerprint"] = json!("changed");
        let request =
            build_request(&page, "Find a stay", &history, true).unwrap();
        assert_eq!(
            request.options.len(),
            page["actions"].as_array().unwrap().len() + 2
        );
    }

    #[test]
    fn suppresses_repeated_actions_within_recent_history() {
        let page = page();
        let entry = json!({"before_fingerprint":"same","page_changed":true,"input":page["actions"][2]});
        for count in [1, 2] {
            let request = build_request(
                &page,
                "Find a stay",
                &vec![entry.clone(); count],
                true,
            )
            .unwrap();
            assert_eq!(
                request
                    .options
                    .iter()
                    .any(|(option, _)| option.element_id == "e3"),
                count == 1
            );
        }
        let mut history = vec![entry.clone(), entry];
        history.extend(vec![json!({"before_fingerprint":"old"}); 20]);
        let request =
            build_request(&page, "Find a stay", &history, true).unwrap();
        assert!(
            request
                .options
                .iter()
                .any(|(option, _)| option.element_id == "e3")
        );
    }

    #[test]
    fn offers_terminals_when_no_actions_are_available() {
        let mut page = page();
        page["actions"] = json!([]);
        let request = build_request(&page, "Find a stay", &[], true).unwrap();
        let ids: Vec<_> = request
            .options
            .iter()
            .map(|(option, _)| option.element_id.as_str())
            .collect();
        assert_eq!(ids, ["DONE", "BLOCKED"]);
    }

    #[test]
    fn rejects_duplicate_option_ids() {
        for id in ["e1", "DONE", "BLOCKED"] {
            let mut page = page();
            page["actions"][2]["id"] = json!(id);
            let error = build_request(&page, "Find a stay", &[], true)
                .err()
                .unwrap();
            assert_eq!(error.to_string(), "duplicate action IDs");
        }
    }

    fn build_predictions(
        request: &Request,
        selected: &str,
    ) -> Vec<CuaS1OptionPrediction> {
        request
            .options
            .iter()
            .enumerate()
            .map(|(index, (option, _))| CuaS1OptionPrediction {
                letter: (b'A' + (index % 26) as u8) as char,
                option: option.clone(),
                logit: 0.0,
                probability: if option.element_id == selected {
                    1.0
                } else {
                    0.0
                },
                is_selected: option.element_id == selected,
                forward_passes: 1,
                dropped_state_tokens: 0,
                is_prefix_shared: false,
                prefix_tokens: 0,
                suffix_tokens: 0,
                whole_prompt_tokens: 0,
                prefix_sharing_fallbacks: 0,
            })
            .collect()
    }

    #[test]
    fn native_decisions_preserve_resolved_actions_and_target_indices() {
        let page = page();
        let native = build_request(&page, "Find a stay", &[], true).unwrap();
        let (questions, space) =
            policy::request(&page, "Find a stay", &[], true).unwrap();
        for (id, operation, target) in [
            ("e1", "TYPE_TEXT", Some("1")),
            ("e2", "CLICK", Some("1")),
            ("e3", "CLICK", Some("2")),
            ("e4", "CLICK", Some("3")),
            ("e5", "CLICK", Some("4")),
            ("e6", "CLICK", Some("5")),
            ("e7", "SELECT", Some("6:1")),
            ("e8", "SELECT", Some("6:2")),
            ("e9", "PRESS_KEY", Some("7:ArrowLeft")),
            ("e10", "PRESS_KEY", Some("7:ArrowRight")),
            ("scroll_down", "SCROLL_DOWN", None),
            ("wait", "WAIT", None),
            ("DONE", "DONE", None),
            ("BLOCKED", "BLOCKED", None),
        ] {
            let predictions = build_predictions(&native, id);
            let decision = resolve(&native, &predictions).unwrap();
            let build_answer = |question: &str, choice: &str| {
                let probabilities: Map<String, Value> = questions["questions"]
                    [question]["criteria"]
                    .as_object()
                    .unwrap()
                    .keys()
                    .map(|id| {
                        (
                            id.clone(),
                            json!(if id == choice { 1.0 } else { 0.0 }),
                        )
                    })
                    .collect();
                json!({"choice":choice,"confidence":1.0,"probabilities":probabilities})
            };
            let mut response = json!({"model":vs1::cua_s1::MODEL_NAME,"answers":{"operation":build_answer("operation", operation)}});
            if let Some(target) = target {
                let question = format!("{}_target", operation.to_lowercase());
                response["answers"][&question] =
                    build_answer(&question, target);
            }
            let expected =
                policy::resolve(&questions, &space, &response).unwrap();
            for key in [
                "choice",
                "operation",
                "target",
                "action",
                "confidence",
                "model",
            ] {
                assert_eq!(decision[key], expected[key], "{id}: {key}");
            }
            assert_eq!(
                decision["probabilities"].as_object().unwrap().len(),
                native.options.len()
            );
            assert_eq!(decision["probabilities"][id], 1.0);
            assert_eq!(
                decision["options"],
                serde_json::to_value(&predictions).unwrap()
            );
            assert_eq!(
                decision["usage"],
                json!({"input_tokens":null,"output_tokens":0,"forward_passes":1})
            );
        }
    }

    #[test]
    fn native_targets_use_the_filtered_element_indices() {
        let page = page();
        let history = vec![
            json!({"before_fingerprint":"same","page_changed":false,"input":page["actions"][2]}),
        ];
        let request =
            build_request(&page, "Find a stay", &history, true).unwrap();
        let decision =
            resolve(&request, &build_predictions(&request, "e4")).unwrap();
        assert_eq!(decision["target"], "2");
        assert_eq!(decision["action"], page["actions"][3]);
    }

    #[test]
    fn native_tournament_keeps_all_options_and_uses_the_selected_winner() {
        let mut page = page();
        page["actions"] = json!((1..=28).map(|node| json!({"id":format!("e{node}"),"kind":"click","node":node,"role":"button","label":format!("Button {node}")})).collect::<Vec<_>>());
        let request =
            build_request(&page, "Choose a button", &[], true).unwrap();
        let mut predictions = build_predictions(&request, "e1");
        for (index, prediction) in predictions.iter_mut().enumerate() {
            prediction.letter = (b'A' + (index % 15) as u8) as char;
            prediction.probability = if index < 15 {
                0.04
            } else if index == 15 {
                0.3
            } else {
                0.1 / 14.0
            };
            prediction.forward_passes = 3;
            prediction.dropped_state_tokens = 20;
        }
        let decision = resolve(&request, &predictions).unwrap();
        assert_eq!(decision["choice"], "e1");
        assert_eq!(decision["action"], page["actions"][0]);
        assert!(
            decision["probabilities"]["e16"].as_f64().unwrap()
                > decision["probabilities"]["e1"].as_f64().unwrap()
        );
        assert_eq!(decision["options"].as_array().unwrap().len(), 30);
        assert_eq!(decision["probabilities"].as_object().unwrap().len(), 30);
        assert_eq!(
            decision["options"],
            serde_json::to_value(&predictions).unwrap()
        );
        assert_eq!(decision["usage"]["forward_passes"], 3);
    }

    #[test]
    fn native_decisions_reject_mismatched_candidates_and_invalid_selections() {
        let request = build_request(&page(), "Find a stay", &[], true).unwrap();
        let valid = build_predictions(&request, "e1");
        let mut missing = valid.clone();
        missing.pop();
        let mut reordered = valid.clone();
        reordered.swap(0, 1);
        let mut unknown = valid.clone();
        unknown[0].option.element_id = "invented".into();
        let mut unselected = valid.clone();
        unselected[0].is_selected = false;
        let mut multiple = valid.clone();
        multiple[1].is_selected = true;
        let mut invalid = valid.clone();
        invalid[0].probability = f32::NAN;
        for predictions in
            [missing, reordered, unknown, unselected, multiple, invalid]
        {
            assert!(resolve(&request, &predictions).is_err());
        }
    }
}
