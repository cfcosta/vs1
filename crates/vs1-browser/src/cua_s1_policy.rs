use std::collections::HashSet;

use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use vs1::CuaS1Option;

use crate::policy;

pub struct Request {
    pub app: String,
    pub task_family: String,
    pub goal: String,
    pub ax_tree: String,
    pub options: Vec<(CuaS1Option, Value)>,
}

pub fn build_request(
    page: &Value,
    goal: &str,
    history: &[Value],
    _compact: bool,
) -> Result<Request> {
    // Cua-S1 always needs textual state, including for upstream policy prompts.
    let (body, space) = policy::request(page, goal, history, true)?;
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
        ax_tree: body["state"]
            .as_str()
            .context("missing textual state")?
            .into(),
        options,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
