use std::{
    collections::{HashMap, HashSet},
    sync::LazyLock,
};

use anyhow::{Context, Result, ensure};
use regex::Regex;
use serde::Serialize;
use serde_json::{Map, Value, json};
use vs1::{CuaS1Option, CuaS1OptionPrediction};

const MAX_OPTIONS: usize = 26;
const MAX_ELEMENTS: usize = 11;
const STOPWORDS: &[&str] = &[
    "the", "a", "an", "into", "on", "in", "to", "and", "click", "type",
    "select", "enter", "turn", "press", "page", "icon", "button", "field",
    "cell", "color",
];

#[derive(Serialize)]
pub struct Request {
    pub app: String,
    pub task_family: String,
    pub goal: Option<String>,
    pub ax_tree: String,
    pub options: Vec<(CuaS1Option, Value)>,
    #[serde(skip)]
    values: Vec<String>,
}

struct Element {
    id: String,
    role: String,
    label: String,
    value: Option<String>,
    checked: Option<bool>,
    rect: Value,
    actions: Vec<Value>,
}

fn tokenize_instruction(text: &str) -> HashSet<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_ascii_lowercase() && !c.is_ascii_digit())
        .filter(|token| !token.is_empty() && !STOPWORDS.contains(token))
        .map(str::to_owned)
        .collect()
}

fn extract_quoted_values(goal: &str) -> Vec<String> {
    // Delimiter deviation from training: also accept typographic quotes.
    static QUOTES: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r#""([^"]+)"|“([^”]+)”"#).unwrap());
    let mut seen = HashSet::new();
    QUOTES
        .captures_iter(goal)
        .map(|capture| {
            capture.get(1).or_else(|| capture.get(2)).unwrap().as_str()
        })
        .filter(|value| seen.insert(*value))
        .take(3)
        .map(str::to_owned)
        .collect()
}

fn prune_elements(mut elements: Vec<Element>, goal: &str) -> Vec<Element> {
    if elements.len() > MAX_ELEMENTS {
        static QUOTES: LazyLock<Regex> =
            LazyLock::new(|| Regex::new(r#""[^"]*"|“[^”]*”"#).unwrap());
        let tokens = tokenize_instruction(&QUOTES.replace_all(goal, " "));
        elements.sort_by_cached_key(|element| {
            std::cmp::Reverse(
                tokenize_instruction(&element.label)
                    .intersection(&tokens)
                    .count(),
            )
        });
        elements.truncate(MAX_ELEMENTS);
    }
    elements
}

fn format_role(role: &str) -> String {
    match role {
        "" | "button" => "Button".into(),
        "link" => "Link".into(),
        "textbox" | "searchbox" | "spinbutton" => "TextField".into(),
        "checkbox" => "CheckBox".into(),
        "radio" => "RadioButton".into(),
        "switch" => "Switch".into(),
        "combobox" => "ComboBox".into(),
        "tab" => "Tab".into(),
        "menuitem" => "MenuItem".into(),
        "option" => "Option".into(),
        _ => {
            let mut chars = role.chars();
            format!(
                "{}{}",
                chars.next().unwrap().to_uppercase(),
                chars.as_str()
            )
        }
    }
}

fn collect_elements(page: &Value) -> Result<Vec<Element>> {
    let mut elements: Vec<Element> = vec![];
    let mut indices = HashMap::new();
    for action in page["actions"]
        .as_array()
        .context("snapshot has no actions")?
    {
        if action["node"].is_null() || action["kind"] == "key" {
            continue;
        }
        let kind = action["kind"].as_str().context("action has no kind")?;
        let node = action["node"].as_u64().context("invalid observed node")?;
        let index = *indices.entry(node).or_insert_with(|| {
            let role = format_role(action["role"].as_str().unwrap_or(""));
            elements.push(Element {
                id: format!("el_{}", elements.len() + 1),
                label: role.clone(),
                role,
                value: None,
                checked: None,
                rect: Value::Null,
                actions: vec![],
            });
            elements.len() - 1
        });
        let element = &mut elements[index];
        if element.actions.is_empty() || matches!(kind, "fill" | "select") {
            let label = action["label"].as_str().unwrap_or("");
            let label = if kind == "select" {
                label.split(" → ").next().unwrap_or("")
            } else {
                label
            };
            element.label = if label.is_empty() {
                element.role.clone()
            } else {
                label.into()
            };
        }
        if matches!(kind, "fill" | "select") {
            element.value = action[if kind == "fill" {
                "value"
            } else {
                "current_value"
            }]
            .as_str()
            .map(str::to_owned);
        }
        if let Some(checked) = action.get("checked") {
            element.checked =
                checked.as_bool().or_else(|| checked.as_str()?.parse().ok());
        }
        if let Some(rect) = action.get("rect") {
            element.rect = rect.clone();
        }
        element.actions.push(action.clone());
    }
    Ok(elements)
}

fn build_option_group(
    element: &Element,
    values: &[String],
) -> Vec<(CuaS1Option, Value)> {
    let mut group = vec![];
    let mut add = |verb: &str, entity_id: Option<String>, action: Value| {
        group.push((
            CuaS1Option {
                element_id: element.id.clone(),
                role: element.role.clone(),
                label: element.label.clone(),
                action: verb.into(),
                entity_id,
            },
            action,
        ));
    };
    if let Some(fill) = element
        .actions
        .iter()
        .find(|action| action["kind"] == "fill")
    {
        for (index, value) in values.iter().enumerate() {
            if element.value.as_ref() != Some(value) {
                add("fill", Some(format!("val_{index}")), fill.clone());
            }
        }
    }
    let selects: Vec<_> = element
        .actions
        .iter()
        .filter(|action| action["kind"] == "select")
        .collect();
    if !selects.is_empty() {
        for (index, value) in values.iter().enumerate() {
            let matches = |candidate: &str| {
                candidate.trim().to_lowercase() == value.trim().to_lowercase()
            };
            if let Some(action) = selects.iter().find(|action| {
                let label = action["label"]
                    .as_str()
                    .unwrap_or("")
                    .split_once(" → ")
                    .map(|(_, label)| label)
                    .unwrap_or("");
                let choice = action["value"].as_str().unwrap_or("");
                let is_current =
                    element.value.as_ref().is_some_and(|current| {
                        let current = current.trim().to_lowercase();
                        current == label.trim().to_lowercase()
                            || current == choice.trim().to_lowercase()
                    });
                (matches(label) || matches(choice)) && !is_current
            }) {
                add("select", Some(format!("val_{index}")), (*action).clone());
            }
        }
    } else if let Some(click) = element
        .actions
        .iter()
        .find(|action| action["kind"] == "click")
    {
        add("click", None, click.clone());
    }
    add(
        "skip",
        None,
        json!({"id":"skip","kind":"skip","label":format!("Skip {}", element.label)}),
    );
    group
}

fn describe_element(element: &Element) -> String {
    let state = if let Some(checked) = element.checked {
        format!(" checked={checked}")
    } else if let Some(value) = &element.value {
        format!(" value=\"{value}\"")
    } else {
        String::new()
    };
    let x = element.rect["x"].as_f64().unwrap_or(0.0);
    let y = element.rect["y"].as_f64().unwrap_or(0.0);
    let right = x + element.rect["w"].as_f64().unwrap_or(0.0);
    let bottom = y + element.rect["h"].as_f64().unwrap_or(0.0);
    let frame = [x, y, right, bottom]
        .map(|coordinate| coordinate.round_ties_even() as i64);
    format!(
        "- {} \"{}\"{state} @ {frame:?}",
        element.role, element.label
    )
}

pub fn build_request(page: &Value, goal: &str) -> Result<Request> {
    let values = extract_quoted_values(goal);
    let elements = prune_elements(collect_elements(page)?, goal);
    let mut options = vec![];
    let mut lines = vec![];
    for element in elements {
        let group = build_option_group(&element, &values);
        if group.len() == 1 {
            continue;
        }
        if options.len() + group.len() > MAX_OPTIONS - 1 {
            break;
        }
        options.extend(group);
        lines.push(describe_element(&element));
    }
    options.push((
        CuaS1Option {
            element_id: "__episode__".into(),
            role: "Button".into(),
            label: "Finish - task complete".into(),
            action: "done".into(),
            entity_id: None,
        },
        json!({"id":"DONE","kind":"terminal","label":"DONE"}),
    ));
    Ok(Request {
        app: "cua_bench_basic".into(),
        task_family: "multi_step_submit".into(),
        goal: Some(goal.into()),
        ax_tree: if lines.is_empty() {
            "- (no interactive elements detected)".into()
        } else {
            lines.join("\n")
        },
        options,
        values,
    })
}

pub fn resolve(
    request: &Request,
    predictions: &[CuaS1OptionPrediction],
) -> Result<Value> {
    ensure!(
        predictions.len() == request.options.len()
            && predictions.iter().zip(&request.options).enumerate().all(
                |(index, (prediction, (option, _)))| {
                    prediction.letter == (b'A' + index as u8) as char
                        && prediction.option.element_id == option.element_id
                        && prediction.option.role == option.role
                        && prediction.option.label == option.label
                        && prediction.option.action == option.action
                        && prediction.option.entity_id == option.entity_id
                }
            ),
        "native-training predictions do not match observed candidates"
    );
    ensure!(
        predictions.iter().filter(|p| p.is_selected).count() == 1,
        "native-training predictions must select exactly one option"
    );
    let probs: Vec<_> = predictions.iter().map(|p| p.probability).collect();
    ensure!(
        probs
            .iter()
            .all(|p| p.is_finite() && (0.0..=1.0).contains(p))
            && (probs.iter().sum::<f32>() - 1.0).abs() < 0.02,
        "invalid native-training probability distribution"
    );
    // Letters identify options uniquely even when element IDs repeat.
    let probabilities: Map<String, Value> = predictions
        .iter()
        .map(|p| (p.letter.to_string(), json!(p.probability)))
        .collect();
    let selected = predictions.iter().position(|p| p.is_selected).unwrap();
    let (option, action) = &request.options[selected];
    let operation = match option.action.as_str() {
        "fill" => "TYPE_TEXT",
        "click" => "CLICK",
        "select" => "SELECT",
        "skip" => "SKIP",
        "done" => "DONE",
        _ => anyhow::bail!("unsupported native-training action"),
    };
    let target = if matches!(operation, "DONE" | "SKIP") {
        Value::Null
    } else {
        json!(option.element_id)
    };
    let mut decision = json!({"choice":predictions[selected].letter.to_string(),"operation":operation,"target":target,"action":action,
        "confidence":vs1::head::confidence_from_probs(&probs, probs.len()),"probabilities":probabilities,
        "options":predictions,"model":vs1::cua_s1::MODEL_NAME,
        "usage":{"input_tokens":null,"output_tokens":0,"forward_passes":predictions[selected].forward_passes}});
    if operation == "TYPE_TEXT" {
        let index: usize = option
            .entity_id
            .as_deref()
            .and_then(|id| id.strip_prefix("val_"))
            .context("fill option has no value index")?
            .parse()?;
        decision["text"] =
            json!(request.values.get(index).context("fill value is missing")?);
    }
    Ok(decision)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn page() -> Value {
        json!({"actions":[
            {"id":"e1","kind":"fill","label":"Destination","role":"searchbox","value":"Lisbon","node":10,"rect":{"x":12,"y":70,"w":200,"h":24}},
            {"id":"e2","kind":"click","label":"Open Destination","role":"searchbox","node":10},
            {"id":"e3","kind":"click","label":"Breakfast","role":"checkbox","checked":"true","node":20,"rect":{"x":12.5,"y":39.5,"w":200,"h":24}},
            {"id":"e4","kind":"select","label":"Room → Suite","role":"combobox","value":"suite_room","current_value":"Standard","node":30,"rect":{"x":12,"y":100,"w":200,"h":24}},
            {"id":"e5","kind":"select","label":"Room → Standard","role":"combobox","value":"standard_room","current_value":"Standard","node":30},
            {"id":"e6","kind":"key","label":"Calendar → Enter","role":"region","key":"Enter","node":40},
            {"id":"scroll_down","kind":"scroll","label":"Scroll down"},
            {"id":"wait","kind":"wait","label":"Wait"}
        ]})
    }

    #[test]
    fn extracts_unique_quotes_in_order_with_a_three_value_limit() {
        assert_eq!(
            extract_quoted_values(
                r#"Type "Lisbon", "Lisbon", “Porto”, "Suite", “ignored”"#
            ),
            ["Lisbon", "Porto", "Suite"]
        );
        assert_eq!(
            extract_quoted_values("“same” and \"same\" and “ next ”"),
            ["same", " next "]
        );
        assert!(
            extract_quoted_values("No 'single quoted' values or “”.")
                .is_empty()
        );
        assert_eq!(
            extract_quoted_values("Type \"two\nlines\""),
            ["two\nlines"]
        );
    }

    #[test]
    fn tokenizes_like_training_and_removes_the_same_stopwords() {
        assert!(tokenize_instruction(&STOPWORDS.join(" ")).is_empty());
        assert_eq!(
            tokenize_instruction("C3 c3 e-mail CAFÉ 42_17"),
            ["c3", "e", "mail", "caf", "42", "17"]
                .map(str::to_owned)
                .into_iter()
                .collect()
        );
    }

    #[test]
    fn prunes_in_rank_order_excluding_quoted_spans_and_breaks_ties_by_page_order()
     {
        let labels = [
            "SUM A1 A10",
            "click type enter",
            "Other 3",
            "Other 4",
            "Other 5",
            "Other 6",
            "Other 7",
            "Other 8",
            "Other 9",
            "Other 10",
            "C3 C3",
            "Destination C3",
            "C3",
        ];
        let page = json!({"actions":labels.iter().enumerate().map(|(index, label)| json!({
            "id":format!("e{index}"),"node":index+1,"kind":"click","role":"button","label":label
        })).collect::<Vec<_>>()});
        for goal in [
            r#"Enter "=SUM(A1:A10)" into cell C3 and destination"#,
            "Enter “=SUM(A1:A10)” into cell C3 and destination",
        ] {
            let request = build_request(&page, goal).unwrap();
            let ids: Vec<_> = request
                .options
                .iter()
                .filter(|(option, _)| option.action == "click")
                .map(|(option, _)| option.element_id.as_str())
                .collect();
            assert_eq!(
                ids,
                [
                    "el_12", "el_11", "el_13", "el_1", "el_2", "el_3", "el_4",
                    "el_5", "el_6", "el_7", "el_8"
                ]
            );
            assert_eq!(request.ax_tree.lines().count(), MAX_ELEMENTS);
            assert!(request.ax_tree.starts_with("- Button \"Destination C3\""));
        }
        let mut small_page = page;
        small_page["actions"]
            .as_array_mut()
            .unwrap()
            .truncate(MAX_ELEMENTS);
        let request = build_request(&small_page, "Click C3").unwrap();
        assert_eq!(request.options[0].0.element_id, "el_1");
        assert_eq!(request.options[20].0.element_id, "el_11");
    }

    #[test]
    fn suppresses_fills_with_the_current_value_and_preserves_value_indices() {
        let page = page();
        let request =
            build_request(&page, r#"Replace "Lisbon" with "Porto""#).unwrap();
        let fills: Vec<_> = request
            .options
            .iter()
            .filter(|(option, _)| option.action == "fill")
            .collect();
        assert_eq!(fills.len(), 1);
        assert_eq!(fills[0].0.entity_id.as_deref(), Some("val_1"));
        assert_eq!(fills[0].1, page["actions"][0]);
        let actions: Vec<_> = request
            .options
            .iter()
            .map(|(option, _)| option.action.as_str())
            .collect();
        assert_eq!(actions, ["fill", "click", "skip", "click", "skip", "done"]);
        assert_eq!(request.options[1].1, page["actions"][1]);
        assert_eq!(request.options[1].0.label, "Destination");
    }

    #[test]
    fn selects_only_matching_quotes_that_are_not_current_and_never_offers_select_clicks()
     {
        for value in [" sUiTe ", " SUITE_ROOM "] {
            let request = build_request(
                &page(),
                &format!("Select \"{value}\", \"standard_room\", \"missing\""),
            )
            .unwrap();
            let options: Vec<_> = request
                .options
                .iter()
                .filter(|(option, _)| option.element_id == "el_3")
                .collect();
            assert_eq!(options.len(), 2);
            assert_eq!(options[0].0.action, "select");
            assert_eq!(options[0].0.entity_id.as_deref(), Some("val_0"));
            assert_eq!(options[0].0.role, "ComboBox");
            assert_eq!(options[0].0.label, "Room");
            assert_eq!(options[0].1["id"], "e4");
            assert_eq!(options[1].0.action, "skip");
            assert!(request.ax_tree.contains(
                "- ComboBox \"Room\" value=\"Standard\" @ [12, 100, 212, 124]"
            ));
        }
        let request = build_request(&page(), r#"Select "Standard""#).unwrap();
        assert!(
            request
                .options
                .iter()
                .all(|(option, _)| option.element_id != "el_3")
        );
    }

    fn build_field_actions(index: usize) -> Vec<Value> {
        vec![
            json!({"id":format!("fill{index}"),"kind":"fill","node":index,"role":"textbox","label":format!("Field {index}"),"value":""}),
            json!({"id":format!("click{index}"),"kind":"click","node":index,"role":"textbox","label":format!("Open Field {index}")}),
        ]
    }

    #[test]
    fn admits_whole_groups_and_stops_at_the_first_that_exceeds_the_budget() {
        let mut actions: Vec<_> =
            (1..=4).flat_map(build_field_actions).collect();
        actions.push(json!({"id":"b5","kind":"click","node":5,"role":"button","label":"Fifth"}));
        actions.extend(build_field_actions(6));
        actions.push(json!({"id":"b7","kind":"click","node":7,"role":"button","label":"Seventh"}));
        let request = build_request(
            &json!({"actions":actions}),
            r#"Type "one", "two", "three""#,
        )
        .unwrap();
        assert_eq!(request.options.len(), 23);
        assert_eq!(request.ax_tree.lines().count(), 5);
        assert!(!request.ax_tree.contains("Field 6"));
        assert!(!request.ax_tree.contains("Seventh"));
        let actions: Vec<_> = (1..=6).flat_map(build_field_actions).collect();
        let full = build_request(
            &json!({"actions":actions}),
            r#"Type "one", "two", "three""#,
        )
        .unwrap();
        assert_eq!(full.options.len(), MAX_OPTIONS);
        for request in [request, full] {
            let last = &request.options.last().unwrap().0;
            assert_eq!(last.element_id, "__episode__");
            assert_eq!(last.role, "Button");
            assert_eq!(last.label, "Finish - task complete");
            assert_eq!(last.action, "done");
            assert!(last.entity_id.is_none());
        }
    }

    #[test]
    fn describes_checked_filled_and_empty_fields_with_python_frame_rounding() {
        let mut page = page();
        let actions = page["actions"].as_array_mut().unwrap();
        actions.swap(0, 2);
        actions[0]["value"] = json!("ignored for checkbox");
        actions.push(json!({"id":"e7","kind":"fill","label":"Name","role":"textbox","node":50,"value":"","rect":{"x":1.4,"y":2.6,"w":10.2,"h":20.2}}));
        let request = build_request(&page, r#"Type "Porto""#).unwrap();
        assert_eq!(
            request.ax_tree,
            "- CheckBox \"Breakfast\" checked=true @ [12, 40, 212, 64]\n- TextField \"Destination\" value=\"Lisbon\" @ [12, 70, 212, 94]\n- TextField \"Name\" value=\"\" @ [1, 3, 12, 23]"
        );
        page["actions"][0]["checked"] = json!(false);
        let request = build_request(&page, r#"Type "Porto""#).unwrap();
        assert!(
            request
                .ax_tree
                .starts_with("- CheckBox \"Breakfast\" checked=false @")
        );
    }

    #[test]
    fn maps_roles_and_uses_the_role_for_an_empty_label() {
        for (role, expected) in [
            ("button", "Button"),
            ("link", "Link"),
            ("textbox", "TextField"),
            ("searchbox", "TextField"),
            ("checkbox", "CheckBox"),
            ("radio", "RadioButton"),
            ("switch", "Switch"),
            ("combobox", "ComboBox"),
            ("spinbutton", "TextField"),
            ("tab", "Tab"),
            ("menuitem", "MenuItem"),
            ("option", "Option"),
            ("gridcell", "Gridcell"),
            ("", "Button"),
        ] {
            assert_eq!(format_role(role), expected);
            let request = build_request(&json!({"actions":[{"id":"e1","kind":"click","node":1,"role":role,"label":""}]}), "Click").unwrap();
            assert_eq!(request.options[0].0.role, expected);
            assert_eq!(request.options[0].0.label, expected);
        }
        let mut actions = build_field_actions(1);
        actions[0]["role"] = json!("combobox");
        let request =
            build_request(&json!({"actions":actions}), "Type \"Porto\"")
                .unwrap();
        assert_eq!(request.options[0].0.role, "ComboBox");
        assert_eq!(request.options[1].0.action, "click");
    }

    #[test]
    fn omits_unobserved_clicks_and_drops_elements_with_only_skip() {
        let mut actions = build_field_actions(1);
        actions.pop();
        actions[0]["value"] = json!("Porto");
        for goal in ["No quotes", "Type \"Porto\""] {
            let request =
                build_request(&json!({"actions":actions}), goal).unwrap();
            assert_eq!(request.options.len(), 1);
            assert_eq!(request.ax_tree, "- (no interactive elements detected)");
        }
        let request =
            build_request(&json!({"actions":actions}), "Type \"Lisbon\"")
                .unwrap();
        assert_eq!(
            request
                .options
                .iter()
                .map(|(option, _)| option.action.as_str())
                .collect::<Vec<_>>(),
            ["fill", "skip", "done"]
        );
        let request = build_request(&json!({"actions":[]}), "Finish").unwrap();
        assert_eq!(request.options.len(), 1);
        assert_eq!(request.ax_tree, "- (no interactive elements detected)");
    }

    #[test]
    fn serializes_training_prompt_fields_without_internal_values() {
        let request = build_request(&page(), "Type \"Porto\"").unwrap();
        let serialized = serde_json::to_value(&request).unwrap();
        assert_eq!(serialized["app"], "cua_bench_basic");
        assert_eq!(serialized["task_family"], "multi_step_submit");
        assert_eq!(request.goal.as_deref(), Some("Type \"Porto\""));
        assert_eq!(serialized["goal"], "Type \"Porto\"");
        assert_eq!(
            serialized["options"].as_array().unwrap().len(),
            request.options.len()
        );
        assert!(serialized.get("values").is_none());
    }

    fn build_predictions(
        request: &Request,
        selected: usize,
    ) -> Vec<CuaS1OptionPrediction> {
        request
            .options
            .iter()
            .enumerate()
            .map(|(index, (option, _))| CuaS1OptionPrediction {
                letter: (b'A' + index as u8) as char,
                option: option.clone(),
                logit: 0.0,
                probability: if index == selected { 1.0 } else { 0.0 },
                is_selected: index == selected,
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
    fn resolves_each_option_by_position_including_repeated_element_ids() {
        let request =
            build_request(&page(), r#"Enter "Porto" or "Suite""#).unwrap();
        for (index, (option, action)) in request.options.iter().enumerate() {
            let decision =
                resolve(&request, &build_predictions(&request, index)).unwrap();
            let operation = match option.action.as_str() {
                "fill" => "TYPE_TEXT",
                "click" => "CLICK",
                "select" => "SELECT",
                "skip" => "SKIP",
                "done" => "DONE",
                _ => unreachable!(),
            };
            assert_eq!(decision["operation"], operation);
            assert_eq!(decision["action"], *action);
            let key = ((b'A' + index as u8) as char).to_string();
            assert_eq!(decision["choice"], key);
            assert_eq!(decision["probabilities"][&key], 1.0);
            assert_eq!(
                decision["probabilities"].as_object().unwrap().len(),
                request.options.len()
            );
            assert_eq!(
                decision["options"].as_array().unwrap().len(),
                request.options.len()
            );
            assert_eq!(decision["model"], vs1::cua_s1::MODEL_NAME);
            assert_eq!(decision["usage"]["forward_passes"], 1);
            if option.action == "fill" {
                assert_eq!(
                    decision["text"],
                    if option.entity_id.as_deref() == Some("val_0") {
                        "Porto"
                    } else {
                        "Suite"
                    }
                );
            } else {
                assert!(decision.get("text").is_none());
            }
            if option.action == "skip" {
                assert_eq!(
                    *action,
                    json!({"id":"skip","kind":"skip","label":format!("Skip {}", option.label)})
                );
            }
            if matches!(option.action.as_str(), "skip" | "done") {
                assert!(decision["target"].is_null());
            } else {
                assert_eq!(decision["target"], option.element_id);
            }
        }
    }

    #[test]
    fn rejects_reordered_options_and_invalid_predictions() {
        let request =
            build_request(&page(), r#"Type "Porto" or "Suite""#).unwrap();
        let valid = build_predictions(&request, 0);
        let mut reordered = valid.clone();
        let first = reordered[0].option.clone();
        reordered[0].option = reordered[1].option.clone();
        reordered[1].option = first;
        assert!(resolve(&request, &reordered).is_err());
        assert!(resolve(&request, &valid[..valid.len() - 1]).is_err());
        for probability in [f32::NAN, -0.1, 0.5, 1.1] {
            let mut invalid = valid.clone();
            invalid[0].probability = probability;
            assert!(resolve(&request, &invalid).is_err());
        }
        for selected in [false, true] {
            let mut invalid = valid.clone();
            for prediction in &mut invalid {
                prediction.is_selected = selected;
            }
            assert!(resolve(&request, &invalid).is_err());
        }
    }
}
