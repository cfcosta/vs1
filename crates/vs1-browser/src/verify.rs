use anyhow::{Context, Result, ensure};
use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};
use serde_json::{Map, Value, json};

use crate::browser::{Browser, Stale};

pub fn outcome(
    page: &Value,
    task: Option<&str>,
    expected_url: Option<&str>,
    expected_text: &[String],
) -> Value {
    let mut checks = Map::new();
    let url = page["url"].as_str().unwrap_or("");
    let text = page["text"].as_str().unwrap_or("");
    match task {
        Some("hotel") => {
            checks
                .insert("property".into(), json!(url.ends_with("#casa-flora")));
            checks.insert("filters".into(),json!(text.contains("Your filters: Design · Free cancellation enabled · Destination Lisbon")));
        }
        Some("wikipedia") => {
            checks.insert("article".into(),json!(url=="https://en.wikipedia.org/wiki/G%C3%B6del%27s_incompleteness_theorems" || url=="https://en.wikipedia.org/wiki/G%C3%B6del's_incompleteness_theorems"));
        }
        Some("flights") => {
            let parsed = url::Url::parse(url).ok();
            checks.insert(
                "search_page".into(),
                json!(
                    parsed.as_ref().is_some_and(|u| u.host_str()
                        == Some("www.google.com")
                        && u.path() == "/travel/flights/search")
                ),
            );
            let actions =
                page["actions"].as_array().cloned().unwrap_or_default();
            for (key, label, value) in [
                ("one_way", "Change ticket type. One way", "One way"),
                ("origin", "Where from?", "Zürich"),
                ("destination", "Where to?", "London"),
                ("date", "Departure", "Sun, Sep 20"),
            ] {
                checks.insert(
                    key.into(),
                    json!(actions.iter().any(|a| {
                        a["label"].as_str().is_some_and(|l| l.trim() == label)
                            && a["value"] == value
                    })),
                );
            }
            let encoded = parsed.as_ref().and_then(|u| {
                u.query_pairs()
                    .find(|(k, _)| k == "tfs")
                    .map(|(_, v)| v.into_owned())
            });
            let date_in_url = encoded
                .and_then(|s| {
                    URL_SAFE_NO_PAD.decode(s.trim_end_matches('=')).ok()
                })
                .is_some_and(|b| b.windows(10).any(|w| w == b"2026-09-20"));
            checks.insert(
                "year".into(),
                json!(date_in_url || text.contains("departing 2026-09-20")),
            );
            let flights: Vec<&str> = actions
                .iter()
                .filter_map(|a| a["label"].as_str())
                .filter(|l| l.contains("Select flight"))
                .collect();
            checks.insert(
                "results".into(),
                json!(
                    !flights.is_empty()
                        && flights
                            .iter()
                            .all(|f| f.contains("Sunday, September 20"))
                ),
            );
        }
        _ => (),
    }
    if let Some(expected) = expected_url {
        checks.insert("expected_url".into(), json!(url == expected));
    }
    for (i, expected) in expected_text.iter().enumerate() {
        checks.insert(
            format!("expected_text_{i}"),
            json!(text.contains(expected)),
        );
    }
    if checks.is_empty() {
        return json!({"passed":null,"reason":"No independent verifier supplied; DONE is only a model claim."});
    }
    json!({"passed":checks.values().all(|v|v==true),"checks":checks})
}

pub fn check_browser(endpoint: &str) -> Result<()> {
    let html = r#"<!doctype html><title>Guard checks</title><style>body{margin:30px}button{width:180px;height:50px}</style>
        <p id="context">Cart total: $10</p><button id="target" onclick="window.clicks=(window.clicks||0)+1">Continue</button>
        <label>City<input id="field" value="Zurich"></label><label><input id="toggle" type="checkbox">Refundable</label>
        <select aria-label="Category"><option>All</option><option>Design</option></select>"#;
    let url = format!(
        "data:text/html;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(html)
    );
    let mut browser = Browser::connect(endpoint, &url)?;
    let mut passed = vec![];
    let page = browser.observe()?;
    let action = find(&page, "Continue", "click")?;
    browser.evaluate(
        "document.querySelector('#target').style.transform='translateY(150px)'",
    )?;
    ensure!(
        browser.fresh(&page, Some(&action))?,
        "movement invalidated semantics"
    );
    browser.act(&action, &page, None)?;
    ensure!(
        browser.evaluate("window.clicks")? == 1,
        "moving target did not click once"
    );
    passed.push("moving target uses current geometry");
    for (name, script) in [
        (
            "context",
            "document.querySelector('#context').textContent='Cart total: $100'",
        ),
        ("field", "document.querySelector('#field').value='London'"),
        ("checkbox", "document.querySelector('#toggle').checked=true"),
        (
            "disabled",
            "document.querySelector('#target').disabled=true",
        ),
        (
            "replaced",
            "document.querySelector('#target').outerHTML=document.querySelector('#target').outerHTML",
        ),
    ] {
        browser.evaluate("document.querySelector('#target').disabled=false")?;
        let page = browser.observe()?;
        let action = find(&page, "Continue", "click")?;
        browser.evaluate(script)?;
        ensure!(
            !browser.fresh(&page, Some(&action))?,
            "{name} was not invalidated"
        );
        ensure!(
            browser
                .act(&action, &page, None)
                .unwrap_err()
                .downcast_ref::<Stale>()
                .is_some(),
            "stale target reached mutation"
        );
        passed.push(name);
    }
    browser.evaluate("document.querySelector('#target').disabled=false")?;
    let page = browser.observe()?;
    let action = find(&page, "Continue", "click")?;
    browser.evaluate("const cover=document.createElement('div');cover.id='cover';cover.style.cssText='position:fixed;inset:0;z-index:9999;background:white';document.body.append(cover)")?;
    ensure!(
        browser
            .act(&action, &page, None)
            .unwrap_err()
            .downcast_ref::<Stale>()
            .is_some(),
        "overlay allowed mutation"
    );
    ensure!(
        browser.evaluate("window.clicks")? == 1,
        "stale/covered action was repeated"
    );
    passed.push("overlay rejects before input");
    browser.evaluate("document.querySelector('#cover').remove()")?;
    let page = browser.observe()?;
    let action = find(&page, "City", "fill")?;
    browser.act(&action, &page, Some("Lisbon"))?;
    ensure!(
        browser.evaluate("document.querySelector('#field').value")? == "Lisbon",
        "text was not replaced"
    );
    passed.push("native text replacement");
    let page = browser.observe()?;
    let action = find(&page, "Category → Design", "select")?;
    browser.act(&action, &page, None)?;
    ensure!(
        browser.evaluate("document.querySelector('select').value")? == "Design",
        "native select failed"
    );
    passed.push("native dropdown selection");
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"passed":passed,"browser":browser.version()})
        )?
    );
    Ok(())
}
fn find(page: &Value, label: &str, kind: &str) -> Result<Value> {
    page["actions"]
        .as_array()
        .context("missing actions")?
        .iter()
        .find(|a| a["label"] == label && a["kind"] == kind)
        .cloned()
        .with_context(|| format!("missing {kind} {label}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn done_alone_is_not_verification() {
        assert!(outcome(&json!({}), None, None, &[])["passed"].is_null());
    }
    #[test]
    fn hotel_requires_applied_filters_as_well_as_property() {
        let mut page = json!({"url":"file:///fixture.html#casa-flora","text":"Your filters: Design · Free cancellation enabled · Destination Lisbon"});
        assert_eq!(outcome(&page, Some("hotel"), None, &[])["passed"], true);
        page["text"] = json!("Free cancellation included");
        assert_eq!(outcome(&page, Some("hotel"), None, &[])["passed"], false);
    }
}
