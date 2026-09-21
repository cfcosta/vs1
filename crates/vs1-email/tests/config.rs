use vs1_email::Config;

const RULES: &str = r#"
[owner]
capture_venues = ["Nubank", "Clear"]
spouse = "Example Person"

[[rules]]
category = "capture"
what = "A periodic statement for owner.capture_venues"
not_for = "A single purchase"
examples = ["Extrato da fatura do Cartão Nubank"]

[[rules]]
category = "other"
what = "Fits none of the above"
"#;

#[test]
fn parses_ordered_rules_and_arbitrary_owner_context() {
    let config = Config::parse(RULES).unwrap();
    assert_eq!(config.rules()[0].category, "capture");
    assert_eq!(config.rules()[1].category, "other");
    assert_eq!(
        config.rules()[0].not_for.as_deref(),
        Some("A single purchase")
    );
    assert_eq!(
        config.rules()[0].examples,
        ["Extrato da fatura do Cartão Nubank"]
    );
    assert_eq!(config.owner()["capture_venues"][1], "Clear");
    assert_eq!(config.owner()["spouse"], "Example Person");
}

#[test]
fn rejects_missing_empty_duplicate_and_misspelled_rules() {
    for raw in [
        "",
        "rules = []",
        "[[rules]]\ncategory = 'only'\nwhat = 'One choice'",
        &RULES.replace("category = \"other\"", "category = \"capture\""),
        &RULES.replace("category = \"capture\"", "category = \" \""),
        &RULES.replace("Fits none of the above", "  "),
        &RULES.replace("not_for", "not_fro"),
        &format!("typo = true\n{RULES}"),
    ] {
        assert!(Config::parse(raw).is_err(), "accepted {raw}");
    }
}

#[test]
fn owner_and_optional_rule_details_can_be_omitted() {
    let config = Config::parse("[[rules]]\ncategory = 'a'\nwhat = 'A'\n[[rules]]\ncategory = 'b'\nwhat = 'B'").unwrap();
    assert!(config.owner().as_object().unwrap().is_empty());
    assert!(config.rules()[0].not_for.is_none());
    assert!(config.rules()[0].examples.is_empty());
}
