use std::{fs, path::Path};

use vs1_email::Config;

#[test]
fn email_and_browser_forward_every_core_acceleration_feature() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let core: toml::Value = toml::from_str(
        &fs::read_to_string(root.join("crates/vs1/Cargo.toml")).unwrap(),
    )
    .unwrap();
    for name in ["vs1-email", "vs1-browser"] {
        let manifest: toml::Value = toml::from_str(
            &fs::read_to_string(root.join(format!("crates/{name}/Cargo.toml")))
                .unwrap(),
        )
        .unwrap();
        for feature in core["features"]
            .as_table()
            .unwrap()
            .keys()
            .filter(|f| f.as_str() != "default")
        {
            let forwarded = manifest
                .get("features")
                .and_then(|v| v.get(feature))
                .and_then(|v| v.as_array());
            assert!(
                forwarded.is_some_and(|values| values
                    .iter()
                    .any(|v| v.as_str() == Some(&format!("vs1/{feature}")))),
                "{name} does not forward {feature}"
            );
        }
    }
}

#[test]
fn supplied_example_has_all_categories_and_owner_references() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../examples/email-rules.toml");
    let config = Config::parse(&fs::read_to_string(path).unwrap()).unwrap();
    let categories = config
        .rules()
        .iter()
        .map(|r| r.category.as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        categories,
        [
            "capture",
            "bills",
            "fiscal",
            "income",
            "receipts",
            "careers",
            "clients",
            "papers",
            "equity",
            "household",
            "identity",
            "security",
            "ops",
            "health",
            "travel",
            "bulk",
            "other"
        ]
    );
    for key in ["capture_venues", "fixed_bills", "clients", "spouse"] {
        assert!(config.owner().get(key).is_some());
    }
}
