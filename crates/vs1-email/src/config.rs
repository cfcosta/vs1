use std::collections::HashSet;

use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use serde_json::{Map, Value};

/// One mutually exclusive category, in the order presented to laya.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Rule {
    pub category: String,
    pub what: String,
    #[serde(default)]
    pub not_for: Option<String>,
    #[serde(default)]
    pub examples: Vec<String>,
}

/// Validated categorization rules and contextual facts about the owner.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    #[serde(default = "empty_owner")]
    owner: Value,
    rules: Vec<Rule>,
}

fn empty_owner() -> Value {
    Value::Object(Map::new())
}

impl Config {
    pub fn parse(raw: &str) -> Result<Self> {
        let config: Self = toml::from_str(raw).context("invalid rules TOML")?;
        ensure!(config.owner.is_object(), "owner must be a TOML table");
        ensure!(config.rules.len() >= 2, "at least two rules are required");
        let mut categories = HashSet::new();
        for rule in &config.rules {
            ensure!(
                !rule.category.trim().is_empty(),
                "category must not be blank"
            );
            ensure!(
                rule.category.trim() == rule.category,
                "category must not have surrounding whitespace"
            );
            ensure!(
                categories.insert(&rule.category),
                "duplicate category {:?}",
                rule.category
            );
            ensure!(
                !rule.what.trim().is_empty(),
                "what must not be blank for {:?}",
                rule.category
            );
        }
        Ok(config)
    }

    pub fn rules(&self) -> &[Rule] {
        &self.rules
    }

    pub fn owner(&self) -> &Value {
        &self.owner
    }
}
