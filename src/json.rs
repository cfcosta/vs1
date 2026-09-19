//! An order-preserving JSON value.
//!
//! `serde_json::Value` sorts object keys unless the crate-wide
//! `preserve_order` feature is on, and the model is sensitive to key
//! order: laya serialises a state with CPython's `json.dumps`, which
//! keeps insertion order, and the encoder reads the result left to
//! right and truncates it on the right. [`Json`] keeps objects in the
//! order they were written so a request parsed from text round-trips
//! to the same string Python would produce.

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

/// A JSON value whose objects keep insertion order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Json {
    Null,
    Bool(bool),
    Number(serde_json::Number),
    String(String),
    Array(Vec<Json>),
    Object(IndexMap<String, Json>),
}

impl Json {
    /// An object from ordered pairs.
    pub fn object<I, K, V>(pairs: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<Json>,
    {
        Json::Object(
            pairs
                .into_iter()
                .map(|(k, v)| (k.into(), v.into()))
                .collect(),
        )
    }

    /// `true` for `null` and the empty string.
    pub fn is_blank(&self) -> bool {
        matches!(self, Json::Null)
            || matches!(self, Json::String(s) if s.is_empty())
    }

    /// The string inside, if this is one.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::String(s) => Some(s),
            _ => None,
        }
    }
}

impl From<serde_json::Value> for Json {
    /// Converts a `serde_json::Value`. Objects arrive in whatever order
    /// the `Value` stores them, which is sorted unless serde_json was
    /// built with `preserve_order`.
    fn from(value: serde_json::Value) -> Self {
        match value {
            serde_json::Value::Null => Json::Null,
            serde_json::Value::Bool(b) => Json::Bool(b),
            serde_json::Value::Number(n) => Json::Number(n),
            serde_json::Value::String(s) => Json::String(s),
            serde_json::Value::Array(items) => {
                Json::Array(items.into_iter().map(Json::from).collect())
            }
            serde_json::Value::Object(map) => Json::Object(
                map.into_iter().map(|(k, v)| (k, Json::from(v))).collect(),
            ),
        }
    }
}

impl From<Json> for serde_json::Value {
    fn from(value: Json) -> Self {
        match value {
            Json::Null => serde_json::Value::Null,
            Json::Bool(b) => serde_json::Value::Bool(b),
            Json::Number(n) => serde_json::Value::Number(n),
            Json::String(s) => serde_json::Value::String(s),
            Json::Array(items) => serde_json::Value::Array(
                items.into_iter().map(serde_json::Value::from).collect(),
            ),
            Json::Object(map) => serde_json::Value::Object(
                map.into_iter()
                    .map(|(k, v)| (k, serde_json::Value::from(v)))
                    .collect(),
            ),
        }
    }
}

impl From<&str> for Json {
    fn from(s: &str) -> Self {
        Json::String(s.to_string())
    }
}

impl From<String> for Json {
    fn from(s: String) -> Self {
        Json::String(s)
    }
}

impl From<bool> for Json {
    fn from(b: bool) -> Self {
        Json::Bool(b)
    }
}

impl From<i64> for Json {
    fn from(n: i64) -> Self {
        Json::Number(n.into())
    }
}

impl From<u64> for Json {
    fn from(n: u64) -> Self {
        Json::Number(n.into())
    }
}

impl From<f64> for Json {
    fn from(n: f64) -> Self {
        serde_json::Number::from_f64(n)
            .map(Json::Number)
            .unwrap_or(Json::Null)
    }
}

impl<T: Into<Json>> From<Vec<T>> for Json {
    fn from(items: Vec<T>) -> Self {
        Json::Array(items.into_iter().map(Into::into).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parsing_keeps_key_order() {
        let json: Json = serde_json::from_str(
            r#"{"zeta": 1, "alpha": [true, null], "mid": "x"}"#,
        )
        .unwrap();
        let Json::Object(map) = &json else {
            panic!("expected object");
        };
        assert_eq!(map.keys().collect::<Vec<_>>(), ["zeta", "alpha", "mid"]);
        assert_eq!(
            serde_json::to_string(&json).unwrap(),
            r#"{"zeta":1,"alpha":[true,null],"mid":"x"}"#
        );
    }

    #[test]
    fn scalars_parse_to_the_right_variant() {
        assert_eq!(serde_json::from_str::<Json>("null").unwrap(), Json::Null);
        assert_eq!(
            serde_json::from_str::<Json>("true").unwrap(),
            Json::Bool(true)
        );
        assert_eq!(
            serde_json::from_str::<Json>("\"s\"").unwrap(),
            Json::from("s")
        );
        assert!(matches!(
            serde_json::from_str::<Json>("2.5").unwrap(),
            Json::Number(_)
        ));
    }

    #[test]
    fn object_helper_and_blankness() {
        let obj = Json::object([("query", "q"), ("passage", "p")]);
        assert_eq!(
            serde_json::to_string(&obj).unwrap(),
            r#"{"query":"q","passage":"p"}"#
        );
        assert!(Json::Null.is_blank());
        assert!(Json::from("").is_blank());
        assert!(!Json::from(false).is_blank());
        assert!(!Json::from(0i64).is_blank());
    }

    #[test]
    fn round_trips_through_serde_json_value() {
        let value = serde_json::json!({"a": {"b": [1, 2]}, "c": "d"});
        let back: serde_json::Value = Json::from(value.clone()).into();
        assert_eq!(back, value);
    }
}
