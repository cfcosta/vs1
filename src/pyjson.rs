//! `json.dumps` the way CPython writes it.
//!
//! laya serialises a structured state with `json.dumps(state,
//! ensure_ascii=False)` and structured criteria with the explicit
//! separators `(", ", ": ")`, which are also CPython's defaults. The
//! model saw that exact spacing during training, so the Rust port has
//! to reproduce it byte for byte instead of using serde_json's compact
//! `{"a":1}` layout. Non-ASCII text stays as-is (`ensure_ascii=False`),
//! and string escaping matches CPython's `json` module: `\"`, `\\`,
//! `\n`, `\r`, `\t`, `\b`, `\f`, and `\u00XX` for the remaining
//! control characters. Objects keep the key order of the `Value`,
//! which is the written order because serde_json is built with
//! `preserve_order`.

use std::fmt::Write;

use serde_json::Value;

/// Serialises `value` with CPython's default `json.dumps` layout.
///
/// Object keys keep the order the `Value` holds them in.
pub fn dumps(value: &Value) -> String {
    let mut out = String::new();
    write_value(&mut out, value);
    out
}

fn write_value(out: &mut String, value: &Value) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(true) => out.push_str("true"),
        Value::Bool(false) => out.push_str("false"),
        Value::Number(n) => write_number(out, n),
        Value::String(s) => write_string(out, s),
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_value(out, item);
            }
            out.push(']');
        }
        Value::Object(map) => {
            out.push('{');
            for (i, (key, item)) in map.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_string(out, key);
                out.push_str(": ");
                write_value(out, item);
            }
            out.push('}');
        }
    }
}

fn write_number(out: &mut String, n: &serde_json::Number) {
    if let Some(i) = n.as_i64() {
        let _ = write!(out, "{i}");
    } else if let Some(u) = n.as_u64() {
        let _ = write!(out, "{u}");
    } else if let Some(f) = n.as_f64() {
        out.push_str(&python_float_repr(f));
    } else {
        out.push_str(&n.to_string());
    }
}

/// CPython's `float.__repr__`: shortest round-trip digits, a trailing
/// `.0` on integral values, and exponent notation with a sign and at
/// least two exponent digits (`1e-05`, `1e+16`) outside `1e-4 ..
/// 1e16`.
fn python_float_repr(f: f64) -> String {
    if f.is_nan() {
        return "NaN".to_string();
    }
    if f.is_infinite() {
        return if f > 0.0 { "Infinity" } else { "-Infinity" }.to_string();
    }
    if f == 0.0 {
        return if f.is_sign_negative() { "-0.0" } else { "0.0" }.to_string();
    }
    let abs = f.abs();
    if (1e-4..1e16).contains(&abs) {
        let s = format!("{f}");
        if s.contains('.') || s.contains('e') {
            s
        } else {
            format!("{s}.0")
        }
    } else {
        let s = format!("{f:e}");
        let (mantissa, exponent) = s.split_once('e').unwrap_or((&s, "0"));
        let exponent: i32 = exponent.parse().unwrap_or(0);
        let sign = if exponent < 0 { '-' } else { '+' };
        format!("{mantissa}e{sign}{:02}", exponent.abs())
    }
}

fn write_string(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(text: &str) -> Value {
        serde_json::from_str(text).unwrap()
    }

    #[test]
    fn nested_layout_matches_cpython_defaults() {
        let value =
            parse(r#"{"zeta": 1, "b": [1, 2.5, "x"], "a": {"d": null}}"#);
        assert_eq!(
            dumps(&value),
            r#"{"zeta": 1, "b": [1, 2.5, "x"], "a": {"d": null}}"#
        );
    }

    #[test]
    fn strings_escape_like_cpython_with_ensure_ascii_false() {
        let value =
            Value::from("quote \" back \\ nl \n tab \t bell \u{7} café 日本");
        assert_eq!(
            dumps(&value),
            "\"quote \\\" back \\\\ nl \\n tab \\t bell \\u0007 café 日本\""
        );
    }

    #[test]
    fn floats_print_like_python_repr() {
        assert_eq!(python_float_repr(1.0), "1.0");
        assert_eq!(python_float_repr(0.5), "0.5");
        assert_eq!(python_float_repr(1e-5), "1e-05");
        assert_eq!(python_float_repr(1e16), "1e+16");
        assert_eq!(python_float_repr(123456.789), "123456.789");
        assert_eq!(python_float_repr(-0.0), "-0.0");
    }

    #[test]
    fn empty_containers_and_bools() {
        assert_eq!(dumps(&parse("{}")), "{}");
        assert_eq!(dumps(&parse("[]")), "[]");
        assert_eq!(dumps(&parse("[true, false, null]")), "[true, false, null]");
    }

    #[test]
    fn objects_keep_written_order() {
        // Guards the `preserve_order` feature on serde_json: without
        // it these keys would come out sorted.
        let built =
            serde_json::json!({"zeta": 1, "alpha": [true, null], "mid": "x"});
        assert_eq!(
            dumps(&built),
            r#"{"zeta": 1, "alpha": [true, null], "mid": "x"}"#
        );
        let parsed = parse(r#"{"query": "what is rust", "passage": "Rust."}"#);
        assert_eq!(
            dumps(&parsed),
            r#"{"query": "what is rust", "passage": "Rust."}"#
        );
    }
}
