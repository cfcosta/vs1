//! Conservative formatting and URL cleanup before body chunking.
use std::sync::OnceLock;

use regex::Regex;

/// Normalizes layout and strips known URL noise while retaining prose and IDs.
pub fn clean_body(body: &str) -> String {
    let mut lines: Vec<String> = Vec::new();
    for line in body.lines() {
        let trimmed = line.trim();
        let visible: Vec<char> =
            trimmed.chars().filter(|c| !c.is_whitespace()).collect();
        if visible.len() >= 3
            && visible.iter().all(|c| {
                matches!(c, '-' | '_' | '=' | '*' | '+' | '|')
                    || ('\u{2500}'..='\u{257f}').contains(c)
            })
        {
            continue;
        }
        let has_table_border = trimmed.chars().any(is_vertical_border);
        let mut text = String::with_capacity(trimmed.len());
        for c in trimmed.chars() {
            if is_vertical_border(c) {
                text.push_str(" | ");
            } else {
                text.push(c);
            }
        }
        let text = text.split_whitespace().collect::<Vec<_>>().join(" ");
        let text = if has_table_border {
            let text = text.strip_prefix("| ").unwrap_or(&text);
            text.strip_suffix(" |").unwrap_or(text).to_owned()
        } else {
            text
        };
        if text.is_empty() {
            if lines.last().is_some_and(|l| !l.is_empty()) {
                lines.push(String::new());
            }
        } else {
            lines.push(text);
        }
    }
    while lines.last().is_some_and(String::is_empty) {
        lines.pop();
    }
    let normalized = lines.join("\n");
    static URLS: OnceLock<Regex> = OnceLock::new();
    URLS.get_or_init(|| {
        Regex::new(r#"(?i)\b(?:https?://|www\.)[^\s<>"`]+"#)
            .expect("constant URL pattern")
    })
    .replace_all(&normalized, |captures: &regex::Captures<'_>| {
        let matched = &captures[0];
        let mut end = matched.len();
        while let Some(last) = matched[..end].chars().next_back() {
            let prefix = &matched[..end];
            let is_suffix = matches!(last, '.' | ',' | ';' | '!' | '?' | ':')
                || [('(', ')'), ('[', ']'), ('{', '}')].iter().any(
                    |(open, close)| {
                        last == *close
                            && prefix.matches(*close).count()
                                > prefix.matches(*open).count()
                    },
                );
            if !is_suffix {
                break;
            }
            end -= last.len_utf8();
        }
        format!("{}{}", clean_url(&matched[..end]), &matched[end..])
    })
    .into_owned()
}
fn is_vertical_border(c: char) -> bool {
    matches!(c, '│' | '┃' | '║' | '┆' | '┇' | '┊' | '┋' | '╎' | '╏')
}
fn clean_url(original: &str) -> String {
    let parseable = if original.to_ascii_lowercase().starts_with("www.") {
        format!("https://{original}")
    } else {
        original.to_owned()
    };
    let Ok(parsed) = url::Url::parse(&parseable) else {
        return original.to_owned();
    };
    // Redirect payloads describe tracking infrastructure, not the message.
    // Match known shapes rather than dropping arbitrary long transaction URLs.
    let path = parsed.path();
    let host = parsed.host_str().unwrap_or_default();
    let sendgrid = path == "/ls/click"
        && parsed.query_pairs().any(|(key, _)| key == "upn");
    let mailchimp = host.ends_with(".list-manage.com")
        && path.trim_end_matches('/') == "/track/click";
    let compressed_redirect = host.starts_with("email.")
        && path.strip_prefix("/c/").is_some_and(|payload| {
            payload.starts_with("eJ") && is_opaque(payload)
        });
    if sendgrid || mailchimp || compressed_redirect {
        return String::new();
    }
    let (before_fragment, fragment) = original
        .split_once('#')
        .map_or((original, None), |(a, b)| (a, Some(b)));
    let Some((base, query)) = before_fragment.split_once('?') else {
        return original.to_owned();
    };
    let mut removed = false;
    let kept = query
        .split('&')
        .filter(|pair| {
            let Some((key, value)) =
                url::form_urlencoded::parse(pair.as_bytes()).next()
            else {
                return true;
            };
            let key = key.to_ascii_lowercase();
            let tracking = key.starts_with("utm_")
                || matches!(
                    key.as_str(),
                    "fbclid"
                        | "gclid"
                        | "dclid"
                        | "msclkid"
                        | "mc_cid"
                        | "mc_eid"
                        | "igshid"
                        | "_hsenc"
                        | "_hsmi"
                        | "mkt_tok"
                        | "vero_id"
                        | "oly_anon_id"
                        | "oly_enc_id"
                        | "s_cid"
                        | "_kx"
                        | "trk"
                        | "trkemail"
                        | "tracking_id"
                );
            let token_key = matches!(
                key.as_str(),
                "sparams"
                    | "token"
                    | "access_token"
                    | "auth_token"
                    | "signature"
                    | "sig"
                    | "jwt"
                    | "nonce"
                    | "updkey"
            );
            let raw_value = pair.split_once('=').map_or("", |(_, v)| v);
            let drop = tracking
                || (token_key && (is_opaque(&value) || is_opaque(raw_value)));
            removed |= drop;
            !drop
        })
        .collect::<Vec<_>>();
    if !removed {
        return original.to_owned();
    }
    let query = kept.join("&");
    let mut result = base.to_owned();
    if !query.is_empty() {
        result.push('?');
        result.push_str(&query);
    }
    if let Some(fragment) = fragment {
        result.push('#');
        result.push_str(fragment);
    }
    result
}
fn is_opaque(value: &str) -> bool {
    value.len() >= 80
        && value
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || "-_=+/%.".contains(c))
        && value.chars().any(|c| c.is_ascii_alphabetic())
        && value.chars().any(|c| c.is_ascii_digit())
}
