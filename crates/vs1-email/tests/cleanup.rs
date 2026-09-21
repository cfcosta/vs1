use vs1_email::{clean_body, parse_email};

#[test]
fn removes_decorative_borders_but_preserves_table_values_and_paragraphs() {
    let input = "\r\n┌──────────────┬─────────┐\r\n│ Invoice     │ AB-1234 │\r\n├──────────────┼─────────┤\r\n│ Due date    │ 09/03/2026 │\r\n│ Total       │ R$\u{a0}1.234,56 │\r\n└──────────────┴─────────┘\r\n\r\n\r\n  Pay\t before   09/03/2026. \r\n\r\nSecurity code: 001234\r\n---\r\n";
    assert_eq!(
        clean_body(input),
        "Invoice | AB-1234\nDue date | 09/03/2026\nTotal | R$ 1.234,56\n\nPay before 09/03/2026.\n\nSecurity code: 001234"
    );
}
#[test]
fn preserves_quotes_signatures_and_meaningful_punctuation() {
    let text = "> Previous message\n> Amount: -42.50\n\n--\nA + B = C\nOrder AB--123__45\nhttps://example.test/a-b_c#section-2";
    assert_eq!(clean_body(text), text);
    assert_eq!(clean_body(&clean_body(text)), clean_body(text));
}
#[test]
fn removes_tracking_and_named_opaque_tokens_but_keeps_semantic_url_fields() {
    let opaque = "Ab9_xY".repeat(60);
    let input = format!(
        "[Receipt](https://store.example.test/email/PurchaseReceipt?sparams={opaque}&order_id=AB-123&utm_source=email#details).\nhttps://example.test/pay?amount=1234.56&due=2026-03-09&gclid=tracking\nhttps://example.test/items?reference={opaque}"
    );
    let expected = format!(
        "[Receipt](https://store.example.test/email/PurchaseReceipt?order_id=AB-123#details).\nhttps://example.test/pay?amount=1234.56&due=2026-03-09\nhttps://example.test/items?reference={opaque}"
    );
    assert_eq!(clean_body(&input), expected);
}
#[test]
fn keeps_link_labels_and_balanced_parentheses_and_does_not_rewrite_clean_urls()
{
    let text = "Read https://example.test/wiki/Thing_(example)?id=001&ref=invoice.\n<https://example.test/receipt?utm_medium=mail>\nwww.example.test/path?UTM_campaign=sale&id=04";
    assert_eq!(
        clean_body(text),
        "Read https://example.test/wiki/Thing_(example)?id=001&ref=invoice.\n<https://example.test/receipt>\nwww.example.test/path?id=04"
    );
}
#[test]
fn cleanup_happens_after_mime_decoding_for_plain_and_html() {
    let plain=parse_email("plain",b"Content-Type: text/plain\r\n\r\nCode:   001234\r\nhttps://example.test/receipt?utm_source=mail&id=42\r\n").unwrap();
    assert_eq!(
        plain.body,
        "Code: 001234\nhttps://example.test/receipt?id=42"
    );
    let html=parse_email("html",b"Content-Type: text/html\r\n\r\n<table><tr><td>Amount</td><td>123.45</td></tr><tr><td>Invoice</td><td>AB-42</td></tr></table>").unwrap();
    assert!(html.body.contains("123.45"));
    assert!(html.body.contains("AB-42"));
    assert!(!html.body.contains('─'));
    assert!(!html.body.contains("   "));
}

#[test]
fn table_cleanup_preserves_empty_cells_next_to_zero_values() {
    assert_eq!(
        clean_body("│ Name │ Count │ Note │\n│ Item │ 0 │ │"),
        "Name | Count | Note\nItem | 0 |"
    );
}

#[test]
fn removes_tracking_redirects_but_preserves_direct_transaction_links() {
    let body = "Confirm your email: https://u171.ct.sendgrid.net/ls/click?upn=opaque123\nNews https://mail.example.test/ls/click?upn=opaque456\nRead https://example.list-manage.com/track/click?u=abc&id=123&e=456\nReceipt https://shop.test/orders/42?amount=123.45&reference=ABC\nCode 001234";
    let body = format!(
        "{body}\nLearn More https://email.email.example.test/c/eJy{}",
        "Ab12".repeat(40)
    );
    let cleaned = clean_body(&body);
    assert!(!cleaned.contains("/c/eJy"));
    assert!(!cleaned.contains("upn="));
    assert!(!cleaned.contains("/track/click"));
    assert!(!cleaned.contains("/ls/click"));
    assert!(cleaned.contains("Confirm your email:"));
    assert!(
        cleaned.contains(
            "https://shop.test/orders/42?amount=123.45&reference=ABC"
        )
    );
    assert!(cleaned.contains("Code 001234"));
    assert_eq!(
        clean_body("https://shop.test/ls/click?order_id=42"),
        "https://shop.test/ls/click?order_id=42"
    );
}
