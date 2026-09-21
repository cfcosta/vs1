use vs1_email::parse_email;

#[test]
fn decodes_headers_and_prefers_plain_text_over_html_and_attachments() {
    let raw = b"From: bank@example.test\r\nTo: owner@example.test\r\nSubject: =?UTF-8?Q?Extrato_S=C3=A3o?=\r\nMessage-ID: <x@test>\r\nContent-Type: multipart/mixed; boundary=outer\r\n\r\n--outer\r\nContent-Type: multipart/alternative; boundary=inner\r\n\r\n--inner\r\nContent-Type: text/plain; charset=utf-8\r\nContent-Transfer-Encoding: quoted-printable\r\n\r\nSaldo S=C3=A3o\r\n--inner\r\nContent-Type: text/html\r\n\r\n<p>duplicate HTML</p>\r\n--inner--\r\n--outer\r\nContent-Type: text/plain\r\nContent-Disposition: attachment; filename=secret.txt\r\n\r\nIgnore attachment\r\n--outer--\r\n";
    let email = parse_email("cur/message:2,S", raw).unwrap();
    assert_eq!(email.path, std::path::Path::new("cur/message:2,S"));
    assert_eq!(email.subject, "Extrato São");
    assert_eq!(email.message_id, "<x@test>");
    assert_eq!(email.from, "bank@example.test");
    assert!(email.body.contains("Saldo São"));
    assert!(!email.body.contains("duplicate HTML"));
    assert!(!email.body.contains("Ignore attachment"));
}

#[test]
fn html_only_and_base64_messages_are_readable() {
    let email = parse_email("new/one", b"Content-Type: text/html; charset=utf-8\r\nContent-Transfer-Encoding: base64\r\n\r\nPHA+UGF5bWVudCByZWNlaXZlZDwvcD4=").unwrap();
    assert!(email.body.contains("Payment received"));
    let email = parse_email("new/two", b"Subject: Empty\r\n\r\n").unwrap();
    assert!(email.body.is_empty());
}

#[test]
fn html_mail_exposes_visible_text_without_style_or_script_noise() {
    let raw = b"Subject: Sign-in\r\nContent-Type: text/html; charset=utf-8\r\n\r\n<html><head><style>.noise { color:red; }</style><script>tracking()</script></head><body><p>Your verification code is <b>123456</b>.</p><p>Do not share it &amp; keep it safe.</p></body></html>";
    let email = parse_email("new/html", raw).unwrap();
    assert!(email.body.contains("Your verification code is"));
    assert!(email.body.contains("123456"));
    assert!(email.body.contains("& keep it safe"));
    assert!(!email.body.contains(".noise"));
    assert!(!email.body.contains("tracking()"));
    assert!(!email.body.contains("<p>"));
}
