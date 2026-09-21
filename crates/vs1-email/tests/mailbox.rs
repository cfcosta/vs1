use std::{
    io::{Cursor, Read, Write},
    sync::{Arc, Mutex},
};

use vs1_email::{parse_email, read_mailbox};

#[test]
fn decodes_headers_and_prefers_plain_text_over_html_and_attachments() {
    let raw = b"From: bank@example.test\r\nTo: owner@example.test\r\nSubject: =?UTF-8?Q?Extrato_S=C3=A3o?=\r\nMessage-ID: <x@test>\r\nContent-Type: multipart/mixed; boundary=outer\r\n\r\n--outer\r\nContent-Type: multipart/alternative; boundary=inner\r\n\r\n--inner\r\nContent-Type: text/plain; charset=utf-8\r\nContent-Transfer-Encoding: quoted-printable\r\n\r\nSaldo S=C3=A3o\r\n--inner\r\nContent-Type: text/html\r\n\r\n<p>duplicate HTML</p>\r\n--inner--\r\n--outer\r\nContent-Type: text/plain\r\nContent-Disposition: attachment; filename=secret.txt\r\n\r\nIgnore attachment\r\n--outer--\r\n";
    let email = parse_email(42, raw).unwrap();
    assert_eq!(email.uid, 42);
    assert_eq!(email.subject, "Extrato São");
    assert_eq!(email.message_id, "<x@test>");
    assert_eq!(email.from, "bank@example.test");
    assert!(email.body.contains("Saldo São"));
    assert!(!email.body.contains("duplicate HTML"));
    assert!(!email.body.contains("Ignore attachment"));
}

#[test]
fn html_only_and_base64_messages_are_readable() {
    let email = parse_email(1, b"Content-Type: text/html; charset=utf-8\r\nContent-Transfer-Encoding: base64\r\n\r\nPHA+UGF5bWVudCByZWNlaXZlZDwvcD4=").unwrap();
    assert!(email.body.contains("Payment received"));
    let email = parse_email(2, b"Subject: Empty\r\n\r\n").unwrap();
    assert!(email.body.is_empty());
}

struct Transport {
    input: Cursor<Vec<u8>>,
    output: Arc<Mutex<Vec<u8>>>,
}
impl Read for Transport {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.input.read(buf)
    }
}
impl Write for Transport {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.output.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn session(replies: &str) -> (imap::Session<Transport>, Arc<Mutex<Vec<u8>>>) {
    let output = Arc::new(Mutex::new(Vec::new()));
    let transport = Transport {
        input: Cursor::new(
            format!("* OK ready\r\na1 OK logged in\r\n{replies}").into_bytes(),
        ),
        output: output.clone(),
    };
    let mut client = imap::Client::new(transport);
    client.read_greeting().unwrap();
    let session = client
        .login("user", "password")
        .map_err(|(e, _)| e)
        .unwrap();
    (session, output)
}

#[test]
fn reads_sorted_uids_with_examine_and_peek_and_honors_limit() {
    let raw = "Subject: Bill\r\n\r\nAmount due";
    let replies = format!(
        "* 3 EXISTS\r\n* OK [UIDVALIDITY 99] valid\r\na2 OK [READ-ONLY] examined\r\n* SEARCH 30 10 20\r\na3 OK searched\r\n* 1 FETCH (UID 10 BODY[] {{{}}}\r\n{})\r\na4 OK fetched\r\n",
        raw.len(),
        raw
    );
    let (mut session, output) = session(&replies);
    let mailbox = read_mailbox(&mut session, "INBOX", "UNSEEN", 1).unwrap();
    assert_eq!(mailbox.uid_validity, 99);
    assert_eq!(mailbox.emails.len(), 1);
    assert_eq!(mailbox.emails[0].uid, 10);
    let commands = String::from_utf8(output.lock().unwrap().clone()).unwrap();
    assert!(commands.contains("EXAMINE \"INBOX\"\r\n"), "{commands}");
    assert!(commands.contains("UID SEARCH UNSEEN\r\n"), "{commands}");
    assert!(
        commands.contains("UID FETCH 10 (UID BODY.PEEK[])\r\n"),
        "{commands}"
    );
    for forbidden in ["SELECT", "STORE", "MOVE", "COPY", "EXPUNGE", "CLOSE"] {
        assert!(!commands.contains(forbidden), "{commands}");
    }
}

#[test]
fn empty_mailbox_does_not_fetch_and_missing_body_is_an_error() {
    let (mut empty, output) = session(
        "* 0 EXISTS\r\n* OK [UIDVALIDITY 99] valid\r\na2 OK examined\r\n* SEARCH\r\na3 OK searched\r\n",
    );
    assert!(
        read_mailbox(&mut empty, "INBOX", "ALL", 10)
            .unwrap()
            .emails
            .is_empty()
    );
    assert!(
        !String::from_utf8(output.lock().unwrap().clone())
            .unwrap()
            .contains("FETCH")
    );
    let (mut missing, _) = session(
        "* 1 EXISTS\r\n* OK [UIDVALIDITY 99] valid\r\na2 OK examined\r\n* SEARCH 1\r\na3 OK searched\r\n* 1 FETCH (UID 1)\r\na4 OK fetched\r\n",
    );
    assert!(read_mailbox(&mut missing, "INBOX", "ALL", 10).is_err());
}

#[test]
fn rejects_command_injection_and_zero_limit_before_io() {
    for (mailbox, search, limit) in [
        ("INBOX", "ALL\r\na9 EXPUNGE", 1),
        ("INBOX\n", "ALL", 1),
        ("INBOX", "ALL", 0),
    ] {
        let (mut session, output) = session("");
        let before = output.lock().unwrap().len();
        assert!(read_mailbox(&mut session, mailbox, search, limit).is_err());
        assert_eq!(output.lock().unwrap().len(), before);
    }
}
