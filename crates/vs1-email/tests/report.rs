use serde_json::json;
use vs1_email::{Config, Email, Mailbox, dry_run};

#[test]
fn report_categorizes_each_message_and_retains_mailbox_identity() {
    let config = Config::parse("[[rules]]\ncategory='income'\nwhat='Money arriving'\n[[rules]]\ncategory='other'\nwhat='Other'").unwrap();
    let mailbox = Mailbox {
        uid_validity: 123,
        emails: vec![Email {
            uid: 7,
            message_id: "<test>".into(),
            subject: "Payment received".into(),
            from: "test@example.test".into(),
            to: "owner@example.test".into(),
            date: String::new(),
            body: "Paid 100".into(),
        }],
    };
    let mut calls = 0;
    let report = dry_run(&config, "imap.example.test", "owner", "INBOX", &mailbox, &mut |_| {
        calls += 1;
        Ok(serde_json::from_value(json!({"model":"laya", "usage":{"input_tokens":20,"output_tokens":0},"answers":{"category":{"type":"choice","choice":"income","probabilities":{"income":0.9,"other":0.1},"confidence":0.5}}})).unwrap())
    }).unwrap();
    assert_eq!(calls, 1);
    let report = serde_json::to_value(report).unwrap();
    assert_eq!(report["dry_run"], true);
    assert_eq!(report["host"], "imap.example.test");
    assert_eq!(report["username"], "owner");
    assert_eq!(report["mailbox"], "INBOX");
    assert_eq!(report["uid_validity"], 123);
    assert_eq!(report["classifications"][0]["uid"], 7);
    assert_eq!(report["classifications"][0]["category"], "income");
    let empty = Mailbox {
        uid_validity: 123,
        emails: vec![],
    };
    let report = dry_run(&config, "host", "user", "INBOX", &empty, &mut |_| {
        panic!("empty mailbox must not invoke model")
    })
    .unwrap();
    assert!(
        serde_json::to_value(report).unwrap()["classifications"]
            .as_array()
            .unwrap()
            .is_empty()
    );
}
