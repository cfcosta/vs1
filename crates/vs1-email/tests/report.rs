use serde_json::json;
use vs1_email::{Config, Email, Mailbox, dry_run};

#[test]
fn report_categorizes_each_message_and_retains_mailbox_identity() {
    let config = Config::parse("[[rules]]\ncategory='income'\nwhat='Money arriving'\n[[rules]]\ncategory='other'\nwhat='Other'").unwrap();
    let mailbox = Mailbox {
        path: "/mail/INBOX".into(),
        failures: vec![vs1_email::MessageFailure {
            path: "/mail/INBOX/new/bad".into(),
            error: "invalid MIME".into(),
        }],
        emails: vec![Email {
            path: "/mail/INBOX/new/message".into(),
            message_id: "<test>".into(),
            subject: "Payment received".into(),
            from: "test@example.test".into(),
            to: "owner@example.test".into(),
            date: String::new(),
            body: "Paid 100".into(),
        }],
    };
    let mut calls = 0;
    let report = dry_run(&config, &mailbox, 2, &mut |_| Ok(true), &mut |_| {
        calls += 1;
        Ok(vec![serde_json::from_value(json!({"model":"laya", "usage":{"input_tokens":20,"output_tokens":0},"answers":{"category":{"type":"choice","choice":"income","probabilities":{"income":0.9,"other":0.1},"confidence":0.5}}})).unwrap()])
    }).unwrap();
    assert_eq!(calls, 1);
    let report = serde_json::to_value(report).unwrap();
    assert_eq!(report["dry_run"], true);
    assert_eq!(report["failures"][0]["error"], "invalid MIME");
    assert_eq!(report["mailbox"], "/mail/INBOX");
    assert_eq!(
        report["classifications"][0]["path"],
        "/mail/INBOX/new/message"
    );
    assert_eq!(report["classifications"][0]["category"], "income");
    let empty = Mailbox {
        path: "/mail/INBOX".into(),
        failures: vec![vs1_email::MessageFailure {
            path: "/mail/INBOX/new/bad".into(),
            error: "invalid MIME".into(),
        }],
        emails: vec![],
    };
    let report = dry_run(&config, &empty, 2, &mut |_| Ok(true), &mut |_| {
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

#[test]
fn batches_messages_and_retains_order_including_partial_last_batch() {
    let config = Config::parse("[[rules]]\ncategory='income'\nwhat='Income'\n[[rules]]\ncategory='other'\nwhat='Other'").unwrap();
    let mailbox = Mailbox {
        path: "/mail".into(),
        failures: vec![],
        emails: (0..5)
            .map(|i| Email {
                path: format!("/mail/new/{i}").into(),
                message_id: i.to_string(),
                subject: i.to_string(),
                from: String::new(),
                to: String::new(),
                date: String::new(),
                body: String::new(),
            })
            .collect(),
    };
    let mut sizes = Vec::new();
    let report = dry_run(&config, &mailbox, 2, &mut |_| Ok(true), &mut |requests| {
        sizes.push(requests.len());
        Ok(requests.iter().map(|request| {
            let state = serde_json::to_value(&request.state).unwrap();
            let category = if state["email"]["subject"] == "1" { "other" } else { "income" };
            let probabilities = if category == "other" { json!({"income":0.1,"other":0.9}) } else { json!({"income":0.9,"other":0.1}) };
            serde_json::from_value(json!({"model":"laya", "usage":{"input_tokens":20,"output_tokens":0},"answers":{"category":{"type":"choice","choice":category,"probabilities":probabilities,"confidence":0.5}}})).unwrap()
        }).collect())
    }).unwrap();
    assert_eq!(sizes, [2, 2, 1]);
    assert_eq!(
        report
            .classifications
            .iter()
            .map(|c| c.message_id.as_str())
            .collect::<Vec<_>>(),
        ["0", "1", "2", "3", "4"]
    );
    assert_eq!(report.classifications[1].category, "other");
    assert!(
        dry_run(&config, &mailbox, 0, &mut |_| Ok(true), &mut |_| panic!(
            "must reject zero before inference"
        ))
        .is_err()
    );
    assert!(
        dry_run(&config, &mailbox, 2, &mut |_| Ok(true), &mut |_| Ok(vec![]))
            .is_err()
    );
    assert!(
        dry_run(
            &config,
            &mailbox,
            2,
            &mut |_| Ok(true),
            &mut |_| anyhow::bail!("backend failure")
        )
        .is_err()
    );
}

#[test]
fn combines_every_chunk_by_length_instead_of_using_only_first_chunk() {
    let config = Config::parse("[[rules]]\ncategory='income'\nwhat='Income'\n[[rules]]\ncategory='other'\nwhat='Other'").unwrap();
    let mailbox = Mailbox {
        path: "/mail".into(),
        failures: vec![],
        emails: vec![Email {
            path: "/mail/new/a".into(),
            message_id: "a".into(),
            subject: "Subject".into(),
            from: String::new(),
            to: String::new(),
            date: String::new(),
            body: "aaaa bbbb cc".into(),
        }],
    };
    let mut bodies = Vec::new();
    let report = dry_run(&config, &mailbox, 2,
        &mut |r| Ok(serde_json::to_value(&r.state)?["email"]["body"].as_str().unwrap().len() <= 5),
        &mut |requests| Ok(requests.iter().map(|r| {
            let state = serde_json::to_value(&r.state).unwrap();
            let body = state["email"]["body"].as_str().unwrap();
            bodies.push(body.to_owned());
            let income = body.starts_with('a');
            serde_json::from_value(json!({"model":"laya", "usage":{"input_tokens":10,"output_tokens":0},"answers":{"category":{"type":"choice","choice":if income {"income"} else {"other"},"probabilities":{"income":if income {0.9} else {0.1},"other":if income {0.1} else {0.9}},"confidence":0.5}}})).unwrap()
        }).collect())
    ).unwrap();
    assert_eq!(bodies.concat(), "aaaa bbbb cc");
    let result = &report.classifications[0];
    assert_eq!(result.category, "other");
    assert_eq!(result.chunks.len(), 3);
    assert_eq!(result.usage.input_tokens, 30);
    let expected = (5.0 * 0.9 + 7.0 * 0.1) / 12.0;
    assert!(
        (result.probabilities["income"].as_f64().unwrap() - expected).abs()
            < 0.0001
    );
    assert_eq!(
        result.chunks.iter().map(|c| c.body_chars).sum::<usize>(),
        12
    );
}
