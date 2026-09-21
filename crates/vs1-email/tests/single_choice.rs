use serde_json::json;
use vs1_email::{Config, Email, Mailbox, dry_run_single_choice_with_progress};
#[test]
fn whole_email_uses_one_question_and_preserves_progress_and_rounded_evidence() {
    let config =
        Config::parse(include_str!("../../../examples/email-rules.toml"))
            .unwrap();
    let email = Email {
        path: "/mail/new/one".into(),
        message_id: "one".into(),
        subject: "subject".into(),
        from: "sender".into(),
        to: "recipient".into(),
        date: "date".into(),
        body: "word ".repeat(4000),
    };
    let mailbox = Mailbox {
        path: "/mail".into(),
        emails: vec![email.clone()],
        failures: vec![],
    };
    let mut calls = 0;
    let mut progress = 0;
    let report=dry_run_single_choice_with_progress(&config,&mailbox,16,&mut |rs|{
  calls+=rs.len();assert_eq!(rs.len(),1);assert_eq!(rs[0].questions.len(),1);
  let r=serde_json::to_value(&rs[0]).unwrap();assert_eq!(r["state"]["email"]["body"],email.body);assert!(r["state"]["email"].get("to").is_none());
  let criteria=r["questions"]["category"]["criteria"].as_object().unwrap();assert_eq!(criteria.len(),17);
  let mut p=serde_json::Map::new();for name in criteria.keys(){p.insert(name.clone(),json!(0.0));}p["ops"]=json!(0.50);p["other"]=json!(0.49);
  Ok(vec![serde_json::from_value(json!({"model":"jev-test","usage":{"input_tokens":300,"output_tokens":10},"answers":{"category":{"type":"choice","choice":"other","confidence":0.1,"probabilities":p}}})).unwrap()])
 },&mut |c|{progress+=1;assert_eq!(c.category,"ops");Ok(())}).unwrap();
    assert_eq!((calls, progress), (1, 1));
    assert_eq!(report.classifications[0].chunks.len(), 1);
    assert_eq!(
        report.classifications[0].chunks[0].body_chars,
        email.body.chars().count()
    );
}
