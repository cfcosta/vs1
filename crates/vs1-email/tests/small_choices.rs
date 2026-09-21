use serde_json::{Map, Value, json};
use vs1_email::{Config, Email, classification_request, classify};
fn config(n: usize) -> Config {
    Config::parse(
        &(0..n)
            .map(|i| {
                format!("[[rules]]\ncategory='c{i}'\nwhat='Purpose {i}'\n")
            })
            .collect::<String>(),
    )
    .unwrap()
}
fn email() -> Email {
    Email {
        path: "test".into(),
        message_id: String::new(),
        subject: "test".into(),
        from: String::new(),
        to: String::new(),
        date: String::new(),
        body: "body".into(),
    }
}
#[test]
fn every_question_has_two_to_five_choices_even_for_large_taxonomies() {
    for n in [2, 5, 6, 17, 26, 127] {
        let r = classification_request(&config(n), &email());
        for q in r.questions.values() {
            assert!(
                (2..=5).contains(&q.render_options().len()),
                "{n}: {:?}",
                q.render_options()
            );
        }
    }
}
#[test]
fn compares_group_winners_and_preserves_evidence_and_total_usage() {
    let c = config(6);
    let mut rounds = Vec::new();
    let result=classify(&c,&email(),&mut |r|{
  let mut answers=Map::new();let mut candidates=Vec::new();
  for (id,q) in &r.questions {
   let labels=q.render_options().iter().map(|s|s.split(':').next().unwrap().to_string()).collect::<Vec<_>>();
   candidates.extend(labels.clone());
   let winner=if labels.contains(&"c3".into()) {"c3"} else {"c0"};
   let probs:Map<String,Value>=labels.iter().map(|label|(label.clone(),json!(if label==winner {0.8} else {0.2/(labels.len()-1) as f64}))).collect();
   answers.insert(id.clone(),json!({"type":"choice","choice":winner,"probabilities":probs,"confidence":0.1}));
  }
  rounds.push(candidates);
  Ok(serde_json::from_value(json!({"model":"mock","usage":{"input_tokens":90,"output_tokens":0},"answers":answers})).unwrap())
 }).unwrap();
    assert_eq!(
        rounds,
        vec![vec!["c0", "c1", "c2", "c3", "c4", "c5"], vec!["c0", "c3"]]
    );
    assert_eq!(result.category, "c3");
    assert!((result.probabilities["c3"].as_f64().unwrap() - 0.8).abs() < 1e-6);
    assert_eq!(result.probabilities["c1"], 0.0);
    assert_eq!(result.probabilities.as_object().unwrap().len(), 6);
    assert_eq!(result.usage.input_tokens, 180);
    assert_eq!(result.decisions.as_array().unwrap().len(), 2);
}

#[test]
fn rejects_invalid_final_round_instead_of_accepting_preliminary_winner() {
    let mut calls = 0;
    let result = classify(&config(6), &email(), &mut |r| {
        calls += 1;
        let mut answers = Map::new();
        for (id, q) in &r.questions {
            let labels = q
                .render_options()
                .iter()
                .map(|s| s.split(':').next().unwrap().to_owned())
                .collect::<Vec<_>>();
            let probabilities: Map<String, Value> = labels
                .iter()
                .enumerate()
                .map(|(i, l)| {
                    (l.clone(), json!(if i == 0 { 1.0 } else { 0.0 }))
                })
                .collect();
            answers.insert(id.clone(),json!({"type":"choice","choice":if calls==2 {"not_a_finalist"} else {&labels[0]},"probabilities":probabilities,"confidence":1.0}));
        }
        Ok(serde_json::from_value(json!({"model":"mock","usage":{"input_tokens":1,"output_tokens":0},"answers":answers})).unwrap())
    });
    assert_eq!(calls, 2);
    assert!(result.is_err());
}

#[test]
fn large_tournament_terminates_and_retains_eliminated_rounds() {
    let result=classify(&config(127),&email(),&mut |r| {
        let answers:Map<String,Value>=r.questions.iter().map(|(id,q)| {
            let labels=q.render_options().iter().map(|s|s.split(':').next().unwrap().to_owned()).collect::<Vec<_>>();
            let probabilities:Map<String,Value>=labels.iter().enumerate().map(|(i,l)|(l.clone(),json!(if i==0 {1.0} else {0.0}))).collect();
            (id.clone(),json!({"type":"choice","choice":labels[0],"probabilities":probabilities,"confidence":1.0}))
        }).collect();
        Ok(serde_json::from_value(json!({"model":"mock","usage":{"input_tokens":1,"output_tokens":0},"answers":answers})).unwrap())
    }).unwrap();
    assert_eq!(result.category, "c0");
    assert_eq!(
        result.decisions[0]["candidates"].as_array().unwrap().len(),
        127
    );
    assert_eq!(result.decisions.as_array().unwrap().len(), 4);
    assert_eq!(result.usage.input_tokens, 4);
}

#[test]
fn closest_relative_runner_up_can_win_without_an_extra_round() {
    let mut rounds = 0;
    let result = classify(&config(17), &email(), &mut |r| {
        rounds += 1;
        let mut answers = Map::new();
        for (id, q) in &r.questions {
            let labels = q.render_options().iter().map(|s| s.split(':').next().unwrap().to_owned()).collect::<Vec<_>>();
            let weights = if rounds == 1 {
                match labels[0].as_str() {
                    "c0" => vec![0.7, 0.075, 0.075, 0.075, 0.075],
                    "c5" => vec![0.4, 0.39, 0.105, 0.105],
                    "c9" => vec![0.8, 0.2, 0.0, 0.0],
                    "c13" => vec![0.26, 0.25, 0.25, 0.24],
                    _ => panic!("unexpected group"),
                }
            } else {
                assert_eq!(labels, ["c0", "c5", "c6", "c9", "c13"]);
                vec![0.0, 0.0, 1.0, 0.0, 0.0]
            };
            let choice = if rounds == 1 { &labels[0] } else { &labels[2] };
            let probabilities: Map<String,Value> = labels.iter().cloned().zip(weights.into_iter().map(|p| json!(p))).collect();
            answers.insert(id.clone(),json!({"type":"choice","choice":choice,"probabilities":probabilities,"confidence":0.0}));
        }
        Ok(serde_json::from_value(json!({"model":"mock","usage":{"input_tokens":1,"output_tokens":0},"answers":answers})).unwrap())
    }).unwrap();
    assert_eq!(rounds, 2);
    assert_eq!(result.category, "c6");
}

#[test]
fn spare_final_slot_breaks_equal_ratios_in_configuration_order() {
    let mut rounds = 0;
    classify(&config(17), &email(), &mut |r| {
        rounds += 1;
        let answers: Map<String,Value> = r.questions.iter().map(|(id,q)| {
            let labels=q.render_options().iter().map(|s|s.split(':').next().unwrap().to_owned()).collect::<Vec<_>>();
            if rounds == 2 { assert_eq!(labels, ["c0", "c1", "c5", "c9", "c13"]); }
            let probabilities:Map<String,Value>=labels.iter().map(|s|(s.clone(),json!(1.0/labels.len() as f64))).collect();
            (id.clone(),json!({"type":"choice","choice":labels[0],"probabilities":probabilities,"confidence":0.0}))
        }).collect();
        Ok(serde_json::from_value(json!({"model":"mock","usage":{"input_tokens":1,"output_tokens":0},"answers":answers})).unwrap())
    }).unwrap();
    assert_eq!(rounds, 2);
}
