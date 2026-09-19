//! Request and response shapes, modelled on the TypeSafe `systemone`
//! endpoint that laya is API-compatible with.
//!
//! A request pairs one *state* (text or JSON) with a map of named,
//! typed *questions*. The three primitives are:
//!
//! - [`Question::Choice`]: pick one option out of a labelled set;
//! - [`Question::Score`]: place the state on an ordered rubric;
//! - [`Question::Noul`]: a calibrated yes/no probability.
//!
//! Every question is evaluated independently against the same state,
//! and all of them run in one forward pass, so adding questions is
//! nearly free. The response mirrors the request: one [`Answer`] per
//! question id, plus token usage.

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::{json::Json, pyjson};

/// What the model reads: plain text, or a JSON document that is
/// serialised the way CPython's `json.dumps` would print it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum State {
    /// Raw text, passed to the model verbatim.
    Text(String),
    /// A JSON object, array, or scalar. Objects keep the key order
    /// they were parsed or built in; a bare string is passed through
    /// like [`State::Text`].
    Json(Json),
}

impl State {
    /// The text the model actually sees.
    pub fn render(&self) -> String {
        match self {
            State::Text(text) => text.clone(),
            State::Json(Json::String(text)) => text.clone(),
            State::Json(value) => pyjson::dumps(value),
        }
    }
}

impl From<&str> for State {
    fn from(text: &str) -> Self {
        State::Text(text.to_string())
    }
}

impl From<String> for State {
    fn from(text: String) -> Self {
        State::Text(text)
    }
}

impl From<Json> for State {
    fn from(value: Json) -> Self {
        State::Json(value)
    }
}

impl From<serde_json::Value> for State {
    /// Objects arrive in the `Value`'s order, which is sorted unless
    /// serde_json was built with `preserve_order`; parse into [`Json`]
    /// directly when the order matters.
    fn from(value: serde_json::Value) -> Self {
        State::Json(value.into())
    }
}

/// Free text or a JSON structure used for instructions and criteria.
///
/// Structured values are rendered with CPython's `json.dumps` layout
/// before they reach the model, matching what laya does in Python.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Description {
    /// Plain text.
    Text(String),
    /// Any other JSON value.
    Json(Json),
}

impl Description {
    /// Text form of the description.
    pub fn render(&self) -> String {
        match self {
            Description::Text(text) => text.clone(),
            Description::Json(Json::String(text)) => text.clone(),
            Description::Json(value) => pyjson::dumps(value),
        }
    }

    /// `true` for `""`, `null`, and nothing else.
    fn is_blank(&self) -> bool {
        match self {
            Description::Text(text) => text.is_empty(),
            Description::Json(value) => value.is_blank(),
        }
    }
}

impl From<&str> for Description {
    fn from(text: &str) -> Self {
        Description::Text(text.to_string())
    }
}

impl From<String> for Description {
    fn from(text: String) -> Self {
        Description::Text(text)
    }
}

/// Options for a choice question: either labels with descriptions or
/// a bare list of labels.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ChoiceCriteria {
    /// `label -> description`; a `null` or empty description leaves the
    /// label to speak for itself.
    Labelled(IndexMap<String, Option<Description>>),
    /// Labels only.
    Labels(Vec<String>),
}

impl ChoiceCriteria {
    /// Labels in declaration order.
    pub fn labels(&self) -> Vec<&str> {
        match self {
            ChoiceCriteria::Labelled(map) => {
                map.keys().map(String::as_str).collect()
            }
            ChoiceCriteria::Labels(labels) => {
                labels.iter().map(String::as_str).collect()
            }
        }
    }

    /// Option texts as the model reads them: `label` on its own, or
    /// `label: description`.
    pub fn render_options(&self) -> Vec<String> {
        match self {
            ChoiceCriteria::Labelled(map) => map
                .iter()
                .map(|(label, description)| match description {
                    Some(d) if !d.is_blank() => {
                        format!("{label}: {}", d.render())
                    }
                    _ => label.clone(),
                })
                .collect(),
            ChoiceCriteria::Labels(labels) => labels.clone(),
        }
    }
}

/// Optional wording for the two sides of a noul question.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct NoulCriteria {
    /// What a `true` answer means.
    #[serde(rename = "true", default, skip_serializing_if = "Option::is_none")]
    pub yes: Option<Description>,
    /// What a `false` answer means.
    #[serde(
        rename = "false",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub no: Option<Description>,
}

/// Pick one option from a labelled set.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChoiceQuestion {
    /// The question put to the model.
    pub instructions: Description,
    /// The options to choose between.
    pub criteria: ChoiceCriteria,
}

/// Place the state on an ordered rubric; level `i` is `criteria[i]`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScoreQuestion {
    /// The question put to the model.
    pub instructions: Description,
    /// Rubric levels from lowest to highest; at least two.
    pub criteria: Vec<Description>,
}

/// A yes/no statement answered with a probability of `true`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NoulQuestion {
    /// The statement or question put to the model.
    pub instructions: Description,
    /// Optional wording for each side.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub criteria: Option<NoulCriteria>,
}

/// One typed question.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Question {
    /// See [`ChoiceQuestion`].
    Choice(ChoiceQuestion),
    /// See [`ScoreQuestion`].
    Score(ScoreQuestion),
    /// See [`NoulQuestion`].
    Noul(NoulQuestion),
}

impl Question {
    /// A choice question over labelled options.
    pub fn choice<I, K, V>(
        instructions: impl Into<Description>,
        options: I,
    ) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<Description>,
    {
        Question::Choice(ChoiceQuestion {
            instructions: instructions.into(),
            criteria: ChoiceCriteria::Labelled(
                options
                    .into_iter()
                    .map(|(k, v)| (k.into(), Some(v.into())))
                    .collect(),
            ),
        })
    }

    /// A score question over ordered levels.
    pub fn score<I, V>(instructions: impl Into<Description>, levels: I) -> Self
    where
        I: IntoIterator<Item = V>,
        V: Into<Description>,
    {
        Question::Score(ScoreQuestion {
            instructions: instructions.into(),
            criteria: levels.into_iter().map(Into::into).collect(),
        })
    }

    /// A noul question with default wording for both sides.
    pub fn noul(instructions: impl Into<Description>) -> Self {
        Question::Noul(NoulQuestion {
            instructions: instructions.into(),
            criteria: None,
        })
    }

    /// A noul question with explicit wording for `true` and `false`.
    pub fn noul_with_criteria(
        instructions: impl Into<Description>,
        yes: impl Into<Description>,
        no: impl Into<Description>,
    ) -> Self {
        Question::Noul(NoulQuestion {
            instructions: instructions.into(),
            criteria: Some(NoulCriteria {
                yes: Some(yes.into()),
                no: Some(no.into()),
            }),
        })
    }

    /// The primitive's wire name: `choice`, `score`, or `noul`.
    pub fn kind(&self) -> QuestionKind {
        match self {
            Question::Choice(_) => QuestionKind::Choice,
            Question::Score(_) => QuestionKind::Score,
            Question::Noul(_) => QuestionKind::Noul,
        }
    }

    /// The instructions, whichever primitive this is.
    pub fn instructions(&self) -> &Description {
        match self {
            Question::Choice(q) => &q.instructions,
            Question::Score(q) => &q.instructions,
            Question::Noul(q) => &q.instructions,
        }
    }

    /// Option texts in label order, exactly as laya renders them.
    ///
    /// Noul always yields `[false, true]`, score yields `level i: ...`.
    pub fn render_options(&self) -> Vec<String> {
        match self {
            Question::Choice(q) => q.criteria.render_options(),
            Question::Score(q) => q
                .criteria
                .iter()
                .enumerate()
                .map(|(i, level)| format!("level {i}: {}", level.render()))
                .collect(),
            Question::Noul(q) => {
                let criteria = q.criteria.clone().unwrap_or_default();
                let no = criteria
                    .no
                    .filter(|d| !d.is_blank())
                    .map(|d| d.render())
                    .unwrap_or_else(|| {
                        "no, the statement does not hold".to_string()
                    });
                let yes = criteria
                    .yes
                    .filter(|d| !d.is_blank())
                    .map(|d| d.render())
                    .unwrap_or_else(|| "yes, the statement holds".to_string());
                vec![format!("false: {no}"), format!("true: {yes}")]
            }
        }
    }
}

/// The three primitives, with the integer ids the checkpoint uses for
/// its type embedding and temperature table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuestionKind {
    Choice,
    Score,
    Noul,
}

impl QuestionKind {
    /// Index into the checkpoint's `type_emb` and `temperature`.
    pub fn index(self) -> usize {
        match self {
            QuestionKind::Choice => 0,
            QuestionKind::Score => 1,
            QuestionKind::Noul => 2,
        }
    }

    /// Wire name.
    pub fn as_str(self) -> &'static str {
        match self {
            QuestionKind::Choice => "choice",
            QuestionKind::Score => "score",
            QuestionKind::Noul => "noul",
        }
    }
}

/// One state, many questions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SystemOneRequest {
    /// Text or JSON to evaluate.
    pub state: State,
    /// Model name override. Informational: the loaded checkpoint
    /// always answers, and the response echoes its own name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Questions keyed by the ids their answers come back under.
    pub questions: IndexMap<String, Question>,
}

impl SystemOneRequest {
    /// A request with no questions yet.
    pub fn new(state: impl Into<State>) -> Self {
        Self {
            state: state.into(),
            model: None,
            questions: IndexMap::new(),
        }
    }

    /// Adds a question under `id`.
    pub fn question(
        mut self,
        id: impl Into<String>,
        question: Question,
    ) -> Self {
        self.questions.insert(id.into(), question);
        self
    }
}

/// Extra laya signal shipped with every answer.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Action {
    /// Probability that acting on the answer beats escalating to a
    /// human. The checkpoint's action head learned this against a
    /// wrong-answer cost, so it is a second, coarser trust signal next
    /// to `confidence`.
    pub act_probability: f32,
}

/// Answer to a [`ChoiceQuestion`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChoiceAnswer {
    /// The most probable label.
    pub choice: String,
    /// Calibrated probability per label, in option order.
    pub probabilities: IndexMap<String, f32>,
    /// `1 - H(p) / log(k)`: `1.0` when one label takes everything,
    /// `0.0` when the distribution is uniform.
    pub confidence: f32,
    /// See [`Action`].
    pub action: Action,
}

/// Answer to a [`ScoreQuestion`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScoreAnswer {
    /// Expected level, `sum(i * p_i)`, so `0.0 ..= levels - 1`.
    pub score: f32,
    /// `"i" -> level text`.
    pub legend: IndexMap<String, String>,
    /// `"i" -> probability`.
    pub probabilities: IndexMap<String, f32>,
    /// Normalised-entropy confidence, as for choice.
    pub confidence: f32,
    /// See [`Action`].
    pub action: Action,
}

/// Answer to a [`NoulQuestion`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NoulAnswer {
    /// Calibrated probability that the statement holds.
    pub noul: f32,
    /// `max(noul, 1 - noul)`.
    pub confidence: f32,
    /// See [`Action`].
    pub action: Action,
}

/// One answer, tagged with its primitive.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Answer {
    Choice(ChoiceAnswer),
    Score(ScoreAnswer),
    Noul(NoulAnswer),
}

impl Answer {
    /// The `noul` probability, when this is a noul answer.
    pub fn noul(&self) -> Option<f32> {
        match self {
            Answer::Noul(a) => Some(a.noul),
            _ => None,
        }
    }

    /// The expected score, when this is a score answer.
    pub fn score(&self) -> Option<f32> {
        match self {
            Answer::Score(a) => Some(a.score),
            _ => None,
        }
    }

    /// The chosen label, when this is a choice answer.
    pub fn choice(&self) -> Option<&str> {
        match self {
            Answer::Choice(a) => Some(a.choice.as_str()),
            _ => None,
        }
    }

    /// Confidence, whichever primitive this is.
    pub fn confidence(&self) -> f32 {
        match self {
            Answer::Choice(a) => a.confidence,
            Answer::Score(a) => a.confidence,
            Answer::Noul(a) => a.confidence,
        }
    }

    /// Action probability, whichever primitive this is.
    pub fn action(&self) -> Action {
        match self {
            Answer::Choice(a) => a.action,
            Answer::Score(a) => a.action,
            Answer::Noul(a) => a.action,
        }
    }
}

/// Token accounting for one request.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize,
)]
pub struct Usage {
    /// Tokens the encoder read, summed over every question sequence.
    pub input_tokens: usize,
    /// Always zero: nothing is generated.
    pub output_tokens: usize,
}

/// Answers for one request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SystemOneResponse {
    /// Name of the checkpoint that answered.
    pub model: String,
    /// One answer per question id, in request order.
    pub answers: IndexMap<String, Answer>,
    /// Token accounting.
    pub usage: Usage,
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn request_round_trips_through_jev_json() {
        // Parsed from text, not from a `Value`: serde_json's default
        // map sorts keys, and question order is part of the contract.
        let raw = r#"{
            "state": {"from": "a@b.c", "body": "refund me"},
            "model": "laya",
            "questions": {
                "department": {
                    "type": "choice",
                    "instructions": "Which department?",
                    "criteria": {"billing": "invoices", "other": null}
                },
                "urgency": {
                    "type": "score",
                    "instructions": "How urgent?",
                    "criteria": ["not urgent", "soon", "critical"]
                },
                "churn": {
                    "type": "noul",
                    "instructions": "Will they leave?",
                    "criteria": {"true": "yes they will", "false": "no"}
                },
                "spam": {"type": "noul", "instructions": "Is it spam?"}
            }
        }"#;
        let request: SystemOneRequest = serde_json::from_str(raw).unwrap();
        assert_eq!(request.questions.len(), 4);
        assert_eq!(
            request.questions.keys().collect::<Vec<_>>(),
            ["department", "urgency", "churn", "spam"]
        );
        let back = serde_json::to_value(&request).unwrap();
        let expected: serde_json::Value = serde_json::from_str(raw).unwrap();
        assert_eq!(back, expected);
    }

    #[test]
    fn choice_options_render_label_or_label_colon_description() {
        let q: Question = serde_json::from_value(json!({
            "type": "choice",
            "instructions": "?",
            "criteria": {"a": "first", "b": null, "c": "", "d": 0, "e": false}
        }))
        .unwrap();
        // Only null and "" mean "no description"; 0 and false are
        // legitimate criterion values, like in laya.
        assert_eq!(
            q.render_options(),
            ["a: first", "b", "c", "d: 0", "e: false"]
        );
    }

    #[test]
    fn choice_labels_list_renders_bare_labels() {
        let q: Question = serde_json::from_value(json!({
            "type": "choice",
            "instructions": "?",
            "criteria": ["x", "y"]
        }))
        .unwrap();
        assert_eq!(q.render_options(), ["x", "y"]);
    }

    #[test]
    fn score_options_are_numbered_levels() {
        let q = Question::score("?", ["calm", "annoyed"]);
        assert_eq!(q.render_options(), ["level 0: calm", "level 1: annoyed"]);
    }

    #[test]
    fn noul_options_default_and_custom() {
        assert_eq!(
            Question::noul("?").render_options(),
            [
                "false: no, the statement does not hold",
                "true: yes, the statement holds"
            ]
        );
        assert_eq!(
            Question::noul_with_criteria("?", "it is so", "it is not")
                .render_options(),
            ["false: it is not", "true: it is so"]
        );
    }

    #[test]
    fn structured_instructions_render_as_python_json() {
        let q: Question = serde_json::from_value(json!({
            "type": "noul",
            "instructions": {"ask": "is it?", "n": 2}
        }))
        .unwrap();
        assert_eq!(q.instructions().render(), r#"{"ask": "is it?", "n": 2}"#);
    }

    #[test]
    fn state_renders_text_verbatim_and_json_like_python() {
        assert_eq!(State::from("hi \"there\"").render(), "hi \"there\"");
        assert_eq!(
            State::from(json!({"a": [1, 2], "b": "x"})).render(),
            r#"{"a": [1, 2], "b": "x"}"#
        );
        assert_eq!(State::from(json!("bare")).render(), "bare");
        // Parsed from text, keys keep their written order.
        let state: State = serde_json::from_str(
            r#"{"from": "a", "subject": "s", "body": "b"}"#,
        )
        .unwrap();
        assert_eq!(
            state.render(),
            r#"{"from": "a", "subject": "s", "body": "b"}"#
        );
    }

    #[test]
    fn answers_serialise_with_type_tag() {
        let answer = Answer::Noul(NoulAnswer {
            noul: 0.9,
            confidence: 0.9,
            action: Action {
                act_probability: 0.5,
            },
        });
        let value = serde_json::to_value(&answer).unwrap();
        assert_eq!(value["type"], "noul");
        assert_eq!(value["noul"], 0.9f32 as f64);
    }
}
