use std::{
    collections::{BTreeMap, BTreeSet},
    path::PathBuf,
};

use candle_core::{DType, Device};
use hf_hub::{Repo, RepoType, api::sync::Api};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

use super::{CuaS1Input, CuaS1Option, TextConfig, TextModel, TextWeights};
use crate::{
    Answer,
    ChoiceAnswer,
    ChoiceCriteria,
    Description,
    NoulAnswer,
    Question,
    Result,
    ScoreAnswer,
    SystemOneError,
    SystemOneRequest,
    SystemOneResponse,
    Usage,
    head::confidence_from_probs,
};

pub const DEFAULT_REPO_ID: &str = "cua-ai/cua-s1-4b-0.2";
pub const ADAPTER_REVISION: &str = "16818868b0cc7813808aae4e87b417657046ab79";
pub const BASE_REPO_ID: &str = "Qwen/Qwen3.5-4B";
pub const BASE_REVISION: &str = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a";
pub const MODEL_NAME: &str = "Cua-S1-4B";
pub const DEFAULT_MAX_LEN: usize = 4096;

/// One option's final-position letter logit and option-only probability.
#[derive(Debug, Clone, Serialize)]
pub struct CuaS1OptionPrediction {
    pub letter: char,
    pub option: CuaS1Option,
    pub logit: f32,
    pub probability: f32,
    /// State tokens dropped for this question; identical for every option.
    pub dropped_state_tokens: usize,
}

/// Local Cua-S1 text decision model, supported on CPU/F32 and CUDA/BF16.
pub struct CuaS1 {
    model: TextModel,
    tokenizer: Tokenizer,
    app: String,
    device: Device,
    dtype: DType,
    max_len: usize,
}

pub struct CuaS1Builder {
    adapter_repo: String,
    device: Device,
    dtype: Option<DType>,
    local_directories: Option<(PathBuf, PathBuf)>,
    app: String,
    max_len: usize,
}

impl CuaS1Builder {
    /// Limits the complete prompt, truncating only the end of the state.
    pub fn with_max_len(mut self, tokens: usize) -> Self {
        self.max_len = tokens;
        self
    }
    pub fn with_app(mut self, app: impl Into<String>) -> Self {
        self.app = app.into();
        self
    }
    pub fn with_device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }
    /// Defaults to BF16 on CUDA and F32 on CPU.
    pub fn with_dtype(mut self, dtype: DType) -> Self {
        self.dtype = Some(dtype);
        self
    }
    /// Loads both checkpoints locally without contacting the Hub.
    /// `adapter_directory` points directly to the `text` adapter directory.
    pub fn with_local_directories(
        mut self,
        base_directory: impl Into<PathBuf>,
        adapter_directory: impl Into<PathBuf>,
    ) -> Self {
        self.local_directories =
            Some((base_directory.into(), adapter_directory.into()));
        self
    }
}

impl TryFrom<CuaS1Builder> for CuaS1 {
    type Error = SystemOneError;

    fn try_from(builder: CuaS1Builder) -> Result<Self> {
        let dtype = builder.dtype.unwrap_or_else(|| {
            if builder.device.is_cuda() && cfg!(feature = "cuda") {
                DType::BF16
            } else {
                DType::F32
            }
        });
        if builder.device.is_cuda() && dtype == DType::F32 {
            return Err(SystemOneError::Config(
                "Cua-S1 F32 weights on CUDA do not fit in 12 GB of GPU memory; use BF16".into(),
            ));
        }
        if !(builder.device.is_cpu() && dtype == DType::F32
            || builder.device.is_cuda()
                && cfg!(feature = "cuda")
                && dtype == DType::BF16)
        {
            return Err(SystemOneError::Config(
                "Cua-S1 supports only CPU with F32 or CUDA with BF16 dtype (requires the cuda feature)".into(),
            ));
        }
        let (base_directory, adapter_directory) =
            match builder.local_directories {
                Some(directories) => directories,
                None => download_checkpoints(&builder.adapter_repo)?,
            };
        let config = TextConfig::from_slice(&std::fs::read(
            base_directory.join("config.json"),
        )?)?;
        let tokenizer =
            Tokenizer::from_file(base_directory.join("tokenizer.json"))?;
        let mut weights = TextWeights::load(
            &base_directory,
            Some(&adapter_directory),
            &builder.device,
            dtype,
        )?;
        Ok(Self {
            model: TextModel::load(&mut weights, &config)?,
            tokenizer,
            app: builder.app,
            device: builder.device,
            dtype,
            max_len: builder.max_len,
        })
    }
}

#[derive(Deserialize)]
struct CheckpointIndex {
    weight_map: BTreeMap<String, String>,
}

fn download_checkpoints(adapter_repo: &str) -> Result<(PathBuf, PathBuf)> {
    let api = Api::new()?;
    let base = api.repo(Repo::with_revision(
        BASE_REPO_ID.into(),
        RepoType::Model,
        BASE_REVISION.into(),
    ));
    let mut base_directory = base.get("config.json")?;
    base_directory.pop();
    base.get("tokenizer.json")?;
    let index: CheckpointIndex = serde_json::from_slice(&std::fs::read(
        base.get("model.safetensors.index.json")?,
    )?)?;
    let shards: BTreeSet<_> = index
        .weight_map
        .iter()
        .filter(|(name, _)| name.starts_with("model.language_model."))
        .map(|(_, filename)| filename)
        .collect();
    for shard in shards {
        base.get(shard)?;
    }
    let adapter = api.repo(Repo::with_revision(
        adapter_repo.into(),
        RepoType::Model,
        if adapter_repo == DEFAULT_REPO_ID {
            ADAPTER_REVISION
        } else {
            "main"
        }
        .into(),
    ));
    let mut adapter_directory = adapter.get("text/adapter_config.json")?;
    adapter_directory.pop();
    adapter.get("text/adapter_model.safetensors")?;
    Ok((base_directory, adapter_directory))
}

impl CuaS1 {
    /// Loads the default adapter at [`ADAPTER_REVISION`], other repos at main.
    /// All adapters use the pinned base.
    /// Defaults to CPU/F32, or BF16 when selecting CUDA with the `cuda` feature.
    pub fn from(adapter_repo: &str) -> CuaS1Builder {
        CuaS1Builder {
            adapter_repo: adapter_repo.into(),
            device: Device::Cpu,
            dtype: None,
            local_directories: None,
            app: "vs1".into(),
            max_len: DEFAULT_MAX_LEN,
        }
    }
    pub fn model_name(&self) -> &str {
        MODEL_NAME
    }
    pub fn device(&self) -> &Device {
        &self.device
    }
    pub fn dtype(&self) -> DType {
        self.dtype
    }
    pub fn max_len(&self) -> usize {
        self.max_len
    }

    pub fn system_one(
        &self,
        request: &SystemOneRequest,
    ) -> Result<SystemOneResponse> {
        self.system_one_batch(std::slice::from_ref(request))?
            .pop()
            .ok_or_else(|| SystemOneError::Config("missing response".into()))
    }
    pub fn system_one_batch(
        &self,
        requests: &[SystemOneRequest],
    ) -> Result<Vec<SystemOneResponse>> {
        let mut inputs = Vec::new();
        let mut locations = Vec::new();
        for (r, request) in requests.iter().enumerate() {
            let state = request.state.render();
            for (id, question) in &request.questions {
                let (goal, options) = format_input(id, question)?;
                inputs.push(CuaS1Input::encode_with_max_len(
                    &self.tokenizer,
                    &options,
                    &self.app,
                    id,
                    &state,
                    Some(&goal),
                    self.max_len,
                )?);
                locations.push((r, id, question, options));
            }
        }
        let mut responses: Vec<_> = requests
            .iter()
            .map(|_| SystemOneResponse {
                model: MODEL_NAME.into(),
                answers: IndexMap::new(),
                usage: Usage::default(),
            })
            .collect();
        for ((r, id, q, options), (input, dropped_state_tokens)) in
            locations.into_iter().zip(inputs)
        {
            let prediction = self.model.forward(&input)?;
            responses[r].usage.input_tokens += input.input_ids.len();
            if dropped_state_tokens > 0 {
                responses[r]
                    .usage
                    .dropped_state_tokens
                    .insert(id.clone(), dropped_state_tokens);
            }
            responses[r].answers.insert(
                id.clone(),
                answer(q, &options, &prediction.probabilities),
            );
        }
        Ok(responses)
    }

    /// Scores 1..=26 options in one forward pass, preserving option order.
    /// Probabilities are normalized over only the option-letter logits.
    pub fn score_options(
        &self,
        app: &str,
        task_family: &str,
        ax_tree: &str,
        goal: Option<&str>,
        options: &[CuaS1Option],
    ) -> Result<Vec<CuaS1OptionPrediction>> {
        let (input, dropped_state_tokens) = CuaS1Input::encode_with_max_len(
            &self.tokenizer,
            options,
            app,
            task_family,
            ax_tree,
            goal,
            self.max_len,
        )?;
        let prediction = self.model.forward(&input)?;
        Ok(options
            .iter()
            .zip(input.letters)
            .zip(prediction.logits)
            .zip(prediction.probabilities)
            .map(|(((option, letter), logit), probability)| {
                CuaS1OptionPrediction {
                    letter,
                    option: option.clone(),
                    logit,
                    probability,
                    dropped_state_tokens,
                }
            })
            .collect())
    }
}

fn question_error(id: &str, reason: impl Into<String>) -> SystemOneError {
    SystemOneError::Question {
        id: id.into(),
        reason: reason.into(),
    }
}

fn format_input(id: &str, q: &Question) -> Result<(String, Vec<CuaS1Option>)> {
    let (goal, candidates): (_, Vec<(String, String)>) = match q {
        Question::Choice(q) => {
            let candidates = match &q.criteria {
                ChoiceCriteria::Labels(labels) => {
                    labels.iter().map(|id| (id.clone(), id.clone())).collect()
                }
                ChoiceCriteria::Labelled(options) => options
                    .iter()
                    .map(|(id, d)| {
                        let label = d
                            .as_ref()
                            .filter(|x| !x.render().trim().is_empty())
                            .map(Description::render)
                            .unwrap_or_else(|| id.clone());
                        (id.clone(), label)
                    })
                    .collect(),
            };
            (q.instructions.render(), candidates)
        }
        Question::Score(q) => (
            q.instructions.render(),
            q.criteria
                .iter()
                .enumerate()
                .map(|(i, d)| (i.to_string(), d.render()))
                .collect(),
        ),
        Question::Noul(q) => {
            if q.criteria
                .as_ref()
                .is_some_and(|c| c.yes.is_some() || c.no.is_some())
            {
                return Err(question_error(
                    id,
                    "Cua-S1 noul uses yes/no labels; custom true/false criteria are unsupported",
                ));
            }
            (
                q.instructions.render(),
                vec![("yes".into(), "yes".into()), ("no".into(), "no".into())],
            )
        }
    };
    if !(2..=26).contains(&candidates.len()) {
        return Err(question_error(
            id,
            "Cua-S1 requires 2..=26 candidates (one letter each, A..Z)",
        ));
    }
    if candidates.iter().any(|(id, _)| id.is_empty())
        || candidates
            .iter()
            .map(|(id, _)| id)
            .collect::<BTreeSet<_>>()
            .len()
            != candidates.len()
    {
        return Err(question_error(
            id,
            "candidate IDs must be nonempty and distinct",
        ));
    }
    Ok((
        goal,
        candidates
            .into_iter()
            .map(|(element_id, label)| CuaS1Option {
                element_id,
                role: "option".into(),
                label,
                action: "select".into(),
                entity_id: None,
            })
            .collect(),
    ))
}

fn answer(q: &Question, options: &[CuaS1Option], probs: &[f32]) -> Answer {
    let confidence = confidence_from_probs(probs, probs.len());
    let probabilities = options
        .iter()
        .map(|option| option.element_id.clone())
        .zip(probs.iter().copied())
        .collect();
    match q {
        Question::Choice(_) => {
            let mut best = 0;
            for i in 1..probs.len() {
                if probs[i] > probs[best] {
                    best = i;
                }
            }
            Answer::Choice(ChoiceAnswer {
                choice: options[best].element_id.clone(),
                probabilities,
                confidence,
                action: None,
            })
        }
        Question::Score(q) => Answer::Score(ScoreAnswer {
            score: probs.iter().enumerate().map(|(i, p)| i as f32 * p).sum(),
            legend: q
                .criteria
                .iter()
                .enumerate()
                .map(|(i, d)| (i.to_string(), d.clone()))
                .collect(),
            probabilities,
            confidence,
            action: None,
        }),
        Question::Noul(_) => Answer::Noul(NoulAnswer {
            noul: probs[0],
            action: None,
        }),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn parse_question(value: serde_json::Value) -> Question {
        serde_json::from_value(value).unwrap()
    }

    #[test]
    fn defaults_app_to_vs1_and_accepts_an_override() {
        assert_eq!(CuaS1::from("unused/repo").app, "vs1");
        assert_eq!(CuaS1::from("unused/repo").with_app("mail").app, "mail");
    }

    #[test]
    fn defaults_max_len_to_4096_and_accepts_an_override() {
        assert_eq!(DEFAULT_MAX_LEN, 4096);
        assert_eq!(CuaS1::from("unused/repo").max_len, DEFAULT_MAX_LEN);
        assert_eq!(CuaS1::from("unused/repo").with_max_len(512).max_len, 512);
    }

    #[test]
    fn maps_choice_descriptions_in_criteria_order() {
        let q = parse_question(json!({
            "type": "choice", "instructions": "Route?",
            "criteria": {"z": "last department", "a": {"team": "first"}}
        }));
        let (goal, options) = format_input("route", &q).unwrap();
        assert_eq!(goal, "Route?");
        assert_eq!(
            serde_json::to_value(options).unwrap(),
            json!([
                {"element_id": "z", "role": "option", "label": "last department", "action": "select", "entity_id": null},
                {"element_id": "a", "role": "option", "label": "{\"team\": \"first\"}", "action": "select", "entity_id": null}
            ])
        );
    }

    #[test]
    fn uses_choice_ids_when_descriptions_are_missing_or_blank() {
        for criteria in [
            json!(["z", "a", "b"]),
            json!({"z": null, "a": "", "b": " \n "}),
        ] {
            let q =
                parse_question(json!({"type": "choice", "criteria": criteria}));
            let (_, options) = format_input("route", &q).unwrap();
            assert_eq!(
                options
                    .iter()
                    .map(|o| o.element_id.as_str())
                    .collect::<Vec<_>>(),
                ["z", "a", "b"]
            );
            for option in options {
                assert_eq!(option.label, option.element_id);
                assert_eq!(option.role, "option");
                assert_eq!(option.action, "select");
                assert!(option.entity_id.is_none());
            }
        }
    }

    #[test]
    fn maps_score_levels_and_renders_instructions() {
        let q = parse_question(json!({
            "type": "score", "instructions": {"rate": "quality"},
            "criteria": ["bad", {"quality": "good"}]
        }));
        let (goal, options) = format_input("quality", &q).unwrap();
        assert_eq!(goal, "{\"rate\": \"quality\"}");
        assert_eq!(
            options
                .iter()
                .map(|o| (o.element_id.as_str(), o.label.as_str()))
                .collect::<Vec<_>>(),
            [("0", "bad"), ("1", "{\"quality\": \"good\"}")]
        );
    }

    #[test]
    fn maps_noul_to_yes_then_no() {
        for criteria in
            [json!(null), json!({}), json!({"true": null, "false": null})]
        {
            let q = parse_question(json!({
                "type": "noul", "instructions": "Paid?", "criteria": criteria
            }));
            let (goal, options) = format_input("paid", &q).unwrap();
            assert_eq!(goal, "Paid?");
            assert_eq!(
                options
                    .iter()
                    .map(|o| (o.element_id.as_str(), o.label.as_str()))
                    .collect::<Vec<_>>(),
                [("yes", "yes"), ("no", "no")]
            );
        }
    }

    #[test]
    fn rejects_custom_noul_criteria() {
        for criteria in [
            json!({"true": "paid"}),
            json!({"false": "unpaid"}),
            json!({"true": ""}),
        ] {
            let q =
                parse_question(json!({"type": "noul", "criteria": criteria}));
            let error = format_input("paid", &q).unwrap_err();
            assert!(matches!(error, SystemOneError::Question { id, reason }
                if id == "paid" && reason.contains("custom true/false criteria are unsupported")));
        }
    }

    #[test]
    fn requires_two_through_twenty_six_candidates() {
        for kind in ["choice", "score"] {
            for count in [0, 1, 2, 26, 27] {
                let criteria: Vec<_> =
                    (0..count).map(|i| i.to_string()).collect();
                let q =
                    parse_question(json!({"type": kind, "criteria": criteria}));
                let result = format_input("bounded", &q);
                if (2..=26).contains(&count) {
                    assert_eq!(result.unwrap().1.len(), count);
                } else {
                    assert!(
                        matches!(result, Err(SystemOneError::Question { id, reason })
                        if id == "bounded" && reason.contains("2..=26") && reason.contains("A..Z"))
                    );
                }
            }
        }
    }

    #[test]
    fn rejects_empty_or_duplicate_candidate_ids() {
        for criteria in [json!(["a", "a"]), json!(["a", ""])] {
            let q =
                parse_question(json!({"type": "choice", "criteria": criteria}));
            assert!(
                matches!(format_input("route", &q), Err(SystemOneError::Question { id, .. }) if id == "route")
            );
        }
    }

    #[test]
    fn answers_with_probabilities_confidence_and_expected_level() {
        let q = Question::choice("Route?", [("z", "last"), ("a", "first")]);
        let (_, options) = format_input("route", &q).unwrap();
        let Answer::Choice(a) = answer(&q, &options, &[0.25, 0.75]) else {
            panic!("choice")
        };
        assert_eq!(a.choice, "a");
        assert_eq!(
            a.probabilities,
            IndexMap::from([("z".into(), 0.25), ("a".into(), 0.75)])
        );
        assert!((a.confidence - 0.1887219).abs() < 1e-6);
        assert!(a.action.is_none());
        assert_eq!(answer(&q, &options, &[0.5, 0.5]).choice(), Some("z"));

        let q = parse_question(
            json!({"type": "score", "criteria": ["bad", {"quality": "neutral"}, "good"]}),
        );
        let (_, options) = format_input("quality", &q).unwrap();
        let Answer::Score(a) = answer(&q, &options, &[0.25, 0.25, 0.5]) else {
            panic!("score")
        };
        assert_eq!(a.score, 1.25);
        assert_eq!(
            a.probabilities,
            IndexMap::from([
                ("0".into(), 0.25),
                ("1".into(), 0.25),
                ("2".into(), 0.5)
            ])
        );
        assert_eq!(
            a.legend["1"],
            Description::Json(json!({"quality": "neutral"}))
        );
        assert!((a.confidence - 0.05360537).abs() < 1e-6);
        assert!(a.action.is_none());

        let q = Question::noul("Paid?");
        let (_, options) = format_input("paid", &q).unwrap();
        let a = answer(&q, &options, &[0.25, 0.75]);
        assert_eq!(
            serde_json::to_value(&a).unwrap(),
            json!({"type": "noul", "noul": 0.25})
        );
        assert!(a.action().is_none());
    }
}
