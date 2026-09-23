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

/// One option's first-round letter/logit and hierarchical probability.
#[derive(Debug, Clone, Serialize)]
pub struct CuaS1OptionPrediction {
    pub letter: char,
    pub option: CuaS1Option,
    pub logit: f32,
    pub probability: f32,
    /// Whether this option won the final round, with ties going to the first.
    pub is_selected: bool,
    /// Total forward passes for this decision; identical for every option.
    pub forward_passes: usize,
    /// State tokens dropped for this option's first-round group.
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
        let mut tokenizer =
            Tokenizer::from_file(base_directory.join("tokenizer.json"))?;
        tokenizer.with_truncation(None)?;
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
    /// Configured maximum prompt size in tokens.
    pub fn context_tokens(&self) -> usize {
        self.max_len
    }
    /// Whether every first-round chat prompt fits without truncating the state.
    /// Finalist prompts enforce the same limit when scored.
    pub fn request_fits(&self, request: &SystemOneRequest) -> Result<bool> {
        request_fits(&self.tokenizer, &self.app, self.max_len, request)
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
        let mut locations = Vec::new();
        for (r, request) in requests.iter().enumerate() {
            let state = request.state.render();
            for (id, question) in &request.questions {
                let (goal, options) = format_input(id, question)?;
                locations.push((r, id, question, options, goal, state.clone()));
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
        for (r, id, q, options, goal, state) in locations {
            let (predictions, usage) = self.score_options_with_usage(
                &self.app,
                id,
                &state,
                Some(&goal),
                &options,
            )?;
            responses[r].usage.input_tokens += usage.input_tokens;
            responses[r]
                .usage
                .dropped_state_tokens
                .extend(usage.dropped_state_tokens);
            responses[r]
                .answers
                .insert(id.clone(), answer(q, &predictions));
        }
        Ok(responses)
    }

    /// Scores options in their original order; up to 26 use one forward pass.
    /// Larger sets use balanced groups, recursively scoring their winners.
    /// Probabilities multiply each group's softmax by its finalist's probability.
    /// Use `is_selected` for the final-round winner, which can differ from argmax.
    pub fn score_options(
        &self,
        app: &str,
        task_family: &str,
        ax_tree: &str,
        goal: Option<&str>,
        options: &[CuaS1Option],
    ) -> Result<Vec<CuaS1OptionPrediction>> {
        self.score_options_with_usage(app, task_family, ax_tree, goal, options)
            .map(|(predictions, _)| predictions)
    }

    fn score_options_with_usage(
        &self,
        app: &str,
        task_family: &str,
        ax_tree: &str,
        goal: Option<&str>,
        options: &[CuaS1Option],
    ) -> Result<(Vec<CuaS1OptionPrediction>, Usage)> {
        let mut usage = Usage::default();
        let predictions = score_tournament(options, &mut |group| {
            let (input, dropped_state_tokens) =
                CuaS1Input::encode_with_max_len(
                    &self.tokenizer,
                    group,
                    app,
                    task_family,
                    ax_tree,
                    goal,
                    self.max_len,
                )?;
            let prediction = self.model.forward(&input)?;
            usage.input_tokens += input.input_ids.len();
            if dropped_state_tokens > 0 {
                let dropped = usage
                    .dropped_state_tokens
                    .entry(task_family.into())
                    .or_default();
                *dropped = (*dropped).max(dropped_state_tokens);
            }
            Ok(group
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
                        is_selected: false,
                        forward_passes: 1,
                        dropped_state_tokens,
                    }
                })
                .collect())
        })?;
        Ok((predictions, usage))
    }
}

fn split_options<T>(options: &[T]) -> impl Iterator<Item = &[T]> {
    let groups = options.len().div_ceil(26).max(1);
    let smaller_groups = groups - options.len() % groups;
    (0..groups).scan(0, move |start, group| {
        let size =
            options.len() / groups + usize::from(group >= smaller_groups);
        let end = *start + size;
        let options = &options[*start..end];
        *start = end;
        Some(options)
    })
}

fn score_tournament(
    options: &[CuaS1Option],
    score_group: &mut impl FnMut(
        &[CuaS1Option],
    ) -> Result<Vec<CuaS1OptionPrediction>>,
) -> Result<Vec<CuaS1OptionPrediction>> {
    if options.is_empty() {
        return Err(SystemOneError::Config(
            "Cua-S1 requires at least one option".into(),
        ));
    }
    if options.len() <= 26 {
        let mut predictions = score_group(options)?;
        let mut best = 0;
        for i in 1..predictions.len() {
            if predictions[i].probability > predictions[best].probability {
                best = i;
            }
        }
        for (i, prediction) in predictions.iter_mut().enumerate() {
            prediction.is_selected = i == best;
            prediction.forward_passes = 1;
        }
        return Ok(predictions);
    }
    let mut groups = Vec::new();
    let mut finalists = Vec::new();
    for options in split_options(options) {
        let predictions = score_tournament(options, score_group)?;
        finalists.extend(
            predictions
                .iter()
                .filter(|p| p.is_selected)
                .map(|p| p.option.clone()),
        );
        groups.push(predictions);
    }
    let finalists = score_tournament(&finalists, score_group)?;
    let forward_passes = groups.len() + finalists[0].forward_passes;
    let mut predictions = Vec::with_capacity(options.len());
    for (group, finalist) in groups.into_iter().zip(finalists) {
        for mut prediction in group {
            prediction.probability *= finalist.probability;
            prediction.is_selected &= finalist.is_selected;
            prediction.forward_passes = forward_passes;
            predictions.push(prediction);
        }
    }
    Ok(predictions)
}

fn question_error(id: &str, reason: impl Into<String>) -> SystemOneError {
    SystemOneError::Question {
        id: id.into(),
        reason: reason.into(),
    }
}

fn request_fits(
    tokenizer: &Tokenizer,
    app: &str,
    max_len: usize,
    request: &SystemOneRequest,
) -> Result<bool> {
    let state = request.state.render();
    for (id, question) in &request.questions {
        let (goal, options) = format_input(id, question)?;
        for group in split_options(&options) {
            let input = CuaS1Input::encode(
                tokenizer,
                group,
                app,
                id,
                &state,
                Some(&goal),
            )?;
            if input.input_ids.len() > max_len {
                return Ok(false);
            }
        }
    }
    Ok(true)
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
    if candidates.len() < 2 {
        return Err(question_error(
            id,
            "Cua-S1 requires at least 2 candidates",
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

fn answer(q: &Question, predictions: &[CuaS1OptionPrediction]) -> Answer {
    let probs: Vec<_> = predictions.iter().map(|p| p.probability).collect();
    let confidence = confidence_from_probs(&probs, probs.len());
    let probabilities = predictions
        .iter()
        .map(|p| (p.option.element_id.clone(), p.probability))
        .collect();
    match q {
        Question::Choice(_) => {
            let selected = predictions.iter().find(|p| p.is_selected).unwrap();
            Answer::Choice(ChoiceAnswer {
                choice: selected.option.element_id.clone(),
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
    use tokenizers::{
        models::wordlevel::WordLevel,
        pre_tokenizers::whitespace::Whitespace,
    };

    use super::*;

    fn parse_question(value: serde_json::Value) -> Question {
        serde_json::from_value(value).unwrap()
    }

    fn predict_group(
        options: &[CuaS1Option],
        probabilities: &[f32],
    ) -> Vec<CuaS1OptionPrediction> {
        assert_eq!(options.len(), probabilities.len());
        options
            .iter()
            .zip('A'..='Z')
            .zip(probabilities)
            .map(|((option, letter), &probability)| CuaS1OptionPrediction {
                letter,
                option: option.clone(),
                logit: probability.ln(),
                probability,
                is_selected: false,
                forward_passes: 1,
                dropped_state_tokens: 0,
            })
            .collect()
    }

    fn answer_with_probabilities(
        q: &Question,
        options: &[CuaS1Option],
        probabilities: &[f32],
    ) -> Answer {
        let predictions = score_tournament(options, &mut |group| {
            Ok(predict_group(group, probabilities))
        })
        .unwrap();
        answer(q, &predictions)
    }

    #[test]
    fn scores_up_to_twenty_six_options_in_one_pass() {
        for count in [1, 2, 26] {
            let options: Vec<_> = (0..count)
                .map(|i| CuaS1Option {
                    element_id: i.to_string(),
                    role: "button".into(),
                    label: format!("Option {i}"),
                    action: "click".into(),
                    entity_id: None,
                })
                .collect();
            let probabilities = vec![1.0 / count as f32; count];
            let mut calls = 0;
            let predictions = score_tournament(&options, &mut |group| {
                calls += 1;
                Ok(predict_group(group, &probabilities))
            })
            .unwrap();
            assert_eq!(calls, 1);
            for (i, p) in predictions.iter().enumerate() {
                assert_eq!(p.probability, probabilities[i]);
                assert_eq!(p.logit, probabilities[i].ln());
                assert_eq!(p.letter, (b'A' + i as u8) as char);
                assert_eq!(p.is_selected, i == 0);
                assert_eq!(p.forward_passes, 1);
                assert_eq!(
                    serde_json::to_value(&p.option).unwrap(),
                    serde_json::to_value(&options[i]).unwrap()
                );
            }
        }
    }

    #[test]
    fn scores_twenty_seven_options_and_selects_the_final_round_winner() {
        let q = Question::choice(
            "Pick",
            (0..27).map(|i| (i.to_string(), format!("Option {i}"))),
        );
        let (_, options) = format_input("pick", &q).unwrap();
        let mut groups = Vec::new();
        let predictions = score_tournament(&options, &mut |group| {
            groups.push(
                group
                    .iter()
                    .map(|o| o.element_id.clone())
                    .collect::<Vec<_>>(),
            );
            let probabilities = match groups.len() {
                1 => {
                    let mut probabilities = vec![0.1 / 12.0; 13];
                    probabilities[0] = 0.9;
                    probabilities
                }
                2 => vec![1.0 / 14.0; 14],
                3 => vec![0.4, 0.6],
                _ => panic!("unexpected group"),
            };
            Ok(predict_group(group, &probabilities))
        })
        .unwrap();
        assert_eq!(
            groups[0],
            (0..13).map(|i| i.to_string()).collect::<Vec<_>>()
        );
        assert_eq!(
            groups[1],
            (13..27).map(|i| i.to_string()).collect::<Vec<_>>()
        );
        assert_eq!(groups[2], ["0", "13"]);
        assert_eq!(predictions[0].letter, 'A');
        assert_eq!(predictions[13].letter, 'A');
        assert_eq!(predictions[0].logit, 0.9_f32.ln());
        assert!((predictions[0].probability - 0.36).abs() < 1e-6);
        assert!(predictions[0].probability > predictions[13].probability);
        assert!(
            (predictions.iter().map(|p| p.probability).sum::<f32>() - 1.0)
                .abs()
                < 1e-6
        );
        for (i, p) in predictions.iter().enumerate() {
            assert_eq!(p.option.element_id, i.to_string());
            assert_eq!(p.is_selected, i == 13);
            assert_eq!(p.forward_passes, 3);
            let expected = if i == 0 {
                0.36
            } else if i < 13 {
                0.4 * 0.1 / 12.0
            } else {
                0.6 / 14.0
            };
            assert!((p.probability - expected).abs() < 1e-6);
        }
        let Answer::Choice(a) = answer(&q, &predictions) else {
            panic!("choice")
        };
        assert_eq!(a.choice, "13");
        assert_eq!(
            a.probabilities.keys().collect::<Vec<_>>(),
            options.iter().map(|o| &o.element_id).collect::<Vec<_>>()
        );
        let q = Question::score("Rate", (0..27).map(|i| i.to_string()));
        let Answer::Score(a) = answer(&q, &predictions) else {
            panic!("score")
        };
        assert!((a.score - 11.96).abs() < 1e-5);
        assert_eq!(a.legend.len(), 27);
        assert_eq!(a.probabilities.len(), 27);
    }

    #[test]
    fn scores_seven_hundred_options_recursively_in_original_order() {
        let q = Question::choice(
            "Pick",
            (0..700).map(|i| (i.to_string(), format!("Option {i}"))),
        );
        let (_, options) = format_input("pick", &q).unwrap();
        let mut groups = Vec::new();
        let predictions = score_tournament(&options, &mut |group| {
            groups.push(
                group
                    .iter()
                    .map(|o| o.element_id.clone())
                    .collect::<Vec<_>>(),
            );
            let probabilities = if group.len() == 2 {
                vec![0.2, 0.8]
            } else {
                vec![1.0 / group.len() as f32; group.len()]
            };
            Ok(predict_group(group, &probabilities))
        })
        .unwrap();
        assert_eq!(groups.len(), 30);
        assert!(groups[..2].iter().all(|g| g.len() == 25));
        assert!(groups[2..27].iter().all(|g| g.len() == 26));
        assert_eq!(groups[27].len(), 13);
        assert_eq!(groups[28].len(), 14);
        assert_eq!(groups[29], ["0", "336"]);
        assert_eq!(predictions.len(), 700);
        assert!(
            (predictions.iter().map(|p| p.probability).sum::<f32>() - 1.0)
                .abs()
                < 1e-5
        );
        for (i, p) in predictions.iter().enumerate() {
            assert_eq!(p.option.element_id, i.to_string());
            assert_eq!(p.is_selected, i == 336);
            assert_eq!(p.forward_passes, 30);
            let expected = if i < 50 {
                0.2 / 13.0 / 25.0
            } else if i < 336 {
                0.2 / 13.0 / 26.0
            } else {
                0.8 / 14.0 / 26.0
            };
            assert!((p.probability - expected).abs() < 1e-7);
        }
        assert_eq!(answer(&q, &predictions).choice(), Some("336"));
    }

    #[test]
    fn rejects_empty_tournaments_without_scoring() {
        let result = score_tournament(&[], &mut |_| panic!("no options"));
        assert!(
            matches!(result, Err(SystemOneError::Config(reason)) if reason.contains("at least one option"))
        );
    }

    #[test]
    fn request_fits_counts_the_full_chat_prompt_for_every_question() {
        let mut tokenizer = Tokenizer::new(
            WordLevel::builder()
                .vocab([("[UNK]".into(), 0)].into_iter().collect())
                .unk_token("[UNK]".into())
                .build()
                .unwrap(),
        );
        tokenizer.with_pre_tokenizer(Some(Whitespace));
        let request = SystemOneRequest::new(json!({"body": "paid"}))
            .question("paid", Question::noul("Paid?"));
        let (goal, options) =
            format_input("paid", &request.questions["paid"]).unwrap();
        let input = CuaS1Input::encode(
            &tokenizer,
            &options,
            "mail",
            "paid",
            &request.state.render(),
            Some(&goal),
        )
        .unwrap();
        let tokens = input.input_ids.len();
        assert!(request_fits(&tokenizer, "mail", tokens, &request).unwrap());
        assert!(
            !request_fits(&tokenizer, "mail", tokens - 1, &request).unwrap()
        );
        assert!(!request_fits(&tokenizer, "mail", 0, &request).unwrap());
        for count in [27, 700] {
            let request = SystemOneRequest::new("state").question(
                "pick",
                Question::choice(
                    "Pick",
                    (0..count).map(|i| (i.to_string(), "option")),
                ),
            );
            assert!(
                request_fits(&tokenizer, "mail", DEFAULT_MAX_LEN, &request)
                    .unwrap()
            );
            assert!(
                !request_fits(&tokenizer, "mail", tokens, &request).unwrap()
            );
        }
        let mut long_state = request.clone();
        long_state.state = "background ".repeat(tokens).into();
        assert!(
            !request_fits(&tokenizer, "mail", tokens, &long_state).unwrap()
        );
        for question in [
            Question::choice("word ".repeat(tokens), [("a", "x"), ("b", "y")]),
            Question::score("Rate", ["word ".repeat(tokens), "good".into()]),
            Question::noul("word ".repeat(tokens)),
        ] {
            let request = request.clone().question("long", question);
            assert!(
                !request_fits(&tokenizer, "mail", tokens, &request).unwrap()
            );
        }
        assert!(
            request_fits(
                &tokenizer,
                "mail",
                tokens,
                &request.question(
                    "invalid",
                    Question::noul_with_criteria("?", "x", "y")
                ),
            )
            .is_err()
        );
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
    fn requires_at_least_two_candidates() {
        for kind in ["choice", "score"] {
            for count in [0, 1, 2, 26, 27, 700] {
                let criteria: Vec<_> =
                    (0..count).map(|i| i.to_string()).collect();
                let q =
                    parse_question(json!({"type": kind, "criteria": criteria}));
                let result = format_input("bounded", &q);
                if count >= 2 {
                    assert_eq!(result.unwrap().1.len(), count);
                } else {
                    assert!(
                        matches!(result, Err(SystemOneError::Question { id, reason })
                        if id == "bounded" && reason.contains("at least 2"))
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
        let Answer::Choice(a) =
            answer_with_probabilities(&q, &options, &[0.25, 0.75])
        else {
            panic!("choice")
        };
        assert_eq!(a.choice, "a");
        assert_eq!(
            a.probabilities,
            IndexMap::from([("z".into(), 0.25), ("a".into(), 0.75)])
        );
        assert!((a.confidence - 0.1887219).abs() < 1e-6);
        assert!(a.action.is_none());
        assert_eq!(
            answer_with_probabilities(&q, &options, &[0.5, 0.5]).choice(),
            Some("z")
        );

        let q = parse_question(
            json!({"type": "score", "criteria": ["bad", {"quality": "neutral"}, "good"]}),
        );
        let (_, options) = format_input("quality", &q).unwrap();
        let Answer::Score(a) =
            answer_with_probabilities(&q, &options, &[0.25, 0.25, 0.5])
        else {
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
        let a = answer_with_probabilities(&q, &options, &[0.25, 0.75]);
        assert_eq!(
            serde_json::to_value(&a).unwrap(),
            json!({"type": "noul", "noul": 0.25})
        );
        assert!(a.action().is_none());
    }
}
