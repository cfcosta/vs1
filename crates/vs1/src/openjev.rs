//! OpenJev (Verdict): GLiClass decisions using the shared ModernBERT encoder.
//! Native predictions retain abstention mass. Typed non-abstained answers are
//! conditional on sufficient evidence, keeping the existing vs1 answer contract.
use std::{
    collections::{HashMap, HashSet},
    path::{Path, PathBuf},
};

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{Linear, Module, VarBuilder, linear};
use hf_hub::{Repo, RepoType, api::sync::Api};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use tokenizers::{Tokenizer, TruncationParams};

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
    config::EncoderConfig,
    head::{confidence_from_probs, softmax},
    modernbert::ModernBert,
};

pub const DEFAULT_REPO_ID: &str = "heman10x/rlcd-modernbert-151m";
pub const DEFAULT_REVISION: &str = "8af2496eb63c7fa66d7d234e1f62629380030eb4";
pub const MODEL_NAME: &str = "OpenJev (Verdict)";
pub const ABSTENTION_ID: &str = "__insufficient_evidence__";
const LABEL: &str = "<<LABEL>>";
const SEPARATOR: &str = "<<SEP>>";

fn config_error(message: impl Into<String>) -> SystemOneError {
    SystemOneError::Config(message.into())
}
fn question_error(id: &str, reason: impl Into<String>) -> SystemOneError {
    SystemOneError::Question {
        id: id.into(),
        reason: reason.into(),
    }
}

#[derive(Debug, Deserialize)]
struct Config {
    model_type: String,
    architecture_type: String,
    encoder_config: serde_json::Value,
    hidden_size: usize,
    class_token_index: u32,
    text_token_index: u32,
    max_num_classes: usize,
    pooling_strategy: String,
    class_token_pooling: String,
    scorer_type: String,
    projector_hidden_act: String,
    embed_class_token: bool,
    extract_text_features: bool,
    normalize_features: bool,
    use_lstm: bool,
    use_segment_embeddings: bool,
    layer_wise: bool,
    squeeze_layers: bool,
    prompt_first: bool,
    encoder_layer_id: i64,
}
impl Config {
    fn encoder(&self) -> Result<EncoderConfig> {
        if self.model_type != "GLiClass"
            || self.architecture_type != "uni-encoder"
            || self.pooling_strategy != "first"
            || self.class_token_pooling != "first"
            || self.scorer_type != "simple"
            || self.projector_hidden_act != "gelu"
            || !self.embed_class_token
            || self.extract_text_features
            || self.normalize_features
            || self.use_lstm
            || self.use_segment_embeddings
            || self.layer_wise
            || self.squeeze_layers
            || !self.prompt_first
            || self.encoder_layer_id != -1
        {
            return Err(config_error(
                "unsupported OpenJev/GLiClass architecture settings",
            ));
        }
        let enc = EncoderConfig::from_slice(&serde_json::to_vec(
            &self.encoder_config,
        )?)?;
        if enc.hidden_size != self.hidden_size
            || !(3..=25).contains(&self.max_num_classes)
        {
            return Err(config_error(
                "unsupported OpenJev projector width or class capacity",
            ));
        }
        Ok(enc)
    }
}

#[derive(Debug, Deserialize)]
struct Calibration {
    temperature: f32,
    #[serde(default)]
    per_k: HashMap<String, f32>,
}
impl Calibration {
    fn validate(&self) -> Result<()> {
        if std::iter::once(&self.temperature)
            .chain(self.per_k.values())
            .any(|t| !t.is_finite() || *t <= 0.0)
        {
            return Err(config_error(
                "OpenJev temperatures must be finite and positive",
            ));
        }
        Ok(())
    }
    fn temperature(&self, k: usize) -> f32 {
        self.per_k
            .get(&k.to_string())
            .copied()
            .unwrap_or(self.temperature)
    }
}

struct Projector {
    first: Linear,
    second: Linear,
}
impl Projector {
    fn load(vb: VarBuilder<'_>, hidden: usize) -> candle_core::Result<Self> {
        Ok(Self {
            first: linear(hidden, hidden, vb.pp("linear_1"))?,
            second: linear(hidden, hidden, vb.pp("linear_2"))?,
        })
    }
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        self.second.forward(&self.first.forward(x)?.gelu_erf()?)
    }
}

/// Tokens and class-marker positions for one OpenJev question.
#[derive(Debug, Clone, Serialize)]
pub struct OpenJevInput {
    pub prompt: String,
    pub ids: Vec<u32>,
    pub markers: Vec<usize>,
    pub candidates: Vec<String>,
}

/// Lossless native output, before conversion to conditional typed answers.
#[derive(Debug, Clone, Serialize)]
pub struct OpenJevPrediction {
    pub logits: Vec<f32>,
    pub probabilities: IndexMap<String, f32>,
    pub selected: String,
    pub abstained: bool,
    pub temperature: f32,
    pub input_tokens: usize,
}

/// Local OpenJev (Verdict) model. Separate instances can coexist with Laya/Jev.
pub struct OpenJev {
    encoder: ModernBert,
    text: Projector,
    classes: Projector,
    tokenizer: Tokenizer,
    calibration: Calibration,
    config: Config,
    device: Device,
    dtype: DType,
    pad: u32,
    batch_size: usize,
    max_len: usize,
}

pub struct OpenJevBuilder {
    repo: String,
    device: Device,
    dtype: Option<DType>,
    batch_size: usize,
    max_len: usize,
}
impl OpenJevBuilder {
    pub fn with_device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }
    /// Defaults to BF16 on CUDA with FlashAttention, F32 otherwise.
    /// BF16 is approximate and requires CUDA.
    pub fn with_dtype(mut self, dtype: DType) -> Self {
        self.dtype = Some(dtype);
        self
    }
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }
    pub fn with_max_len(mut self, max_len: usize) -> Self {
        self.max_len = max_len;
        self
    }
}
impl TryFrom<OpenJevBuilder> for OpenJev {
    type Error = SystemOneError;
    fn try_from(builder: OpenJevBuilder) -> Result<Self> {
        let dtype = builder.dtype.unwrap_or_else(|| {
            if builder.device.is_cuda() && cfg!(feature = "cuda") {
                DType::BF16
            } else {
                DType::F32
            }
        });
        if !matches!(dtype, DType::F32 | DType::BF16)
            || (dtype == DType::BF16
                && (!builder.device.is_cuda() || !cfg!(feature = "cuda")))
        {
            return Err(config_error("OpenJev supports f32, or bf16 on CUDA"));
        }
        if builder.batch_size == 0 || builder.max_len < 2 {
            return Err(config_error(
                "batch size must be positive and max length at least two",
            ));
        }
        let root = PathBuf::from(&builder.repo);
        let files: Vec<PathBuf> = if root.is_dir() {
            [
                "config.json",
                "calibrator.json",
                "tokenizer.json",
                "model.safetensors",
            ]
            .iter()
            .map(|f| root.join(f))
            .collect()
        } else {
            let revision = if builder.repo == DEFAULT_REPO_ID {
                DEFAULT_REVISION
            } else {
                "main"
            };
            let repo = Api::new()?.repo(Repo::with_revision(
                builder.repo,
                RepoType::Model,
                revision.into(),
            ));
            [
                "config.json",
                "calibrator.json",
                "tokenizer.json",
                "model.safetensors",
            ]
            .iter()
            .map(|f| repo.get(f))
            .collect::<std::result::Result<_, _>>()?
        };
        let config: Config =
            serde_json::from_slice(&std::fs::read(&files[0])?)?;
        let enc = config.encoder()?;
        if builder.max_len > enc.max_position_embeddings {
            return Err(config_error(
                "max length exceeds encoder position capacity",
            ));
        }
        let calibration: Calibration =
            serde_json::from_slice(&std::fs::read(&files[1])?)?;
        calibration.validate()?;
        let mut tokenizer = Tokenizer::from_file(&files[2])?;
        if tokenizer.token_to_id(LABEL) != Some(config.class_token_index)
            || tokenizer.token_to_id(SEPARATOR) != Some(config.text_token_index)
        {
            return Err(config_error(
                "OpenJev tokenizer markers do not match config",
            ));
        }
        tokenizer.with_padding(None);
        tokenizer.with_truncation(Some(TruncationParams {
            max_length: builder.max_len,
            ..Default::default()
        }))?;
        let vb = weights(&files[3], dtype, &builder.device)?;
        let encoder = ModernBert::load(
            vb.pp("model.encoder_model"),
            &enc.to_modernbert_config(),
        )?;
        Ok(Self {
            encoder,
            text: Projector::load(
                vb.pp("model.text_projector"),
                enc.hidden_size,
            )?,
            classes: Projector::load(
                vb.pp("model.classes_projector"),
                enc.hidden_size,
            )?,
            tokenizer,
            calibration,
            config,
            device: builder.device,
            dtype,
            pad: enc.pad_token_id,
            batch_size: builder.batch_size,
            max_len: builder.max_len,
        })
    }
}
fn weights(
    path: &Path,
    dtype: DType,
    device: &Device,
) -> Result<VarBuilder<'static>> {
    // SAFETY: checkpoint is mapped read-only and must not be modified while loaded.
    Ok(unsafe { VarBuilder::from_mmaped_safetensors(&[path], dtype, device)? })
}

impl OpenJev {
    pub fn from(repo: &str) -> OpenJevBuilder {
        OpenJevBuilder {
            repo: repo.into(),
            device: Device::Cpu,
            dtype: None,
            batch_size: 16,
            max_len: 512,
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

    pub fn build_input(
        &self,
        state: &str,
        id: &str,
        question: &Question,
    ) -> Result<OpenJevInput> {
        let (prompt, candidates) =
            format_input(state, id, question, self.config.max_num_classes)?;
        let encoded = self.tokenizer.encode(prompt.as_str(), true)?;
        let ids = encoded.get_ids().to_vec();
        let markers: Vec<_> = ids
            .iter()
            .enumerate()
            .filter_map(|(i, &t)| {
                (t == self.config.class_token_index).then_some(i)
            })
            .collect();
        if markers.len() != candidates.len()
            || !ids.contains(&self.config.text_token_index)
        {
            return Err(question_error(
                id,
                "candidate header was truncated; shorten the descriptions",
            ));
        }
        Ok(OpenJevInput {
            prompt,
            ids,
            markers,
            candidates,
        })
    }

    /// Evaluate already prepared inputs. Inputs are validated before inference.
    pub fn predict(
        &self,
        inputs: &[OpenJevInput],
    ) -> Result<Vec<OpenJevPrediction>> {
        for input in inputs {
            let actual: Vec<_> = input
                .ids
                .iter()
                .enumerate()
                .filter_map(|(i, &t)| {
                    (t == self.config.class_token_index).then_some(i)
                })
                .collect();
            if input.ids.is_empty()
                || input.ids.len() > self.max_len
                || actual != input.markers
                || input.markers.len() != input.candidates.len()
                || !(3..=self.config.max_num_classes).contains(&actual.len())
                || input.candidates.last().map(String::as_str)
                    != Some(ABSTENTION_ID)
                || input.candidates.iter().collect::<HashSet<_>>().len()
                    != input.candidates.len()
                || !input.ids.contains(&self.config.text_token_index)
                || input
                    .candidates
                    .iter()
                    .any(|id| id.is_empty() || id == "insufficient_evidence")
                || input.ids.iter().any(|&x| {
                    x as usize
                        >= self.config.encoder_config["vocab_size"]
                            .as_u64()
                            .unwrap_or(0) as usize
                })
            {
                return Err(config_error("invalid prepared OpenJev input"));
            }
        }
        let mut results = Vec::with_capacity(inputs.len());
        for chunk in inputs.chunks(self.batch_size) {
            let len = chunk.iter().map(|x| x.ids.len()).max().unwrap_or(0);
            let mut ids = vec![self.pad; chunk.len() * len];
            let mut mask = vec![0u32; ids.len()];
            for (i, x) in chunk.iter().enumerate() {
                ids[i * len..i * len + x.ids.len()].copy_from_slice(&x.ids);
                mask[i * len..i * len + x.ids.len()].fill(1);
            }
            let ids = Tensor::from_vec(ids, (chunk.len(), len), &self.device)?;
            #[cfg(feature = "cuda")]
            let packed = self.device.is_cuda() && self.dtype == DType::BF16;
            #[cfg(not(feature = "cuda"))]
            let packed = false;
            let lens: Vec<_> = chunk.iter().map(|x| x.ids.len()).collect();
            let hidden = if packed {
                #[cfg(feature = "cuda")]
                {
                    self.encoder.forward_varlen_packed(&ids, &lens)?
                }
                #[cfg(not(feature = "cuda"))]
                {
                    unreachable!()
                }
            } else {
                self.encoder
                    .forward(
                        &ids,
                        &Tensor::from_vec(
                            mask,
                            (chunk.len(), len),
                            &self.device,
                        )?,
                    )?
                    .reshape((chunk.len() * len, self.config.hidden_size))?
            };
            let mut offset = 0;
            for (i, input) in chunk.iter().enumerate() {
                let start = if packed { offset } else { i * len };
                let marker_ids: Vec<u32> =
                    input.markers.iter().map(|&m| (start + m) as u32).collect();
                let labels = hidden.index_select(
                    &Tensor::from_vec(
                        marker_ids,
                        input.markers.len(),
                        &self.device,
                    )?,
                    0,
                )?;
                let text =
                    self.text.forward(&hidden.i(start)?.unsqueeze(0)?)?;
                let classes = self.classes.forward(&labels)?;
                let logits = classes
                    .matmul(&text.t()?)?
                    .flatten_all()?
                    .to_dtype(DType::F32)?
                    .to_vec1::<f32>()?;
                if logits.iter().any(|x| !x.is_finite()) {
                    return Err(config_error(
                        "OpenJev returned non-finite logits",
                    ));
                }
                results.push(prediction(input, logits, &self.calibration));
                offset += lens[i];
            }
        }
        Ok(results)
    }

    pub fn system_one(
        &self,
        request: &SystemOneRequest,
    ) -> Result<SystemOneResponse> {
        self.system_one_batch(std::slice::from_ref(request))?
            .pop()
            .ok_or_else(|| config_error("missing response"))
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
                inputs.push(self.build_input(&state, id, question)?);
                locations.push((r, id, question));
            }
        }
        let predictions = self.predict(&inputs)?;
        let mut responses: Vec<_> = requests
            .iter()
            .map(|_| SystemOneResponse {
                model: MODEL_NAME.into(),
                answers: IndexMap::new(),
                usage: Usage::default(),
            })
            .collect();
        for ((r, id, q), p) in locations.into_iter().zip(predictions) {
            responses[r].usage.input_tokens += p.input_tokens;
            responses[r].answers.insert(id.clone(), answer(q, &p)?);
        }
        Ok(responses)
    }
}

fn format_input(
    state: &str,
    id: &str,
    q: &Question,
    capacity: usize,
) -> Result<(String, Vec<String>)> {
    let (body, mut labels, mut ids) = match q {
        Question::Choice(q) => {
            let ids: Vec<String> =
                q.criteria.labels().into_iter().map(str::to_owned).collect();
            let descriptions: Vec<String> = match &q.criteria {
                ChoiceCriteria::Labels(labels) => labels.clone(),
                ChoiceCriteria::Labelled(options) => options
                    .iter()
                    .map(|(id, d)| {
                        d.as_ref()
                            .filter(|x| !x.render().trim().is_empty())
                            .map(Description::render)
                            .unwrap_or_else(|| id.clone())
                    })
                    .collect(),
            };
            (
                format!(
                    "Question: {}\n\nContext:\n{state}",
                    q.instructions.render()
                ),
                descriptions
                    .iter()
                    .map(|d| format!("It is {d}"))
                    .collect::<Vec<_>>(),
                ids,
            )
        }
        Question::Score(q) => (
            format!(
                "Question: {}\n\nContext:\n{state}",
                q.instructions.render()
            ),
            q.criteria
                .iter()
                .enumerate()
                .map(|(i, d)| format!("{} (Value: {i}.0)", d.render()))
                .collect(),
            (0..q.criteria.len()).map(|i| i.to_string()).collect(),
        ),
        Question::Noul(q) => {
            if q.criteria
                .as_ref()
                .is_some_and(|c| c.yes.is_some() || c.no.is_some())
            {
                return Err(question_error(
                    id,
                    "OpenJev noul uses proposition-based labels; custom true/false criteria are unsupported",
                ));
            }
            let proposition = q.instructions.render();
            (
                format!(
                    "Context:\n{state}\n\nEvaluate proposition: {proposition}"
                ),
                vec![
                    format!("true: {proposition}"),
                    format!("false: not {proposition}"),
                ],
                vec!["true".into(), "false".into()],
            )
        }
    };
    if ids.len() < 2 || ids.len() >= capacity {
        return Err(question_error(
            id,
            format!(
                "OpenJev requires 2..={} substantive candidates",
                capacity - 1
            ),
        ));
    }
    if ids.iter().any(|id| {
        id.is_empty() || id == ABSTENTION_ID || id == "insufficient_evidence"
    }) || ids.iter().collect::<HashSet<_>>().len() != ids.len()
    {
        return Err(question_error(
            id,
            "candidate IDs must be distinct and cannot use reserved abstention IDs",
        ));
    }
    if std::iter::once(&body)
        .chain(labels.iter())
        .any(|s| s.contains(LABEL) || s.contains(SEPARATOR))
    {
        return Err(question_error(
            id,
            "input contains reserved OpenJev marker text",
        ));
    }
    labels.push("insufficient evidence".into());
    ids.push(ABSTENTION_ID.into());
    Ok((
        format!(
            "{}{SEPARATOR}{body}",
            labels
                .iter()
                .map(|x| format!("{LABEL}{x}"))
                .collect::<String>()
        ),
        ids,
    ))
}
fn prediction(
    input: &OpenJevInput,
    logits: Vec<f32>,
    calibration: &Calibration,
) -> OpenJevPrediction {
    let temperature = calibration.temperature(logits.len());
    let probs =
        softmax(&logits.iter().map(|x| x / temperature).collect::<Vec<_>>());
    let mut best = 0;
    for i in 1..logits.len() {
        if logits[i] > logits[best] {
            best = i;
        }
    }
    OpenJevPrediction {
        selected: input.candidates[best].clone(),
        abstained: input.candidates[best] == ABSTENTION_ID,
        probabilities: input.candidates.iter().cloned().zip(probs).collect(),
        logits,
        temperature,
        input_tokens: input.ids.len(),
    }
}
fn answer(q: &Question, p: &OpenJevPrediction) -> Result<Answer> {
    if p.abstained {
        return Ok(Answer::Abstain(crate::AbstentionAnswer {
            question_type: q.kind(),
            probabilities: p.probabilities.clone(),
            reason: "insufficient_evidence".into(),
        }));
    }
    let mass: f32 = p
        .probabilities
        .iter()
        .filter(|(id, _)| id.as_str() != ABSTENTION_ID)
        .map(|(_, p)| p)
        .sum();
    if !mass.is_finite() || mass <= 0.0 {
        return Err(config_error("invalid substantive probability mass"));
    }
    let probs: IndexMap<String, f32> = p
        .probabilities
        .iter()
        .filter(|(id, _)| id.as_str() != ABSTENTION_ID)
        .map(|(id, p)| (id.clone(), p / mass))
        .collect();
    let confidence = confidence_from_probs(
        &probs.values().copied().collect::<Vec<_>>(),
        probs.len(),
    );
    Ok(match q {
        Question::Choice(_) => Answer::Choice(ChoiceAnswer {
            choice: p.selected.clone(),
            probabilities: probs,
            confidence,
            action: None,
        }),
        Question::Score(q) => Answer::Score(ScoreAnswer {
            score: probs.values().enumerate().map(|(i, p)| i as f32 * p).sum(),
            legend: q
                .criteria
                .iter()
                .enumerate()
                .map(|(i, d)| (i.to_string(), d.clone()))
                .collect(),
            probabilities: probs,
            confidence,
            action: None,
        }),
        Question::Noul(_) => Answer::Noul(NoulAnswer {
            noul: probs["true"],
            action: None,
        }),
    })
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn question(value: serde_json::Value) -> Question {
        serde_json::from_value(value).unwrap()
    }
    fn choice() -> Question {
        question(
            json!({"type":"choice","instructions":"Route?","criteria":{"a":"first","b":"second"}}),
        )
    }
    fn native(q: &Question, logits: Vec<f32>) -> OpenJevPrediction {
        let (prompt, candidates) = format_input("state", "q", q, 25).unwrap();
        prediction(
            &OpenJevInput {
                prompt,
                candidates,
                ids: vec![1],
                markers: vec![],
            },
            logits,
            &Calibration {
                temperature: 2.,
                per_k: HashMap::from([("3".into(), 1.)]),
            },
        )
    }

    #[test]
    fn upstream_prompts_and_candidate_order() {
        let (prompt, ids) = format_input("state", "q", &choice(), 25).unwrap();
        assert_eq!(
            prompt,
            "<<LABEL>>It is first<<LABEL>>It is second<<LABEL>>insufficient evidence<<SEP>>Question: Route?\n\nContext:\nstate"
        );
        assert_eq!(ids, ["a", "b", ABSTENTION_ID]);
        let q = question(
            json!({"type":"score","instructions":"Rate","criteria":["bad","good"]}),
        );
        let (prompt, ids) = format_input("state", "q", &q, 25).unwrap();
        assert_eq!(
            prompt,
            "<<LABEL>>bad (Value: 0.0)<<LABEL>>good (Value: 1.0)<<LABEL>>insufficient evidence<<SEP>>Question: Rate\n\nContext:\nstate"
        );
        assert_eq!(ids, ["0", "1", ABSTENTION_ID]);
        let q = question(json!({"type":"noul","instructions":"Paid"}));
        let (prompt, ids) = format_input("state", "q", &q, 25).unwrap();
        assert_eq!(
            prompt,
            "<<LABEL>>true: Paid<<LABEL>>false: not Paid<<LABEL>>insufficient evidence<<SEP>>Context:\nstate\n\nEvaluate proposition: Paid"
        );
        assert_eq!(ids, ["true", "false", ABSTENTION_ID]);
    }

    #[test]
    fn calibration_counts_abstention_and_falls_back_without_interpolation() {
        let calibration = Calibration {
            temperature: 2.8039,
            per_k: HashMap::from([("3".into(), 5.0069), ("5".into(), 3.056)]),
        };
        assert_eq!(calibration.temperature(3), 5.0069);
        assert_eq!(calibration.temperature(4), 2.8039);
        assert!(calibration.validate().is_ok());
        assert!(
            Calibration {
                temperature: 0.,
                per_k: HashMap::new()
            }
            .validate()
            .is_err()
        );
        assert!(
            Calibration {
                temperature: 1.,
                per_k: HashMap::from([("3".into(), f32::NAN)])
            }
            .validate()
            .is_err()
        );
        let p = native(&choice(), vec![1., 1., 1.]);
        assert_eq!(p.temperature, 1.);
        assert_eq!(p.selected, "a");
        assert!(!p.abstained);
        assert!((p.probabilities.values().sum::<f32>() - 1.).abs() < 1e-6);
    }

    #[test]
    fn abstention_is_explicit_and_round_trips_for_every_primitive() {
        for q in [
            choice(),
            question(json!({"type":"score","criteria":["bad","good"]})),
            question(json!({"type":"noul","instructions":"Paid"})),
        ] {
            let p = native(&q, vec![-2., -1., 4.]);
            let answer = answer(&q, &p).unwrap();
            assert!(
                answer.choice().is_none()
                    && answer.noul().is_none()
                    && answer.score().is_none()
            );
            assert!(answer.confidence().is_none() && answer.action().is_none());
            assert_eq!(answer.abstention().unwrap().question_type, q.kind());
            assert_eq!(
                answer.abstention().unwrap().probabilities,
                p.probabilities
            );
            let wire = serde_json::to_value(&answer).unwrap();
            assert_eq!(wire["type"], "abstain");
            assert_eq!(serde_json::from_value::<Answer>(wire).unwrap(), answer);
        }
    }

    #[test]
    fn normal_answers_condition_on_sufficient_evidence() {
        let q = choice();
        let p = native(&q, vec![0., 2., 1.]);
        let Answer::Choice(a) = answer(&q, &p).unwrap() else {
            panic!("choice")
        };
        assert_eq!(a.choice, "b");
        assert_eq!(a.probabilities.len(), 2);
        assert!((a.probabilities.values().sum::<f32>() - 1.).abs() < 1e-6);
        assert!(a.action.is_none());
        let q = question(json!({"type":"score","criteria":["bad","good"]}));
        let p = native(&q, vec![0., 2., 1.]);
        assert!(
            (answer(&q, &p).unwrap().score().unwrap() - 0.880797).abs() < 1e-6
        );
        let q = question(json!({"type":"noul","instructions":"Paid"}));
        let p = native(&q, vec![2., 0., 1.]);
        assert!(
            (answer(&q, &p).unwrap().noul().unwrap() - 0.880797).abs() < 1e-6
        );
    }

    #[test]
    fn reject_ambiguous_or_unsupported_inputs() {
        assert!(
            OpenJev::try_from(OpenJev::from("unused").with_dtype(DType::BF16))
                .is_err()
        );
        assert!(
            OpenJev::try_from(OpenJev::from("unused").with_batch_size(0))
                .is_err()
        );
        assert!(format_input("<<LABEL>>spoof", "q", &choice(), 25).is_err());
        for criteria in [
            json!(["one"]),
            json!(["same", "same"]),
            json!(["a", ABSTENTION_ID]),
            json!((0..25).map(|i| i.to_string()).collect::<Vec<_>>()),
        ] {
            let q = question(json!({"type":"choice","criteria":criteria}));
            assert!(format_input("state", "q", &q, 25).is_err());
        }
        let q = question(json!({"type":"noul","criteria":{"true":"custom"}}));
        assert!(format_input("state", "q", &q, 25).is_err());
        let q = question(
            json!({"type":"choice","criteria":(0..24).map(|i|i.to_string()).collect::<Vec<_>>()}),
        );
        assert_eq!(format_input("state", "q", &q, 25).unwrap().1.len(), 25);
    }
}
