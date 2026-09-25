//! GLiNER2.5-Decide: zero-shot label classification on DeBERTa-v3-large.
//!
//! Each request is one upstream `classify_text` call: its questions become
//! classification tasks scored together in a single forward pass. Every
//! task reads the `[L]` marker in front of each label through the
//! checkpoint's classifier MLP and takes a softmax over its labels.
use std::{collections::HashSet, path::PathBuf, sync::LazyLock};

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{Linear, Module, VarBuilder, linear};
use hf_hub::{Repo, RepoType, api::sync::Api};
use indexmap::IndexMap;
use regex::Regex;
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

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
    deberta::{Config as DebertaConfig, DebertaV2Model},
    head::{confidence_from_probs, softmax},
};

pub const DEFAULT_REPO_ID: &str = "fastino/GLiNER2.5-Decide";
pub const DEFAULT_REVISION: &str = "7ee5da4c2415e32259bcdc0b1a7367c32ce8d6f6";
pub const MODEL_NAME: &str = "GLiNER2.5-Decide";

const SEP_STRUCT: &str = "[SEP_STRUCT]";
const SEP_TEXT: &str = "[SEP_TEXT]";
const PROMPT: &str = "[P]";
const LABEL: &str = "[L]";
const DESCRIPTION: &str = "[DESCRIPTION]";
/// Every marker the upstream processor adds; none may appear in caller text.
const RESERVED: [&str; 10] = [
    SEP_STRUCT,
    SEP_TEXT,
    PROMPT,
    "[C]",
    "[E]",
    "[R]",
    LABEL,
    "[EXAMPLE]",
    "[OUTPUT]",
    DESCRIPTION,
];
const YES: &str = "yes";
const NO: &str = "no";

/// Upstream `WhitespaceTokenSplitter`, case-insensitive like its Python form.
static WORDS: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)(?:https?://\S+|www\.\S+)|[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}|@[a-z0-9_]+|\w+(?:[-_]\w+)*|\S",
    )
    .expect("valid word pattern")
});

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
    architecture: String,
    token_pooling: String,
    #[serde(default)]
    use_moe: bool,
}

/// `Linear -> ReLU -> Linear` scoring one `[L]` embedding.
struct Classifier {
    first: Linear,
    second: Linear,
}
impl Classifier {
    fn load(vb: VarBuilder<'_>, hidden: usize) -> candle_core::Result<Self> {
        Ok(Self {
            first: linear(hidden, hidden * 2, vb.pp("0"))?,
            second: linear(hidden * 2, 1, vb.pp("2"))?,
        })
    }
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        self.second.forward(&self.first.forward(x)?.relu()?)
    }
}

/// One classification task as the upstream schema describes it.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct GlinerTask {
    /// Task name: the question ID.
    pub name: String,
    /// Question instructions, appended to the name as `name: prompt`.
    pub prompt: Option<String>,
    /// Label IDs, in the order they are scored.
    pub labels: Vec<String>,
    /// Descriptions for the labels that have one.
    pub descriptions: IndexMap<String, String>,
}
impl GlinerTask {
    /// Upstream `_transform_schema` for inference (`example_mode="both"`).
    fn schema_tokens(&self) -> Vec<String> {
        let mut prompt = self.name.clone();
        if let Some(p) = &self.prompt {
            prompt = format!("{prompt}: {p}");
        }
        for (label, description) in &self.descriptions {
            prompt.push_str(&format!(" {DESCRIPTION} {label}: {description}"));
        }
        let mut tokens = vec!["(".into(), PROMPT.into(), prompt, "(".into()];
        for label in &self.labels {
            tokens.push(LABEL.into());
            tokens.push(label.clone());
        }
        tokens.extend([")".into(), ")".into()]);
        tokens
    }
}

/// Token IDs and `[L]` positions for one request.
#[derive(Debug, Clone, Serialize)]
pub struct GlinerDecideInput {
    pub ids: Vec<u32>,
    pub tasks: Vec<GlinerTask>,
    /// `[L]` positions per task, in label order.
    pub markers: Vec<Vec<usize>>,
    /// State subword tokens dropped from the right to fit the context.
    pub dropped_tokens: usize,
}

/// Native per-task output, before conversion to typed answers.
#[derive(Debug, Clone, Serialize)]
pub struct GlinerTaskPrediction {
    pub logits: Vec<f32>,
    pub probabilities: IndexMap<String, f32>,
    pub selected: String,
}

/// Local GLiNER2.5-Decide model. Separate instances can coexist with others.
pub struct GlinerDecide {
    encoder: DebertaV2Model,
    classifier: Classifier,
    tokenizer: Tokenizer,
    label: u32,
    vocab_size: usize,
    device: Device,
    dtype: DType,
    batch_size: usize,
    max_len: usize,
}

pub struct GlinerDecideBuilder {
    repo: String,
    device: Device,
    dtype: Option<DType>,
    batch_size: usize,
    max_len: usize,
}
impl GlinerDecideBuilder {
    pub fn with_device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }
    /// Defaults to BF16 on CUDA, F32 otherwise. BF16 is approximate and
    /// requires CUDA; attention scores and softmax stay in F32.
    pub fn with_dtype(mut self, dtype: DType) -> Self {
        self.dtype = Some(dtype);
        self
    }
    /// Requests per forward pass.
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }
    /// Tokens per request, schema included. State is truncated to fit.
    pub fn with_max_len(mut self, max_len: usize) -> Self {
        self.max_len = max_len;
        self
    }
}
impl TryFrom<GlinerDecideBuilder> for GlinerDecide {
    type Error = SystemOneError;
    fn try_from(builder: GlinerDecideBuilder) -> Result<Self> {
        let dtype = builder.dtype.unwrap_or_else(|| {
            if builder.device.is_cuda() {
                DType::BF16
            } else {
                DType::F32
            }
        });
        if !matches!(dtype, DType::F32 | DType::BF16)
            || (dtype == DType::BF16 && !builder.device.is_cuda())
        {
            return Err(config_error(
                "GLiNER2.5-Decide supports f32, or bf16 on CUDA",
            ));
        }
        if builder.batch_size == 0 || builder.max_len < 2 {
            return Err(config_error(
                "batch size must be positive and max length at least two",
            ));
        }
        const FILES: [&str; 4] = [
            "config.json",
            "encoder_config/config.json",
            "tokenizer.json",
            "model.safetensors",
        ];
        let root = PathBuf::from(&builder.repo);
        let files: Vec<PathBuf> = if root.is_dir() {
            FILES.iter().map(|f| root.join(f)).collect()
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
            FILES
                .iter()
                .map(|f| repo.get(f))
                .collect::<std::result::Result<_, _>>()?
        };
        let config: Config =
            serde_json::from_slice(&std::fs::read(&files[0])?)?;
        if config.architecture != "span"
            || config.token_pooling != "first"
            || config.use_moe
        {
            return Err(config_error(
                "unsupported GLiNER2 architecture settings",
            ));
        }
        let encoder_config: DebertaConfig =
            serde_json::from_slice(&std::fs::read(&files[1])?)?;
        if encoder_config.conv_kernel_size.unwrap_or(0) > 0
            || encoder_config
                .embedding_size
                .is_some_and(|e| e != encoder_config.hidden_size)
        {
            return Err(config_error("unsupported DeBERTa encoder settings"));
        }
        let mut tokenizer = Tokenizer::from_file(&files[2])?;
        tokenizer.with_padding(None);
        tokenizer.with_truncation(None)?;
        if let Some(t) =
            RESERVED.iter().find(|t| tokenizer.token_to_id(t).is_none())
        {
            return Err(config_error(format!("tokenizer lacks {t}")));
        }
        let label = tokenizer.token_to_id(LABEL).expect("checked above");
        // SAFETY: checkpoint is mapped read-only and must not be modified while loaded.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[&files[3]],
                dtype,
                &builder.device,
            )?
        };
        Ok(Self {
            encoder: DebertaV2Model::load(vb.pp("encoder"), &encoder_config)?,
            classifier: Classifier::load(
                vb.pp("classifier"),
                encoder_config.hidden_size,
            )?,
            tokenizer,
            label,
            vocab_size: encoder_config.vocab_size,
            device: builder.device,
            dtype,
            batch_size: builder.batch_size,
            max_len: builder.max_len,
        })
    }
}
impl GlinerDecide {
    pub fn from(repo: &str) -> GlinerDecideBuilder {
        GlinerDecideBuilder {
            repo: repo.into(),
            device: Device::Cpu,
            dtype: None,
            batch_size: 8,
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
    /// Configured maximum input size in tokens, schema included.
    pub fn context_tokens(&self) -> usize {
        self.max_len
    }
    /// Whether every request fits without truncating its state.
    pub fn request_fits(&self, request: &SystemOneRequest) -> Result<bool> {
        Ok(self.build_input(request)?.dropped_tokens == 0)
    }

    fn subwords(&self, piece: &str) -> Result<Vec<u32>> {
        Ok(self.tokenizer.encode(piece, false)?.get_ids().to_vec())
    }

    /// Upstream `_format_input_with_mapping`, truncating state on the right.
    pub fn build_input(
        &self,
        request: &SystemOneRequest,
    ) -> Result<GlinerDecideInput> {
        let tasks = request
            .questions
            .iter()
            .map(|(id, q)| task(id, q))
            .collect::<Result<Vec<_>>>()?;
        if tasks.is_empty() {
            return Err(config_error("request has no questions"));
        }
        let mut ids = Vec::new();
        let mut markers = Vec::with_capacity(tasks.len());
        for (i, t) in tasks.iter().enumerate() {
            if i > 0 {
                ids.extend(self.subwords(SEP_STRUCT)?);
            }
            let mut positions = Vec::with_capacity(t.labels.len());
            for (j, piece) in t.schema_tokens().iter().enumerate() {
                // Upstream routes only these structural `[L]` slots.
                if j >= 4 && j % 2 == 0 && j < 4 + 2 * t.labels.len() {
                    positions.push(ids.len());
                }
                ids.extend(self.subwords(piece)?);
            }
            markers.push(positions);
        }
        ids.extend(self.subwords(SEP_TEXT)?);
        if ids.len() > self.max_len {
            return Err(config_error(format!(
                "questions need {} tokens, over the {}-token context",
                ids.len(),
                self.max_len
            )));
        }
        let text = normalize(&request.state.render());
        if reserved(&text) {
            return Err(config_error(
                "state contains a reserved GLiNER2 marker",
            ));
        }
        let mut dropped_tokens = 0;
        for word in WORDS.find_iter(&text) {
            let sub = self.subwords(&word.as_str().to_lowercase())?;
            if dropped_tokens > 0 || ids.len() + sub.len() > self.max_len {
                dropped_tokens += sub.len();
            } else {
                ids.extend(sub);
            }
        }
        Ok(GlinerDecideInput {
            ids,
            tasks,
            markers,
            dropped_tokens,
        })
    }

    /// Evaluate prepared inputs, one forward pass per `batch_size` requests.
    pub fn predict(
        &self,
        inputs: &[GlinerDecideInput],
    ) -> Result<Vec<Vec<GlinerTaskPrediction>>> {
        for input in inputs {
            let valid = !input.ids.is_empty()
                && input.ids.len() <= self.max_len
                && input.markers.len() == input.tasks.len()
                && input.ids.iter().all(|&x| (x as usize) < self.vocab_size)
                && input.tasks.iter().zip(&input.markers).all(|(t, m)| {
                    m.len() == t.labels.len()
                        && m.iter()
                            .all(|&p| input.ids.get(p) == Some(&self.label))
                })
                && input.markers.iter().map(Vec::len).sum::<usize>()
                    == input.ids.iter().filter(|&&x| x == self.label).count();
            if !valid {
                return Err(config_error("invalid prepared GLiNER2 input"));
            }
        }
        // Batch similar lengths together: every row pads to its batch's longest.
        let mut order: Vec<usize> = (0..inputs.len()).collect();
        order.sort_by_key(|&i| inputs[i].ids.len());
        let mut results = vec![Vec::new(); inputs.len()];
        for indices in order.chunks(self.batch_size) {
            let chunk: Vec<&GlinerDecideInput> =
                indices.iter().map(|&i| &inputs[i]).collect();
            let len = chunk.iter().map(|x| x.ids.len()).max().unwrap_or(0);
            let mut ids = vec![0u32; chunk.len() * len];
            let mut mask = vec![0f32; ids.len()];
            for (i, x) in chunk.iter().enumerate() {
                ids[i * len..i * len + x.ids.len()].copy_from_slice(&x.ids);
                mask[i * len..i * len + x.ids.len()].fill(1.);
            }
            let shape = (chunk.len(), len);
            let hidden = self.encoder.forward(
                &Tensor::from_vec(ids, shape, &self.device)?,
                None,
                Some(Tensor::from_vec(mask, shape, &self.device)?),
            )?;
            for (i, input) in chunk.iter().enumerate() {
                let hidden = hidden.i(i)?;
                let mut tasks = Vec::with_capacity(input.tasks.len());
                for (task, positions) in input.tasks.iter().zip(&input.markers)
                {
                    let index: Vec<u32> =
                        positions.iter().map(|&p| p as u32).collect();
                    let labels = hidden.index_select(
                        &Tensor::from_vec(
                            index,
                            positions.len(),
                            &self.device,
                        )?,
                        0,
                    )?;
                    let logits = self
                        .classifier
                        .forward(&labels)?
                        .flatten_all()?
                        .to_dtype(DType::F32)?
                        .to_vec1::<f32>()?;
                    if logits.iter().any(|x| !x.is_finite()) {
                        return Err(config_error(
                            "GLiNER2 returned non-finite logits",
                        ));
                    }
                    tasks.push(prediction(task, logits));
                }
                results[indices[i]] = tasks;
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
        let inputs = requests
            .iter()
            .map(|r| self.build_input(r))
            .collect::<Result<Vec<_>>>()?;
        let predictions = self.predict(&inputs)?;
        requests
            .iter()
            .zip(inputs.iter().zip(predictions))
            .map(|(request, (input, tasks))| {
                let mut usage = Usage {
                    input_tokens: input.ids.len(),
                    ..Usage::default()
                };
                let mut answers = IndexMap::new();
                for ((id, q), p) in request.questions.iter().zip(&tasks) {
                    if input.dropped_tokens > 0 {
                        usage
                            .dropped_state_tokens
                            .insert(id.clone(), input.dropped_tokens);
                    }
                    answers.insert(id.clone(), answer(q, p));
                }
                Ok(SystemOneResponse {
                    model: MODEL_NAME.into(),
                    answers,
                    usage,
                })
            })
            .collect()
    }
}

/// Upstream `_normalize_text`: collation expects closing punctuation.
fn normalize(text: &str) -> String {
    if text.is_empty() {
        ".".into()
    } else if text.ends_with(['.', '!', '?']) {
        text.into()
    } else {
        format!("{text}.")
    }
}

fn reserved(s: &str) -> bool {
    RESERVED.iter().any(|m| s.contains(m))
}

fn text(d: &Description) -> Option<String> {
    Some(d.render()).filter(|s| !s.is_empty() && !d.is_blank())
}

/// Map a typed question to an upstream classification task.
fn task(id: &str, q: &Question) -> Result<GlinerTask> {
    let mut descriptions = IndexMap::new();
    let labels: Vec<String> = match q {
        Question::Choice(q) => {
            if let ChoiceCriteria::Labelled(options) = &q.criteria {
                for (label, d) in options {
                    if let Some(d) = d.as_ref().and_then(text) {
                        descriptions.insert(label.clone(), d);
                    }
                }
            }
            q.criteria.labels().into_iter().map(str::to_owned).collect()
        }
        Question::Score(q) => {
            for (i, d) in q.criteria.iter().enumerate() {
                if let Some(d) = text(d) {
                    descriptions.insert(i.to_string(), d);
                }
            }
            (0..q.criteria.len()).map(|i| i.to_string()).collect()
        }
        Question::Noul(q) => {
            let criteria = q.criteria.clone().unwrap_or_default();
            for (label, d) in [(YES, criteria.yes), (NO, criteria.no)] {
                if let Some(d) = d.as_ref().and_then(text) {
                    descriptions.insert(label.into(), d);
                }
            }
            vec![YES.into(), NO.into()]
        }
    };
    if labels.len() < 2 {
        return Err(question_error(id, "needs at least two labels"));
    }
    if labels.iter().any(String::is_empty)
        || labels.iter().collect::<HashSet<_>>().len() != labels.len()
    {
        return Err(question_error(
            id,
            "labels must be distinct and non-empty",
        ));
    }
    let prompt = text(q.instructions());
    if std::iter::once(id)
        .chain(prompt.as_deref())
        .chain(labels.iter().map(String::as_str))
        .chain(descriptions.values().map(String::as_str))
        .any(reserved)
    {
        return Err(question_error(id, "contains a reserved GLiNER2 marker"));
    }
    Ok(GlinerTask {
        name: id.into(),
        prompt,
        labels,
        descriptions,
    })
}

fn prediction(task: &GlinerTask, logits: Vec<f32>) -> GlinerTaskPrediction {
    let probs = softmax(&logits);
    let mut best = 0;
    for i in 1..probs.len() {
        if probs[i] > probs[best] {
            best = i;
        }
    }
    GlinerTaskPrediction {
        selected: task.labels[best].clone(),
        probabilities: task.labels.iter().cloned().zip(probs).collect(),
        logits,
    }
}

fn answer(q: &Question, p: &GlinerTaskPrediction) -> Answer {
    let probs: Vec<f32> = p.probabilities.values().copied().collect();
    let confidence = confidence_from_probs(&probs, probs.len());
    match q {
        Question::Choice(_) => Answer::Choice(ChoiceAnswer {
            choice: p.selected.clone(),
            probabilities: p.probabilities.clone(),
            confidence,
            action: None,
        }),
        Question::Score(q) => Answer::Score(ScoreAnswer {
            score: probs.iter().enumerate().map(|(i, p)| i as f32 * p).sum(),
            legend: q
                .criteria
                .iter()
                .enumerate()
                .map(|(i, d)| (i.to_string(), d.clone()))
                .collect(),
            probabilities: p.probabilities.clone(),
            confidence,
            action: None,
        }),
        Question::Noul(_) => Answer::Noul(NoulAnswer {
            noul: p.probabilities[YES],
            action: None,
        }),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn question(value: serde_json::Value) -> Question {
        serde_json::from_value(value).unwrap()
    }
    fn words(text: &str) -> Vec<String> {
        WORDS
            .find_iter(text)
            .map(|m| m.as_str().to_lowercase())
            .collect()
    }

    #[test]
    fn words_follow_the_upstream_whitespace_splitter() {
        assert_eq!(
            words(
                "Mail Ann@Example.com, see https://x.io/a?b=1 or @ops_team re: re-open_now!"
            ),
            [
                "mail",
                "ann@example.com",
                ",",
                "see",
                "https://x.io/a?b=1",
                "or",
                "@ops_team",
                "re",
                ":",
                "re-open_now",
                "!"
            ]
        );
        assert_eq!(normalize(""), ".");
        assert_eq!(normalize("done?"), "done?");
        assert_eq!(normalize("done"), "done.");
    }

    #[test]
    fn questions_map_to_upstream_schema_tokens() {
        let q = question(json!({
            "type": "choice",
            "instructions": "Which team?",
            "criteria": {"legal": "contracts", "support": null, "billing": ""}
        }));
        let t = task("route", &q).unwrap();
        assert_eq!(
            t.schema_tokens(),
            [
                "(",
                "[P]",
                "route: Which team? [DESCRIPTION] legal: contracts",
                "(",
                "[L]",
                "legal",
                "[L]",
                "support",
                "[L]",
                "billing",
                ")",
                ")"
            ]
        );
        let q = question(json!({"type": "score", "criteria": ["bad", "good"]}));
        let t = task("rating", &q).unwrap();
        assert_eq!(t.prompt, None);
        assert_eq!(t.labels, ["0", "1"]);
        assert_eq!(
            t.schema_tokens()[2],
            "rating [DESCRIPTION] 0: bad [DESCRIPTION] 1: good"
        );
        let q = question(json!({
            "type": "noul",
            "instructions": "Paid?",
            "criteria": {"false": "unpaid"}
        }));
        let t = task("paid", &q).unwrap();
        assert_eq!(t.labels, [YES, NO]);
        assert_eq!(
            t.schema_tokens()[2],
            "paid: Paid? [DESCRIPTION] no: unpaid"
        );
    }

    #[test]
    fn reject_ambiguous_or_reserved_inputs() {
        for criteria in [json!(["one"]), json!(["a", "a"]), json!(["", "b"])] {
            let q = question(json!({"type": "choice", "criteria": criteria}));
            assert!(task("q", &q).is_err());
        }
        let q = question(json!({"type": "choice", "criteria": ["a", "b"]}));
        assert!(task("x [L] y", &q).is_err());
        let q = question(
            json!({"type": "choice", "instructions": "[SEP_TEXT]", "criteria": ["a", "b"]}),
        );
        assert!(task("q", &q).is_err());
        assert!(
            GlinerDecide::try_from(
                GlinerDecide::from("unused").with_dtype(DType::BF16)
            )
            .is_err()
        );
        assert!(
            GlinerDecide::try_from(
                GlinerDecide::from("unused").with_batch_size(0)
            )
            .is_err()
        );
    }

    #[test]
    fn answers_keep_the_softmax_over_labels() {
        let q = question(json!({"type": "score", "criteria": ["bad", "good"]}));
        let p = prediction(&task("q", &q).unwrap(), vec![0., 2.]);
        assert_eq!(p.selected, "1");
        assert!((answer(&q, &p).score().unwrap() - 0.880797).abs() < 1e-6);
        let q = question(json!({"type": "noul", "instructions": "Paid"}));
        let p = prediction(&task("q", &q).unwrap(), vec![2., 0.]);
        assert!((answer(&q, &p).noul().unwrap() - 0.880797).abs() < 1e-6);
        let q = question(json!({"type": "choice", "criteria": ["a", "b"]}));
        let p = prediction(&task("q", &q).unwrap(), vec![1., 1.]);
        let a = answer(&q, &p);
        assert_eq!(a.choice(), Some("a"));
        assert!(a.action().is_none());
        assert!(a.confidence().unwrap() < 1e-6);
    }
}
