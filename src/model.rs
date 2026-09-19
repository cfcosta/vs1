//! The decision model: encoder, head, and the batched
//! `system_one(state, questions)` evaluation.

use std::path::Path;

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use indexmap::IndexMap;
use tokenizers::Tokenizer;

use crate::{
    builder::SystemOneBuilder,
    config::{AgentConfig, EncoderConfig, TokenizerConfig},
    error::{Result, SystemOneError},
    head::{
        ActionHead,
        HeadLayer,
        Scorer,
        action_features,
        confidence_from_probs,
        softmax,
    },
    modernbert::ModernBert,
    sequence::{
        Collated,
        EncodedItem,
        EncodedState,
        SpecialTokens,
        build_sequence,
        collate,
    },
    types::{
        Action,
        Answer,
        ChoiceAnswer,
        NoulAnswer,
        Question,
        QuestionKind,
        ScoreAnswer,
        SystemOneRequest,
        SystemOneResponse,
        Usage,
    },
};

/// Default items per forward pass on an accelerator.
pub const DEFAULT_ACCELERATED_BATCH_SIZE: usize = 32;

/// Default items per forward pass on the CPU.
pub const DEFAULT_CPU_BATCH_SIZE: usize = 8;

/// The files a checkpoint consists of, already read from disk.
#[derive(Debug)]
pub struct CheckpointAssets {
    /// `rl_agent_config.json`.
    pub agent_config: Vec<u8>,
    /// `encoder/config.json`.
    pub encoder_config: Vec<u8>,
    /// `tokenizer/tokenizer.json`.
    pub tokenizer: Vec<u8>,
    /// `tokenizer/tokenizer_config.json`.
    pub tokenizer_config: Vec<u8>,
    /// Path of `model.safetensors`; memory-mapped rather than read.
    pub weights_path: std::path::PathBuf,
}

/// Raw per-item output of one forward pass.
#[derive(Debug, Clone)]
struct ItemOutput {
    /// Marker logits, `option_count` of them.
    logits: Vec<f32>,
    /// Probability of the default ("act") action.
    act_probability: f32,
}

/// A laya-style System One decision model.
///
/// Build one with [`SystemOne::from`], then call
/// [`system_one`](Self::system_one) or
/// [`system_one_batch`](Self::system_one_batch).
pub struct SystemOne {
    encoder: ModernBert,
    type_emb: Tensor,
    head: Vec<HeadLayer>,
    scorer: Scorer,
    act_head: ActionHead,
    tokenizer: Tokenizer,
    special: SpecialTokens,
    config: AgentConfig,
    model_name: String,
    device: Device,
    dtype: DType,
    batch_size: usize,
}

impl SystemOne {
    /// Starts configuring a model from a Hub repo id or a local
    /// checkpoint directory.
    pub fn from(repo_id: &str) -> SystemOneBuilder {
        SystemOneBuilder::new(repo_id)
    }

    /// Builds the model from checkpoint bytes.
    ///
    /// `dtype` defaults to BF16 on CUDA and F32 elsewhere. F16 is
    /// accepted but not recommended: ModernBERT activations overflow
    /// it.
    pub fn new(
        assets: CheckpointAssets,
        model_name: Option<String>,
        device: &Device,
        dtype: Option<DType>,
        batch_size: Option<usize>,
        max_len: Option<usize>,
        head_max_len: Option<usize>,
    ) -> Result<Self> {
        let mut config = AgentConfig::from_slice(&assets.agent_config)?;
        if let Some(max_len) = max_len {
            config.max_len = max_len;
        }
        if let Some(head_max_len) = head_max_len {
            config.head_max_len = head_max_len;
        }
        if config.head_max_len >= config.max_len {
            return Err(SystemOneError::Config(format!(
                "head_max_len ({}) must be smaller than max_len ({})",
                config.head_max_len, config.max_len
            )));
        }
        let encoder_config = EncoderConfig::from_slice(&assets.encoder_config)?;
        let tokenizer_config =
            TokenizerConfig::from_slice(&assets.tokenizer_config)?;

        let dtype = dtype.unwrap_or(if device.is_cuda() {
            DType::BF16
        } else {
            DType::F32
        });

        let vb = load_weights(&assets.weights_path, dtype, device)?;
        let hidden = encoder_config.hidden_size;
        let encoder = ModernBert::load(
            vb.pp("encoder"),
            &encoder_config.to_modernbert_config(),
        )?;
        let type_emb = vb.get((3, hidden), "type_emb.weight")?;
        let head = (0..config.head_layers)
            .map(|i| HeadLayer::load(vb.pp(format!("head.layers.{i}")), hidden))
            .collect::<candle_core::Result<Vec<_>>>()?;
        let scorer = Scorer::load(vb.pp("scorer"), hidden)?;
        let act_head =
            ActionHead::load(vb.pp("act_head"), hidden, config.action_count())?;

        let tokenizer = Tokenizer::from_bytes(&assets.tokenizer)?;
        let special = SpecialTokens::resolve(
            &tokenizer,
            &tokenizer_config.cls_token,
            &tokenizer_config.sep_token,
            &tokenizer_config.mask_token,
            &tokenizer_config.pad_token,
        )?;

        let batch_size = match batch_size {
            Some(0) => {
                return Err(SystemOneError::Config(
                    "batch size must be greater than zero".to_string(),
                ));
            }
            Some(n) => n,
            None if device.is_cpu() => DEFAULT_CPU_BATCH_SIZE,
            None => DEFAULT_ACCELERATED_BATCH_SIZE,
        };

        let model_name = model_name
            .filter(|name| !name.is_empty())
            .or_else(|| Some(config.model_name.clone()))
            .filter(|name| !name.is_empty())
            .unwrap_or_else(|| "laya".to_string());

        Ok(Self {
            encoder,
            type_emb,
            head,
            scorer,
            act_head,
            tokenizer,
            special,
            config,
            model_name,
            device: device.clone(),
            dtype,
            batch_size,
        })
    }

    /// The checkpoint's calibration and length settings.
    pub fn config(&self) -> &AgentConfig {
        &self.config
    }

    /// Name reported in every response.
    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    /// Device the model runs on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Dtype the encoder and head run in.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Items per forward pass.
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Tokenises a state once so several questions can share it.
    pub fn encode_state(
        &self,
        state: &crate::types::State,
    ) -> Result<EncodedState> {
        EncodedState::encode(&self.tokenizer, &self.special, state)
    }

    /// Builds the token sequence for one question over a tokenised
    /// state. Exposed for inspection and tests.
    pub fn build_sequence(
        &self,
        state: &EncodedState,
        question_id: &str,
        question: &Question,
    ) -> Result<EncodedItem> {
        build_sequence(
            &self.tokenizer,
            &self.special,
            state,
            question_id,
            question,
            self.config.max_len,
            self.config.head_max_len,
        )
    }

    /// Answers every question in `request` in one forward pass.
    pub fn system_one(
        &self,
        request: &SystemOneRequest,
    ) -> Result<SystemOneResponse> {
        let mut responses =
            self.system_one_batch(std::slice::from_ref(request))?;
        Ok(responses.pop().expect("one response per request"))
    }

    /// Answers every question of every request, packing all the
    /// `(state, question)` sequences into as few forward passes as the
    /// batch size allows. Sequences are sorted by length first so a
    /// batch pads as little as possible.
    pub fn system_one_batch(
        &self,
        requests: &[SystemOneRequest],
    ) -> Result<Vec<SystemOneResponse>> {
        // (request index, question id, encoded sequence)
        let mut items: Vec<(usize, &str, EncodedItem)> = Vec::new();
        for (r, request) in requests.iter().enumerate() {
            let state = self.encode_state(&request.state)?;
            for (id, question) in &request.questions {
                items.push((
                    r,
                    id.as_str(),
                    self.build_sequence(&state, id, question)?,
                ));
            }
        }

        let mut order: Vec<usize> = (0..items.len()).collect();
        order.sort_by_key(|&i| std::cmp::Reverse(items[i].2.ids.len()));

        let mut outputs: Vec<Option<ItemOutput>> = vec![None; items.len()];
        for chunk in order.chunks(self.batch_size) {
            let batch: Vec<&EncodedItem> =
                chunk.iter().map(|&i| &items[i].2).collect();
            let results = self.forward_batch(&batch)?;
            for (&i, output) in chunk.iter().zip(results) {
                outputs[i] = Some(output);
            }
        }

        let mut responses: Vec<SystemOneResponse> = requests
            .iter()
            .map(|_| SystemOneResponse {
                model: self.model_name.clone(),
                answers: IndexMap::new(),
                usage: Usage::default(),
            })
            .collect();
        for ((r, id, item), output) in items.into_iter().zip(outputs) {
            let output = output.expect("every item ran in some batch");
            let question = &requests[r].questions[id];
            let answer = self.answer(question, &item, &output);
            let response = &mut responses[r];
            response.usage.input_tokens += item.ids.len();
            response.answers.insert(id.to_string(), answer);
        }
        Ok(responses)
    }

    fn forward_batch(&self, items: &[&EncodedItem]) -> Result<Vec<ItemOutput>> {
        let collated = collate(items, self.special.pad);
        let logits = self.marker_logits(&collated)?;
        let (pooled, logits_rows) = logits;
        let features: Vec<f32> = logits_rows
            .iter()
            .zip(&collated.option_counts)
            .flat_map(|(row, &k)| action_features(row, k))
            .collect();
        let features = Tensor::from_vec(
            features,
            (collated.batch, crate::head::ACTION_FEATURES),
            &self.device,
        )?;
        let act = self
            .act_head
            .forward(&pooled, &features)?
            .to_vec2::<f32>()?;
        Ok(logits_rows
            .into_iter()
            .zip(&collated.option_counts)
            .zip(act)
            .map(|((row, &k), act_row)| ItemOutput {
                logits: row[..k].to_vec(),
                act_probability: act_row.first().copied().unwrap_or(0.0),
            })
            .collect())
    }

    /// Runs the encoder and head; returns the F32 pooled `[CLS]`
    /// state `(batch, hidden)` and the raw marker logits per item
    /// (`max_markers` wide, slots past each item's option count are
    /// meaningless).
    fn marker_logits(
        &self,
        collated: &Collated,
    ) -> Result<(Tensor, Vec<Vec<f32>>)> {
        let shape = (collated.batch, collated.seq_len);
        let input_ids =
            Tensor::from_slice(&collated.input_ids, shape, &self.device)?;
        let attention_mask =
            Tensor::from_slice(&collated.attention_mask, shape, &self.device)?;
        let kinds =
            Tensor::from_slice(&collated.kinds, collated.batch, &self.device)?;

        let mut hidden = self.encoder.forward(&input_ids, &attention_mask)?;
        let type_vec = self.type_emb.index_select(&kinds, 0)?.unsqueeze(1)?;
        hidden = hidden.broadcast_add(&type_vec)?;

        let key_bias = key_padding_bias(&attention_mask, self.dtype)?;
        for layer in &self.head {
            hidden = layer.forward(&hidden, &key_bias)?;
        }

        let hidden_size = hidden.dim(2)?;
        let marker_pos = Tensor::from_slice(
            &collated.marker_pos,
            (collated.batch, collated.max_markers),
            &self.device,
        )?
        .unsqueeze(2)?
        .expand((collated.batch, collated.max_markers, hidden_size))?
        .contiguous()?;
        let marked = hidden.gather(&marker_pos, 1)?;
        let logits = self.scorer.forward(&marked)?.to_vec2::<f32>()?;
        let pooled =
            hidden.i((.., 0, ..))?.to_dtype(DType::F32)?.contiguous()?;
        Ok((pooled, logits))
    }

    fn answer(
        &self,
        question: &Question,
        item: &EncodedItem,
        output: &ItemOutput,
    ) -> Answer {
        let k = item.option_count();
        let temperature = self.config.temperature_for(item.kind, k);
        let z: Vec<f32> =
            output.logits.iter().map(|l| l / temperature).collect();
        let p = softmax(&z);
        let action = Some(Action {
            act_probability: output.act_probability,
        });
        match (question, item.kind) {
            (Question::Choice(q), QuestionKind::Choice) => {
                let labels = q.criteria.labels();
                let best = argmax(&p);
                Answer::Choice(ChoiceAnswer {
                    choice: labels[best].to_string(),
                    probabilities: labels
                        .iter()
                        .zip(&p)
                        .map(|(label, &prob)| (label.to_string(), prob))
                        .collect(),
                    confidence: confidence_from_probs(&p, k),
                    action,
                })
            }
            (Question::Score(q), QuestionKind::Score) => {
                Answer::Score(ScoreAnswer {
                    score: p
                        .iter()
                        .enumerate()
                        .map(|(i, &v)| i as f32 * v)
                        .sum(),
                    legend: q
                        .criteria
                        .iter()
                        .enumerate()
                        .map(|(i, level)| (i.to_string(), level.clone()))
                        .collect(),
                    probabilities: p
                        .iter()
                        .enumerate()
                        .map(|(i, &v)| (i.to_string(), v))
                        .collect(),
                    confidence: confidence_from_probs(&p, k),
                    action,
                })
            }
            (Question::Noul(_), QuestionKind::Noul) => {
                let noul = p.get(1).copied().unwrap_or(0.0);
                Answer::Noul(NoulAnswer { noul, action })
            }
            _ => unreachable!("item kind always matches its question"),
        }
    }
}

fn argmax(p: &[f32]) -> usize {
    p.iter()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
            if v > bv { (i, v) } else { (bi, bv) }
        })
        .0
}

/// Additive `(batch, 1, 1, seq)` mask for the head's attention: zero
/// on real tokens, a large finite negative on padding. Finite rather
/// than `-inf` so an all-padding row (impossible here, but cheap to
/// guard) softmaxes to uniform instead of NaN.
fn key_padding_bias(attention_mask: &Tensor, dtype: DType) -> Result<Tensor> {
    let min_value = match dtype {
        DType::F16 => -65504.0,
        DType::BF16 => -1e38,
        _ => f32::MIN as f64,
    };
    let inverted = (1.0 - attention_mask.to_dtype(DType::F32)?)?;
    Ok((inverted * min_value)?
        .to_dtype(dtype)?
        .unsqueeze(1)?
        .unsqueeze(1)?)
}

fn load_weights(
    path: &Path,
    dtype: DType,
    device: &Device,
) -> Result<VarBuilder<'static>> {
    // SAFETY: the checkpoint file is opened read-only and is not
    // modified for the lifetime of the model; the mapping is only
    // read through the VarBuilder.
    Ok(unsafe { VarBuilder::from_mmaped_safetensors(&[path], dtype, device)? })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_picks_first_maximum() {
        assert_eq!(argmax(&[0.1, 0.7, 0.7, 0.2]), 1);
        assert_eq!(argmax(&[3.0]), 0);
    }

    #[test]
    fn key_padding_bias_shape_and_values() {
        let mask = Tensor::new(&[[1u32, 1, 0]], &Device::Cpu).unwrap();
        let bias = key_padding_bias(&mask, DType::F32).unwrap();
        assert_eq!(bias.dims(), [1, 1, 1, 3]);
        let values = bias.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(values[0], 0.0);
        assert_eq!(values[1], 0.0);
        assert_eq!(values[2], f32::MIN);
    }
}
