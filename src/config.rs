//! Checkpoint configuration: laya's `rl_agent_config.json` and the
//! encoder's Hugging Face `config.json`.

use std::collections::HashMap;

use serde::Deserialize;

use crate::{
    error::{Result, SystemOneError},
    modernbert::Config as ModernBertConfig,
    types::QuestionKind,
};

fn default_head_layers() -> usize {
    2
}

fn default_max_len() -> usize {
    512
}

fn default_head_max_len() -> usize {
    192
}

fn default_temperature() -> Vec<f32> {
    vec![1.0, 1.0, 1.0]
}

fn default_amp_dtype() -> String {
    "fp16".to_string()
}

/// `rl_agent_config.json`: how the decision head was built and how its
/// logits are calibrated.
#[derive(Debug, Clone, Deserialize)]
pub struct AgentConfig {
    /// Hugging Face id of the encoder the head was trained on. Only
    /// used as a fallback when the checkpoint ships no encoder config.
    pub encoder: String,
    /// Post-encoder transformer layers in the decision head.
    #[serde(default = "default_head_layers")]
    pub head_layers: usize,
    /// Longest token sequence the model reads; the state is truncated
    /// to whatever the question header leaves.
    #[serde(default = "default_max_len")]
    pub max_len: usize,
    /// Token budget shared by the instructions and every option.
    #[serde(default = "default_head_max_len")]
    pub head_max_len: usize,
    /// Cost of each non-default action; the action head has one output
    /// per entry plus one for "act".
    #[serde(default)]
    pub act_costs: HashMap<String, f64>,
    /// Autocast dtype used in training (`bf16` or `fp16`).
    #[serde(default = "default_amp_dtype")]
    pub amp_dtype: String,
    /// Display name of the checkpoint.
    #[serde(default)]
    pub model_name: String,
    /// Per-primitive temperature, indexed by [`QuestionKind::index`].
    #[serde(default = "default_temperature")]
    pub temperature: Vec<f32>,
    /// Finer temperatures keyed by `<primitive>:<option bucket>`, see
    /// [`temperature_bucket`].
    #[serde(default)]
    pub temperature_by_options: HashMap<String, f32>,
}

impl AgentConfig {
    /// Parses `rl_agent_config.json`.
    pub fn from_slice(bytes: &[u8]) -> Result<Self> {
        let config: AgentConfig = serde_json::from_slice(bytes)?;
        if config.temperature.len() < 3 {
            return Err(SystemOneError::Config(format!(
                "temperature must list one value per primitive, got {}",
                config.temperature.len()
            )));
        }
        if config.head_max_len >= config.max_len {
            return Err(SystemOneError::Config(format!(
                "head_max_len ({}) must be smaller than max_len ({})",
                config.head_max_len, config.max_len
            )));
        }
        Ok(config)
    }

    /// Number of actions the action head chooses between.
    pub fn action_count(&self) -> usize {
        self.act_costs.len() + 1
    }

    /// Temperature to divide the logits of a `kind` question with `k`
    /// options by. Bucketed values win over the per-primitive default,
    /// and the result is floored at `1e-3` like laya does.
    pub fn temperature_for(&self, kind: QuestionKind, k: usize) -> f32 {
        let bucket = temperature_bucket(kind, k);
        self.temperature_by_options
            .get(&bucket)
            .copied()
            .unwrap_or(self.temperature[kind.index()])
            .max(1e-3)
    }
}

/// The `temperature_by_options` key for a question with `k` options:
/// `choice:2`, `score:3-5`, `noul:2`, `choice:6-10`, `choice:11+`.
pub fn temperature_bucket(kind: QuestionKind, k: usize) -> String {
    let size = if k <= 2 {
        "2"
    } else if k <= 5 {
        "3-5"
    } else if k <= 10 {
        "6-10"
    } else {
        "11+"
    };
    format!("{}:{size}", kind.as_str())
}

#[derive(Debug, Clone, Deserialize)]
struct RopeSpec {
    rope_theta: f64,
}

#[derive(Debug, Clone, Deserialize)]
struct RopeParameters {
    full_attention: Option<RopeSpec>,
    sliding_attention: Option<RopeSpec>,
}

fn default_global_attn_every_n_layers() -> usize {
    3
}

fn default_local_attention() -> usize {
    128
}

/// The subset of a ModernBERT `config.json` the encoder needs, in both
/// the transformers v4 (`global_rope_theta`) and v5 (`rope_parameters`)
/// spellings.
#[derive(Debug, Clone, Deserialize)]
pub struct EncoderConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub pad_token_id: u32,
    layer_norm_eps: Option<f64>,
    norm_eps: Option<f64>,
    #[serde(default = "default_global_attn_every_n_layers")]
    pub global_attn_every_n_layers: usize,
    #[serde(default = "default_local_attention")]
    pub local_attention: usize,
    global_rope_theta: Option<f64>,
    local_rope_theta: Option<f64>,
    rope_parameters: Option<RopeParameters>,
    #[serde(default)]
    layer_types: Vec<String>,
    #[serde(default)]
    pub model_type: String,
    #[serde(default)]
    pub architectures: Vec<String>,
}

impl EncoderConfig {
    /// Parses `encoder/config.json`.
    pub fn from_slice(bytes: &[u8]) -> Result<Self> {
        let config: EncoderConfig = serde_json::from_slice(bytes)?;
        if config.model_type != "modernbert" {
            return Err(SystemOneError::Config(format!(
                "encoder model_type must be modernbert, got {:?}",
                config.model_type
            )));
        }
        if !config
            .hidden_size
            .is_multiple_of(config.num_attention_heads)
        {
            return Err(SystemOneError::Config(format!(
                "hidden_size {} is not divisible by num_attention_heads {}",
                config.hidden_size, config.num_attention_heads
            )));
        }
        config.check_layer_types()?;
        Ok(config)
    }

    /// The encoder alternates global and sliding layers by
    /// `global_attn_every_n_layers`; a checkpoint whose explicit
    /// `layer_types` disagree would silently mis-attend, so refuse it.
    fn check_layer_types(&self) -> Result<()> {
        for (layer, kind) in self.layer_types.iter().enumerate() {
            let expected = if layer % self.global_attn_every_n_layers == 0 {
                "full_attention"
            } else {
                "sliding_attention"
            };
            if kind != expected {
                return Err(SystemOneError::Config(format!(
                    "layer_types[{layer}] is {kind:?} but \
                     global_attn_every_n_layers={} implies {expected:?}",
                    self.global_attn_every_n_layers
                )));
            }
        }
        Ok(())
    }

    /// LayerNorm epsilon, from whichever key the config spells it with.
    pub fn layer_norm_eps(&self) -> f64 {
        self.layer_norm_eps.or(self.norm_eps).unwrap_or(1e-5)
    }

    /// RoPE theta for global-attention layers.
    pub fn global_rope_theta(&self) -> f64 {
        self.global_rope_theta
            .or_else(|| {
                self.rope_parameters
                    .as_ref()
                    .and_then(|p| p.full_attention.as_ref())
                    .map(|s| s.rope_theta)
            })
            .unwrap_or(160_000.0)
    }

    /// RoPE theta for sliding-window layers.
    pub fn local_rope_theta(&self) -> f64 {
        self.local_rope_theta
            .or_else(|| {
                self.rope_parameters
                    .as_ref()
                    .and_then(|p| p.sliding_attention.as_ref())
                    .map(|s| s.rope_theta)
            })
            .unwrap_or(10_000.0)
    }

    /// The config docbert-pylate's ModernBERT port loads from.
    pub fn to_modernbert_config(&self) -> ModernBertConfig {
        ModernBertConfig {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            intermediate_size: self.intermediate_size,
            max_position_embeddings: self.max_position_embeddings,
            layer_norm_eps: self.layer_norm_eps(),
            pad_token_id: self.pad_token_id,
            global_attn_every_n_layers: self.global_attn_every_n_layers,
            global_rope_theta: self.global_rope_theta(),
            local_attention: self.local_attention,
            local_rope_theta: self.local_rope_theta(),
            classifier_config: None,
        }
    }
}

/// `tokenizer/tokenizer_config.json`: the special token strings.
#[derive(Debug, Clone, Deserialize)]
pub struct TokenizerConfig {
    #[serde(default = "default_cls")]
    pub cls_token: String,
    #[serde(default = "default_sep")]
    pub sep_token: String,
    #[serde(default = "default_mask")]
    pub mask_token: String,
    #[serde(default = "default_pad")]
    pub pad_token: String,
}

fn default_cls() -> String {
    "[CLS]".to_string()
}

fn default_sep() -> String {
    "[SEP]".to_string()
}

fn default_mask() -> String {
    "[MASK]".to_string()
}

fn default_pad() -> String {
    "[PAD]".to_string()
}

impl TokenizerConfig {
    /// Parses `tokenizer_config.json`.
    pub fn from_slice(bytes: &[u8]) -> Result<Self> {
        Ok(serde_json::from_slice(bytes)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LAYA_AGENT: &str = r#"{
        "encoder": "answerdotai/ModernBERT-large",
        "head_layers": 2, "max_len": 512, "head_max_len": 192,
        "act_costs": {"escalate": 0.5}, "amp_dtype": "bf16",
        "model_name": "rl-agent",
        "temperature": [1.6369, 1.2514, 1.9834],
        "temperature_by_options": {
            "choice:3-5": 1.76, "choice:6-10": 1.0, "score:3-5": 1.25,
            "noul:2": 1.98, "choice:11+": 0.1, "choice:2": 1.9
        }
    }"#;

    #[test]
    fn agent_config_parses_and_buckets_temperatures() {
        let config = AgentConfig::from_slice(LAYA_AGENT.as_bytes()).unwrap();
        assert_eq!(config.action_count(), 2);
        assert_eq!(config.temperature_for(QuestionKind::Noul, 2), 1.98);
        assert_eq!(config.temperature_for(QuestionKind::Choice, 4), 1.76);
        assert_eq!(config.temperature_for(QuestionKind::Choice, 8), 1.0);
        assert_eq!(config.temperature_for(QuestionKind::Choice, 40), 0.1);
        // No `score:2` bucket: falls back to the per-primitive value.
        assert_eq!(config.temperature_for(QuestionKind::Score, 2), 1.2514);
        // No `score:6-10` bucket either.
        assert_eq!(config.temperature_for(QuestionKind::Score, 7), 1.2514);
    }

    #[test]
    fn temperature_is_floored() {
        let mut config =
            AgentConfig::from_slice(LAYA_AGENT.as_bytes()).unwrap();
        config.temperature[2] = 0.0;
        config.temperature_by_options.clear();
        assert_eq!(config.temperature_for(QuestionKind::Noul, 2), 1e-3);
    }

    #[test]
    fn temperature_bucket_names() {
        assert_eq!(temperature_bucket(QuestionKind::Noul, 2), "noul:2");
        assert_eq!(temperature_bucket(QuestionKind::Choice, 3), "choice:3-5");
        assert_eq!(temperature_bucket(QuestionKind::Choice, 5), "choice:3-5");
        assert_eq!(temperature_bucket(QuestionKind::Score, 10), "score:6-10");
        assert_eq!(temperature_bucket(QuestionKind::Choice, 11), "choice:11+");
    }

    #[test]
    fn encoder_config_reads_transformers_v5_rope_parameters() {
        let raw = r#"{
            "model_type": "modernbert", "vocab_size": 50368,
            "hidden_size": 1024, "num_hidden_layers": 28,
            "num_attention_heads": 16, "intermediate_size": 2624,
            "max_position_embeddings": 8192, "pad_token_id": 50283,
            "layer_norm_eps": 1e-05, "norm_eps": 1e-05,
            "global_attn_every_n_layers": 3, "local_attention": 128,
            "layer_types": ["full_attention", "sliding_attention",
                            "sliding_attention", "full_attention"],
            "rope_parameters": {
                "full_attention": {"rope_theta": 160000.0},
                "sliding_attention": {"rope_theta": 10000.0}
            }
        }"#;
        let config = EncoderConfig::from_slice(raw.as_bytes()).unwrap();
        let mb = config.to_modernbert_config();
        assert_eq!(mb.global_rope_theta, 160_000.0);
        assert_eq!(mb.local_rope_theta, 10_000.0);
        assert_eq!(mb.layer_norm_eps, 1e-5);
        assert_eq!(mb.pad_token_id, 50283);
        assert!(mb.classifier_config.is_none());
    }

    #[test]
    fn encoder_config_reads_transformers_v4_rope_theta() {
        let raw = r#"{
            "model_type": "modernbert", "vocab_size": 8,
            "hidden_size": 64, "num_hidden_layers": 2,
            "num_attention_heads": 4, "intermediate_size": 96,
            "max_position_embeddings": 512, "pad_token_id": 0,
            "layer_norm_eps": 1e-05,
            "global_rope_theta": 100.0, "local_rope_theta": 50.0
        }"#;
        let config = EncoderConfig::from_slice(raw.as_bytes()).unwrap();
        assert_eq!(config.global_rope_theta(), 100.0);
        assert_eq!(config.local_rope_theta(), 50.0);
    }

    #[test]
    fn encoder_config_rejects_inconsistent_layer_types() {
        let raw = r#"{
            "model_type": "modernbert", "vocab_size": 8,
            "hidden_size": 64, "num_hidden_layers": 2,
            "num_attention_heads": 4, "intermediate_size": 96,
            "max_position_embeddings": 512, "pad_token_id": 0,
            "layer_types": ["sliding_attention", "full_attention"]
        }"#;
        let err = EncoderConfig::from_slice(raw.as_bytes()).unwrap_err();
        assert!(err.to_string().contains("layer_types[0]"), "{err}");
    }

    #[test]
    fn encoder_config_rejects_other_model_types() {
        let raw = r#"{
            "model_type": "bert", "vocab_size": 8, "hidden_size": 64,
            "num_hidden_layers": 2, "num_attention_heads": 4,
            "intermediate_size": 96, "max_position_embeddings": 512,
            "pad_token_id": 0
        }"#;
        assert!(EncoderConfig::from_slice(raw.as_bytes()).is_err());
    }

    #[test]
    fn tokenizer_config_defaults_to_modernbert_specials() {
        let config = TokenizerConfig::from_slice(b"{}").unwrap();
        assert_eq!(config.cls_token, "[CLS]");
        assert_eq!(config.mask_token, "[MASK]");
        let config =
            TokenizerConfig::from_slice(br#"{"cls_token": "<bos>", "sep_token": "<eos>", "mask_token": "<mask>", "pad_token": "<pad>"}"#)
                .unwrap();
        assert_eq!(config.sep_token, "<eos>");
    }
}
