use serde::Deserialize;

use crate::{Result, SystemOneError};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerType {
    LinearAttention,
    FullAttention,
}

#[derive(Debug, Clone, Deserialize)]
struct RopeParameters {
    rope_theta: f64,
    partial_rotary_factor: f64,
}

/// The text decoder settings from a Qwen3.5 `config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct TextConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub layer_types: Vec<LayerType>,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub vocab_size: usize,
    pub tie_word_embeddings: bool,
    pub attn_output_gate: bool,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_conv_kernel_dim: usize,
    rope_parameters: RopeParameters,
}

#[derive(Deserialize)]
struct Config {
    text_config: TextConfig,
    tie_word_embeddings: bool,
}

impl TextConfig {
    /// Parses the `Qwen3_5ForConditionalGeneration` config layout.
    pub fn from_slice(bytes: &[u8]) -> Result<Self> {
        let config: Config = serde_json::from_slice(bytes)?;
        let text = config.text_config;
        if text.layer_types.len() != text.num_hidden_layers {
            return Err(SystemOneError::Config(format!(
                "layer_types must list one type per layer, got {} for {} layers",
                text.layer_types.len(),
                text.num_hidden_layers
            )));
        }
        if !text.attn_output_gate {
            return Err(SystemOneError::Config(
                "Qwen3.5 text decoder requires attn_output_gate=true".into(),
            ));
        }
        if !config.tie_word_embeddings || !text.tie_word_embeddings {
            return Err(SystemOneError::Config(
                "Qwen3.5 requires tie_word_embeddings=true in both config and text_config"
                    .into(),
            ));
        }
        if text.num_attention_heads == 0
            || text.num_key_value_heads == 0
            || !text
                .num_attention_heads
                .is_multiple_of(text.num_key_value_heads)
        {
            return Err(SystemOneError::Config(format!(
                "num_attention_heads ({}) must be a positive multiple of num_key_value_heads ({})",
                text.num_attention_heads, text.num_key_value_heads
            )));
        }
        if text.linear_num_value_heads == 0
            || text.linear_num_key_heads == 0
            || !text
                .linear_num_value_heads
                .is_multiple_of(text.linear_num_key_heads)
        {
            return Err(SystemOneError::Config(format!(
                "linear_num_value_heads ({}) must be a positive multiple of linear_num_key_heads ({})",
                text.linear_num_value_heads, text.linear_num_key_heads
            )));
        }
        Ok(text)
    }

    /// RoPE theta for text positions; mRoPE section settings are unused.
    pub fn rope_theta(&self) -> f64 {
        self.rope_parameters.rope_theta
    }

    pub fn partial_rotary_factor(&self) -> f64 {
        self.rope_parameters.partial_rotary_factor
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const QWEN_CONFIG: &[u8] =
        include_bytes!("../../../../research/cua-s1/qwen3.5-4b-config.json");

    #[test]
    fn pinned_text_config() {
        let config = TextConfig::from_slice(QWEN_CONFIG).unwrap();
        assert_eq!(config.num_hidden_layers, 32);
        assert_eq!(config.layer_types.len(), 32);
        assert_eq!(config.hidden_size, 2560);
        assert_eq!(config.intermediate_size, 9216);
        assert_eq!(config.num_attention_heads, 16);
        assert_eq!(config.num_key_value_heads, 4);
        assert_eq!(config.head_dim, 256);
        assert_eq!(config.rms_norm_eps, 1e-6);
        assert_eq!(config.vocab_size, 248320);
        assert!(config.tie_word_embeddings);
        assert!(config.attn_output_gate);
        assert_eq!(config.linear_num_key_heads, 16);
        assert_eq!(config.linear_num_value_heads, 32);
        assert_eq!(config.linear_key_head_dim, 128);
        assert_eq!(config.linear_value_head_dim, 128);
        assert_eq!(config.linear_conv_kernel_dim, 4);
        assert_eq!(config.rope_theta(), 10_000_000.0);
        assert_eq!(config.partial_rotary_factor(), 0.25);
        for (layer, kind) in config.layer_types.iter().enumerate() {
            let expected = if layer % 4 == 3 {
                LayerType::FullAttention
            } else {
                LayerType::LinearAttention
            };
            assert_eq!(*kind, expected, "layer {layer}");
        }
    }

    #[test]
    fn rejects_unsupported_text_config() {
        for (pointer, value, message) in [
            ("/text_config/num_hidden_layers", "31", "layer_types"),
            ("/text_config/attn_output_gate", "false", "attn_output_gate"),
            (
                "/text_config/tie_word_embeddings",
                "false",
                "tie_word_embeddings",
            ),
            ("/tie_word_embeddings", "false", "tie_word_embeddings"),
            (
                "/text_config/num_attention_heads",
                "15",
                "num_attention_heads",
            ),
            (
                "/text_config/num_attention_heads",
                "0",
                "num_attention_heads",
            ),
            (
                "/text_config/num_key_value_heads",
                "0",
                "num_key_value_heads",
            ),
            (
                "/text_config/linear_num_value_heads",
                "31",
                "linear_num_value_heads",
            ),
            (
                "/text_config/linear_num_value_heads",
                "0",
                "linear_num_value_heads",
            ),
            (
                "/text_config/linear_num_key_heads",
                "0",
                "linear_num_key_heads",
            ),
        ] {
            let mut raw: serde_json::Value =
                serde_json::from_slice(QWEN_CONFIG).unwrap();
            *raw.pointer_mut(pointer).unwrap() =
                serde_json::from_str(value).unwrap();
            let err =
                TextConfig::from_slice(&serde_json::to_vec(&raw).unwrap())
                    .unwrap_err();
            assert!(matches!(err, SystemOneError::Config(_)), "{err}");
            assert!(err.to_string().contains(message), "{pointer}: {err}");
        }
    }
}
