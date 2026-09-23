use candle_core::{DType, Device, Tensor};
use candle_nn::{Module, ops};

use super::{
    CuaS1Input,
    FullAttention,
    GatedDeltaNet,
    LayerType,
    Mlp,
    TextConfig,
    TextWeights,
    normalize_rms,
};
use crate::{Result, SystemOneError};

enum Mixer {
    LinearAttention(GatedDeltaNet),
    FullAttention(FullAttention),
}

struct DecoderLayer {
    mixer: Mixer,
    mlp: Mlp,
    input_layernorm: Tensor,
    post_attention_layernorm: Tensor,
    rms_norm_eps: f64,
}

impl DecoderLayer {
    fn load(
        weights: &mut TextWeights,
        config: &TextConfig,
        layer: usize,
    ) -> Result<Self> {
        let prefix = format!("model.language_model.layers.{layer}");
        let mixer = match config.layer_types[layer] {
            LayerType::LinearAttention => Mixer::LinearAttention(
                GatedDeltaNet::load(weights, config, layer)?,
            ),
            LayerType::FullAttention => Mixer::FullAttention(
                FullAttention::load(weights, config, layer)?,
            ),
        };
        Ok(Self {
            mixer,
            mlp: Mlp::new(
                weights
                    .linear_weight(&format!("{prefix}.mlp.gate_proj.weight"))?,
                weights
                    .linear_weight(&format!("{prefix}.mlp.up_proj.weight"))?,
                weights
                    .linear_weight(&format!("{prefix}.mlp.down_proj.weight"))?,
            )?,
            input_layernorm: weights
                .tensor(&format!("{prefix}.input_layernorm.weight"))?,
            post_attention_layernorm: weights
                .tensor(&format!("{prefix}.post_attention_layernorm.weight"))?,
            rms_norm_eps: config.rms_norm_eps,
        })
    }

    fn forward(&self, hidden: &Tensor) -> candle_core::Result<Tensor> {
        let normalized =
            normalize_rms(hidden, &self.input_layernorm, self.rms_norm_eps)?;
        let mixed = match &self.mixer {
            Mixer::LinearAttention(mixer) => mixer.forward(&normalized)?,
            Mixer::FullAttention(mixer) => mixer.forward(&normalized)?,
        };
        let hidden = (hidden + mixed)?;
        let normalized = normalize_rms(
            &hidden,
            &self.post_attention_layernorm,
            self.rms_norm_eps,
        )?;
        hidden + self.mlp.forward(&normalized)?
    }
}

/// Raw letter logits and probabilities, both in the input's option order.
#[derive(Debug, Clone)]
pub struct CuaS1Prediction {
    pub logits: Vec<f32>,
    pub probabilities: Vec<f32>,
}

/// Qwen3.5 text decoder for one unpadded CPU/F32 sequence, without a cache.
pub struct TextModel {
    embed_tokens: Tensor,
    layers: Vec<DecoderLayer>,
    norm: Tensor,
    rms_norm_eps: f64,
}

impl TextModel {
    /// Loads the decoder and rejects any unused LoRA modules.
    pub fn load(
        weights: &mut TextWeights,
        config: &TextConfig,
    ) -> Result<Self> {
        if config.num_hidden_layers == 0
            || config.layer_types.len() != config.num_hidden_layers
            || !config.tie_word_embeddings
        {
            return Err(SystemOneError::Config(
                "text decoder requires tied embeddings and one mixer type per layer".into(),
            ));
        }
        let embed_tokens =
            weights.tensor("model.language_model.embed_tokens.weight")?;
        if !embed_tokens.device().is_cpu() || embed_tokens.dtype() != DType::F32
        {
            return Err(SystemOneError::Config(
                "text decoder requires CPU/F32 text weights".into(),
            ));
        }
        if embed_tokens.dims() != [config.vocab_size, config.hidden_size] {
            return Err(SystemOneError::Config(
                "embedding dimensions do not match the text config".into(),
            ));
        }
        let layers = (0..config.num_hidden_layers)
            .map(|layer| DecoderLayer::load(weights, config, layer))
            .collect::<Result<Vec<_>>>()?;
        let norm = weights.tensor("model.language_model.norm.weight")?;
        weights.check_all_used()?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            rms_norm_eps: config.rms_norm_eps,
        })
    }

    /// Scores only the requested letter tokens at the final prompt position.
    pub fn forward(&self, input: &CuaS1Input) -> Result<CuaS1Prediction> {
        if input.input_ids.is_empty()
            || !(1..=26).contains(&input.letter_ids.len())
            || input.letters.len() != input.letter_ids.len()
        {
            return Err(SystemOneError::Config(
                "text decoder requires a nonempty prompt and 1..=26 matching letters and token ids".into(),
            ));
        }
        let vocab_size = self.embed_tokens.dim(0)?;
        if input
            .input_ids
            .iter()
            .chain(&input.letter_ids)
            .any(|&id| id as usize >= vocab_size)
        {
            return Err(SystemOneError::Config(
                "text decoder token id is outside the embedding vocabulary"
                    .into(),
            ));
        }
        let ids = Tensor::new(input.input_ids.as_slice(), &Device::Cpu)?;
        let mut hidden = self.embed_tokens.index_select(&ids, 0)?;
        for layer in &self.layers {
            hidden = layer.forward(&hidden)?;
        }
        self.read_options(&hidden, &input.letter_ids)
    }

    fn read_options(
        &self,
        hidden: &Tensor,
        letter_ids: &[u32],
    ) -> Result<CuaS1Prediction> {
        let last = hidden.narrow(0, hidden.dim(0)? - 1, 1)?;
        let last = normalize_rms(&last, &self.norm, self.rms_norm_eps)?;
        let ids = Tensor::new(letter_ids, &Device::Cpu)?;
        // Gather before projecting: never materialize full-vocabulary logits.
        let embeddings = self.embed_tokens.index_select(&ids, 0)?;
        let logits = last.matmul(&embeddings.t()?)?.squeeze(0)?;
        let probabilities = ops::softmax_last_dim(&logits)?.to_vec1::<f32>()?;
        Ok(CuaS1Prediction {
            logits: logits.to_vec1::<f32>()?,
            probabilities,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use candle_core::safetensors::MmapedSafetensors;
    use serde::Deserialize;

    use super::*;

    #[derive(Deserialize)]
    struct Prompts {
        cases: Vec<Prompt>,
    }

    #[derive(Deserialize)]
    struct Prompt {
        name: String,
        chat_text: String,
        input_ids: Vec<u32>,
        letters: Vec<char>,
        letter_ids: Vec<u32>,
    }

    impl Prompt {
        fn into_input(self) -> CuaS1Input {
            CuaS1Input {
                chat_text: self.chat_text,
                input_ids: self.input_ids,
                letters: self.letters,
                letter_ids: self.letter_ids,
            }
        }
    }

    #[derive(Deserialize)]
    struct Reference {
        cases: Vec<Case>,
    }

    #[derive(Deserialize)]
    struct Case {
        name: String,
        input_tokens: usize,
        options: Vec<OptionPrediction>,
    }

    #[derive(Deserialize)]
    struct OptionPrediction {
        letter: char,
        token_id: u32,
        logit: f32,
        probability: f32,
    }

    fn load_model() -> TextModel {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../artifacts/cua-s1");
        let base_directory =
            root.join("base/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a");
        let adapter_directory =
            root.join("adapter/16818868b0cc7813808aae4e87b417657046ab79/text");
        let config = TextConfig::from_slice(
            &std::fs::read(base_directory.join("config.json")).unwrap(),
        )
        .unwrap();
        let mut weights = TextWeights::load(
            &base_directory,
            Some(&adapter_directory),
            &Device::Cpu,
            DType::F32,
        )
        .unwrap();
        let model = TextModel::load(&mut weights, &config).unwrap();
        assert_eq!(model.layers.len(), 32);
        model
    }

    fn load_prompts() -> Vec<Prompt> {
        let prompts: Prompts = serde_json::from_str(include_str!(
            "../../../../research/cua-s1/prompts.json"
        ))
        .unwrap();
        assert_eq!(prompts.cases.len(), 6);
        prompts.cases
    }

    fn max_absolute_difference(actual: &[f32], expected: &[f32]) -> f32 {
        assert_eq!(actual.len(), expected.len());
        actual
            .iter()
            .zip(expected)
            .map(|(&actual, &expected)| {
                assert!(actual.is_finite() && expected.is_finite());
                (actual - expected).abs()
            })
            .fold(0., f32::max)
    }

    #[test]
    fn reads_only_requested_embeddings_in_option_order_at_the_last_position() {
        let model = TextModel {
            embed_tokens: Tensor::new(
                &[[1000f32, 1000.], [3., 4.], [-4., 3.]],
                &Device::Cpu,
            )
            .unwrap(),
            layers: Vec::new(),
            norm: Tensor::new(&[1f32, -0.5], &Device::Cpu).unwrap(),
            rms_norm_eps: 3.5,
        };
        let input = CuaS1Input {
            chat_text: String::new(),
            input_ids: vec![0, 1],
            letters: vec!['B', 'A'],
            letter_ids: vec![2, 1],
        };
        let prediction = model.forward(&input).unwrap();
        // Last embedding [3, 4] normalizes to [1.5, 0.5].
        assert_eq!(prediction.logits, [-4.5, 6.5]);
        let first = (-11f32).exp() / (1. + (-11f32).exp());
        assert!(
            max_absolute_difference(
                &prediction.probabilities,
                &[first, 1. - first],
            ) < 1e-7
        );
        let mut singleton = input.clone();
        singleton.letters = vec!['B'];
        singleton.letter_ids = vec![2];
        assert_eq!(model.forward(&singleton).unwrap().probabilities, [1.]);
        for invalid in [
            CuaS1Input {
                input_ids: vec![],
                ..input.clone()
            },
            CuaS1Input {
                input_ids: vec![3],
                ..input.clone()
            },
            CuaS1Input {
                letters: vec![],
                ..input.clone()
            },
            CuaS1Input {
                letter_ids: vec![],
                ..input.clone()
            },
            CuaS1Input {
                letter_ids: vec![2, 3],
                ..input.clone()
            },
            CuaS1Input {
                letters: vec!['A'; 27],
                letter_ids: vec![1; 27],
                ..input
            },
        ] {
            assert!(model.forward(&invalid).is_err());
        }
    }

    #[test]
    #[ignore = "requires pinned artifacts/cua-s1 weights and layer dump, about 20 GB RAM"]
    fn reproduces_every_dumped_decoder_layer() {
        let model = load_model();
        let input = load_prompts().remove(0).into_input();
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../artifacts/cua-s1");
        // SAFETY: the reference dump is mapped read-only and not modified.
        let dump = unsafe {
            MmapedSafetensors::new(root.join("layers-f32.safetensors"))
        }
        .unwrap();
        let ids =
            Tensor::new(input.input_ids.as_slice(), &Device::Cpu).unwrap();
        assert_eq!(
            dump.load("input_ids", &Device::Cpu)
                .unwrap()
                .squeeze(0)
                .unwrap()
                .to_dtype(DType::U32)
                .unwrap()
                .to_vec1::<u32>()
                .unwrap(),
            input.input_ids
        );
        let mut hidden = model.embed_tokens.index_select(&ids, 0).unwrap();
        let mut has_matching_layers = true;
        for (layer, decoder) in model.layers.iter().enumerate() {
            hidden = decoder.forward(&hidden).unwrap();
            let name = format!("layers.{layer}.output");
            let expected =
                dump.load(&name, &Device::Cpu).unwrap().squeeze(0).unwrap();
            assert_eq!(hidden.dtype(), DType::F32);
            assert_eq!(hidden.dims(), expected.dims());
            let actual =
                hidden.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let expected =
                expected.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let difference = max_absolute_difference(&actual, &expected);
            println!("{name}: max absolute difference {difference:e}");
            has_matching_layers &= actual
                .iter()
                .zip(&expected)
                .all(|(a, e)| (a - e).abs() <= 1e-3 + 1e-4 * e.abs());
        }
        assert!(
            has_matching_layers,
            "layer outputs exceed atol=1e-3, rtol=1e-4"
        );
    }

    #[test]
    #[ignore = "requires pinned artifacts/cua-s1 weights, about 20 GB RAM"]
    fn reproduces_all_reference_option_predictions() {
        let model = load_model();
        let reference: Reference = serde_json::from_str(include_str!(
            "../../../../research/cua-s1/probabilities-f32.json"
        ))
        .unwrap();
        let prompts = load_prompts();
        assert_eq!(reference.cases.len(), prompts.len());
        let mut max_logit_difference = 0f32;
        let mut max_probability_difference = 0f32;
        let mut has_matching_top_options = true;
        for (prompt, case) in prompts.into_iter().zip(reference.cases) {
            assert_eq!(prompt.name, case.name);
            let input = prompt.into_input();
            assert_eq!(input.input_ids.len(), case.input_tokens);
            assert_eq!(input.letter_ids.len(), case.options.len());
            for (i, option) in case.options.iter().enumerate() {
                assert_eq!(input.letters[i], option.letter);
                assert_eq!(input.letter_ids[i], option.token_id);
            }
            let prediction = model.forward(&input).unwrap();
            let logits: Vec<_> = case.options.iter().map(|o| o.logit).collect();
            let probabilities: Vec<_> =
                case.options.iter().map(|o| o.probability).collect();
            let logit_difference =
                max_absolute_difference(&prediction.logits, &logits);
            let probability_difference = max_absolute_difference(
                &prediction.probabilities,
                &probabilities,
            );
            max_logit_difference = max_logit_difference.max(logit_difference);
            max_probability_difference =
                max_probability_difference.max(probability_difference);
            let top_option = |values: &[f32]| {
                values
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.total_cmp(b))
                    .unwrap()
                    .0
            };
            let top = top_option(&prediction.probabilities);
            let expected_top = top_option(&probabilities);
            has_matching_top_options &= top == expected_top;
            println!(
                "{}: top {} (expected {}), max absolute logit difference {logit_difference:e}, probability difference {probability_difference:e}",
                case.name, input.letters[top], input.letters[expected_top]
            );
        }
        println!(
            "all cases: max absolute logit difference {max_logit_difference:e}, probability difference {max_probability_difference:e}"
        );
        assert!(
            has_matching_top_options,
            "top option differs from reference"
        );
        assert!(
            max_logit_difference <= 1e-3,
            "letter logits exceed atol=1e-3"
        );
        assert!(
            max_probability_difference <= 1e-4,
            "probabilities exceed atol=1e-4"
        );
    }
}
