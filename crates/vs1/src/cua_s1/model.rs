use candle_core::{DType, Tensor};
use candle_nn::{Module, ops};

use super::{
    CuaS1Input,
    DeltaNetState,
    FullAttention,
    GatedDeltaNet,
    KvCache,
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

#[derive(Debug, Clone)]
enum MixerState {
    LinearAttention(DeltaNetState),
    FullAttention(KvCache),
}

/// Reusable per-layer state for a nonempty prefix encoded by one text model.
#[derive(Debug, Clone)]
pub struct PrefixState {
    layer_states: Vec<MixerState>,
    token_count: usize,
}

impl PrefixState {
    pub fn token_count(&self) -> usize {
        self.token_count
    }
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

    fn forward_with_state(
        &self,
        hidden: &Tensor,
        state: Option<&MixerState>,
    ) -> candle_core::Result<(Tensor, MixerState)> {
        let normalized =
            normalize_rms(hidden, &self.input_layernorm, self.rms_norm_eps)?;
        let (mixed, state) = match &self.mixer {
            Mixer::LinearAttention(mixer) => {
                let state = match state {
                    Some(MixerState::LinearAttention(state)) => Some(state),
                    None => None,
                    _ => candle_core::bail!("expected DeltaNet prefix state"),
                };
                let (mixed, state) =
                    mixer.forward_with_state(&normalized, state)?;
                (mixed, MixerState::LinearAttention(state))
            }
            Mixer::FullAttention(mixer) => {
                let cache = match state {
                    Some(MixerState::FullAttention(cache)) => Some(cache),
                    None => None,
                    _ => candle_core::bail!(
                        "expected full-attention prefix cache"
                    ),
                };
                let (mixed, cache) =
                    mixer.forward_with_cache(&normalized, cache)?;
                (mixed, MixerState::FullAttention(cache))
            }
        };
        let hidden = (hidden + mixed)?;
        let normalized = normalize_rms(
            &hidden,
            &self.post_attention_layernorm,
            self.rms_norm_eps,
        )?;
        Ok(((hidden + self.mlp.forward(&normalized)?)?, state))
    }
}

/// Raw letter logits and probabilities, both in the input's option order.
#[derive(Debug, Clone)]
pub struct CuaS1Prediction {
    pub logits: Vec<f32>,
    pub probabilities: Vec<f32>,
}

/// Qwen3.5 text decoder for one unpadded sequence with optional prefix reuse.
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
        // Keep the tied embedding table on the CPU; only selected rows move.
        let embed_tokens = weights
            .load_cpu_tensor("model.language_model.embed_tokens.weight")?;
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
        let ids = Tensor::new(
            input.input_ids.as_slice(),
            self.embed_tokens.device(),
        )?;
        let mut hidden = self
            .embed_tokens
            .index_select(&ids, 0)?
            .to_device(self.norm.device())?;
        for layer in &self.layers {
            hidden = layer.forward(&hidden)?;
        }
        self.read_options(&hidden, &input.letter_ids)
    }

    /// Encodes a nonempty prefix without the final norm or letter readout.
    pub fn encode_prefix(&self, ids: &[u32]) -> Result<PrefixState> {
        let mut hidden = self.embed(ids)?;
        let mut layer_states = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            let (output, state) = layer.forward_with_state(&hidden, None)?;
            hidden = output;
            layer_states.push(state);
        }
        Ok(PrefixState {
            layer_states,
            token_count: ids.len(),
        })
    }

    /// Scores a nonempty suffix using an unchanged prefix from this model.
    pub fn score_suffix(
        &self,
        prefix: &PrefixState,
        suffix_ids: &[u32],
        letter_ids: &[u32],
    ) -> Result<CuaS1Prediction> {
        if prefix.layer_states.len() != self.layers.len() {
            return Err(SystemOneError::Config(
                "prefix state must contain one state per decoder layer".into(),
            ));
        }
        if !(1..=26).contains(&letter_ids.len()) {
            return Err(SystemOneError::Config(
                "text decoder requires 1..=26 letter token ids".into(),
            ));
        }
        let vocab_size = self.embed_tokens.dim(0)?;
        if letter_ids.iter().any(|&id| id as usize >= vocab_size) {
            return Err(SystemOneError::Config(
                "text decoder token id is outside the embedding vocabulary"
                    .into(),
            ));
        }
        let mut hidden = self.embed(suffix_ids)?;
        for (layer, state) in self.layers.iter().zip(&prefix.layer_states) {
            (hidden, _) = layer.forward_with_state(&hidden, Some(state))?;
        }
        self.read_options(&hidden, letter_ids)
    }

    fn embed(&self, ids: &[u32]) -> Result<Tensor> {
        if ids.is_empty() {
            return Err(SystemOneError::Config(
                "text decoder requires a nonempty token sequence".into(),
            ));
        }
        let vocab_size = self.embed_tokens.dim(0)?;
        if ids.iter().any(|&id| id as usize >= vocab_size) {
            return Err(SystemOneError::Config(
                "text decoder token id is outside the embedding vocabulary"
                    .into(),
            ));
        }
        let ids = Tensor::new(ids, self.embed_tokens.device())?;
        Ok(self
            .embed_tokens
            .index_select(&ids, 0)?
            .to_device(self.norm.device())?)
    }

    fn read_options(
        &self,
        hidden: &Tensor,
        letter_ids: &[u32],
    ) -> Result<CuaS1Prediction> {
        let last = hidden.narrow(0, hidden.dim(0)? - 1, 1)?;
        let last = normalize_rms(&last, &self.norm, self.rms_norm_eps)?;
        let ids = Tensor::new(letter_ids, self.embed_tokens.device())?;
        // Gather before projecting: never materialize full-vocabulary logits.
        let embeddings = self
            .embed_tokens
            .index_select(&ids, 0)?
            .to_device(hidden.device())?;
        let logits = last
            .matmul(&embeddings.t()?)?
            .squeeze(0)?
            .to_dtype(DType::F32)?;
        let probabilities = ops::softmax_last_dim(&logits)?.to_vec1::<f32>()?;
        Ok(CuaS1Prediction {
            logits: logits.to_vec1::<f32>()?,
            probabilities,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, path::Path};

    use candle_core::{
        Device,
        safetensors::{self, MmapedSafetensors},
    };
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

    fn load_model(device: &Device, dtype: DType) -> TextModel {
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
            device,
            dtype,
        )
        .unwrap();
        let model = TextModel::load(&mut weights, &config).unwrap();
        assert_eq!(model.layers.len(), 32);
        model
    }

    fn sample_model() -> TextModel {
        let config: TextConfig = serde_json::from_value(serde_json::json!({
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 4,
            "layer_types": ["linear_attention", "full_attention", "linear_attention", "full_attention"],
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "rms_norm_eps": 1e-6,
            "vocab_size": 32,
            "tie_word_embeddings": true,
            "attn_output_gate": true,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_key_head_dim": 4,
            "linear_value_head_dim": 4,
            "linear_conv_kernel_dim": 4,
            "rope_parameters": {
                "rope_theta": 10_000_000.,
                "partial_rotary_factor": 0.25
            }
        }))
        .unwrap();
        let mut tensors = HashMap::new();
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut sample_weight = |name: String, dims: &[usize]| {
            let values: Vec<_> = (0..dims.iter().product::<usize>())
                .map(|_| {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    ((seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.5
                })
                .collect();
            tensors.insert(
                name,
                Tensor::from_vec(values, dims, &Device::Cpu).unwrap(),
            );
        };
        sample_weight(
            "model.language_model.embed_tokens.weight".into(),
            &[32, 16],
        );
        sample_weight("model.language_model.norm.weight".into(), &[16]);
        for (layer, kind) in config.layer_types.iter().enumerate() {
            let prefix = format!("model.language_model.layers.{layer}");
            for (name, dims) in [
                ("input_layernorm.weight", &[16][..]),
                ("post_attention_layernorm.weight", &[16][..]),
                ("mlp.gate_proj.weight", &[32, 16][..]),
                ("mlp.up_proj.weight", &[32, 16][..]),
                ("mlp.down_proj.weight", &[16, 32][..]),
            ] {
                sample_weight(format!("{prefix}.{name}"), dims);
            }
            let mixer_weights: &[(&str, &[usize])] = match kind {
                LayerType::LinearAttention => &[
                    ("linear_attn.in_proj_qkv.weight", &[32, 16]),
                    ("linear_attn.in_proj_z.weight", &[16, 16]),
                    ("linear_attn.in_proj_b.weight", &[4, 16]),
                    ("linear_attn.in_proj_a.weight", &[4, 16]),
                    ("linear_attn.out_proj.weight", &[16, 16]),
                    ("linear_attn.conv1d.weight", &[32, 1, 4]),
                    ("linear_attn.A_log", &[4]),
                    ("linear_attn.dt_bias", &[4]),
                    ("linear_attn.norm.weight", &[4]),
                ],
                LayerType::FullAttention => &[
                    ("self_attn.q_proj.weight", &[64, 16]),
                    ("self_attn.k_proj.weight", &[16, 16]),
                    ("self_attn.v_proj.weight", &[16, 16]),
                    ("self_attn.o_proj.weight", &[16, 32]),
                    ("self_attn.q_norm.weight", &[8]),
                    ("self_attn.k_norm.weight", &[8]),
                ],
            };
            for (name, dims) in mixer_weights {
                sample_weight(format!("{prefix}.{name}"), dims);
            }
        }
        let directory = std::env::temp_dir().join(format!(
            "vs1-cua-s1-prefix-{}-{:?}",
            std::process::id(),
            std::thread::current().id(),
        ));
        std::fs::create_dir_all(&directory).unwrap();
        safetensors::save(&tensors, directory.join("model.safetensors"))
            .unwrap();
        let weight_map: HashMap<_, _> = tensors
            .keys()
            .map(|name| (name, "model.safetensors"))
            .collect();
        std::fs::write(
            directory.join("model.safetensors.index.json"),
            serde_json::to_vec(&serde_json::json!({"weight_map": weight_map}))
                .unwrap(),
        )
        .unwrap();
        let mut weights =
            TextWeights::load(&directory, None, &Device::Cpu, DType::F32)
                .unwrap();
        let model = TextModel::load(&mut weights, &config).unwrap();
        drop(weights);
        std::fs::remove_dir_all(directory).unwrap();
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
    fn reuses_prefix_state_for_different_suffixes() {
        let model = sample_model();
        let ids = [1, 5, 9, 2, 14, 7, 3, 18, 6, 21, 12, 4];
        let letter_ids = [8, 2, 19, 5];
        for split in [1, 2, 3, 4, 7, 11] {
            let prefix = model.encode_prefix(&ids[..split]).unwrap();
            assert_eq!(prefix.token_count(), split);
            assert_eq!(prefix.layer_states.len(), model.layers.len());
            let mut predictions = Vec::new();
            for suffix in [&ids[split..], &[23, 17, 6][..], &ids[split..]] {
                let input = CuaS1Input {
                    chat_text: String::new(),
                    input_ids: [&ids[..split], suffix].concat(),
                    letters: vec!['A', 'B', 'C', 'D'],
                    letter_ids: letter_ids.to_vec(),
                };
                let whole = model.forward(&input).unwrap();
                let prediction =
                    model.score_suffix(&prefix, suffix, &letter_ids).unwrap();
                let difference = max_absolute_difference(
                    &prediction.probabilities,
                    &whole.probabilities,
                );
                println!(
                    "split={split}, suffix={suffix:?}: max probability difference {difference:e}"
                );
                assert!(difference <= 1e-5, "split={split}: exceeds atol=1e-5");
                predictions.push(prediction.probabilities);
            }
            assert_eq!(predictions[0], predictions[2]);
        }
    }

    #[test]
    fn rejects_invalid_prefix_and_suffix_inputs() {
        let model = sample_model();
        assert!(model.encode_prefix(&[]).is_err());
        assert!(model.encode_prefix(&[32]).is_err());
        let prefix = model.encode_prefix(&[1, 2]).unwrap();
        for (suffix, letters) in [
            (&[][..], &[1][..]),
            (&[32][..], &[1][..]),
            (&[3][..], &[][..]),
            (&[3][..], &[32][..]),
            (&[3][..], &[1; 27][..]),
        ] {
            assert!(model.score_suffix(&prefix, suffix, letters).is_err());
        }
        assert_eq!(
            model
                .score_suffix(&prefix, &[3], &[1])
                .unwrap()
                .probabilities,
            [1.]
        );
        let mut incomplete = prefix.clone();
        incomplete.layer_states.pop();
        assert!(model.score_suffix(&incomplete, &[3], &[1]).is_err());
        let mut mismatched = prefix.clone();
        mismatched.layer_states.swap(0, 1);
        assert!(model.score_suffix(&mismatched, &[3], &[1]).is_err());
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires CUDA"]
    fn prefix_matches_whole_fixture_predictions_on_cuda_bf16() {
        let device = Device::new_cuda(0).unwrap();
        let model = load_model(&device, DType::BF16);
        let tokenizer = tokenizers::Tokenizer::from_file(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../artifacts/cua-s1/base")
                .join(super::super::BASE_REVISION)
                .join("tokenizer.json"),
        )
        .unwrap();
        let mut has_matching_top_options = true;
        let mut max_probability_difference = 0f32;
        for prompt in load_prompts() {
            let name = prompt.name.clone();
            let input = prompt.into_input();
            let options_start =
                input.chat_text.find("\nOptions:\n").unwrap() + 1;
            let prefix_ids = tokenizer
                .encode(&input.chat_text[..options_start], true)
                .unwrap();
            let split = prefix_ids.len();
            assert_eq!(prefix_ids.get_ids(), &input.input_ids[..split]);
            let prefix = model.encode_prefix(prefix_ids.get_ids()).unwrap();
            let prediction = model
                .score_suffix(
                    &prefix,
                    &input.input_ids[split..],
                    &input.letter_ids,
                )
                .unwrap();
            let whole = model.forward(&input).unwrap();
            let difference = max_absolute_difference(
                &prediction.probabilities,
                &whole.probabilities,
            );
            max_probability_difference =
                max_probability_difference.max(difference);
            let top_option = |values: &[f32]| {
                values
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.total_cmp(b))
                    .unwrap()
                    .0
            };
            let top = top_option(&prediction.probabilities);
            let expected_top = top_option(&whole.probabilities);
            let has_matching_top_option = top == expected_top;
            has_matching_top_options &= has_matching_top_option;
            let tolerance = 1. / 64.;
            let is_close = prediction
                .probabilities
                .iter()
                .zip(&whole.probabilities)
                .all(|(actual, expected)| {
                    (actual - expected).abs()
                        <= tolerance + tolerance * expected.abs()
                });
            println!(
                "{name}: split={split}, max probability difference {difference:e}, top {} (whole {}), top agrees={has_matching_top_option}, within atol=rtol=2^-6: {is_close}",
                input.letters[top], input.letters[expected_top]
            );
        }
        println!(
            "all 6 cases: max probability difference {max_probability_difference:e}"
        );
        assert!(
            has_matching_top_options,
            "top option differs from whole-prompt forward"
        );
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
        let model = load_model(&Device::Cpu, DType::F32);
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
        let model = load_model(&Device::Cpu, DType::F32);
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
