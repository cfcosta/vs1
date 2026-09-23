//! Full text attention from Transformers 5.17.0's Qwen3.5 implementation.

use candle_core::{D, DType, Device, Result, Tensor};
use candle_nn::{Linear, Module, ops};

use super::{LayerType, TextConfig, TextWeights, normalize_rms};
use crate::SystemOneError;

/// Gated, causal grouped-query attention for one CPU/F32 sequence.
pub struct FullAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: Tensor,
    k_norm: Tensor,
    inv_freq: Tensor,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rms_norm_eps: f64,
}

impl FullAttention {
    /// Loads projections with merged LoRA weights and unmerged norm weights.
    pub fn load(
        weights: &mut TextWeights,
        config: &TextConfig,
        layer: usize,
    ) -> crate::Result<Self> {
        if config.layer_types.get(layer) != Some(&LayerType::FullAttention) {
            return Err(SystemOneError::Config(format!(
                "layer {layer} is not a full-attention layer"
            )));
        }
        let rotary_dim =
            (config.head_dim as f64 * config.partial_rotary_factor()) as usize;
        if rotary_dim == 0
            || !rotary_dim.is_multiple_of(2)
            || rotary_dim > config.head_dim
        {
            return Err(SystemOneError::Config(
                "rotary dimension must be positive, even and at most head_dim"
                    .into(),
            ));
        }
        let prefix = format!("model.language_model.layers.{layer}.self_attn");
        let q_weight =
            weights.linear_weight(&format!("{prefix}.q_proj.weight"))?;
        if !q_weight.device().is_cpu() || q_weight.dtype() != DType::F32 {
            return Err(SystemOneError::Config(
                "full attention requires CPU/F32 text weights".into(),
            ));
        }
        let inv_freq: Vec<_> = (0..rotary_dim)
            .step_by(2)
            .map(|i| {
                (config.rope_theta() as f32)
                    .powf(i as f32 / rotary_dim as f32)
                    .recip()
            })
            .collect();
        Ok(Self {
            q_proj: Linear::new(q_weight, None),
            k_proj: Linear::new(
                weights.linear_weight(&format!("{prefix}.k_proj.weight"))?,
                None,
            ),
            v_proj: Linear::new(
                weights.linear_weight(&format!("{prefix}.v_proj.weight"))?,
                None,
            ),
            o_proj: Linear::new(
                weights.linear_weight(&format!("{prefix}.o_proj.weight"))?,
                None,
            ),
            q_norm: weights.tensor(&format!("{prefix}.q_norm.weight"))?,
            k_norm: weights.tensor(&format!("{prefix}.k_norm.weight"))?,
            inv_freq: Tensor::new(inv_freq.as_slice(), &Device::Cpu)?,
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            rms_norm_eps: config.rms_norm_eps,
        })
    }

    fn attend(
        &self,
        query_gate: &Tensor,
        key: &Tensor,
        value: &Tensor,
    ) -> Result<Tensor> {
        let seq = query_gate.dim(0)?;
        let query_gate =
            query_gate.reshape((seq, self.num_heads, 2 * self.head_dim))?;
        // The gate follows each head's query, not all queries together.
        let query = query_gate.narrow(D::Minus1, 0, self.head_dim)?;
        let gate = query_gate
            .narrow(D::Minus1, self.head_dim, self.head_dim)?
            .reshape((seq, self.num_heads * self.head_dim))?;
        let query = normalize_rms(&query, &self.q_norm, self.rms_norm_eps)?
            .transpose(0, 1)?;
        let key = key.reshape((seq, self.num_kv_heads, self.head_dim))?;
        let key = normalize_rms(&key, &self.k_norm, self.rms_norm_eps)?
            .transpose(0, 1)?;
        let value = value
            .reshape((seq, self.num_kv_heads, self.head_dim))?
            .transpose(0, 1)?;

        // Text gives all three mRoPE axes the same positions, starting at zero.
        let positions = Tensor::arange(0f32, seq as f32, &Device::Cpu)?;
        let freqs = positions
            .unsqueeze(1)?
            .broadcast_mul(&self.inv_freq.unsqueeze(0)?)?;
        let freqs = Tensor::cat(&[&freqs, &freqs], D::Minus1)?;
        let cos = freqs.cos()?;
        let sin = freqs.sin()?;
        let query = apply_rotary(&query, &cos, &sin)?;
        let key = apply_rotary(&key, &cos, &sin)?;
        let groups = self.num_heads / self.num_kv_heads;
        let key = repeat_kv(&key, groups)?;
        let value = repeat_kv(&value, groups)?;

        let scores = (query.matmul(&key.transpose(1, 2)?)?
            * (self.head_dim as f64).sqrt().recip())?;
        let mask: Vec<_> = (0..seq)
            .flat_map(|row| {
                (0..seq).map(
                    move |col| {
                        if col > row { f32::NEG_INFINITY } else { 0. }
                    },
                )
            })
            .collect();
        let mask = Tensor::from_vec(mask, (seq, seq), &Device::Cpu)?;
        let probabilities =
            ops::softmax_last_dim(&scores.broadcast_add(&mask)?)?;
        let output = probabilities
            .matmul(&value)?
            .transpose(0, 1)?
            .reshape((seq, self.num_heads * self.head_dim))?;
        output * ops::sigmoid(&gate)?
    }
}

impl Module for FullAttention {
    /// Mixes an unpadded `[seq, hidden_size]` sequence at positions `0..seq`.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (seq, _) = x.dims2()?;
        if seq == 0 || !x.device().is_cpu() || x.dtype() != DType::F32 {
            candle_core::bail!(
                "full attention requires a nonempty CPU/F32 sequence"
            )
        }
        let query_gate = self.q_proj.forward(x)?;
        let key = self.k_proj.forward(x)?;
        let value = self.v_proj.forward(x)?;
        self.o_proj
            .forward(&self.attend(&query_gate, &key, &value)?)
    }
}

fn apply_rotary(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let rotary_dim = cos.dim(D::Minus1)?;
    let rotary = x.narrow(D::Minus1, 0, rotary_dim)?;
    let first = rotary.narrow(D::Minus1, 0, rotary_dim / 2)?;
    let second = rotary.narrow(D::Minus1, rotary_dim / 2, rotary_dim / 2)?;
    let rotated = Tensor::cat(&[&second.neg()?, &first], D::Minus1)?;
    let rotary = (rotary.broadcast_mul(cos)? + rotated.broadcast_mul(sin)?)?;
    let tail =
        x.narrow(D::Minus1, rotary_dim, x.dim(D::Minus1)? - rotary_dim)?;
    Tensor::cat(&[&rotary, &tail], D::Minus1)
}

fn repeat_kv(x: &Tensor, groups: usize) -> Result<Tensor> {
    let (heads, seq, head_dim) = x.dims3()?;
    x.unsqueeze(1)?
        .broadcast_as((heads, groups, seq, head_dim))?
        .reshape((heads * groups, seq, head_dim))
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use candle_core::safetensors::MmapedSafetensors;

    use super::*;

    fn assert_close(name: &str, output: &Tensor, expected: &Tensor) {
        assert_eq!(output.dtype(), DType::F32);
        assert_eq!(output.dims(), expected.dims(), "{name}");
        let output = output.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let expected =
            expected.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let mut max_absolute_difference = 0f32;
        let mut max_relative_difference = 0f32;
        let mut is_close = true;
        for (actual, expected) in output.into_iter().zip(expected) {
            let difference = (actual - expected).abs();
            max_absolute_difference = max_absolute_difference.max(difference);
            max_relative_difference = max_relative_difference
                .max(difference / expected.abs().max(1e-6));
            is_close &= actual.is_finite()
                && expected.is_finite()
                && difference <= 1e-4 + 1e-4 * expected.abs();
        }
        println!(
            "{name}: max absolute difference {max_absolute_difference:e}, max relative difference {max_relative_difference:e} (denominator floored at 1e-6)"
        );
        assert!(is_close, "{name}: exceeds atol=1e-4, rtol=1e-4");
    }

    #[test]
    fn rotates_halves_of_only_the_rotary_dimensions() {
        let x = Tensor::new(
            &[[[1f32, 2., 3., 4., 5., 6.], [7., 8., 9., 10., 11., 12.]]],
            &Device::Cpu,
        )
        .unwrap();
        let cos = Tensor::new(&[[1f32; 4], [0.; 4]], &Device::Cpu).unwrap();
        let sin = Tensor::new(&[[0f32; 4], [1.; 4]], &Device::Cpu).unwrap();
        let expected = Tensor::new(
            &[[[1f32, 2., 3., 4., 5., 6.], [-9., -10., 7., 8., 11., 12.]]],
            &Device::Cpu,
        )
        .unwrap();
        assert_close(
            "partial RoPE",
            &apply_rotary(&x, &cos, &sin).unwrap(),
            &expected,
        );
    }

    #[test]
    fn attends_causally_with_grouped_keys_and_per_head_gates() {
        let mut q_weight = vec![0f32; 16 * 4];
        for (head, gate) in
            [0., 3f32.ln(), -3f32.ln(), 0.].into_iter().enumerate()
        {
            for dim in 2..4 {
                q_weight[(head * 4 + dim) * 4] = gate;
            }
        }
        let attention = FullAttention {
            q_proj: Linear::new(
                Tensor::from_vec(q_weight, (16, 4), &Device::Cpu).unwrap(),
                None,
            ),
            k_proj: Linear::new(
                Tensor::zeros((4, 4), DType::F32, &Device::Cpu).unwrap(),
                None,
            ),
            v_proj: Linear::new(
                Tensor::eye(4, DType::F32, &Device::Cpu).unwrap(),
                None,
            ),
            o_proj: Linear::new(
                Tensor::new(
                    &[
                        [1f32, 0., 0., 2., 0., 0., 0., 0.],
                        [0., 1., 0., 0., 0., 0., -1., 0.],
                        [0., 0., 1., 0., 1., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 1., 0., 3.],
                    ],
                    &Device::Cpu,
                )
                .unwrap(),
                None,
            ),
            q_norm: Tensor::zeros(2, DType::F32, &Device::Cpu).unwrap(),
            k_norm: Tensor::zeros(2, DType::F32, &Device::Cpu).unwrap(),
            inv_freq: Tensor::new(&[1f32], &Device::Cpu).unwrap(),
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 2,
            rms_norm_eps: 1e-6,
        };
        let x =
            Tensor::new(&[[1f32, 2., 3., 4.], [1., 6., 7., 8.]], &Device::Cpu)
                .unwrap();
        // Zero keys give uniform attention over the prefix. KV heads repeat
        // consecutively; the four query heads have gates 1/2, 3/4, 1/4, 1/2.
        let expected = Tensor::new(
            &[[3.5f32, -0.5, 1.5, 7.], [6.5, -0.5, 2., 10.5]],
            &Device::Cpu,
        )
        .unwrap();
        for seq in [1, 2] {
            assert_close(
                "causal grouped attention",
                &attention.forward(&x.narrow(0, 0, seq).unwrap()).unwrap(),
                &expected.narrow(0, 0, seq).unwrap(),
            );
        }
        assert!(attention.forward(&x.narrow(0, 0, 0).unwrap()).is_err());
        assert!(
            attention
                .forward(&x.to_dtype(DType::BF16).unwrap())
                .is_err()
        );
        assert!(attention.forward(&x.unsqueeze(0).unwrap()).is_err());
    }

    #[test]
    #[ignore = "requires the pinned local base, adapter and layer dump in artifacts/cua-s1"]
    fn reproduces_dumped_layer_three_full_attention() {
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
        let attention = FullAttention::load(&mut weights, &config, 3).unwrap();
        let input_norm = weights
            .tensor("model.language_model.layers.3.input_layernorm.weight")
            .unwrap();
        // SAFETY: the reference dump is mapped read-only and not modified.
        let dump = unsafe {
            MmapedSafetensors::new(root.join("layers-f32.safetensors"))
        }
        .unwrap();
        let load = |name: &str| {
            dump.load(name, &Device::Cpu).unwrap().squeeze(0).unwrap()
        };
        let input = normalize_rms(
            &load("layers.3.input"),
            &input_norm,
            config.rms_norm_eps,
        )
        .unwrap();
        let query_gate = attention.q_proj.forward(&input).unwrap();
        let key = attention.k_proj.forward(&input).unwrap();
        let value = attention.v_proj.forward(&input).unwrap();
        for (name, output) in [
            ("layers.3.self_attn.q_proj", &query_gate),
            ("layers.3.self_attn.k_proj", &key),
            ("layers.3.self_attn.v_proj", &value),
        ] {
            assert_close(name, output, &load(name));
        }
        let gated = attention.attend(&query_gate, &key, &value).unwrap();
        assert_close(
            "layers.3.self_attn.o_proj.input",
            &gated,
            &load("layers.3.self_attn.o_proj.input"),
        );
        assert_close(
            "layers.3.mixer_output",
            &attention.forward(&input).unwrap(),
            &load("layers.3.mixer_output"),
        );
    }
}
