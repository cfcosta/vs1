//! Recurrent text DeltaNet from Transformers 5.17.0's Qwen3.5 implementation.

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Linear, Module, ops};

use super::{
    LayerType,
    TextConfig,
    TextWeights,
    normalize_l2,
    normalize_rms_gated,
};
use crate::SystemOneError;

/// Gated delta-rule attention for one sequence, without a cache.
pub struct GatedDeltaNet {
    in_proj_qkv: Linear,
    in_proj_z: Linear,
    in_proj_b: Linear,
    in_proj_a: Linear,
    out_proj: Linear,
    conv_weight: Tensor,
    a_log: Tensor,
    dt_bias: Tensor,
    norm_weight: Tensor,
    num_key_heads: usize,
    num_value_heads: usize,
    key_head_dim: usize,
    value_head_dim: usize,
    rms_norm_eps: f64,
}

impl GatedDeltaNet {
    /// Loads projections with merged LoRA weights and unmerged mixer weights.
    pub fn load(
        weights: &mut TextWeights,
        config: &TextConfig,
        layer: usize,
    ) -> crate::Result<Self> {
        if config.layer_types.get(layer) != Some(&LayerType::LinearAttention) {
            return Err(SystemOneError::Config(format!(
                "layer {layer} is not a linear-attention layer"
            )));
        }
        if config.linear_num_key_heads == 0
            || config.linear_num_value_heads == 0
            || !config
                .linear_num_value_heads
                .is_multiple_of(config.linear_num_key_heads)
            || config.linear_key_head_dim == 0
            || config.linear_value_head_dim == 0
            || config.linear_conv_kernel_dim == 0
        {
            return Err(SystemOneError::Config(
                "DeltaNet requires positive dimensions and value heads divisible by key heads".into(),
            ));
        }
        let prefix = format!("model.language_model.layers.{layer}.linear_attn");
        let qkv_weight =
            weights.linear_weight(&format!("{prefix}.in_proj_qkv.weight"))?;
        let conv_weight = weights.tensor(&format!("{prefix}.conv1d.weight"))?;
        let channels =
            2 * config.linear_num_key_heads * config.linear_key_head_dim
                + config.linear_num_value_heads * config.linear_value_head_dim;
        if conv_weight.dims() != [channels, 1, config.linear_conv_kernel_dim] {
            return Err(SystemOneError::Config(
                "DeltaNet convolution dimensions do not match the text config"
                    .into(),
            ));
        }
        Ok(Self {
            in_proj_qkv: Linear::new(qkv_weight, None),
            in_proj_z: Linear::new(
                weights.linear_weight(&format!("{prefix}.in_proj_z.weight"))?,
                None,
            ),
            in_proj_b: Linear::new(
                weights.linear_weight(&format!("{prefix}.in_proj_b.weight"))?,
                None,
            ),
            in_proj_a: Linear::new(
                weights.linear_weight(&format!("{prefix}.in_proj_a.weight"))?,
                None,
            ),
            out_proj: Linear::new(
                weights.linear_weight(&format!("{prefix}.out_proj.weight"))?,
                None,
            ),
            conv_weight,
            a_log: weights.tensor(&format!("{prefix}.A_log"))?,
            dt_bias: weights.tensor(&format!("{prefix}.dt_bias"))?,
            norm_weight: weights.tensor(&format!("{prefix}.norm.weight"))?,
            num_key_heads: config.linear_num_key_heads,
            num_value_heads: config.linear_num_value_heads,
            key_head_dim: config.linear_key_head_dim,
            value_head_dim: config.linear_value_head_dim,
            rms_norm_eps: config.rms_norm_eps,
        })
    }

    fn mix(&self, mixed: &Tensor, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let seq = mixed.dim(0)?;
        let mixed = convolve_causally(mixed, &self.conv_weight)?.silu()?;
        let key_dim = self.num_key_heads * self.key_head_dim;
        let value_dim = self.num_value_heads * self.value_head_dim;
        let query = mixed.narrow(1, 0, key_dim)?.reshape((
            seq,
            self.num_key_heads,
            self.key_head_dim,
        ))?;
        let key = mixed.narrow(1, key_dim, key_dim)?.reshape((
            seq,
            self.num_key_heads,
            self.key_head_dim,
        ))?;
        let value = mixed.narrow(1, 2 * key_dim, value_dim)?.reshape((
            seq,
            self.num_value_heads,
            self.value_head_dim,
        ))?;
        let groups = self.num_value_heads / self.num_key_heads;
        let query = repeat_key_heads(&query, groups)?.to_dtype(DType::F32)?;
        let key = repeat_key_heads(&key, groups)?.to_dtype(DType::F32)?;
        let query = (normalize_l2(&query)?
            * (self.key_head_dim as f64).sqrt().recip())?;
        let key = normalize_l2(&key)?;
        let value = value.to_dtype(DType::F32)?.contiguous()?;
        let beta = ops::sigmoid(b)?.to_dtype(DType::F32)?;
        let g = compute_log_decay(a, &self.a_log, &self.dt_bias)?;
        let decay = g.exp()?;
        #[cfg(feature = "cuda")]
        if mixed.device().is_cuda() {
            return super::delta_rule_cuda::apply_delta_rule(
                &query.contiguous()?,
                &key.contiguous()?,
                &value,
                &beta.contiguous()?,
                &decay.contiguous()?,
            )?
            .to_dtype(mixed.dtype());
        }
        apply_delta_rule(&query, &key, &value, &beta, &decay)?
            .to_dtype(mixed.dtype())
    }
}

impl Module for GatedDeltaNet {
    /// Mixes an unpadded `[seq, hidden_size]` sequence from a zero state.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (seq, _) = x.dims2()?;
        if seq == 0 {
            candle_core::bail!("DeltaNet requires a nonempty sequence")
        }
        let mixed = self.in_proj_qkv.forward(x)?;
        let a = self.in_proj_a.forward(x)?;
        let b = self.in_proj_b.forward(x)?;
        let output = self.mix(&mixed, &a, &b)?;
        let gate = self.in_proj_z.forward(x)?.reshape(output.shape())?;
        let output = normalize_rms_gated(
            &output,
            &self.norm_weight,
            &gate,
            self.rms_norm_eps,
        )?;
        self.out_proj.forward(
            &output
                .reshape((seq, self.num_value_heads * self.value_head_dim))?,
        )
    }
}

fn convolve_causally(x: &Tensor, weight: &Tensor) -> Result<Tensor> {
    let (seq, channels) = x.dims2()?;
    let (weight_channels, channels_per_group, kernel) = weight.dims3()?;
    if weight_channels != channels || channels_per_group != 1 || kernel == 0 {
        candle_core::bail!(
            "causal depthwise convolution requires [channels, 1, kernel] weights"
        )
    }
    // Accumulate in F32 like PyTorch, without launching a convolution per channel.
    let padded = x.to_dtype(DType::F32)?.pad_with_zeros(0, kernel - 1, 0)?;
    let weight = weight.to_dtype(DType::F32)?.squeeze(1)?;
    let mut output = Tensor::zeros((seq, channels), DType::F32, x.device())?;
    for tap in 0..kernel {
        let tap_weight = weight.narrow(1, tap, 1)?.squeeze(1)?;
        let product = padded.narrow(0, tap, seq)?.broadcast_mul(&tap_weight)?;
        output = (output + product)?;
    }
    output.to_dtype(x.dtype())
}

fn repeat_key_heads(x: &Tensor, groups: usize) -> Result<Tensor> {
    let (seq, heads, head_dim) = x.dims3()?;
    // Each key head serves consecutive value heads, rather than a tiled block.
    x.unsqueeze(2)?
        .broadcast_as((seq, heads, groups, head_dim))?
        .reshape((seq, heads * groups, head_dim))
}

fn compute_log_decay(
    a: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
) -> Result<Tensor> {
    let shifted = a
        .to_dtype(DType::F32)?
        .broadcast_add(&dt_bias.to_dtype(DType::F32)?)?;
    // PyTorch softplus switches to its linear branch above 20.
    let softplus =
        (shifted.clamp(f32::NEG_INFINITY, 20.)?.exp()? + 1.)?.log()?;
    let softplus = shifted.gt(20.)?.where_cond(&shifted, &softplus)?;
    softplus.broadcast_mul(&a_log.to_dtype(DType::F32)?.exp()?.neg()?)
}

/// Applies the recurrence from a zero state with precomputed `exp(g)` decay.
pub(crate) fn apply_delta_rule(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    beta: &Tensor,
    decay: &Tensor,
) -> Result<Tensor> {
    let (seq, heads, key_dim) = query.dims3()?;
    let (value_seq, value_heads, value_dim) = value.dims3()?;
    if key.shape() != query.shape()
        || (value_seq, value_heads) != (seq, heads)
        || beta.dims() != [seq, heads]
        || decay.dims() != [seq, heads]
    {
        candle_core::bail!(
            "delta-rule query, key, value and gate dimensions do not match"
        )
    }
    let query = query.to_vec3::<f32>()?;
    let key = key.to_vec3::<f32>()?;
    let value = value.to_vec3::<f32>()?;
    let beta = beta.to_vec2::<f32>()?;
    let decay = decay.to_vec2::<f32>()?;
    let mut output = vec![0f32; seq * heads * value_dim];
    for head in 0..heads {
        let mut state = vec![0f32; key_dim * value_dim];
        let mut delta = vec![0f32; value_dim];
        for t in 0..seq {
            for entry in &mut state {
                *entry *= decay[t][head];
            }
            delta.fill(0.);
            for (row, &key) in state.chunks_exact(value_dim).zip(&key[t][head])
            {
                for (memory, &entry) in delta.iter_mut().zip(row) {
                    *memory += key * entry;
                }
            }
            for (delta, &value) in delta.iter_mut().zip(&value[t][head]) {
                *delta = (value - *delta) * beta[t][head];
            }
            let offset = (t * heads + head) * value_dim;
            let token_output = &mut output[offset..offset + value_dim];
            for ((row, &key), &query) in state
                .chunks_exact_mut(value_dim)
                .zip(&key[t][head])
                .zip(&query[t][head])
            {
                for ((entry, &delta), output) in
                    row.iter_mut().zip(&delta).zip(token_output.iter_mut())
                {
                    *entry += key * delta;
                    *output += query * *entry;
                }
            }
        }
    }
    Tensor::from_vec(output, (seq, heads, value_dim), &Device::Cpu)
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use candle_core::safetensors::MmapedSafetensors;

    use super::*;
    use crate::cua_s1::normalize_rms;

    fn assert_close(
        name: &str,
        output: &Tensor,
        expected: &Tensor,
        tolerance: f32,
    ) {
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
                && difference <= tolerance + tolerance * expected.abs();
        }
        println!(
            "{name}: max absolute difference {max_absolute_difference:e}, max relative difference {max_relative_difference:e} (denominator floored at 1e-6)"
        );
        assert!(
            is_close,
            "{name}: exceeds atol={tolerance}, rtol={tolerance}"
        );
    }

    #[test]
    fn convolves_each_channel_with_left_padding_and_no_future_tokens() {
        let x = Tensor::new(
            &[[1f32, 10.], [2., 20.], [3., 30.], [4., 40.], [5., 50.]],
            &Device::Cpu,
        )
        .unwrap();
        let weight = Tensor::new(
            &[[[1f32, 2., 3., 4.]], [[2., 4., 6., 8.]]],
            &Device::Cpu,
        )
        .unwrap();
        let expected = Tensor::new(
            &[
                [4f32, 80.],
                [11., 220.],
                [20., 400.],
                [30., 600.],
                [40., 800.],
            ],
            &Device::Cpu,
        )
        .unwrap();
        for seq in [1, 3, 5] {
            assert_close(
                "causal convolution",
                &convolve_causally(&x.narrow(0, 0, seq).unwrap(), &weight)
                    .unwrap(),
                &expected.narrow(0, 0, seq).unwrap(),
                0.,
            );
        }
    }

    #[test]
    fn accumulates_bf16_convolution_in_f32_before_casting_back() {
        let x = Tensor::new(&[[256f32], [1.], [-256.], [1.]], &Device::Cpu)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let weight =
            Tensor::ones((1, 1, 4), DType::BF16, &Device::Cpu).unwrap();
        let expected =
            Tensor::new(&[[256f32], [256.], [1.], [2.]], &Device::Cpu).unwrap();
        let output = convolve_causally(&x, &weight).unwrap();
        assert_eq!(output.dtype(), DType::BF16);
        assert_close(
            "BF16 causal convolution",
            &output.to_dtype(DType::F32).unwrap(),
            &expected,
            0.,
        );
    }

    #[test]
    fn repeats_key_heads_consecutively_for_each_token() {
        let x = Tensor::new(
            &[[[1f32, 2.], [3., 4.]], [[5., 6.], [7., 8.]]],
            &Device::Cpu,
        )
        .unwrap();
        let expected = Tensor::new(
            &[
                [[1f32, 2.], [1., 2.], [3., 4.], [3., 4.]],
                [[5., 6.], [5., 6.], [7., 8.], [7., 8.]],
            ],
            &Device::Cpu,
        )
        .unwrap();
        assert_close(
            "repeated key heads",
            &repeat_key_heads(&x, 2).unwrap(),
            &expected,
            0.,
        );
    }

    #[test]
    fn computes_log_decay_with_bias_and_stable_softplus() {
        let a =
            Tensor::new(&[[-1000f32, 0.], [0., 1000.]], &Device::Cpu).unwrap();
        let a_log = Tensor::new(&[0f32, 2f32.ln()], &Device::Cpu).unwrap();
        let dt_bias = Tensor::new(&[0f32, 3f32.ln()], &Device::Cpu).unwrap();
        let expected = Tensor::new(
            &[
                [0f32, -2. * 4f32.ln()],
                [-2f32.ln(), -2. * (1000. + 3f32.ln())],
            ],
            &Device::Cpu,
        )
        .unwrap();
        assert_close(
            "decay",
            &compute_log_decay(&a, &a_log, &dt_bias).unwrap(),
            &expected,
            1e-6,
        );
    }

    #[test]
    fn computes_log_decay_in_f32_before_adding_bf16_bias() {
        let a = Tensor::new(&[[20f32]], &Device::Cpu)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let a_log = Tensor::zeros(1, DType::BF16, &Device::Cpu).unwrap();
        let dt_bias = Tensor::new(&[0.015625f32], &Device::Cpu)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let output = compute_log_decay(&a, &a_log, &dt_bias).unwrap();
        assert_eq!(output.dtype(), DType::F32);
        // BF16 addition would round the shifted input back to 20.
        assert_eq!(output.to_vec2::<f32>().unwrap(), [[-20.015625]]);
    }

    #[test]
    fn decays_state_before_delta_updates_and_reads_each_head_independently() {
        let query = Tensor::new(
            &[
                [[1f32, 0.], [0., 1.]],
                [[0., 1.], [1., 0.]],
                [[1., 1.], [1., 1.]],
            ],
            &Device::Cpu,
        )
        .unwrap();
        let key = Tensor::new(
            &[
                [[1f32, 0.], [0., 1.]],
                [[1., 1.], [1., 0.]],
                [[0., 1.], [1., 0.]],
            ],
            &Device::Cpu,
        )
        .unwrap();
        let value = Tensor::new(
            &[
                [[2f32, 4., 6.], [8., 6., 4.]],
                [[6., 8., 10.], [4., 6., 8.]],
                [[10., 12., 14.], [0., 0., 0.]],
            ],
            &Device::Cpu,
        )
        .unwrap();
        let beta =
            Tensor::new(&[[0.5f32, 1.], [0.5, 0.25], [0., 1.]], &Device::Cpu)
                .unwrap();
        let g = Tensor::new(
            &[[0f32, 0.], [0.5f32.ln(), 0.], [0.25f32.ln(), 0.]],
            &Device::Cpu,
        )
        .unwrap();
        let decay = g.exp().unwrap();
        let expected = Tensor::new(
            &[
                [[1f32, 2., 3.], [8., 6., 4.]],
                [[2.75, 3.5, 4.25], [1., 1.5, 2.]],
                [[1.5, 2., 2.5], [8., 6., 4.]],
            ],
            &Device::Cpu,
        )
        .unwrap();
        for seq in [1, 2, 3] {
            let prefix = |x: &Tensor| x.narrow(0, 0, seq).unwrap();
            assert_close(
                "delta recurrence",
                &apply_delta_rule(
                    &prefix(&query),
                    &prefix(&key),
                    &prefix(&value),
                    &prefix(&beta),
                    &prefix(&decay),
                )
                .unwrap(),
                &prefix(&expected),
                1e-6,
            );
        }
    }

    #[test]
    #[ignore = "requires the pinned local base, adapter and layer dump in artifacts/cua-s1"]
    fn reproduces_dumped_layer_zero_linear_attention() {
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
        let mixer = GatedDeltaNet::load(&mut weights, &config, 0).unwrap();
        let input_norm = weights
            .tensor("model.language_model.layers.0.input_layernorm.weight")
            .unwrap();
        // SAFETY: the reference dump is mapped read-only and not modified.
        let dump = unsafe {
            MmapedSafetensors::new(root.join("layers-f32.safetensors"))
        }
        .unwrap();
        let load = |name: &str| dump.load(name, &Device::Cpu).unwrap();
        let input = normalize_rms(
            &load("layers.0.input").squeeze(0).unwrap(),
            &input_norm,
            config.rms_norm_eps,
        )
        .unwrap();
        let mixed = mixer.in_proj_qkv.forward(&input).unwrap();
        let z = mixer.in_proj_z.forward(&input).unwrap();
        let a = mixer.in_proj_a.forward(&input).unwrap();
        let b = mixer.in_proj_b.forward(&input).unwrap();
        for (name, output) in [
            ("layers.0.linear_attn.in_proj_qkv", &mixed),
            ("layers.0.linear_attn.in_proj_z", &z),
            ("layers.0.linear_attn.in_proj_a", &a),
            ("layers.0.linear_attn.in_proj_b", &b),
        ] {
            assert_close(name, output, &load(name).squeeze(0).unwrap(), 1e-4);
        }
        let seq = input.dim(0).unwrap();
        let norm_shape = (
            seq * config.linear_num_value_heads,
            config.linear_value_head_dim,
        );
        let output = mixer
            .mix(&mixed, &a, &b)
            .unwrap()
            .reshape(norm_shape)
            .unwrap();
        let gate = z.reshape(norm_shape).unwrap();
        assert_close(
            "layers.0.linear_attn.norm.gate",
            &gate,
            &load("layers.0.linear_attn.norm.gate"),
            1e-4,
        );
        // The dump uses the chunked rule, so recurrent rounding can accumulate.
        assert_close(
            "layers.0.linear_attn.norm.input",
            &output,
            &load("layers.0.linear_attn.norm.input"),
            1e-4,
        );
        let normalized = normalize_rms_gated(
            &output,
            &mixer.norm_weight,
            &gate,
            config.rms_norm_eps,
        )
        .unwrap();
        assert_close(
            "layers.0.linear_attn.norm.output",
            &normalized,
            &load("layers.0.linear_attn.norm.output"),
            1e-4,
        );
        let normalized_dump = normalize_rms_gated(
            &load("layers.0.linear_attn.norm.input"),
            &mixer.norm_weight,
            &load("layers.0.linear_attn.norm.gate"),
            config.rms_norm_eps,
        )
        .unwrap();
        assert_close(
            "gated norm with dumped inputs",
            &normalized_dump,
            &load("layers.0.linear_attn.norm.output"),
            1e-4,
        );
        assert_close(
            "layers.0.mixer_output",
            &mixer.forward(&input).unwrap(),
            &load("layers.0.mixer_output").squeeze(0).unwrap(),
            1e-4,
        );
    }
}
