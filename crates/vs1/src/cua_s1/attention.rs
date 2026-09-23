//! Full text attention from Transformers 5.17.0's Qwen3.5 implementation.

use candle_core::{D, DType, Result, Tensor};
use candle_nn::{Linear, Module, ops};

use super::{LayerType, TextConfig, TextWeights, normalize_rms};
use crate::SystemOneError;

/// Saved keys and values for one full-attention sequence.
#[derive(Debug, Clone)]
pub struct KvCache {
    /// Keys after normalization and rotary, shaped `[kv_heads, seq, head_dim]`.
    pub key: Tensor,
    /// Projected values with shape `[kv_heads, seq, head_dim]`.
    pub value: Tensor,
}

/// Gated, causal grouped-query attention for one sequence.
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
        let inv_freq: Vec<_> = (0..rotary_dim)
            .step_by(2)
            .map(|i| {
                (config.rope_theta() as f32)
                    .powf(i as f32 / rotary_dim as f32)
                    .recip()
            })
            .collect();
        let inv_freq = Tensor::new(inv_freq.as_slice(), q_weight.device())?;
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
            inv_freq,
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
        self.attend_with_kernel(query_gate, key, value, None, attend_causally)
            .map(|(output, _)| output)
    }

    fn attend_with_kernel(
        &self,
        query_gate: &Tensor,
        key: &Tensor,
        value: &Tensor,
        cache: Option<&KvCache>,
        attend: impl FnOnce(&Tensor, &Tensor, &Tensor) -> Result<Tensor>,
    ) -> Result<(Tensor, KvCache)> {
        let seq = query_gate.dim(0)?;
        let cached_len = match cache {
            Some(cache) => {
                let (heads, cached_len, head_dim) = cache.key.dims3()?;
                if heads != self.num_kv_heads
                    || head_dim != self.head_dim
                    || cache.value.shape() != cache.key.shape()
                    || cache.key.dtype() != query_gate.dtype()
                    || cache.value.dtype() != query_gate.dtype()
                    || !cache.key.device().same_device(query_gate.device())
                    || !cache.value.device().same_device(query_gate.device())
                {
                    candle_core::bail!(
                        "full attention cache has incompatible shape, dtype or device"
                    )
                }
                cached_len
            }
            None => 0,
        };
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

        // Text gives all three mRoPE axes the same absolute positions.
        let positions = Tensor::arange(
            cached_len as f32,
            (cached_len + seq) as f32,
            query.device(),
        )?;
        let freqs = positions
            .unsqueeze(1)?
            .broadcast_mul(&self.inv_freq.unsqueeze(0)?)?;
        let freqs = Tensor::cat(&[&freqs, &freqs], D::Minus1)?;
        let cos = freqs.cos()?.to_dtype(query.dtype())?;
        let sin = freqs.sin()?.to_dtype(query.dtype())?;
        let query = apply_rotary(&query, &cos, &sin)?;
        let key = apply_rotary(&key, &cos, &sin)?;
        let (key, value) = match cache {
            Some(cache) => (
                Tensor::cat(&[&cache.key, &key], 1)?,
                Tensor::cat(&[&cache.value, &value], 1)?,
            ),
            None => (key, value),
        };
        let output = attend(&query, &key, &value)?
            .reshape((seq, self.num_heads * self.head_dim))?;
        Ok(((output * ops::sigmoid(&gate)?)?, KvCache { key, value }))
    }

    /// Continues an unpadded `[seq, hidden_size]` sequence from cached keys/values.
    pub fn forward_with_cache(
        &self,
        x: &Tensor,
        cache: Option<&KvCache>,
    ) -> Result<(Tensor, KvCache)> {
        let (seq, _) = x.dims2()?;
        if seq == 0 {
            candle_core::bail!("full attention requires a nonempty sequence")
        }
        let query_gate = self.q_proj.forward(x)?;
        let key = self.k_proj.forward(x)?;
        let value = self.v_proj.forward(x)?;
        let (output, cache) = self.attend_with_kernel(
            &query_gate,
            &key,
            &value,
            cache,
            attend_causally,
        )?;
        Ok((self.o_proj.forward(&output)?, cache))
    }
}

impl Module for FullAttention {
    /// Mixes an unpadded `[seq, hidden_size]` sequence at positions `0..seq`.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (seq, _) = x.dims2()?;
        if seq == 0 {
            candle_core::bail!("full attention requires a nonempty sequence")
        }
        let query_gate = self.q_proj.forward(x)?;
        let key = self.k_proj.forward(x)?;
        let value = self.v_proj.forward(x)?;
        self.o_proj
            .forward(&self.attend(&query_gate, &key, &value)?)
    }
}

fn attend_causally(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if query.device().is_cuda()
        && matches!(query.dtype(), DType::BF16 | DType::F16)
    {
        // Flash attention consumes [batch, seq, heads, dim] with grouped KV heads.
        // candle-flash-attn 0.11.0 kernels/mask.h aligns causal masks to the
        // bottom right: keys through query_row + key_len - query_len are visible.
        let head_dim = query.dim(D::Minus1)?;
        return candle_flash_attn::flash_attn(
            &query.transpose(0, 1)?.contiguous()?.unsqueeze(0)?,
            &key.transpose(0, 1)?.contiguous()?.unsqueeze(0)?,
            &value.transpose(0, 1)?.contiguous()?.unsqueeze(0)?,
            (head_dim as f32).sqrt().recip(),
            true,
        )?
        .squeeze(0);
    }
    attend_explicitly(query, key, value)
}

fn attend_explicitly(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
) -> Result<Tensor> {
    let (heads, seq, head_dim) = query.dims3()?;
    let key_seq = key.dim(1)?;
    let cached_len = key_seq - seq;
    // Materialize the per-head layouts for CUDA's batched matmul.
    let query = query.contiguous()?;
    let groups = heads / key.dim(0)?;
    let key = repeat_kv(key, groups)?.transpose(1, 2)?.contiguous()?;
    let value = repeat_kv(value, groups)?.contiguous()?;

    let scores = (query.matmul(&key)? * (head_dim as f64).sqrt().recip())?;
    let mask: Vec<_> = (0..seq)
        .flat_map(|row| {
            (0..key_seq).map(move |col| {
                if col > cached_len + row {
                    f32::NEG_INFINITY
                } else {
                    0.
                }
            })
        })
        .collect();
    let mask = Tensor::from_vec(mask, (seq, key_seq), scores.device())?
        .to_dtype(scores.dtype())?;
    let probabilities = ops::softmax_last_dim(
        &scores.broadcast_add(&mask)?.to_dtype(DType::F32)?,
    )?
    .to_dtype(query.dtype())?;
    probabilities.matmul(&value)?.transpose(0, 1)
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

    use candle_core::{Device, safetensors::MmapedSafetensors};

    use super::*;

    fn sample_attention(
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<FullAttention> {
        let sample_projection = |input: usize, output| -> Result<Linear> {
            let weight = Tensor::randn(
                0f32,
                (input as f32).sqrt().recip(),
                (output, input),
                device,
            )?
            .to_dtype(dtype)?;
            Ok(Linear::new(weight, None))
        };
        let rotary_dim = head_dim / 4;
        let inv_freq: Vec<_> = (0..rotary_dim)
            .step_by(2)
            .map(|i| 10_000_000f32.powf(i as f32 / rotary_dim as f32).recip())
            .collect();
        Ok(FullAttention {
            q_proj: sample_projection(hidden_size, num_heads * head_dim * 2)?,
            k_proj: sample_projection(hidden_size, num_kv_heads * head_dim)?,
            v_proj: sample_projection(hidden_size, num_kv_heads * head_dim)?,
            o_proj: sample_projection(num_heads * head_dim, hidden_size)?,
            q_norm: Tensor::zeros(head_dim, dtype, device)?,
            k_norm: Tensor::zeros(head_dim, dtype, device)?,
            inv_freq: Tensor::new(inv_freq.as_slice(), device)?,
            num_heads,
            num_kv_heads,
            head_dim,
            rms_norm_eps: 1e-6,
        })
    }

    fn compare_continuation(
        name: &str,
        output: &Tensor,
        expected: &Tensor,
        tolerance: f32,
        should_assert_tolerance: bool,
    ) -> Result<()> {
        assert_eq!(output.dtype(), expected.dtype(), "{name}");
        assert_eq!(output.dims(), expected.dims(), "{name}");
        let is_bf16 = expected.dtype() == DType::BF16;
        let output = output
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let expected = expected
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let mut max_absolute_difference = 0f32;
        let mut max_tolerance_ratio = 0f32;
        for (actual, expected) in output.into_iter().zip(expected) {
            assert!(actual.is_finite() && expected.is_finite(), "{name}");
            let difference = (actual - expected).abs();
            max_absolute_difference = max_absolute_difference.max(difference);
            if is_bf16 && tolerance > 0. {
                // BF16 uses the same absolute and relative tolerance.
                let allowed_difference = tolerance + tolerance * expected.abs();
                max_tolerance_ratio =
                    max_tolerance_ratio.max(difference / allowed_difference);
            }
        }
        if is_bf16 && tolerance > 0. {
            println!(
                "{name}: max absolute difference {max_absolute_difference:e}, max tolerance ratio {max_tolerance_ratio:e}"
            );
            if should_assert_tolerance {
                assert!(
                    max_tolerance_ratio <= 1.,
                    "{name}: exceeds atol={tolerance}, rtol={tolerance}"
                );
            }
        } else {
            println!(
                "{name}: max absolute difference {max_absolute_difference:e}"
            );
            if should_assert_tolerance {
                assert!(
                    max_absolute_difference <= tolerance,
                    "{name}: exceeds atol={tolerance}"
                );
            }
        }
        Ok(())
    }

    fn assert_cached_splits_match(
        attention: &FullAttention,
        x: &Tensor,
        splits: &[usize],
        tolerance: f32,
    ) -> Result<()> {
        let seq = x.dim(0)?;
        let whole = attention.forward(x)?;
        let (output, whole_cache) = attention.forward_with_cache(x, None)?;
        compare_continuation("no cache", &output, &whole, 0., true)?;
        assert_eq!(
            whole_cache.key.dims(),
            [attention.num_kv_heads, seq, attention.head_dim]
        );
        assert_eq!(whole_cache.value.dims(), whole_cache.key.dims());
        // BF16 rotary cancellation and shape-dependent projection rounding make keys/values diagnostic only.
        let should_assert_cache_tolerance = x.dtype() != DType::BF16;
        for &split in splits {
            let (prefix, cache) =
                attention.forward_with_cache(&x.narrow(0, 0, split)?, None)?;
            let saved_key = cache.key.copy()?;
            let saved_value = cache.value.copy()?;
            for _ in 0..2 {
                let (suffix, extended) = attention.forward_with_cache(
                    &x.narrow(0, split, seq - split)?,
                    Some(&cache),
                )?;
                compare_continuation(
                    &format!("split={split} output"),
                    &Tensor::cat(&[&prefix, &suffix], 0)?,
                    &whole,
                    tolerance,
                    true,
                )?;
                compare_continuation(
                    &format!("split={split} keys"),
                    &extended.key,
                    &whole_cache.key,
                    tolerance,
                    should_assert_cache_tolerance,
                )?;
                compare_continuation(
                    &format!("split={split} values"),
                    &extended.value,
                    &whole_cache.value,
                    tolerance,
                    should_assert_cache_tolerance,
                )?;
            }
            compare_continuation(
                "saved keys",
                &cache.key,
                &saved_key,
                0.,
                true,
            )?;
            compare_continuation(
                "saved values",
                &cache.value,
                &saved_value,
                0.,
                true,
            )?;
        }
        Ok(())
    }

    #[test]
    fn cached_prefix_matches_full_sequence_on_f32() -> Result<()> {
        let attention =
            sample_attention(32, 4, 2, 16, DType::F32, &Device::Cpu)?;
        let x = Tensor::randn(0f32, 1., (23, 32), &Device::Cpu)?;
        assert_cached_splits_match(&attention, &x, &[1, 2, 7, 16, 22], 1e-5)?;
        let whole = attention.forward(&x)?;
        let mut cache = None;
        for t in 0..23 {
            let (output, next_cache) = attention
                .forward_with_cache(&x.narrow(0, t, 1)?, cache.as_ref())?;
            compare_continuation(
                &format!("token={t} output"),
                &output,
                &whole.narrow(0, t, 1)?,
                1e-5,
                true,
            )?;
            assert_eq!(next_cache.key.dim(1)?, t + 1);
            cache = Some(next_cache);
        }
        Ok(())
    }

    #[test]
    fn rejects_mismatched_full_attention_cache() -> Result<()> {
        let attention =
            sample_attention(32, 4, 2, 16, DType::F32, &Device::Cpu)?;
        let x = Tensor::ones((2, 32), DType::F32, &Device::Cpu)?;
        let (_, cache) = attention.forward_with_cache(&x, None)?;
        for invalid in [
            KvCache {
                key: cache.key.narrow(0, 0, 1)?,
                ..cache.clone()
            },
            KvCache {
                key: cache.key.narrow(2, 0, 8)?,
                ..cache.clone()
            },
            KvCache {
                key: cache.key.to_dtype(DType::BF16)?,
                ..cache.clone()
            },
            KvCache {
                value: cache.value.narrow(1, 0, 1)?,
                ..cache.clone()
            },
            KvCache {
                value: cache.value.to_dtype(DType::BF16)?,
                ..cache.clone()
            },
        ] {
            assert!(attention.forward_with_cache(&x, Some(&invalid)).is_err());
        }
        assert!(
            attention
                .forward_with_cache(&x.narrow(0, 0, 0)?, Some(&cache))
                .is_err()
        );
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires CUDA"]
    fn cached_prefix_matches_full_sequence_on_cuda_bf16() -> Result<()> {
        let device = Device::new_cuda(0)?;
        device.set_seed(42)?;
        let attention =
            sample_attention(2560, 16, 4, 256, DType::BF16, &device)?;
        let x = Tensor::randn(0f32, 1., (1860, 2560), &device)?
            .to_dtype(DType::BF16)?;
        assert_cached_splits_match(
            &attention,
            &x,
            &[1, 129, 930, 1796, 1859],
            1. / 64.,
        )
    }

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

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires CUDA"]
    fn flash_attention_matches_explicit_attention_on_bf16() -> Result<()> {
        let device = Device::new_cuda(0)?;
        device.set_seed(42)?;
        let hidden_size = 2560;
        let attention =
            sample_attention(hidden_size, 16, 4, 256, DType::BF16, &device)?;
        for seq in [1, 7, 129, 257] {
            let input = Tensor::randn(0f32, 1., (seq, hidden_size), &device)?
                .to_dtype(DType::BF16)?;
            let query_gate = attention.q_proj.forward(&input)?;
            let key = attention.k_proj.forward(&input)?;
            let value = attention.v_proj.forward(&input)?;
            let expected = attention.o_proj.forward(
                &attention
                    .attend_with_kernel(
                        &query_gate,
                        &key,
                        &value,
                        None,
                        attend_explicitly,
                    )?
                    .0,
            )?;
            let output = attention.forward(&input)?;
            assert_eq!(output.dtype(), DType::BF16);
            assert_eq!(output.dims(), expected.dims());
            let output = output
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let expected = expected
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let mut max_absolute_difference = 0f32;
            for (actual, expected) in output.into_iter().zip(expected) {
                assert!(actual.is_finite() && expected.is_finite());
                max_absolute_difference =
                    max_absolute_difference.max((actual - expected).abs());
            }
            println!(
                "seq {seq}: flash attention max absolute difference {max_absolute_difference:e}"
            );
            assert!(
                max_absolute_difference <= 2e-2,
                "seq {seq}: exceeds BF16 atol=2e-2"
            );
        }
        Ok(())
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
