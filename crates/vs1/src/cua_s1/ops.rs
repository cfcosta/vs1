//! Text norms and MLP from Transformers 5.17.0's Qwen3.5 implementation.

use candle_core::{D, DType, Result, Tensor};
use candle_nn::{Linear, Module};

/// Zero-centered RMSNorm over the last dimension, with scale `1 + weight`.
pub fn normalize_rms(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let input_dtype = x.dtype();
    let x = x.to_dtype(DType::F32)?;
    let variance = x.sqr()?.mean_keepdim(D::Minus1)?;
    let normalized = x.broadcast_mul(&(variance + eps)?.sqrt()?.recip()?)?;
    normalized
        .broadcast_mul(&(weight.to_dtype(DType::F32)? + 1.)?)?
        .to_dtype(input_dtype)
}

/// RMSNorm over the last dimension, followed by `weight * silu(gate)`.
pub fn normalize_rms_gated(
    x: &Tensor,
    weight: &Tensor,
    gate: &Tensor,
    eps: f64,
) -> Result<Tensor> {
    let input_dtype = x.dtype();
    let x = x.to_dtype(DType::F32)?;
    let variance = x.sqr()?.mean_keepdim(D::Minus1)?;
    let normalized = x.broadcast_mul(&(variance + eps)?.sqrt()?.recip()?)?;
    // Preserve PyTorch's rounding before the weight and gate multiplies,
    // while doing the arithmetic itself in F32.
    let normalized = normalized.to_dtype(input_dtype)?.to_dtype(DType::F32)?;
    let mut weighted =
        normalized.broadcast_mul(&weight.to_dtype(DType::F32)?)?;
    if weight.dtype() == input_dtype {
        weighted = weighted.to_dtype(input_dtype)?.to_dtype(DType::F32)?;
    }
    (weighted * gate.to_dtype(DType::F32)?.silu()?)?.to_dtype(input_dtype)
}

/// L2 normalization over the last dimension, with epsilon `1e-6` inside rsqrt.
pub fn normalize_l2(x: &Tensor) -> Result<Tensor> {
    let input_dtype = x.dtype();
    let x = x.to_dtype(DType::F32)?;
    let squared_norm = x.sqr()?.sum_keepdim(D::Minus1)?;
    x.broadcast_mul(&(squared_norm + 1e-6)?.sqrt()?.recip()?)?
        .to_dtype(input_dtype)
}

/// Bias-free Qwen3.5 MLP, with gate, up and down projections in the model dtype.
pub struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Mlp {
    pub fn new(
        gate_weight: Tensor,
        up_weight: Tensor,
        down_weight: Tensor,
    ) -> Result<Self> {
        Ok(Self {
            gate_proj: Linear::new(gate_weight, None),
            up_proj: Linear::new(up_weight, None),
            down_proj: Linear::new(down_weight, None),
        })
    }
}

impl Module for Mlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gated =
            (self.gate_proj.forward(x)?.silu()? * self.up_proj.forward(x)?)?;
        self.down_proj.forward(&gated)
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use candle_core::{Device, safetensors::MmapedSafetensors};

    use super::*;
    use crate::cua_s1::{TextConfig, TextWeights};

    fn assert_values(output: &Tensor, expected: &[f32], tolerance: f32) {
        let expected = Tensor::new(expected, &Device::Cpu)
            .unwrap()
            .to_dtype(output.dtype())
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let actual = output
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert!(
                (actual - expected).abs() <= tolerance,
                "expected {expected}, got {actual}"
            );
        }
    }

    #[test]
    fn normalizes_rms_with_zero_centered_weights_in_f32() {
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            let x = Tensor::new(
                &[[[300f32, 400.], [-400., 300.], [0., 0.]]],
                &Device::Cpu,
            )
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
            let weight = Tensor::new(&[1f32, -0.5], &Device::Cpu)
                .unwrap()
                .to_dtype(dtype)
                .unwrap();
            // sqrt(mean([300^2, 400^2]) + 35000) = 400.
            let output = normalize_rms(&x, &weight, 35000.).unwrap();
            assert_eq!(output.dtype(), dtype);
            assert_eq!(output.dims(), x.dims());
            assert_values(&output, &[1.5, 0.5, -2., 0.375, 0., 0.], 1e-6);
        }
    }

    #[test]
    fn normalizes_rms_before_weighting_and_applying_silu_gate() {
        let x = Tensor::new(&[[[3f32, 4.], [-4., 3.]]], &Device::Cpu).unwrap();
        let weight = Tensor::new(&[2f32, -0.5], &Device::Cpu).unwrap();
        let log_three = 3f32.ln();
        let gate = Tensor::new(
            &[[[log_three, -log_three], [0., log_three]]],
            &Device::Cpu,
        )
        .unwrap();
        // RMS denominator is 4; silu(ln(3)) = 3 ln(3) / 4.
        let output = normalize_rms_gated(&x, &weight, &gate, 3.5).unwrap();
        assert_eq!(output.dtype(), DType::F32);
        assert_eq!(output.dims(), x.dims());
        assert_values(
            &output,
            &[
                1.125 * log_three,
                0.125 * log_three,
                0.,
                -0.28125 * log_three,
            ],
            1e-6,
        );
    }

    #[test]
    fn rounds_gated_norm_at_the_reference_cast_boundaries() {
        let x = Tensor::new(&[[3f32, 2.]], &Device::Cpu)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let gate = Tensor::new(&[[2f32, 0.]], &Device::Cpu)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        // 3 / sqrt(6.5 + eps) rounds to 1.1796875 in BF16. Multiplying
        // by 1.1015625 gives 1.29949951171875, or 1.296875 in BF16.
        for (weight_dtype, expected) in
            [(DType::F32, 2.296875), (DType::BF16, 2.28125)]
        {
            let weight = Tensor::new(&[1.1015625f32, 2.], &Device::Cpu)
                .unwrap()
                .to_dtype(weight_dtype)
                .unwrap();
            let output = normalize_rms_gated(&x, &weight, &gate, 1e-6).unwrap();
            assert_eq!(output.dtype(), DType::BF16);
            assert_values(&output, &[expected, 0.], 0.);
        }
    }

    #[test]
    fn normalizes_l2_with_epsilon_inside_the_square_root() {
        let x = Tensor::new(
            &[[[3f32, 4.], [0., 0.], [0.0006, 0.0008]]],
            &Device::Cpu,
        )
        .unwrap();
        let output = normalize_l2(&x).unwrap();
        assert_eq!(output.dtype(), DType::F32);
        assert_eq!(output.dims(), x.dims());
        assert_values(
            &output,
            &[0.6, 0.8, 0., 0., 0.6 / 2f32.sqrt(), 0.8 / 2f32.sqrt()],
            1e-7,
        );
        for dtype in [DType::F16, DType::BF16] {
            let x = Tensor::new(&[[300f32, -400.]], &Device::Cpu)
                .unwrap()
                .to_dtype(dtype)
                .unwrap();
            let output = normalize_l2(&x).unwrap();
            assert_eq!(output.dtype(), dtype);
            assert_values(&output, &[0.6, -0.8], 0.);
        }
    }

    #[test]
    fn projects_silu_gated_mlp_in_input_dtype() {
        // CPU matmul supports F16 but not BF16. The F16 expectations
        // include intermediate rounding.
        for (dtype, expected) in [
            (
                DType::F32,
                [8.70167, -0.5795272, -3.162_64, 0.71521753, 0., 0.],
            ),
            (
                DType::F16,
                [
                    8.695_312_5,
                    -0.580_566_4,
                    -3.162_109_4,
                    0.715_332_03,
                    0.,
                    0.,
                ],
            ),
        ] {
            let gate_weight =
                Tensor::new(&[[1f32, 0.], [0., 1.], [1., -1.]], &Device::Cpu)
                    .unwrap()
                    .to_dtype(dtype)
                    .unwrap();
            let up_weight =
                Tensor::new(&[[1f32, 1.], [2., 0.], [0., -1.]], &Device::Cpu)
                    .unwrap()
                    .to_dtype(dtype)
                    .unwrap();
            let down_weight =
                Tensor::new(&[[1f32, 2., -1.], [-1., 0., 3.]], &Device::Cpu)
                    .unwrap()
                    .to_dtype(dtype)
                    .unwrap();
            let mlp = Mlp::new(gate_weight, up_weight, down_weight).unwrap();
            let x =
                Tensor::new(&[[[1f32, 2.], [-1., 1.], [0., 0.]]], &Device::Cpu)
                    .unwrap()
                    .to_dtype(dtype)
                    .unwrap();
            let output = mlp.forward(&x).unwrap();
            assert_eq!(output.dtype(), dtype);
            assert_eq!(output.dims(), x.dims());
            // First row: gate = [1, 2, -1], up = [3, 2, -2].
            // Second row: gate = [-1, 1, -2], up = [0, -2, -1].
            assert_values(&output, &expected, 1e-6);
        }
    }

    #[test]
    #[ignore = "requires the pinned local base and layer dump in artifacts/cua-s1"]
    fn reproduces_dumped_layer_zero_gated_norm() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../artifacts/cua-s1");
        let base_directory =
            root.join("base/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a");
        let weights =
            TextWeights::load(&base_directory, None, &Device::Cpu, DType::F32)
                .unwrap();
        let weight = weights
            .tensor("model.language_model.layers.0.linear_attn.norm.weight")
            .unwrap();
        let config = TextConfig::from_slice(
            &std::fs::read(base_directory.join("config.json")).unwrap(),
        )
        .unwrap();
        // SAFETY: the reference dump is mapped read-only and not modified.
        let dump = unsafe {
            MmapedSafetensors::new(root.join("layers-f32.safetensors"))
        }
        .unwrap();
        let x = dump
            .load("layers.0.linear_attn.norm.input", &Device::Cpu)
            .unwrap();
        let gate = dump
            .load("layers.0.linear_attn.norm.gate", &Device::Cpu)
            .unwrap();
        let expected = dump
            .load("layers.0.linear_attn.norm.output", &Device::Cpu)
            .unwrap();
        let output =
            normalize_rms_gated(&x, &weight, &gate, config.rms_norm_eps)
                .unwrap();
        assert_eq!(output.dtype(), DType::F32);
        assert_eq!(output.dims(), expected.dims());
        let max_absolute_difference = (&output - &expected)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        println!(
            "layer 0 gated norm max absolute difference: {max_absolute_difference:e}"
        );
        assert!(
            max_absolute_difference <= 1e-6,
            "{max_absolute_difference:e}"
        );
    }
}
