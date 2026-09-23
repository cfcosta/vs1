//! BF16 gated RMSNorm with F32 arithmetic and the reference's cast boundaries.
use candle_core::{
    CpuStorage,
    CudaStorage,
    CustomOp3,
    DType,
    Layout,
    Result,
    Shape,
    Tensor,
    backend::BackendStorage,
    cuda_backend::{
        CudaStorageSlice,
        WrapErr,
        cudarc::driver::{LaunchConfig, PushKernelArg},
    },
};

struct GatedRmsNorm {
    eps: f64,
}

impl CustomOp3 for GatedRmsNorm {
    fn name(&self) -> &'static str {
        "vs1_gated_rms_norm"
    }
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("gated RMSNorm kernel requires CUDA")
    }
    fn cuda_fwd(
        &self,
        x: &CudaStorage,
        xl: &Layout,
        weight: &CudaStorage,
        wl: &Layout,
        gate: &CudaStorage,
        gl: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let (CudaStorageSlice::BF16(xs), CudaStorageSlice::BF16(gs)) =
            (&x.slice, &gate.slice)
        else {
            candle_core::bail!("gated RMSNorm kernel requires BF16 inputs")
        };
        let kernel = match &weight.slice {
            CudaStorageSlice::BF16(_) => "normalize_rms_gated_bf16",
            CudaStorageSlice::F32(_) => "normalize_rms_gated_bf16_f32_weight",
            _ => candle_core::bail!(
                "gated RMSNorm kernel requires BF16 or F32 weights"
            ),
        };
        let count = xl.shape().elem_count();
        let dev = x.device();
        let func = dev.get_or_load_custom_func(
            kernel,
            "vs1_gated_rms_norm",
            include_str!(concat!(env!("OUT_DIR"), "/gated_rms_norm.ptx")),
        )?;
        let x = xs.slice(xl.start_offset()..xl.start_offset() + count);
        let gate = gs.slice(gl.start_offset()..gl.start_offset() + count);
        // SAFETY: every output element is written by the kernel.
        let mut output = unsafe { dev.alloc(count)? };
        let eps = self.eps as f32;
        let mut launch = func.builder();
        launch.arg(&x).arg(&gate).arg(&mut output).arg(&eps);
        let weight_bf16;
        let weight_f32;
        match &weight.slice {
            CudaStorageSlice::BF16(ws) => {
                weight_bf16 =
                    ws.slice(wl.start_offset()..wl.start_offset() + 128);
                launch.arg(&weight_bf16);
            }
            CudaStorageSlice::F32(ws) => {
                weight_f32 =
                    ws.slice(wl.start_offset()..wl.start_offset() + 128);
                launch.arg(&weight_f32);
            }
            _ => unreachable!(),
        }
        // SAFETY: normalize_rms_gated checks shapes, devices, dtypes and
        // contiguity; each block reads and writes exactly one 128-value row.
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: ((count / 128) as u32, 1, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .w()?;
        Ok((
            CudaStorage {
                slice: CudaStorageSlice::BF16(output),
                device: dev.clone(),
            },
            xl.shape().clone(),
        ))
    }
}

/// Returns `None` when the fused kernel does not apply.
pub(super) fn normalize_rms_gated(
    x: &Tensor,
    weight: &Tensor,
    gate: &Tensor,
    eps: f64,
) -> Result<Option<Tensor>> {
    if !x.device().is_cuda()
        || x.dtype() != DType::BF16
        || gate.dtype() != DType::BF16
        || !matches!(weight.dtype(), DType::BF16 | DType::F32)
        || x.dims().last() != Some(&128)
        || weight.dims() != [128]
        || gate.shape() != x.shape()
        || x.elem_count() == 0
        || x.elem_count() > u32::MAX as usize
        || [x, weight, gate]
            .iter()
            .any(|t| !t.is_contiguous() || !t.device().same_device(x.device()))
    {
        return Ok(None);
    }
    x.apply_op3_no_bwd(weight, gate, &GatedRmsNorm { eps })
        .map(Some)
}

#[cfg(test)]
mod tests {
    use candle_core::Device;

    use super::*;
    use crate::cua_s1::ops::normalize_rms_gated_candle;

    fn compare_with_candle(
        x: &Tensor,
        weight: &Tensor,
        gate: &Tensor,
        eps: f64,
    ) -> Result<()> {
        let expected = normalize_rms_gated_candle(x, weight, gate, eps)?;
        let actual =
            normalize_rms_gated(x, weight, gate, eps)?.expect("fused path");
        assert_eq!(actual.dtype(), x.dtype());
        assert_eq!(actual.shape(), x.shape());
        let expected = expected
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let actual = actual
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let max_absolute_difference = actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        println!(
            "gated RMSNorm {:?}, weight={:?}, eps={eps:e}: max absolute difference {max_absolute_difference:e}",
            x.dims(),
            weight.dtype(),
        );
        for (i, (actual, expected)) in actual.iter().zip(&expected).enumerate()
        {
            // A changed reduction order can cross each of three BF16 casts.
            let tolerance = 3.
                * f32::from_bits(expected.abs().to_bits() & 0x7f80_0000)
                / 128.;
            assert!(
                (actual - expected).abs() <= tolerance,
                "element {i}: {actual} vs {expected}, tolerance {tolerance}"
            );
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires CUDA"]
    fn normalizes_bf16_value_heads_like_candle() -> Result<()> {
        let device = Device::new_cuda(0)?;
        device.set_seed(42)?;
        for seq in [1, 7, 1860] {
            // Independent nonzero storage offsets for all three inputs.
            let x = Tensor::randn(0f32, 2., (seq + 1, 32, 128), &device)?
                .to_dtype(DType::BF16)?
                .narrow(0, 1, seq)?;
            let gate = Tensor::randn(0f32, 2., (seq + 2, 32, 128), &device)?
                .to_dtype(DType::BF16)?
                .narrow(0, 2, seq)?;
            for dtype in [DType::BF16, DType::F32] {
                let weight = Tensor::rand(-1.5f32, 1.5, 131, &device)?
                    .to_dtype(dtype)?
                    .narrow(0, 3, 128)?;
                compare_with_candle(&x, &weight, &gate, 1e-6)?;
            }
        }
        let x = Tensor::randn(0f32, 1e-4, (1, 32, 128), &device)?
            .to_dtype(DType::BF16)?;
        let gate = Tensor::randn(0f32, 2., x.shape(), &device)?
            .to_dtype(DType::BF16)?;
        for dtype in [DType::BF16, DType::F32] {
            let weight =
                Tensor::rand(-1.5f32, 1.5, 128, &device)?.to_dtype(dtype)?;
            for eps in [1e-6, 1e-3] {
                compare_with_candle(&x, &weight, &gate, eps)?;
                compare_with_candle(&x.zeros_like()?, &weight, &gate, eps)?;
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires CUDA"]
    fn rounds_bf16_at_both_reference_cast_boundaries() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let x = Tensor::from_vec(
            [3f32, 2.].repeat(32 * 64),
            (1, 32, 128),
            &device,
        )?
        .to_dtype(DType::BF16)?;
        let gate =
            Tensor::full(2f32, x.shape(), &device)?.to_dtype(DType::BF16)?;
        for (dtype, expected) in
            [(DType::BF16, 2.28125), (DType::F32, 2.296875)]
        {
            let weight =
                Tensor::full(1.1015625f32, 128, &device)?.to_dtype(dtype)?;
            compare_with_candle(&x, &weight, &gate, 1e-6)?;
            let actual = normalize_rms_gated(&x, &weight, &gate, 1e-6)?
                .expect("fused path")
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            assert_eq!(actual[0], expected);
        }
        Ok(())
    }
}
