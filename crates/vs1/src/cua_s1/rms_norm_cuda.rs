//! Zero-centered BF16 RMSNorm with F32 arithmetic and one final BF16 cast.
use candle_core::{
    CpuStorage,
    CudaStorage,
    CustomOp2,
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

struct RmsNorm {
    eps: f64,
    row_stride: usize,
}

impl CustomOp2 for RmsNorm {
    fn name(&self) -> &'static str {
        "vs1_zero_centered_rms_norm"
    }
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("zero-centered RMSNorm kernel requires CUDA")
    }
    fn cuda_fwd(
        &self,
        x: &CudaStorage,
        xl: &Layout,
        weight: &CudaStorage,
        wl: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let (CudaStorageSlice::BF16(xs), CudaStorageSlice::BF16(ws)) =
            (&x.slice, &weight.slice)
        else {
            candle_core::bail!("zero-centered RMSNorm kernel requires BF16")
        };
        let cols = *xl.dims().last().unwrap();
        let count = xl.shape().elem_count();
        let rows = count / cols;
        let input_count = (rows - 1) * self.row_stride + cols;
        let dev = x.device();
        let func = dev.get_or_load_custom_func(
            "normalize_rms_bf16",
            "vs1_zero_centered_rms_norm",
            include_str!(concat!(
                env!("OUT_DIR"),
                "/zero_centered_rms_norm.ptx"
            )),
        )?;
        let x = xs.slice(xl.start_offset()..xl.start_offset() + input_count);
        let weight = ws.slice(wl.start_offset()..wl.start_offset() + cols);
        // SAFETY: every output element is written by the kernel.
        let mut output = unsafe { dev.alloc(count)? };
        let mean_scale = (1. / cols as f64) as f32;
        let eps = self.eps as f32;
        let (cols, row_stride) = (cols as u32, self.row_stride as u32);
        let mut launch = func.builder();
        launch
            .arg(&x)
            .arg(&weight)
            .arg(&mut output)
            .arg(&cols)
            .arg(&row_stride)
            .arg(&mean_scale)
            .arg(&eps);
        // SAFETY: normalize_rms checks shapes, strides, devices and dtype.
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (rows as u32, 1, 1),
                block_dim: (256, 1, 1),
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
pub(super) fn normalize_rms(
    x: &Tensor,
    weight: &Tensor,
    eps: f64,
) -> Result<Option<Tensor>> {
    let Some(&cols) = x.dims().last() else {
        return Ok(None);
    };
    if !x.device().is_cuda()
        || x.dtype() != DType::BF16
        || weight.dtype() != DType::BF16
        || !weight.device().same_device(x.device())
        || !weight.is_contiguous()
        || weight.dims() != [cols]
        || !matches!(cols, 256 | 2560)
        || x.elem_count() == 0
        || x.elem_count() > u32::MAX as usize
    {
        return Ok(None);
    }
    let row_stride = if x.is_contiguous() {
        cols
    } else {
        // Q is a (seq, heads, head_dim) view of interleaved query/gate rows.
        let [_, heads, _] = x.dims() else {
            return Ok(None);
        };
        let strides = x.stride();
        if strides[2] != 1
            || strides[1] < cols
            || strides[0] != heads * strides[1]
        {
            return Ok(None);
        }
        strides[1]
    };
    let rows = x.elem_count() / cols;
    let Some(input_count) = (rows - 1)
        .checked_mul(row_stride)
        .and_then(|span| span.checked_add(cols))
    else {
        return Ok(None);
    };
    if input_count > u32::MAX as usize || row_stride > u32::MAX as usize {
        return Ok(None);
    }
    x.apply_op2_no_bwd(weight, &RmsNorm { eps, row_stride })
        .map(Some)
}

#[cfg(test)]
mod tests {
    use candle_core::{D, Device};

    use super::*;
    use crate::cua_s1::ops::normalize_rms_candle;

    fn compare_with_candle(
        x: &Tensor,
        weight: &Tensor,
        eps: f64,
    ) -> Result<()> {
        let expected = normalize_rms_candle(x, weight, eps)?;
        let actual = normalize_rms(x, weight, eps)?.expect("fused path");
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
            "zero-centered RMSNorm {:?}, eps={eps:e}: max absolute difference {max_absolute_difference:e}",
            x.dims(),
        );
        for (i, (actual, expected)) in actual.iter().zip(&expected).enumerate()
        {
            // A different F32 reduction order can cross a BF16 rounding boundary.
            let tolerance =
                f32::from_bits(expected.abs().to_bits() & 0x7f80_0000) / 128.;
            assert!(
                (actual - expected).abs() <= tolerance,
                "element {i}: {actual} vs {expected}, tolerance {tolerance}"
            );
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires CUDA"]
    fn normalizes_bf16_rows_like_candle() -> Result<()> {
        let device = Device::new_cuda(0)?;
        device.set_seed(42)?;
        for (seq, heads, cols, segments) in [
            (1, 1, 2560, 1),
            (1860, 1, 2560, 1),
            (1860, 16, 256, 2),
            (1860, 4, 256, 1),
        ] {
            // Nonzero storage offsets for both inputs; Q skips each gate row.
            let x = Tensor::randn(
                0f32,
                2.,
                (seq + 1, heads, cols * segments),
                &device,
            )?
            .to_dtype(DType::BF16)?
            .narrow(0, 1, seq)?
            .narrow(D::Minus1, 0, cols)?;
            let x = if heads == 1 { x.squeeze(1)? } else { x };
            let weight = Tensor::rand(-0.5f32, 0.5, cols + 1, &device)?
                .to_dtype(DType::BF16)?
                .narrow(0, 1, cols)?;
            compare_with_candle(&x, &weight, 1e-6)?;
        }
        for cols in [256, 2560] {
            let weight = Tensor::rand(-0.5f32, 0.5, cols, &device)?
                .to_dtype(DType::BF16)?;
            let x = Tensor::randn(0f32, 1e-4, (7, cols), &device)?
                .to_dtype(DType::BF16)?;
            for eps in [1e-6, 1e-3] {
                compare_with_candle(&x, &weight, eps)?;
                compare_with_candle(&x.zeros_like()?, &weight, eps)?;
            }
        }
        Ok(())
    }
}
