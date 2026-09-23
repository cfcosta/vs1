//! Causal depthwise convolution and SiLU with Candle's BF16 rounding.
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

struct CausalConvSilu;

impl CustomOp2 for CausalConvSilu {
    fn name(&self) -> &'static str {
        "vs1_causal_conv_silu"
    }
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("causal convolution and SiLU kernel requires CUDA")
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
            candle_core::bail!(
                "causal convolution and SiLU kernel requires BF16"
            )
        };
        let (seq, channels) = xl.shape().dims2()?;
        let kernel = wl.shape().dims3()?.2;
        let count = seq * channels;
        let dev = x.device();
        let func = dev.get_or_load_custom_func(
            "convolve_causally_with_silu_bf16",
            "vs1_causal_conv_silu",
            include_str!(concat!(env!("OUT_DIR"), "/causal_conv.ptx")),
        )?;
        let x = xs.slice(xl.start_offset()..xl.start_offset() + count);
        let weight =
            ws.slice(wl.start_offset()..wl.start_offset() + channels * kernel);
        // SAFETY: every output element is written by the kernel.
        let mut output = unsafe { dev.alloc(count)? };
        let (count, channels, kernel) =
            (count as u32, channels as u32, kernel as u32);
        let mut launch = func.builder();
        launch
            .arg(&x)
            .arg(&weight)
            .arg(&mut output)
            .arg(&count)
            .arg(&channels)
            .arg(&kernel);
        // SAFETY: convolve_causally_with_silu checks shapes, devices, dtype
        // and contiguity; the kernel checks the final block's bounds.
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (count.div_ceil(256), 1, 1),
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
pub(super) fn convolve_causally_with_silu(
    x: &Tensor,
    weight: &Tensor,
) -> Result<Option<Tensor>> {
    let Ok((seq, channels)) = x.dims2() else {
        return Ok(None);
    };
    let Ok((weight_channels, channels_per_group, kernel)) = weight.dims3()
    else {
        return Ok(None);
    };
    if !x.device().is_cuda()
        || seq == 0
        || channels == 0
        || kernel == 0
        || weight_channels != channels
        || channels_per_group != 1
        || [x, weight].iter().any(|t| {
            t.dtype() != DType::BF16
                || !t.is_contiguous()
                || !t.device().same_device(x.device())
                || t.elem_count() > u32::MAX as usize
        })
    {
        return Ok(None);
    }
    x.apply_op2_no_bwd(weight, &CausalConvSilu).map(Some)
}

#[cfg(test)]
mod tests {
    use candle_core::Device;

    use super::*;
    use crate::cua_s1::delta_net::convolve_causally;

    #[test]
    #[ignore = "requires CUDA"]
    fn convolves_bf16_and_applies_silu_like_candle() -> Result<()> {
        let device = Device::new_cuda(0)?;
        device.set_seed(42)?;
        for (seq, channels, kernel) in [
            (1, 8192, 4),
            (3, 8192, 4),
            (4, 8192, 4),
            (1860, 8192, 4),
            (4096, 8192, 4),
            (7, 3, 1),
            (9, 5, 7),
        ] {
            // Exercise nonzero storage offsets for both contiguous inputs.
            let x = Tensor::randn(0f32, 2., (seq + 1, channels), &device)?
                .to_dtype(DType::BF16)?
                .narrow(0, 1, seq)?;
            let weight =
                Tensor::rand(-0.5f32, 0.5, (channels + 1, 1, kernel), &device)?
                    .to_dtype(DType::BF16)?
                    .narrow(0, 1, channels)?;
            let expected = convolve_causally(&x, &weight)?.silu()?;
            let actual =
                convolve_causally_with_silu(&x, &weight)?.expect("fused path");
            assert_eq!(actual.dtype(), DType::BF16);
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
                "causal convolution and SiLU seq={seq}, channels={channels}, kernel={kernel}: max absolute difference {max_absolute_difference:e}"
            );
            for (i, (actual, expected)) in
                actual.iter().zip(&expected).enumerate()
            {
                assert_eq!(actual, expected, "element {i}");
            }
        }
        Ok(())
    }
}
