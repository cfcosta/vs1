//! Causal depthwise convolution and SiLU with Candle's BF16 rounding.
use candle_core::{
    CpuStorage,
    CudaStorage,
    CustomOp2,
    DType,
    Layout,
    Result,
    Shape,
    Storage,
    Tensor,
    backend::BackendStorage,
    cuda_backend::{
        CudaStorageSlice,
        WrapErr,
        cudarc::driver::{LaunchConfig, PushKernelArg},
    },
};

struct CausalConvSilu {
    initial_state: Option<Tensor>,
    should_save_state: bool,
}

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
        let initial_storage =
            self.initial_state.as_ref().map(Tensor::storage_and_layout);
        let initial_state = match initial_storage.as_ref() {
            Some((storage, layout)) => {
                let Storage::Cuda(storage) = &**storage else {
                    candle_core::bail!("causal convolution state requires CUDA")
                };
                let CudaStorageSlice::BF16(slice) = &storage.slice else {
                    candle_core::bail!("causal convolution state requires BF16")
                };
                Some(slice.slice(
                    layout.start_offset()
                        ..layout.start_offset() + (kernel - 1) * channels,
                ))
            }
            None => None,
        };
        let state_rows = if self.should_save_state {
            kernel - 1
        } else {
            0
        };
        let output_count = count + state_rows * channels;
        // SAFETY: every output and requested final-state element is written.
        let mut output = unsafe { dev.alloc(output_count)? };
        let should_save_state = u32::from(self.should_save_state);
        let null = 0u64;
        let (count, channels, kernel) =
            (count as u32, channels as u32, kernel as u32);
        let mut launch = func.builder();
        launch.arg(&x).arg(&weight);
        match &initial_state {
            Some(state) => launch.arg(state),
            None => launch.arg(&null),
        };
        launch
            .arg(&mut output)
            .arg(&count)
            .arg(&channels)
            .arg(&kernel)
            .arg(&should_save_state);
        // SAFETY: convolve_causally_with_silu checks shapes, devices, dtype
        // and contiguity; the kernel checks the final block's bounds.
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: ((output_count as u32).div_ceil(256), 1, 1),
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
            Shape::from((seq + state_rows, channels as usize)),
        ))
    }
}

/// Returns `None` when the fused kernel does not apply.
pub(super) fn convolve_causally_with_silu(
    x: &Tensor,
    weight: &Tensor,
) -> Result<Option<Tensor>> {
    CausalConvSilu {
        initial_state: None,
        should_save_state: false,
    }
    .apply(x, weight)
}

pub(super) fn convolve_causally_with_silu_with_state(
    x: &Tensor,
    weight: &Tensor,
    initial_state: Option<&Tensor>,
) -> Result<Option<(Tensor, Tensor)>> {
    let output = CausalConvSilu {
        initial_state: initial_state.cloned(),
        should_save_state: true,
    }
    .apply(x, weight)?;
    output
        .map(|output| {
            let seq = x.dim(0)?;
            let state = output.narrow(0, seq, weight.dim(2)? - 1)?.copy()?;
            Ok((output.narrow(0, 0, seq)?, state))
        })
        .transpose()
}

impl CausalConvSilu {
    fn apply(&self, x: &Tensor, weight: &Tensor) -> Result<Option<Tensor>> {
        let Ok((seq, channels)) = x.dims2() else {
            return Ok(None);
        };
        let Ok((weight_channels, channels_per_group, kernel)) = weight.dims3()
        else {
            return Ok(None);
        };
        if let Some(state) = &self.initial_state {
            super::delta_net::validate_state(
                state,
                &[kernel.saturating_sub(1), channels],
                x.dtype(),
                x.device(),
            )?;
        }
        if !x.device().is_cuda()
            || seq == 0
            || channels == 0
            || kernel == 0
            || weight_channels != channels
            || channels_per_group != 1
            || (seq + kernel - 1) * channels > u32::MAX as usize
            || [x, weight]
                .into_iter()
                .chain(self.initial_state.as_ref())
                .any(|t| {
                    t.dtype() != DType::BF16
                        || !t.is_contiguous()
                        || !t.device().same_device(x.device())
                        || t.elem_count() > u32::MAX as usize
                })
        {
            return Ok(None);
        }
        x.apply_op2_no_bwd(weight, self).map(Some)
    }
}

#[cfg(test)]
mod tests {
    use candle_core::Device;

    use super::*;
    use crate::cua_s1::delta_net::convolve_causally;

    #[test]
    #[ignore = "requires CUDA"]
    fn continues_bf16_convolution_with_identical_output_and_state() -> Result<()>
    {
        use crate::cua_s1::delta_net::{
            convolve_causally_with_state,
            tests::assert_same_bits,
        };

        let device = Device::new_cuda(0)?;
        device.set_seed(42)?;
        let (seq, channels, kernel) = (1860, 8192, 4);
        let x = Tensor::randn(0f32, 2., (seq + 1, channels), &device)?
            .to_dtype(DType::BF16)?
            .narrow(0, 1, seq)?;
        let weight =
            Tensor::rand(-0.5f32, 0.5, (channels + 1, 1, kernel), &device)?
                .to_dtype(DType::BF16)?
                .narrow(0, 1, channels)?;
        let (whole, whole_state) =
            convolve_causally_with_silu_with_state(&x, &weight, None)?
                .expect("fused path");
        assert_same_bits(
            &whole,
            &convolve_causally_with_silu(&x, &weight)?.unwrap(),
        )?;
        assert_same_bits(
            &whole_state,
            &x.narrow(0, seq - kernel + 1, kernel - 1)?,
        )?;
        let zero = Tensor::zeros((kernel - 1, channels), DType::BF16, &device)?;
        let (output, state) =
            convolve_causally_with_silu_with_state(&x, &weight, Some(&zero))?
                .unwrap();
        assert_same_bits(&output, &whole)?;
        assert_same_bits(&state, &whole_state)?;
        for split in [1, 2, 930, seq - 1] {
            let prefix = x.narrow(0, 0, split)?;
            let suffix = x.narrow(0, split, seq - split)?;
            let (prefix_output, prefix_state) =
                convolve_causally_with_silu_with_state(&prefix, &weight, None)?
                    .unwrap();
            // A saved state can be a contiguous view with a nonzero offset.
            let prefix_state = Tensor::cat(&[&zero, &prefix_state], 0)?
                .narrow(0, kernel - 1, kernel - 1)?;
            let saved_state = prefix_state.copy()?;
            let (suffix_output, state) =
                convolve_causally_with_silu_with_state(
                    &suffix,
                    &weight,
                    Some(&prefix_state),
                )?
                .unwrap();
            assert_same_bits(
                &Tensor::cat(&[&prefix_output, &suffix_output], 0)?,
                &whole,
            )?;
            assert_same_bits(&state, &whole_state)?;
            let (candle_output, candle_state) = convolve_causally_with_state(
                &suffix,
                &weight,
                Some(&prefix_state),
            )?;
            assert_same_bits(&suffix_output, &candle_output.silu()?)?;
            assert_same_bits(&state, &candle_state)?;
            assert_same_bits(&prefix_state, &saved_state)?;
        }
        Ok(())
    }

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
