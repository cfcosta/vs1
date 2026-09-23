//! BF16 bias add, optionally followed by ReLU, with Candle's rounding.
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

/// Test-only switch back to Candle's `broadcast_add` and `relu`.
#[cfg(test)]
pub(crate) static REFERENCE_BIAS: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

#[cfg(test)]
static KERNEL_CALLS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

struct BiasAct {
    relu: bool,
}

impl CustomOp2 for BiasAct {
    fn name(&self) -> &'static str {
        "vs1_bias_act"
    }
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("fused bias activation requires CUDA")
    }
    fn cuda_fwd(
        &self,
        x: &CudaStorage,
        xl: &Layout,
        bias: &CudaStorage,
        bl: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let (CudaStorageSlice::BF16(xs), CudaStorageSlice::BF16(bs)) =
            (&x.slice, &bias.slice)
        else {
            candle_core::bail!("fused bias activation requires BF16")
        };
        let (rows, cols) = xl.shape().dims2()?;
        let count = rows * cols;
        let dev = x.device();
        let func = dev.get_or_load_custom_func(
            if self.relu {
                "bias_relu_bf16"
            } else {
                "bias_add_bf16"
            },
            "vs1_bias_act",
            include_str!(concat!(env!("OUT_DIR"), "/bias_act.ptx")),
        )?;
        let x = xs.slice(xl.start_offset()..xl.start_offset() + count);
        let bias = bs.slice(bl.start_offset()..bl.start_offset() + cols);
        // SAFETY: every output element is written by the kernel.
        let mut out = unsafe { dev.alloc(count)? };
        let (count8, cols8) = ((count / 8) as u32, (cols / 8) as u32);
        let mut launch = func.builder();
        launch
            .arg(&x)
            .arg(&bias)
            .arg(&mut out)
            .arg(&count8)
            .arg(&cols8);
        // SAFETY: `forward` checks shape, dtype, contiguity and alignment.
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (count8.div_ceil(256), 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .w()?;
        Ok((
            CudaStorage {
                slice: CudaStorageSlice::BF16(out),
                device: dev.clone(),
            },
            (rows, cols).into(),
        ))
    }
}

/// `x.broadcast_add(bias)`, then `relu()` when asked, over `(rows, cols)`.
/// Returns `None` when the fused kernel does not apply.
pub(crate) fn forward(
    x: &Tensor,
    bias: &Tensor,
    relu: bool,
) -> Result<Option<Tensor>> {
    #[cfg(test)]
    if REFERENCE_BIAS.load(std::sync::atomic::Ordering::Relaxed) {
        return Ok(None);
    }
    let Ok((rows, cols)) = x.dims2() else {
        return Ok(None);
    };
    let aligned = [x, bias].iter().all(|t| {
        t.is_contiguous()
            && t.dtype() == DType::BF16
            && t.layout().start_offset() % 8 == 0
            && t.device().same_device(x.device())
    });
    if !x.device().is_cuda()
        || !aligned
        || rows == 0
        || cols % 8 != 0
        || bias.dims() != [cols]
        || x.elem_count() > u32::MAX as usize
    {
        return Ok(None);
    }
    #[cfg(test)]
    KERNEL_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    x.apply_op2_no_bwd(bias, &BiasAct { relu }).map(Some)
}

/// Candle's `Linear::forward` on `(rows, in)` inputs, with the bias add
/// (and an optional following ReLU) done by the fused kernel.
pub(crate) fn linear(
    x: &Tensor,
    layer: &candle_nn::Linear,
    relu: bool,
) -> Result<Tensor> {
    use candle_core::Module;
    if let (Some(bias), [_, _]) = (layer.bias(), x.dims()) {
        // Same matmul call as candle_nn::Linear for rank-2 inputs.
        let product = x.matmul(&layer.weight().t()?)?;
        if let Some(out) = forward(&product, bias, relu)? {
            return Ok(out);
        }
        let out = product.broadcast_add(bias)?;
        return if relu { out.relu() } else { Ok(out) };
    }
    let out = layer.forward(x)?;
    if relu { out.relu() } else { Ok(out) }
}

#[cfg(test)]
mod tests {
    use super::*;

    // On the CPU, BF16 -> F32 is a 16-bit shift, so NaN payloads survive.
    fn bits(t: &Tensor) -> Result<Vec<u32>> {
        Ok(t.flatten_all()?
            .to_device(&candle_core::Device::Cpu)?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?
            .iter()
            .map(|v| v.to_bits())
            .collect())
    }

    fn bf16s(
        bits: impl IntoIterator<Item = u16>,
        device: &candle_core::Device,
    ) -> Result<Tensor> {
        let values: Vec<f32> = bits
            .into_iter()
            .map(|b| f32::from_bits(u32::from(b) << 16))
            .collect();
        let n = values.len();
        // CPU F32 -> BF16 keeps these exactly representable bit patterns
        // (quiet NaNs included); then copy to the device.
        Tensor::from_vec(values, n, &candle_core::Device::Cpu)?
            .to_dtype(DType::BF16)?
            .to_device(device)
    }

    #[test]
    #[ignore = "requires CUDA"]
    fn bias_relu_matches_candle_bits() -> Result<()> {
        let device = candle_core::Device::new_cuda(0)?;
        // Every BF16 bit pattern, including infinities, zeros and quiet NaNs.
        let all: Vec<u16> = (0u16..=65535).collect();
        let x = bf16s(all.iter().copied(), &device)?.reshape((64, 1024))?;
        // Biases: shifted full sweeps, then special values.
        let special = [
            0x0000u16, 0x8000, 0x3f80, 0xbf80, 0x7f80, 0xff80, 0x7fc0, 0x0001,
            0x7f7f,
        ];
        let mut biases: Vec<Vec<u16>> = (0..64)
            .map(|r| {
                all[r * 1024..(r + 1) * 1024]
                    .iter()
                    .rev()
                    .copied()
                    .collect()
            })
            .collect();
        biases.extend(special.iter().map(|&b| vec![b; 1024]));
        KERNEL_CALLS.store(0, std::sync::atomic::Ordering::Relaxed);
        let mut calls = 0;
        for bias in biases {
            let bias = bf16s(bias, &device)?;
            let sum = x.broadcast_add(&bias)?;
            for relu in [false, true] {
                let expected = if relu { sum.relu()? } else { sum.clone() };
                let actual = forward(&x, &bias, relu)?.expect("fused path");
                calls += 1;
                assert_eq!(bits(&actual)?, bits(&expected)?, "relu={relu}");
            }
        }
        // Wider 4096-column rows and a row-offset view.
        let wide = x.reshape((16, 4096))?;
        let bias = bf16s(all[..4096].iter().copied(), &device)?;
        let view = wide.narrow(0, 3, 9)?;
        for relu in [false, true] {
            let sum = view.broadcast_add(&bias)?;
            let expected = if relu { sum.relu()? } else { sum };
            let actual = forward(&view, &bias, relu)?.expect("fused path");
            calls += 1;
            assert_eq!(bits(&actual)?, bits(&expected)?);
        }
        assert_eq!(
            KERNEL_CALLS.load(std::sync::atomic::Ordering::Relaxed),
            calls
        );
        // Unaligned views fall back.
        let flat = x
            .flatten_all()?
            .narrow(0, 1, 8 * 1024)?
            .reshape((8, 1024))?;
        assert!(forward(&flat, &bias.narrow(0, 0, 1024)?, true)?.is_none());
        Ok(())
    }

    #[test]
    #[ignore = "requires CUDA and checkpoint; run alone"]
    fn paired_bias_act() -> anyhow::Result<()> {
        KERNEL_CALLS.store(0, std::sync::atomic::Ordering::Relaxed);
        crate::model::batch_bench::run_paired_cases(
            &REFERENCE_BIAS,
            crate::model::batch_bench::cases(),
        )?;
        assert!(KERNEL_CALLS.load(std::sync::atomic::Ordering::Relaxed) > 0);
        Ok(())
    }
}
