//! BF16 residual add and zero-bias LayerNorm with Candle's reduction order.
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

#[cfg(test)]
pub(crate) static REFERENCE_NORM: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

#[cfg(test)]
static KERNEL_CALLS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

struct ResidualNorm {
    eps: f32,
}

impl CustomOp3 for ResidualNorm {
    fn name(&self) -> &'static str {
        "vs1_residual_norm"
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
        candle_core::bail!("fused residual norm requires CUDA")
    }
    fn cuda_fwd(
        &self,
        x: &CudaStorage,
        xl: &Layout,
        y: &CudaStorage,
        yl: &Layout,
        weight: &CudaStorage,
        wl: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let (
            CudaStorageSlice::BF16(xs),
            CudaStorageSlice::BF16(ys),
            CudaStorageSlice::BF16(ws),
        ) = (&x.slice, &y.slice, &weight.slice)
        else {
            candle_core::bail!("fused residual norm requires BF16")
        };
        let (rows, cols) = xl.shape().dims2()?;
        let count = rows * cols;
        let dev = x.device();
        let func = dev.get_or_load_custom_func(
            "residual_norm_bf16",
            "vs1_residual_norm",
            include_str!(concat!(env!("OUT_DIR"), "/residual_norm.ptx")),
        )?;
        let x = xs.slice(xl.start_offset()..xl.start_offset() + count);
        let y = ys.slice(yl.start_offset()..yl.start_offset() + count);
        let weight = ws.slice(wl.start_offset()..wl.start_offset() + cols);
        // SAFETY: every output element is written by the kernel.
        let mut out = unsafe { dev.alloc(2 * count)? };
        let (n, c) = (count as u32, cols as i32);
        let mut launch = func.builder();
        launch
            .arg(&x)
            .arg(&y)
            .arg(&weight)
            .arg(&mut out)
            .arg(&n)
            .arg(&c)
            .arg(&self.eps);
        // SAFETY: forward validates shapes, devices and contiguous BF16 storage.
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (rows as u32, 1, 1),
                block_dim: (if cols < 1024 { 32 } else { 1024 }, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .w()?;
        Ok((
            CudaStorage {
                slice: CudaStorageSlice::BF16(out),
                device: dev.clone(),
            },
            (2, rows, cols).into(),
        ))
    }
}

/// The weight belongs to a LayerNorm whose bias is all positive zero.
pub(crate) fn forward(
    x: &Tensor,
    y: &Tensor,
    weight: &Tensor,
    eps: f64,
) -> Result<(Tensor, Tensor)> {
    let (rows, cols) = x.dims2()?;
    if rows == 0
        || cols == 0
        || cols > 1024
        || x.elem_count() > u32::MAX as usize / 2
        || x.dims() != y.dims()
        || weight.dims() != [cols]
        || [x, y, weight].iter().any(|t| {
            !t.is_contiguous()
                || t.dtype() != DType::BF16
                || !t.device().same_device(x.device())
        })
    {
        candle_core::bail!("invalid BF16 residual norm inputs")
    }
    let out =
        x.apply_op3_no_bwd(y, weight, &ResidualNorm { eps: eps as f32 })?;
    #[cfg(test)]
    KERNEL_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    Ok((out.get(0)?, out.get(1)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check(x: &Tensor, y: &Tensor, weight: &Tensor, eps: f64) -> Result<()> {
        let sum = (x + y)?;
        let bias = weight.zeros_like()?;
        let expected =
            sum.apply(&candle_nn::LayerNorm::new(weight.clone(), bias, eps))?;
        let (actual_sum, actual_norm) = forward(x, y, weight, eps)?;
        for (name, a, e) in
            [("sum", actual_sum, sum), ("norm", actual_norm, expected)]
        {
            let a = a.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
            let e = e.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
            for (i, (a, e)) in a.iter().zip(&e).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    e.to_bits(),
                    "{name} {:?} eps={eps}, element {i}: {a} vs {e}",
                    x.dims()
                );
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires CUDA"]
    fn residual_norm_matches_candle_bits() -> Result<()> {
        let device = candle_core::Device::new_cuda(0)?;
        for cols in [1, 7, 31, 32, 33, 64, 128, 768, 1023, 1024] {
            let rows = 9;
            let values: Vec<_> = (0..rows * cols)
                .map(|i| match i / cols {
                    0 => 0.0,
                    1 => -0.0,
                    2 => 1.0,
                    _ => ((i * 37 % 4093) as f32 - 2046.0) / 127.0,
                })
                .collect();
            let x = Tensor::from_vec(values.clone(), (rows, cols), &device)?
                .to_dtype(DType::BF16)?;
            let y = Tensor::from_vec(
                values.into_iter().rev().collect::<Vec<_>>(),
                (rows, cols),
                &device,
            )?
            .to_dtype(DType::BF16)?;
            let weight = Tensor::from_vec(
                (0..cols)
                    .map(|i| ((i * 13 % 257) as f32 - 128.0) / 97.0)
                    .collect::<Vec<_>>(),
                cols,
                &device,
            )?
            .to_dtype(DType::BF16)?;
            // Offset each storage independently, including an odd weight offset.
            let x = Tensor::cat(&[&x, &x], 0)?.narrow(0, rows, rows)?;
            let y = Tensor::cat(&[&y, &y], 0)?.narrow(0, rows, rows)?;
            let weight =
                Tensor::cat(&[&weight, &weight], 0)?.narrow(0, cols, cols)?;
            for eps in [1e-5, 1e-12, 0.1] {
                check(&x, &y, &weight, eps)?;
                check(&x, &x, &weight, eps)?;
            }
        }
        let values: Vec<_> = (0u32..=65535)
            .filter(|v| v & 0x7f80 != 0x7f80)
            .map(|v| f32::from_bits(v << 16))
            .collect();
        let x = Tensor::from_vec(values.clone(), (85, 768), &device)?
            .to_dtype(DType::BF16)?;
        let y = Tensor::from_vec(
            values.into_iter().rev().collect::<Vec<_>>(),
            (85, 768),
            &device,
        )?
        .to_dtype(DType::BF16)?;
        let weight = Tensor::ones(768, DType::BF16, &device)?;
        check(&x, &y, &weight, 1e-5)?;
        check(&x, &x, &weight, 1e-5)?;
        // Include every finite value in the 1024-wide reduction too.
        let values: Vec<_> = (0u32..=65535)
            .map(|v| {
                if v & 0x7f80 == 0x7f80 {
                    0.0
                } else {
                    f32::from_bits(v << 16)
                }
            })
            .collect();
        let x = Tensor::from_vec(values, (64, 1024), &device)?
            .to_dtype(DType::BF16)?;
        let weight = Tensor::ones(1024, DType::BF16, &device)?;
        check(&x, &x, &weight, 1e-5)?;
        Ok(())
    }

    #[test]
    #[ignore = "requires CUDA and checkpoint; run alone"]
    fn paired_norm_latency() -> anyhow::Result<()> {
        KERNEL_CALLS.store(0, std::sync::atomic::Ordering::Relaxed);
        crate::geglu_bench::run_paired(&REFERENCE_NORM)?;
        assert!(
            KERNEL_CALLS.load(std::sync::atomic::Ordering::Relaxed) > 0,
            "benchmark did not exercise the fused residual norm"
        );
        Ok(())
    }
}
