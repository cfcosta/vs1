//! Rounded BF16 GeGLU GEMM epilogue for the measured large-matrix path.
use std::ffi::c_void;

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
        cudarc::driver::{DevicePtr, DevicePtrMut},
    },
};

unsafe extern "C" {
    fn vs1_cutlass_geglu(
        x: *const c_void,
        w: *const c_void,
        g: *const c_void,
        out: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
        stream: *mut c_void,
        fused: i32,
    ) -> i32;
}
unsafe extern "C" {
    fn vs1_cutlass_dual_geglu(
        x: *const c_void,
        wa: *const c_void,
        wg: *const c_void,
        out: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
        stream: *mut c_void,
    ) -> i32;
}
/// Test-only switch back to the separate gate GEMM plus fused epilogue.
#[cfg(test)]
pub(crate) static REFERENCE_DUAL: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
#[cfg(test)]
pub(crate) static REFERENCE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
pub(crate) fn enabled(xs: &Tensor, weight: &Tensor) -> bool {
    #[cfg(test)]
    if REFERENCE.load(std::sync::atomic::Ordering::Relaxed) {
        return false;
    }
    xs.device().is_cuda()
        && xs.dtype() == DType::BF16
        && xs.is_contiguous()
        && xs.rank() == 2
        && (2048..=32768).contains(&xs.dims()[0])
        && xs.dims()[1] == 1024
        && weight.dims() == [2624, 1024]
        && weight.is_contiguous()
        && !candle_core::cuda_backend::gemm_reduced_precision_bf16()
        && crate::parallel_cuda::supported_device(xs)
}
pub(crate) fn forward(
    xs: &Tensor,
    weight: &Tensor,
    gate: &Tensor,
) -> Result<Tensor> {
    xs.apply_op3_no_bwd(weight, gate, &Fused(true))
}
/// Both FFN projections in one dual GEMM, rounded like two separate
/// Candle GEMMs, then the rounded GeGLU. Same eligibility as `enabled`.
pub(crate) fn dual_enabled() -> bool {
    #[cfg(test)]
    if REFERENCE_DUAL.load(std::sync::atomic::Ordering::Relaxed) {
        return false;
    }
    true
}
/// Test-only in-model check: compare each dual product against Candle's
/// two GEMMs plus the rounded GeGLU on the model's real activations.
#[cfg(test)]
pub(crate) static DUAL_CHECK: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
#[cfg(test)]
pub(crate) static DUAL_CHECKS: std::sync::Mutex<Vec<(usize, usize)>> =
    std::sync::Mutex::new(Vec::new());
#[cfg(test)]
pub(crate) fn check_dual(
    xs: &Tensor,
    dual: &Tensor,
    act: &candle_nn::Linear,
    gate: &candle_nn::Linear,
) -> Result<()> {
    if !DUAL_CHECK.load(std::sync::atomic::Ordering::Relaxed) {
        return Ok(());
    }
    let expected =
        crate::geglu_cuda::forward(&xs.apply(act)?, &xs.apply(gate)?)?;
    let differing = dual
        .ne(&expected)?
        .to_dtype(DType::U32)?
        .sum_all()?
        .to_scalar::<u32>()?;
    DUAL_CHECKS
        .lock()
        .unwrap()
        .push((xs.dim(0)?, differing as usize));
    Ok(())
}
pub(crate) fn dual_forward(
    xs: &Tensor,
    act: &Tensor,
    gate: &Tensor,
) -> Result<Tensor> {
    if act.dims() != gate.dims() || !gate.is_contiguous() {
        candle_core::bail!("dual GeGLU weights must match")
    }
    xs.apply_op3_no_bwd(act, gate, &Dual)
}
struct Dual;
impl CustomOp3 for Dual {
    fn name(&self) -> &'static str {
        "vs1_cutlass_dual_geglu"
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
        candle_core::bail!("CUDA only")
    }
    fn cuda_fwd(
        &self,
        x: &CudaStorage,
        xl: &Layout,
        a: &CudaStorage,
        al: &Layout,
        g: &CudaStorage,
        gl: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let (m, k) = xl.shape().dims2()?;
        let (n, ak) = al.shape().dims2()?;
        if ak != k
            || gl.dims() != [n, k]
            || !xl.is_contiguous()
            || !al.is_contiguous()
            || !gl.is_contiguous()
        {
            candle_core::bail!("invalid dual GEMM layouts")
        }
        let (
            CudaStorageSlice::BF16(xb),
            CudaStorageSlice::BF16(ab),
            CudaStorageSlice::BF16(gb),
        ) = (&x.slice, &a.slice, &g.slice)
        else {
            candle_core::bail!("BF16 only")
        };
        let dev = x.device();
        let stream = dev.cuda_stream();
        let xv = xb.slice(xl.start_offset()..xl.start_offset() + m * k);
        let av = ab.slice(al.start_offset()..al.start_offset() + n * k);
        let gv = gb.slice(gl.start_offset()..gl.start_offset() + n * k);
        // SAFETY: the checked GEMM writes all m*n BF16 elements before use.
        let mut out = unsafe { dev.alloc(m * n)? };
        let (xp, xguard) = xv.device_ptr(&stream);
        let (ap, aguard) = av.device_ptr(&stream);
        let (gp, gguard) = gv.device_ptr(&stream);
        let (yp, yguard) = out.device_ptr_mut(&stream);
        stream.context().bind_to_thread().w()?;
        // SAFETY: contiguous BF16 buffers match the checked matrix dimensions.
        // Pointer guards order allocation/read/write on this same CUDA stream.
        let status = unsafe {
            vs1_cutlass_dual_geglu(
                xp as *const _,
                ap as *const _,
                gp as *const _,
                yp as *mut _,
                i32::try_from(m).map_err(candle_core::Error::wrap)?,
                i32::try_from(n).map_err(candle_core::Error::wrap)?,
                i32::try_from(k).map_err(candle_core::Error::wrap)?,
                stream.cu_stream().cast(),
            )
        };
        if status != 0 {
            candle_core::bail!("CUTLASS dual status {status}")
        }
        drop((xguard, aguard, gguard, yguard));
        Ok((
            CudaStorage {
                slice: CudaStorageSlice::BF16(out),
                device: dev.clone(),
            },
            (m, n).into(),
        ))
    }
}
struct Fused(bool);
impl CustomOp3 for Fused {
    fn name(&self) -> &'static str {
        "vs1_cutlass_geglu"
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
        candle_core::bail!("CUDA only")
    }
    fn cuda_fwd(
        &self,
        x: &CudaStorage,
        xl: &Layout,
        w: &CudaStorage,
        wl: &Layout,
        g: &CudaStorage,
        gl: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let (m, k) = xl.shape().dims2()?;
        let (n, wk) = wl.shape().dims2()?;
        if wk != k
            || gl.dims() != [m, n]
            || !xl.is_contiguous()
            || !wl.is_contiguous()
            || !gl.is_contiguous()
        {
            candle_core::bail!("invalid fused GEMM layouts")
        }
        let (
            CudaStorageSlice::BF16(xb),
            CudaStorageSlice::BF16(wb),
            CudaStorageSlice::BF16(gb),
        ) = (&x.slice, &w.slice, &g.slice)
        else {
            candle_core::bail!("BF16 only")
        };
        let dev = x.device();
        let stream = dev.cuda_stream();
        let xv = xb.slice(xl.start_offset()..xl.start_offset() + m * k);
        let wv = wb.slice(wl.start_offset()..wl.start_offset() + n * k);
        let gv = gb.slice(gl.start_offset()..gl.start_offset() + m * n);
        // SAFETY: the checked GEMM writes all m*n BF16 elements before use.
        let mut out = unsafe { dev.alloc(m * n)? };
        let (xp, xguard) = xv.device_ptr(&stream);
        let (wp, wguard) = wv.device_ptr(&stream);
        let (gp, gguard) = gv.device_ptr(&stream);
        let (yp, yguard) = out.device_ptr_mut(&stream);
        stream.context().bind_to_thread().w()?;
        // SAFETY: contiguous BF16 buffers match the checked matrix dimensions.
        // Pointer guards order allocation/read/write on this same CUDA stream.
        let status = unsafe {
            vs1_cutlass_geglu(
                xp as *const _,
                wp as *const _,
                gp as *const _,
                yp as *mut _,
                i32::try_from(m).map_err(candle_core::Error::wrap)?,
                i32::try_from(n).map_err(candle_core::Error::wrap)?,
                i32::try_from(k).map_err(candle_core::Error::wrap)?,
                stream.cu_stream().cast(),
                i32::from(self.0),
            )
        };
        if status != 0 {
            candle_core::bail!("CUTLASS status {status}")
        }
        drop((xguard, wguard, gguard, yguard));
        Ok((
            CudaStorage {
                slice: CudaStorageSlice::BF16(out),
                device: dev.clone(),
            },
            (m, n).into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use candle_nn::Linear;
    use serde_json::{Value, json};

    use super::*;
    fn compare(a: &Tensor, b: &Tensor) -> Result<Value> {
        let a = a.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        let b = b.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        let differences = a
            .iter()
            .zip(&b)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        let max = a
            .iter()
            .zip(&b)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        Ok(
            json!({"values":a.len(),"differing_values":differences,"max_absolute_error":max}),
        )
    }
    #[test]
    #[ignore = "requires CUDA; run alone"]
    fn rounded_products_match_candle() -> anyhow::Result<()> {
        let device = candle_core::Device::new_cuda(0)?;
        let mut seed = 713u32;
        let mut tensor = |rows, columns| -> Result<Tensor> {
            let data: Vec<f32> = (0..rows * columns)
                .map(|i| {
                    seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                    ((seed >> 8) as f32 / 16777216.0 - 0.5)
                        * [0.001, 0.1, 1.0, 8.0][i % 4]
                })
                .collect();
            Tensor::from_vec(data, (rows, columns), &device)?
                .to_dtype(DType::BF16)
        };
        let a = Linear::new(tensor(2624, 1024)?, None);
        let g = Linear::new(tensor(2624, 1024)?, None);
        let mut report = vec![];
        for rows in [
            129, 1032, 1536, 2047, 2048, 2049, 3870, 4128, 5400, 8192, 16384,
            32768,
        ] {
            let input = tensor(rows, 1024)?;
            if crate::parallel_cuda::supported_device(&input) {
                assert_eq!(enabled(&input, a.weight()), rows >= 2048);
            }
            for xs in [&input, &input.neg()?] {
                let gate = xs.apply(&g)?;
                let expected_act = xs.apply(&a)?;
                let plain =
                    xs.apply_op3_no_bwd(a.weight(), &gate, &Fused(false))?;
                let fused =
                    xs.apply_op3_no_bwd(a.weight(), &gate, &Fused(true))?;
                report.push(json!({"rows":rows,"gemm_vs_candle":compare(&plain,&expected_act)?,
                "fused_vs_candle":compare(&fused,&crate::geglu_cuda::forward(&expected_act,&gate)?)?,
                "epilogue_rounding_control":compare(&fused,&crate::geglu_cuda::forward(&plain,&gate)?)?}));
            }
        }
        assert!(!enabled(
            &Tensor::zeros((32769, 1024), DType::BF16, &device)?,
            a.weight()
        ));
        std::fs::write(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(
            "../../research/inference-followups/03-final-seeded-exactness.json",
        ),
        serde_json::to_vec_pretty(&report)?,
    )?;
        eprintln!("SEEDED_REPORT={}", serde_json::to_string(&report)?);
        assert!(report.iter().all(
            |r| r["fused_vs_candle"]["differing_values"] == 0
                && r["gemm_vs_candle"]["differing_values"] == 0
                && r["epilogue_rounding_control"]["differing_values"] == 0
        ));
        Ok(())
    }

    #[test]
    #[ignore = "requires CUDA; run alone"]
    fn dual_products_match_candle() -> anyhow::Result<()> {
        let device = candle_core::Device::new_cuda(0)?;
        let mut seed = 917u32;
        let mut tensor = |rows, columns| -> Result<Tensor> {
            let data: Vec<f32> = (0..rows * columns)
                .map(|i| {
                    seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                    ((seed >> 8) as f32 / 16777216.0 - 0.5)
                        * [0.001, 0.1, 1.0, 8.0, 60.0][i % 5]
                })
                .collect();
            Tensor::from_vec(data, (rows, columns), &device)?
                .to_dtype(DType::BF16)
        };
        let mut report = vec![];
        for weights in 0..3 {
            let a = Linear::new(tensor(2624, 1024)?, None);
            let g = Linear::new(tensor(2624, 1024)?, None);
            for rows in [129, 1032, 2047, 2048, 2049, 4128, 5400, 16384, 32768]
            {
                let input = tensor(rows, 1024)?;
                for xs in [&input, &input.neg()?] {
                    let expected = crate::geglu_cuda::forward(
                        &xs.apply(&a)?,
                        &xs.apply(&g)?,
                    )?;
                    let dual = dual_forward(xs, a.weight(), g.weight())?;
                    report.push(json!({"weights":weights,"rows":rows,"dual_vs_candle":compare(&dual,&expected)?}));
                }
            }
        }
        eprintln!("DUAL_REPORT={}", serde_json::to_string(&report)?);
        assert!(
            report
                .iter()
                .all(|r| r["dual_vs_candle"]["differing_values"] == 0)
        );
        Ok(())
    }

    /// Real encoder activations have outlier channels that synthetic
    /// products do not; cuBLAS's reduction order also varies by shape.
    /// Check every FFN layer across many packed row counts at or above
    /// the 2048-row guard.
    #[test]
    #[ignore = "requires CUDA and checkpoint; run alone"]
    fn dual_matches_candle_on_real_activations() -> anyhow::Result<()> {
        use crate::{Question, SystemOneRequest};
        let model: crate::SystemOne =
            crate::SystemOne::from(crate::DEFAULT_REPO_ID)
                .with_device(candle_core::Device::new_cuda(0)?)
                .with_dtype(DType::BF16)
                .try_into()?;
        let paragraph = "The indexing pipeline compares content hashes and updates changed documents. Unchanged files are skipped. ";
        DUAL_CHECKS.lock().unwrap().clear();
        DUAL_CHECK.store(true, std::sync::atomic::Ordering::Relaxed);
        for repeats in [1usize, 2, 3, 5, 7, 9, 12, 16, 20, 26] {
            for n in [5usize, 8, 10, 12, 14, 16, 17, 19, 21, 24, 27, 30, 32] {
                let requests: Vec<_> = (0..n)
                    .map(|i| {
                        SystemOneRequest::new(format!(
                            "Doc {i}. {}",
                            paragraph.repeat(repeats)
                        ))
                        .question(
                            "q",
                            Question::noul("Does this explain how changed files are selected?"),
                        )
                    })
                    .collect();
                model.system_one_batch(&requests)?;
            }
        }
        DUAL_CHECK.store(false, std::sync::atomic::Ordering::Relaxed);
        let checks = std::mem::take(&mut *DUAL_CHECKS.lock().unwrap());
        let mut shapes: Vec<_> = checks.iter().map(|&(rows, _)| rows).collect();
        shapes.sort_unstable();
        shapes.dedup();
        let bad: Vec<_> = checks.iter().filter(|&&(_, d)| d > 0).collect();
        eprintln!(
            "DUAL_REAL shapes={} products={} rows={:?}..{:?} mismatched={bad:?}",
            shapes.len(),
            checks.len(),
            shapes.first(),
            shapes.last()
        );
        assert!(shapes.len() > 50 && bad.is_empty());
        Ok(())
    }

    #[test]
    #[ignore = "requires CUDA and checkpoint; run alone"]
    fn paired_dual_geglu() -> anyhow::Result<()> {
        crate::model::batch_bench::run_paired_cases(
            &REFERENCE_DUAL,
            crate::model::batch_bench::cases(),
        )
    }

    #[test]
    #[ignore = "requires CUDA and checkpoint; run alone"]
    fn paired_rounded_epilogue() -> anyhow::Result<()> {
        crate::model::batch_bench::run_paired_cases(
            &REFERENCE,
            crate::model::batch_bench::cases(),
        )
    }
    #[test]
    #[ignore = "requires CUDA and checkpoint; run alone"]
    fn paired_rounded_parallel_batches() -> anyhow::Result<()> {
        let model: crate::SystemOne =
            crate::SystemOne::from(crate::DEFAULT_REPO_ID)
                .with_device(candle_core::Device::new_cuda(0)?)
                .with_dtype(DType::BF16)
                .with_parallel_cuda_batches(true)
                .try_into()?;
        let cases = crate::model::batch_bench::cases()
            .into_iter()
            .filter(|(n, _)| ["64", "shared128"].contains(&n.as_str()))
            .collect();
        crate::model::batch_bench::run_paired_model(&REFERENCE, cases, model)
    }
}
