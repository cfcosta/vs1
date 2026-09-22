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
