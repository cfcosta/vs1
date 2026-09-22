//! Overlap independent encoder projections on the measured CUDA configuration.
//! Matrix dimensions, BF16 output boundaries and the Candle cuBLAS call stay
//! unchanged. Small products retain their original serial implementation.
use std::{cell::RefCell, sync::Arc};

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
        cudarc::{
            cublas::{CudaBlas, result, sys},
            driver::{CudaStream, DevicePtr, DevicePtrMut},
        },
    },
};
use candle_nn::Linear;

struct Lane {
    blas: CudaBlas,
    stream: Arc<CudaStream>,
}
struct Pool {
    context: usize,
    lanes: Vec<Lane>,
}
impl Drop for Pool {
    fn drop(&mut self) {
        for lane in &self.lanes {
            let _ = lane.stream.synchronize();
        }
    }
}
thread_local! {
    // At most three streams/handles per calling thread, reused across layers and
    // models in the same context. A context switch replaces the previous pool.
    static POOL: RefCell<Option<Pool>> = const { RefCell::new(None) };
    static SUPPORTED: RefCell<Option<(usize, bool)>> = const { RefCell::new(None) };
}
#[cfg(test)]
pub(crate) static REFERENCE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

pub(crate) fn enabled(kind: &str, xs: &Tensor) -> bool {
    #[cfg(test)]
    if REFERENCE.load(std::sync::atomic::Ordering::Relaxed) {
        return false;
    }
    if !xs.device().is_cuda()
        || xs.dtype() != DType::BF16
        || xs.rank() != 2
        || !xs.is_contiguous()
        || candle_core::cuda_backend::gemm_reduced_precision_bf16()
    {
        return false;
    }
    let minimum_rows = if kind == "qkv" { 1536 } else { 2048 };
    if xs.dims()[0] < minimum_rows || xs.dims()[1] != 1024 {
        return false;
    }
    supported_device(xs)
}

pub(crate) fn supported_device(xs: &Tensor) -> bool {
    let Ok(dev) = xs.device().as_cuda_device() else {
        return false;
    };
    let stream = dev.cuda_stream();
    let context = stream.context().cu_ctx() as usize;
    SUPPORTED.with(|supported| {
        let mut supported = supported.borrow_mut();
        if let Some((key, value)) = *supported
            && key == context
        {
            return value;
        }
        let value = (|| -> Result<bool> {
            stream.context().bind_to_thread().w()?;
            let mut version = 0;
            // SAFETY: a live cuBLAS handle and writable integer output.
            unsafe {
                sys::cublasGetVersion_v2(
                    *dev.cublas_handle().handle(),
                    &mut version,
                )
                .result()
                .w()?;
            }
            Ok(stream.context().name().w()? == "NVIDIA GeForce RTX 3080 Ti"
                && version == 120901)
        })()
        .unwrap_or(false);
        *supported = Some((context, value));
        value
    })
}
struct Projection<'a>(&'a Lane);
impl CustomOp2 for Projection<'_> {
    fn name(&self) -> &'static str {
        "vs1_parallel_projection"
    }
    fn cpu_fwd(
        &self,
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
    ) -> Result<(CudaStorage, Shape)> {
        let (m, k) = xl.shape().dims2()?;
        let (n, wk) = wl.shape().dims2()?;
        if wk != k || !xl.is_contiguous() || !wl.is_contiguous() {
            candle_core::bail!("projection layout")
        }
        let (CudaStorageSlice::BF16(xb), CudaStorageSlice::BF16(wb)) =
            (&x.slice, &w.slice)
        else {
            candle_core::bail!("BF16 only")
        };
        let stream = &self.0.stream;
        let xv = xb.slice(xl.start_offset()..xl.start_offset() + m * k);
        let wv = wb.slice(wl.start_offset()..wl.start_offset() + n * k);
        // SAFETY: beta=0; GEMM writes the entire allocation before any read.
        let mut output = unsafe { stream.alloc(m * n).w()? };
        let (xp, xguard) = xv.device_ptr(stream);
        let (wp, wguard) = wv.device_ptr(stream);
        let (yp, yguard) = output.device_ptr_mut(stream);
        let alpha = 1f32;
        let beta = 0f32;
        // Same call, strides, precision and algorithm selector as Candle's BF16
        // matrix product. Separate handles/streams permit independent launches.
        // SAFETY: checked contiguous BF16 buffers have m*k, n*k and m*n
        // elements; guards and explicit stream joins order allocation/use.
        unsafe {
            result::gemm_strided_batched_ex(
                *self.0.blas.handle(),
                sys::cublasOperation_t::CUBLAS_OP_T,
                sys::cublasOperation_t::CUBLAS_OP_N,
                n as i32,
                m as i32,
                k as i32,
                (&alpha as *const f32).cast(),
                wp as *const _,
                sys::cudaDataType_t::CUDA_R_16BF,
                k as i32,
                (n * k) as i64,
                xp as *const _,
                sys::cudaDataType_t::CUDA_R_16BF,
                k as i32,
                (m * k) as i64,
                (&beta as *const f32).cast(),
                yp as *mut _,
                sys::cudaDataType_t::CUDA_R_16BF,
                n as i32,
                (m * n) as i64,
                1,
                sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
            )
            .w()?;
        }
        drop((xguard, wguard, yguard));
        Ok((
            CudaStorage {
                slice: CudaStorageSlice::BF16(output),
                device: x.device().clone(),
            },
            (m, n).into(),
        ))
    }
}
pub(crate) fn project(xs: &Tensor, weights: &[&Linear]) -> Result<Vec<Tensor>> {
    if weights.is_empty() || weights.len() > 3 {
        candle_core::bail!(
            "projection group must contain one to three matrices"
        )
    }
    let main = xs.device().as_cuda_device()?.cuda_stream();
    main.context().bind_to_thread().w()?;
    let context = main.context().cu_ctx() as usize;
    POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        if pool.as_ref().is_none_or(|p| p.context != context) {
            let mut lanes = Vec::new();
            for _ in 0..3 {
                let stream = main.context().new_stream().w()?;
                let blas = CudaBlas::new(stream.clone()).w()?;
                lanes.push(Lane { blas, stream });
            }
            *pool = Some(Pool { context, lanes });
        }
        let pool = pool.as_ref().unwrap();
        let mut outputs = Vec::with_capacity(weights.len());
        for (linear, lane) in weights.iter().zip(&pool.lanes) {
            if linear.bias().is_some() {
                candle_core::bail!("bias unsupported")
            }
            lane.stream.join(&main).w()?;
            outputs
                .push(xs.apply_op2_no_bwd(linear.weight(), &Projection(lane))?);
        }
        for lane in pool.lanes.iter().take(weights.len()) {
            main.join(&lane.stream).w()?;
        }
        Ok(outputs)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires CUDA; run alone"]
    fn independent_products_match_candle() -> Result<()> {
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
        for n in [1024, 2624] {
            let weights: Vec<_> = (0..3)
                .map(|_| Ok(Linear::new(tensor(n, 1024)?, None)))
                .collect::<Result<_>>()?;
            let refs: Vec<_> = weights.iter().collect();
            for m in [512, 597, 832, 1536, 2048, 3870, 5400] {
                let xs = tensor(m, 1024)?;
                assert_eq!(enabled("qkv", &xs), m >= 1536);
                for input in [&xs, &xs.neg()?] {
                    let actual = project(input, &refs)?;
                    for (weight, output) in weights.iter().zip(actual) {
                        let expected = input
                            .apply(weight)?
                            .flatten_all()?
                            .to_dtype(DType::F32)?
                            .to_vec1::<f32>()?;
                        let actual = output
                            .flatten_all()?
                            .to_dtype(DType::F32)?
                            .to_vec1::<f32>()?;
                        assert_eq!(
                            actual
                                .iter()
                                .zip(&expected)
                                .filter(|(a, b)| a.to_bits() != b.to_bits())
                                .count(),
                            0,
                            "m={m}, n={n}"
                        );
                    }
                }
            }
        }
        assert!(!enabled(
            "qkv",
            &Tensor::zeros((129, 1024), DType::BF16, &device)?
        ));
        assert!(!enabled(
            "ffn",
            &Tensor::zeros((1536, 1024), DType::BF16, &device)?
        ));
        Ok(())
    }
}
