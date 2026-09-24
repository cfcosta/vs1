//! Dual GEMM with Candle's CUDA BF16 SwiGLU rounding for Qwen3.5-4B.
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
    fn vs1_cutlass_dual_swiglu(
        x: *const c_void,
        gate: *const c_void,
        up: *const c_void,
        out: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
        stream: *mut c_void,
    ) -> i32;
}

#[derive(Debug, thiserror::Error)]
#[error("CUTLASS cannot implement the SwiGLU operands")]
struct UnsupportedSwiGluOperands;

struct DualSwiGlu;

impl CustomOp3 for DualSwiGlu {
    fn name(&self) -> &'static str {
        "vs1_cutlass_dual_swiglu"
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
        candle_core::bail!("dual SwiGLU requires CUDA")
    }
    fn cuda_fwd(
        &self,
        x: &CudaStorage,
        xl: &Layout,
        gate: &CudaStorage,
        gl: &Layout,
        up: &CudaStorage,
        ul: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let (m, k) = xl.shape().dims2()?;
        let (n, gk) = gl.shape().dims2()?;
        if k != 2560
            || n != 9216
            || gk != k
            || ul.dims() != [n, k]
            || !xl.is_contiguous()
            || !gl.is_contiguous()
            || !ul.is_contiguous()
        {
            candle_core::bail!("invalid dual SwiGLU layouts")
        }
        let (
            CudaStorageSlice::BF16(xs),
            CudaStorageSlice::BF16(gs),
            CudaStorageSlice::BF16(us),
        ) = (&x.slice, &gate.slice, &up.slice)
        else {
            candle_core::bail!("dual SwiGLU requires BF16")
        };
        let dev = x.device();
        let stream = dev.cuda_stream();
        let x = xs.slice(xl.start_offset()..xl.start_offset() + m * k);
        let gate = gs.slice(gl.start_offset()..gl.start_offset() + n * k);
        let up = us.slice(ul.start_offset()..ul.start_offset() + n * k);
        // SAFETY: a successful GEMM writes every output element before use.
        let mut output = unsafe { dev.alloc(m * n)? };
        let (xp, xguard) = x.device_ptr(&stream);
        let (gp, gguard) = gate.device_ptr(&stream);
        let (up, uguard) = up.device_ptr(&stream);
        let (yp, yguard) = output.device_ptr_mut(&stream);
        stream.context().bind_to_thread().w()?;
        // SAFETY: checked contiguous BF16 matrices and dimensions; pointer
        // guards order reads and writes on the same CUDA stream.
        let status = unsafe {
            vs1_cutlass_dual_swiglu(
                xp as *const _,
                gp as *const _,
                up as *const _,
                yp as *mut _,
                i32::try_from(m).map_err(candle_core::Error::wrap)?,
                n as i32,
                k as i32,
                stream.cu_stream().cast(),
            )
        };
        if status == -1 {
            return Err(candle_core::Error::Cuda(Box::new(
                UnsupportedSwiGluOperands,
            )));
        }
        if status != 0 {
            candle_core::bail!("CUTLASS dual SwiGLU status {status}")
        }
        drop((xguard, gguard, uguard, yguard));
        Ok((
            CudaStorage {
                slice: CudaStorageSlice::BF16(output),
                device: dev.clone(),
            },
            (m, n).into(),
        ))
    }
}

/// Returns `None` for unsupported shapes, layouts or devices before launch.
pub(super) fn project_swiglu(
    x: &Tensor,
    gate: &Tensor,
    up: &Tensor,
) -> Result<Option<Tensor>> {
    if !x.device().is_cuda()
        || !(2..=4).contains(&x.rank())
        || x.dims().last() != Some(&2560)
        || gate.dims() != [9216, 2560]
        || up.dims() != gate.dims()
        || x.elem_count() == 0
        || x.elem_count() / 2560 > i32::MAX as usize / 9216
        || candle_core::cuda_backend::gemm_reduced_precision_bf16()
        || [x, gate, up].iter().any(|t| {
            t.dtype() != DType::BF16
                || !t.is_contiguous()
                || !t.layout().start_offset().is_multiple_of(8)
                || !t.device().same_device(x.device())
        })
    {
        return Ok(None);
    }
    let dev = x.device().as_cuda_device()?;
    if dev.cuda_stream().context().compute_capability().w()?.0 < 8 {
        return Ok(None);
    }
    let rows = x.elem_count() / 2560;
    let flat = x.reshape((rows, 2560))?;
    let output = match flat.apply_op3_no_bwd(gate, up, &DualSwiGlu) {
        Ok(output) => output,
        Err(candle_core::Error::Cuda(error))
            if error.is::<UnsupportedSwiGluOperands>() =>
        {
            return Ok(None);
        }
        Err(error) => return Err(error),
    };
    let mut shape = x.dims().to_vec();
    *shape.last_mut().unwrap() = 9216;
    output.reshape(shape).map(Some)
}
