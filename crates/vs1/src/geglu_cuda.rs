//! BF16 CUDA GELU followed by a gate, with Candle's rounding preserved.
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

struct GeGlu;

// Test-only switch for adjacent baseline/candidate calls on one loaded model.
#[cfg(test)]
pub(crate) static REFERENCE_MLP: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

#[cfg(test)]
pub(crate) static REFERENCE_VECTOR: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

impl CustomOp2 for GeGlu {
    fn name(&self) -> &'static str {
        "vs1_geglu_bf16"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("GeGLU CUDA operation received CPU storage")
    }

    fn cuda_fwd(
        &self,
        activation: &CudaStorage,
        al: &Layout,
        gate: &CudaStorage,
        gl: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let (CudaStorageSlice::BF16(a), CudaStorageSlice::BF16(g)) =
            (&activation.slice, &gate.slice)
        else {
            candle_core::bail!("fused GeGLU requires BF16 inputs")
        };
        if !al.is_contiguous()
            || !gl.is_contiguous()
            || al.shape() != gl.shape()
        {
            candle_core::bail!(
                "fused GeGLU requires contiguous, equally shaped inputs"
            )
        }
        let ptx = include_str!(concat!(env!("OUT_DIR"), "/geglu.ptx"));
        let dev = activation.device();
        let paired = al.start_offset().is_multiple_of(2)
            && gl.start_offset().is_multiple_of(2);
        #[cfg(test)]
        let paired = paired
            && !REFERENCE_VECTOR.load(std::sync::atomic::Ordering::Relaxed);
        let name = if paired {
            "geglu_bf16_pair"
        } else {
            "geglu_bf16"
        };
        let func = dev.get_or_load_custom_func(name, "vs1_geglu", ptx)?;
        let count = al.shape().elem_count();
        let count_u32 = u32::try_from(count)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let a = a.slice(al.start_offset()..al.start_offset() + count);
        let g = g.slice(gl.start_offset()..gl.start_offset() + count);
        // SAFETY: the launch writes every element before returning the tensor.
        let mut output = unsafe { dev.alloc(count)? };
        let mut launch = func.builder();
        launch.arg(&a).arg(&g).arg(&mut output).arg(&count_u32);
        // SAFETY: matching contiguous BF16 buffers have exactly count elements.
        let config = if paired {
            LaunchConfig {
                grid_dim: (count_u32.div_ceil(2).div_ceil(256), 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            }
        } else {
            LaunchConfig::for_num_elems(count_u32)
        };
        unsafe { launch.launch(config) }.w()?;
        Ok((
            CudaStorage {
                slice: CudaStorageSlice::BF16(output),
                device: dev.clone(),
            },
            al.shape().clone(),
        ))
    }
}

pub(crate) fn forward(activation: &Tensor, gate: &Tensor) -> Result<Tensor> {
    if activation.device().is_cuda()
        && activation.dtype() == DType::BF16
        && gate.dtype() == DType::BF16
        && activation.is_contiguous()
        && gate.is_contiguous()
        && activation.elem_count() > 0
        && activation.elem_count() <= u32::MAX as usize
    {
        activation.apply_op2_no_bwd(gate, &GeGlu)
    } else {
        activation.gelu_erf()? * gate
    }
}

#[cfg(test)]
mod tests {
    use candle_core::Device;

    use super::*;

    fn assert_exact(activation: &Tensor, gate: &Tensor) -> Result<()> {
        let expected = (activation.gelu_erf()? * gate)?
            .flatten_all()?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?;
        let actual = forward(activation, gate)?
            .flatten_all()?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?;
        let mismatches: Vec<_> = actual
            .iter()
            .zip(&expected)
            .enumerate()
            .filter(|(_, (a, e))| a.to_bits() != e.to_bits())
            .take(5)
            .collect();
        assert!(
            mismatches.is_empty(),
            "first differing values: {mismatches:?}"
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires an Ampere or newer CUDA GPU"]
    fn all_finite_bf16_activations_match_candle_exactly() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let values: Vec<f32> = (0u32..=u16::MAX as u32)
            .filter(|bits| bits & 0x7f80 != 0x7f80)
            .map(|bits| f32::from_bits(bits << 16))
            .collect();
        let n = values.len();
        let activation = Tensor::from_vec(values.clone(), n, &device)?
            .to_dtype(DType::BF16)?;
        // Every finite BF16 value on both inputs, with several different pairings.
        for offset in [0, 1, 127, 8191, 32767] {
            let gates: Vec<_> =
                (0..n).map(|i| values[(i + offset) % n]).collect();
            let gate =
                Tensor::from_vec(gates, n, &device)?.to_dtype(DType::BF16)?;
            assert_exact(&activation, &gate)?;
        }
        for scalar in [
            0.0f32,
            -0.0,
            1.0,
            -1.0,
            0.5,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
        ] {
            let gate = Tensor::from_vec(vec![scalar; n], n, &device)?
                .to_dtype(DType::BF16)?;
            assert_exact(&activation, &gate)?;
        }
        // Offsets, launch tails and the fallback for non-contiguous/F32 inputs.
        let a = activation.narrow(0, 16123, 257)?.reshape((1, 257))?;
        let g = activation.narrow(0, 17017, 257)?.reshape((1, 257))?;
        assert_exact(&a, &g)?;
        let even = activation.narrow(0, 16124, 257)?;
        assert_exact(&even, &even)?;
        let strided =
            activation.narrow(0, 15401, 1024)?.reshape((32, 32))?.t()?;
        assert_exact(&strided, &strided)?;
        assert_exact(&a.to_dtype(DType::F32)?, &a.to_dtype(DType::F32)?)?;
        for scalar in [f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
            let a = Tensor::from_vec(vec![scalar; 7], 7, &device)?
                .to_dtype(DType::BF16)?;
            let g = Tensor::from_vec(
                vec![
                    0.0f32,
                    -0.0,
                    1.0,
                    -1.0,
                    f32::INFINITY,
                    f32::NEG_INFINITY,
                    f32::NAN,
                ],
                7,
                &device,
            )?
            .to_dtype(DType::BF16)?;
            assert_exact(&a, &g)?;
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires CUDA and the Laya checkpoint; run alone"]
    fn paired_vector_latency() -> anyhow::Result<()> {
        crate::geglu_bench::run_paired(&REFERENCE_VECTOR)
    }
}
