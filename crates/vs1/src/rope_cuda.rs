//! Paired rotary application for contiguous BF16 packed Q/K tensors.
use candle_core::{
    CpuStorage,
    CudaStorage,
    CustomOp3,
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

#[cfg(test)]
pub(crate) static REFERENCE_ROPE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

struct RopePair {
    sin: Tensor,
}

impl CustomOp3 for RopePair {
    fn name(&self) -> &'static str {
        "vs1_rope_pair"
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
        candle_core::bail!("paired rotary requires CUDA")
    }
    fn cuda_fwd(
        &self,
        q: &CudaStorage,
        ql: &Layout,
        k: &CudaStorage,
        kl: &Layout,
        cos: &CudaStorage,
        cl: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let (sin_storage, sl) = self.sin.storage_and_layout();
        let Storage::Cuda(sin) = &*sin_storage else {
            candle_core::bail!("rotary sine table requires CUDA")
        };
        let (
            CudaStorageSlice::BF16(qs),
            CudaStorageSlice::BF16(ks),
            CudaStorageSlice::BF16(cs),
            CudaStorageSlice::BF16(ss),
        ) = (&q.slice, &k.slice, &cos.slice, &sin.slice)
        else {
            candle_core::bail!("paired rotary requires BF16")
        };
        let (tokens, heads, dim) = ql.shape().dims3()?;
        let count = ql.shape().elem_count();
        let dev = q.device();
        let func = dev.get_or_load_custom_func(
            "rope_pair_bf16",
            "vs1_rope_pair",
            include_str!(concat!(env!("OUT_DIR"), "/rope_pair.ptx")),
        )?;
        let q = qs.slice(ql.start_offset()..ql.start_offset() + count);
        let k = ks.slice(kl.start_offset()..kl.start_offset() + count);
        let cs_count = tokens * dim / 2;
        let cos = cs.slice(cl.start_offset()..cl.start_offset() + cs_count);
        let sin = ss.slice(sl.start_offset()..sl.start_offset() + cs_count);
        // SAFETY: every output element is initialized by the launch.
        let mut out = unsafe { dev.alloc(2 * count)? };
        let (n, h, d) = (count as u32, heads as u32, dim as u32);
        let mut launch = func.builder();
        launch
            .arg(&q)
            .arg(&k)
            .arg(&cos)
            .arg(&sin)
            .arg(&mut out)
            .arg(&n)
            .arg(&h)
            .arg(&d);
        let config = LaunchConfig {
            grid_dim: ((n / 2).div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: forward validates dimensions, layouts and devices before launch.
        unsafe { launch.launch(config) }.w()?;
        Ok((
            CudaStorage {
                slice: CudaStorageSlice::BF16(out),
                device: dev.clone(),
            },
            (2, tokens, heads, dim).into(),
        ))
    }
}

pub(crate) fn forward(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let (tokens, heads, dim) = q.dims3()?;
    if tokens == 0
        || heads == 0
        || dim == 0
        || !dim.is_multiple_of(2)
        || q.elem_count() > u32::MAX as usize / 2
        || q.dims() != k.dims()
        || cos.dims() != [tokens, dim / 2]
        || cos.dims() != sin.dims()
        || [q, k, cos, sin].iter().any(|t| {
            !t.is_contiguous()
                || t.dtype() != DType::BF16
                || !t.device().same_device(q.device())
        })
    {
        candle_core::bail!("invalid packed BF16 rotary inputs")
    }
    let out = q.apply_op3_no_bwd(k, cos, &RopePair { sin: sin.clone() })?;
    Ok((out.get(0)?, out.get(1)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires CUDA"]
    fn rotary_matches_candle_bits() -> Result<()> {
        let device = candle_core::Device::new_cuda(0)?;
        for (t, h, d) in [(1, 1, 16), (7, 3, 80), (129, 12, 64), (85, 12, 64)] {
            let n = t * h * d;
            let values: Vec<f32> = if t == 85 {
                (0u32..=65535)
                    .filter(|v| v & 0x7f80 != 0x7f80)
                    .map(|v| f32::from_bits(v << 16))
                    .collect()
            } else {
                (0..n)
                    .map(|i| ((i * 37 % 4093) as f32 - 2046.0) / 127.0)
                    .collect()
            };
            let q = Tensor::from_vec(values.clone(), (t, h, d), &device)?
                .to_dtype(DType::BF16)?;
            let k = Tensor::from_vec(
                values.into_iter().rev().collect::<Vec<_>>(),
                (t, h, d),
                &device,
            )?
            .to_dtype(DType::BF16)?;
            let angles: Vec<_> =
                (0..t * d / 2).map(|i| i as f32 / 19.0).collect();
            let cos = Tensor::from_vec(
                angles.iter().map(|v| v.cos()).collect::<Vec<_>>(),
                (t, d / 2),
                &device,
            )?
            .to_dtype(DType::BF16)?;
            let sin = Tensor::from_vec(
                angles.iter().map(|v| v.sin()).collect::<Vec<_>>(),
                (t, d / 2),
                &device,
            )?
            .to_dtype(DType::BF16)?;
            // Exercise nonzero input/table offsets without changing the data.
            let q = Tensor::cat(&[&q, &q], 0)?.narrow(0, t, t)?;
            let cos = Tensor::cat(&[&cos, &cos], 0)?.narrow(0, t, t)?;
            let sin = Tensor::cat(&[&sin, &sin], 0)?.narrow(0, t, t)?;
            let (aq, ak) = forward(&q, &k, &cos, &sin)?;
            for (input, actual) in [(&q, aq), (&k, ak)] {
                let expected = candle_nn::rotary_emb::rope_thd(
                    &input.unsqueeze(0)?,
                    &cos,
                    &sin,
                )?;
                let expected = expected
                    .flatten_all()?
                    .to_dtype(DType::F32)?
                    .to_vec1::<f32>()?;
                let actual = actual
                    .flatten_all()?
                    .to_dtype(DType::F32)?
                    .to_vec1::<f32>()?;
                for (i, (a, e)) in actual.iter().zip(&expected).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        e.to_bits(),
                        "shape {t}/{h}/{d}, element {i}: {a} vs {e}"
                    );
                }
            }
        }
        Ok(())
    }
    #[test]
    #[ignore = "requires CUDA and checkpoint; run alone"]
    fn paired_rope_latency() -> anyhow::Result<()> {
        crate::geglu_bench::run_paired(&REFERENCE_ROPE)
    }
}
