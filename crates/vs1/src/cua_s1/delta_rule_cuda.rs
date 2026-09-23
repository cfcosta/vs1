//! F32 delta-rule recurrence with the CPU loop's arithmetic order.
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

struct DeltaRule {
    beta: Tensor,
    decay: Tensor,
}

impl CustomOp3 for DeltaRule {
    fn name(&self) -> &'static str {
        "vs1_delta_rule"
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
        candle_core::bail!("delta-rule kernel requires CUDA")
    }
    fn cuda_fwd(
        &self,
        query: &CudaStorage,
        ql: &Layout,
        key: &CudaStorage,
        kl: &Layout,
        value: &CudaStorage,
        vl: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let (beta_storage, bl) = self.beta.storage_and_layout();
        let (decay_storage, dl) = self.decay.storage_and_layout();
        let (Storage::Cuda(beta), Storage::Cuda(decay)) =
            (&*beta_storage, &*decay_storage)
        else {
            candle_core::bail!("delta-rule gates require CUDA")
        };
        let (
            CudaStorageSlice::F32(qs),
            CudaStorageSlice::F32(ks),
            CudaStorageSlice::F32(vs),
            CudaStorageSlice::F32(bs),
            CudaStorageSlice::F32(ds),
        ) = (
            &query.slice,
            &key.slice,
            &value.slice,
            &beta.slice,
            &decay.slice,
        )
        else {
            candle_core::bail!("delta-rule kernel requires F32")
        };
        let (seq, heads, key_dim) = ql.shape().dims3()?;
        let value_dim = vl.shape().dims3()?.2;
        let dev = query.device();
        let func = dev.get_or_load_custom_func(
            "apply_delta_rule_f32",
            "vs1_delta_rule",
            include_str!(concat!(env!("OUT_DIR"), "/gated_delta.ptx")),
        )?;
        // The register-resident state can limit a block below 1024 threads.
        let max_threads = func.max_threads_per_block().w()? as usize;
        if value_dim > max_threads {
            candle_core::bail!(
                "delta-rule value_dim {value_dim} exceeds the kernel's block limit {max_threads}"
            )
        }
        let key_count = seq * heads * key_dim;
        let value_count = seq * heads * value_dim;
        let gate_count = seq * heads;
        let query = qs.slice(ql.start_offset()..ql.start_offset() + key_count);
        let key = ks.slice(kl.start_offset()..kl.start_offset() + key_count);
        let value =
            vs.slice(vl.start_offset()..vl.start_offset() + value_count);
        let beta = bs.slice(bl.start_offset()..bl.start_offset() + gate_count);
        let decay = ds.slice(dl.start_offset()..dl.start_offset() + gate_count);
        // SAFETY: every output element is written by the kernel.
        let mut output = unsafe { dev.alloc(value_count)? };
        let (seq, heads, value_dim) =
            (seq as u32, heads as u32, value_dim as u32);
        let mut launch = func.builder();
        launch
            .arg(&query)
            .arg(&key)
            .arg(&value)
            .arg(&beta)
            .arg(&decay)
            .arg(&mut output)
            .arg(&seq)
            .arg(&heads)
            .arg(&value_dim);
        // SAFETY: apply_delta_rule checks shapes, devices, dtype and contiguity;
        // the kernel's block limit is checked above.
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (heads, 1, 1),
                block_dim: (value_dim, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .w()?;
        Ok((
            CudaStorage {
                slice: CudaStorageSlice::F32(output),
                device: dev.clone(),
            },
            vl.shape().clone(),
        ))
    }
}

/// Applies the recurrence with repeated, normalized/scaled Q/K and `exp(g)` decay.
pub(crate) fn apply_delta_rule(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    beta: &Tensor,
    decay: &Tensor,
) -> Result<Tensor> {
    let (seq, heads, key_dim) = query.dims3()?;
    let (value_seq, value_heads, value_dim) = value.dims3()?;
    if seq == 0
        || heads == 0
        || key_dim != 128
        || value_dim == 0
        || value_dim > 1024
        || key.shape() != query.shape()
        || (value_seq, value_heads) != (seq, heads)
        || beta.dims() != [seq, heads]
        || decay.dims() != [seq, heads]
        || query.elem_count() > u32::MAX as usize
        || value.elem_count() > u32::MAX as usize
    {
        candle_core::bail!(
            "delta-rule kernel requires matching nonempty dimensions, key_dim 128 and value_dim fitting one block"
        )
    }
    if !query.device().is_cuda()
        || [query, key, value, beta, decay].iter().any(|t| {
            !t.is_contiguous()
                || t.dtype() != DType::F32
                || !t.device().same_device(query.device())
        })
    {
        candle_core::bail!(
            "delta-rule kernel requires contiguous F32 inputs on one CUDA device"
        )
    }
    query.apply_op3_no_bwd(
        key,
        value,
        &DeltaRule {
            beta: beta.clone(),
            decay: decay.clone(),
        },
    )
}

#[cfg(test)]
mod tests {
    use candle_core::Device;

    use super::*;
    use crate::cua_s1::{delta_net, normalize_l2};

    #[test]
    #[ignore = "requires CUDA"]
    fn delta_rule_matches_cpu_bits() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut sample_values = |count| -> Vec<f32> {
            (0..count)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    (state >> 40) as f32 / (1u64 << 24) as f32
                })
                .collect()
        };
        let heads = 32;
        let key_dim = 128;
        for (seq, value_dim) in [
            (1, 128),
            (7, 128),
            (129, 128),
            (257, 128),
            (3, 1),
            (3, 31),
            (3, 129),
            (3, 256),
        ] {
            let query = Tensor::from_vec(
                sample_values(seq * heads * key_dim),
                (seq, heads, key_dim),
                &Device::Cpu,
            )?
            .affine(2., -1.)?;
            let key = Tensor::from_vec(
                sample_values(seq * heads * key_dim),
                (seq, heads, key_dim),
                &Device::Cpu,
            )?
            .affine(2., -1.)?;
            let query =
                (normalize_l2(&query)? * (key_dim as f64).sqrt().recip())?;
            let key = normalize_l2(&key)?;
            let value = Tensor::from_vec(
                sample_values(seq * heads * value_dim),
                (seq, heads, value_dim),
                &Device::Cpu,
            )?
            .affine(2., -1.)?;
            let beta = Tensor::from_vec(
                sample_values(seq * heads),
                (seq, heads),
                &Device::Cpu,
            )?;
            let decay = Tensor::from_vec(
                sample_values(seq * heads),
                (seq, heads),
                &Device::Cpu,
            )?
            .neg()?
            .exp()?;
            let expected = delta_net::apply_delta_rule(
                &query, &key, &value, &beta, &decay,
            )?;
            // Exercise independent nonzero storage offsets without changing data.
            let to_cuda = |tensor: &Tensor, padding| -> Result<Tensor> {
                let flat = tensor.flatten_all()?.to_device(&device)?;
                let prefix = Tensor::zeros(padding, DType::F32, &device)?;
                Tensor::cat(&[&prefix, &flat], 0)?
                    .narrow(0, padding, tensor.elem_count())?
                    .reshape(tensor.shape())
            };
            let actual = apply_delta_rule(
                &to_cuda(&query, 1)?,
                &to_cuda(&key, 2)?,
                &to_cuda(&value, 3)?,
                &to_cuda(&beta, 4)?,
                &to_cuda(&decay, 5)?,
            )?;
            assert_eq!(actual.dims(), expected.dims());
            assert_eq!(actual.dtype(), DType::F32);
            let actual = actual
                .flatten_all()?
                .to_device(&Device::Cpu)?
                .to_vec1::<f32>()?;
            let expected = expected.flatten_all()?.to_vec1::<f32>()?;
            for (i, (actual, expected)) in
                actual.iter().zip(&expected).enumerate()
            {
                assert_eq!(
                    actual.to_bits(),
                    expected.to_bits(),
                    "seq={seq}, value_dim={value_dim}, element {i}: {actual} vs {expected}"
                );
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires CUDA"]
    fn rejects_unsupported_delta_rule_inputs() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let query = Tensor::zeros((2, 2, 128), DType::F32, &device)?;
        let gate = Tensor::ones((2, 2), DType::F32, &device)?;
        for (index, invalid) in [
            (0, query.narrow(0, 0, 0)?),
            (0, query.narrow(1, 0, 0)?),
            (0, query.narrow(2, 0, 64)?),
            (1, query.narrow(0, 0, 1)?),
            (2, query.narrow(2, 0, 0)?),
            (2, Tensor::zeros((2, 2, 1025), DType::F32, &device)?),
            (3, gate.narrow(0, 0, 1)?),
            (4, gate.narrow(1, 0, 1)?),
            (0, query.transpose(0, 1)?),
            (2, query.to_dtype(DType::BF16)?),
            (4, gate.to_device(&Device::Cpu)?),
        ] {
            let mut inputs = [&query, &query, &query, &gate, &gate];
            inputs[index] = &invalid;
            assert!(
                apply_delta_rule(
                    inputs[0], inputs[1], inputs[2], inputs[3], inputs[4],
                )
                .is_err(),
                "accepted unsupported input {index}: {invalid:?}"
            );
        }
        Ok(())
    }
}
