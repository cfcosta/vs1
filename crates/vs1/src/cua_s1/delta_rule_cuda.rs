//! F32 delta-rule recurrence with parallel reductions and a serial fallback.
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
    is_parallel: bool,
    initial_state: Option<Tensor>,
    should_save_state: bool,
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
        let (kernel, columns_per_block, threads_per_block) = if self.is_parallel
        {
            ("apply_delta_rule_parallel_f32", 32, 256)
        } else {
            ("apply_delta_rule_f32", value_dim, value_dim)
        };
        let func = dev.get_or_load_custom_func(
            kernel,
            "vs1_delta_rule",
            include_str!(concat!(env!("OUT_DIR"), "/gated_delta.ptx")),
        )?;
        // The register-resident state can limit a block below 1024 threads.
        let max_threads = func.max_threads_per_block().w()? as usize;
        if threads_per_block > max_threads {
            candle_core::bail!(
                "delta-rule block size {threads_per_block} exceeds the kernel's block limit {max_threads}"
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
        let initial_storage =
            self.initial_state.as_ref().map(Tensor::storage_and_layout);
        let initial_state = match initial_storage.as_ref() {
            Some((storage, layout)) => {
                let Storage::Cuda(storage) = &**storage else {
                    candle_core::bail!("delta-rule state requires CUDA")
                };
                let CudaStorageSlice::F32(slice) = &storage.slice else {
                    candle_core::bail!("delta-rule state requires F32")
                };
                Some(slice.slice(
                    layout.start_offset()
                        ..layout.start_offset() + heads * key_dim * value_dim,
                ))
            }
            None => None,
        };
        let state_count = if self.should_save_state {
            heads * key_dim * value_dim
        } else {
            0
        };
        // SAFETY: every output and requested final-state element is written.
        let mut output = unsafe { dev.alloc(value_count + state_count)? };
        let should_save_state = u32::from(self.should_save_state);
        let null = 0u64;
        let (seq, heads, value_dim) =
            (seq as u32, heads as u32, value_dim as u32);
        let mut launch = func.builder();
        launch
            .arg(&query)
            .arg(&key)
            .arg(&value)
            .arg(&beta)
            .arg(&decay);
        match &initial_state {
            Some(state) => launch.arg(state),
            None => launch.arg(&null),
        };
        launch
            .arg(&mut output)
            .arg(&seq)
            .arg(&heads)
            .arg(&value_dim)
            .arg(&should_save_state);
        // SAFETY: apply_delta_rule checks shapes, devices, dtype and contiguity;
        // parallel blocks cover complete groups of 32 value columns, and the
        // kernel's block limit is checked above.
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (heads, value_dim / columns_per_block as u32, 1),
                block_dim: (threads_per_block as u32, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .w()?;
        Ok((
            CudaStorage {
                slice: CudaStorageSlice::F32(output),
                device: dev.clone(),
            },
            if self.should_save_state {
                Shape::from(value_count + state_count)
            } else {
                vl.shape().clone()
            },
        ))
    }
}

/// Applies the recurrence with repeated, normalized/scaled Q/K and `exp(g)` decay.
/// Uses parallel reductions for the model's 128 value columns, serial otherwise.
pub(crate) fn apply_delta_rule(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    beta: &Tensor,
    decay: &Tensor,
) -> Result<Tensor> {
    DeltaRule {
        beta: beta.clone(),
        decay: decay.clone(),
        is_parallel: value.dim(2)? == 128,
        initial_state: None,
        should_save_state: false,
    }
    .apply(query, key, value)
}

pub(crate) fn apply_delta_rule_with_state(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    beta: &Tensor,
    decay: &Tensor,
    initial_state: Option<&Tensor>,
) -> Result<(Tensor, Tensor)> {
    let output = DeltaRule {
        beta: beta.clone(),
        decay: decay.clone(),
        is_parallel: value.dim(2)? == 128,
        initial_state: initial_state.cloned(),
        should_save_state: true,
    }
    .apply(query, key, value)?;
    split_output_and_state(&output, value, query.dim(2)?)
}

fn split_output_and_state(
    output: &Tensor,
    value: &Tensor,
    key_dim: usize,
) -> Result<(Tensor, Tensor)> {
    let (_, heads, value_dim) = value.dims3()?;
    let state = output
        .narrow(0, value.elem_count(), heads * key_dim * value_dim)?
        .reshape((heads, key_dim, value_dim))?
        .copy()?;
    Ok((
        output
            .narrow(0, 0, value.elem_count())?
            .reshape(value.shape())?,
        state,
    ))
}

impl DeltaRule {
    fn apply(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
    ) -> Result<Tensor> {
        let (seq, heads, key_dim) = query.dims3()?;
        let (value_seq, value_heads, value_dim) = value.dims3()?;
        if seq == 0
            || heads == 0
            || key_dim != 128
            || value_dim == 0
            || value_dim > 1024
            || (self.is_parallel && value_dim != 128)
            || key.shape() != query.shape()
            || (value_seq, value_heads) != (seq, heads)
            || self.beta.dims() != [seq, heads]
            || self.decay.dims() != [seq, heads]
            || query.elem_count() > u32::MAX as usize
            || (seq + key_dim) * heads * value_dim > u32::MAX as usize
        {
            candle_core::bail!(
                "delta-rule kernel requires matching nonempty dimensions, key_dim 128 and value_dim fitting one block"
            )
        }
        if let Some(state) = &self.initial_state {
            super::delta_net::validate_state(
                state,
                &[heads, key_dim, value_dim],
                DType::F32,
                query.device(),
            )?;
        }
        if !query.device().is_cuda()
            || [query, key, value, &self.beta, &self.decay]
                .into_iter()
                .chain(self.initial_state.as_ref())
                .any(|t| {
                    !t.is_contiguous()
                        || t.dtype() != DType::F32
                        || !t.device().same_device(query.device())
                })
        {
            candle_core::bail!(
                "delta-rule kernel requires contiguous F32 inputs on one CUDA device"
            )
        }
        query.apply_op3_no_bwd(key, value, self)
    }
}

#[cfg(test)]
mod tests {
    use candle_core::Device;

    use super::*;
    use crate::cua_s1::{delta_net, normalize_l2};

    #[test]
    #[ignore = "requires CUDA"]
    fn continues_both_delta_rule_kernels_with_identical_output_and_state()
    -> Result<()> {
        use delta_net::tests::assert_same_bits;

        let device = Device::new_cuda(0)?;
        device.set_seed(42)?;
        let (seq, heads, dim) = (1860, 32, 128);
        let sample = |shape: &[usize]| -> Result<Tensor> {
            Tensor::rand(-1f32, 1., shape, &device)?
                .to_dtype(DType::BF16)?
                .to_dtype(DType::F32)
        };
        let query = (normalize_l2(
            &sample(&[seq + 1, heads, dim])?.narrow(0, 1, seq)?,
        )? * (dim as f64).sqrt().recip())?;
        let key =
            normalize_l2(&sample(&[seq + 2, heads, dim])?.narrow(0, 2, seq)?)?;
        let value = sample(&[seq + 3, heads, dim])?.narrow(0, 3, seq)?;
        let beta = candle_nn::ops::sigmoid(&sample(&[seq, heads])?)?;
        let decay = sample(&[seq, heads])?.abs()?.neg()?.exp()?;
        let zero = Tensor::zeros((heads, dim, dim), DType::F32, &device)?;
        for is_parallel in [false, true] {
            let apply = |start,
                         len,
                         state: Option<&Tensor>,
                         should_save_state|
             -> Result<Tensor> {
                DeltaRule {
                    beta: beta.narrow(0, start, len)?,
                    decay: decay.narrow(0, start, len)?,
                    is_parallel,
                    initial_state: state.cloned(),
                    should_save_state,
                }
                .apply(
                    &query.narrow(0, start, len)?,
                    &key.narrow(0, start, len)?,
                    &value.narrow(0, start, len)?,
                )
            };
            let (whole, whole_state) = split_output_and_state(
                &apply(0, seq, None, true)?,
                &value,
                dim,
            )?;
            assert_same_bits(&whole, &apply(0, seq, None, false)?)?;
            let (output, state) = split_output_and_state(
                &apply(0, seq, Some(&zero), true)?,
                &value,
                dim,
            )?;
            assert_same_bits(&output, &whole)?;
            assert_same_bits(&state, &whole_state)?;
            for split in [1, 2, 930, seq - 1] {
                let (prefix_output, prefix_state) = split_output_and_state(
                    &apply(0, split, None, true)?,
                    &value.narrow(0, 0, split)?,
                    dim,
                )?;
                let prefix_state = Tensor::cat(&[&zero, &prefix_state], 0)?
                    .narrow(0, heads, heads)?;
                let saved_state = prefix_state.copy()?;
                let (suffix_output, state) = split_output_and_state(
                    &apply(split, seq - split, Some(&prefix_state), true)?,
                    &value.narrow(0, split, seq - split)?,
                    dim,
                )?;
                assert_same_bits(
                    &Tensor::cat(&[&prefix_output, &suffix_output], 0)?,
                    &whole,
                )?;
                assert_same_bits(&state, &whole_state)?;
                assert_same_bits(&prefix_state, &saved_state)?;
            }
        }
        Ok(())
    }

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
            let actual = to_cuda(&query, 1)?.apply_op3_no_bwd(
                &to_cuda(&key, 2)?,
                &to_cuda(&value, 3)?,
                &DeltaRule {
                    beta: to_cuda(&beta, 4)?,
                    decay: to_cuda(&decay, 5)?,
                    is_parallel: false,
                    initial_state: None,
                    should_save_state: false,
                },
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
    fn parallel_delta_rule_matches_cpu() -> Result<()> {
        let device = Device::new_cuda(0)?;
        device.set_seed(42)?;
        let heads = 32;
        let key_dim = 128;
        let value_dim = 128;
        for seq in [1, 7, 300, 2000] {
            let query =
                Tensor::rand(-1f32, 1., (seq, heads, key_dim), &device)?
                    .to_device(&Device::Cpu)?;
            let key = Tensor::rand(-1f32, 1., (seq, heads, key_dim), &device)?
                .to_device(&Device::Cpu)?;
            let query =
                (normalize_l2(&query)? * (key_dim as f64).sqrt().recip())?;
            let key = normalize_l2(&key)?;
            let value =
                Tensor::rand(-1f32, 1., (seq, heads, value_dim), &device)?
                    .to_device(&Device::Cpu)?;
            let beta = Tensor::rand(0f32, 1., (seq, heads), &device)?
                .to_device(&Device::Cpu)?;
            let decay = Tensor::rand(0f32, 1., (seq, heads), &device)?
                .to_device(&Device::Cpu)?
                .neg()?
                .exp()?;
            let expected = delta_net::apply_delta_rule(
                &query, &key, &value, &beta, &decay,
            )?;
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
            let mut max_absolute_difference = 0f32;
            let mut max_relative_difference = 0f32;
            let mut is_close = true;
            for (actual, expected) in actual.into_iter().zip(expected) {
                let difference = (actual - expected).abs();
                max_absolute_difference =
                    max_absolute_difference.max(difference);
                max_relative_difference = max_relative_difference
                    .max(difference / expected.abs().max(1e-6));
                is_close &= actual.is_finite()
                    && expected.is_finite()
                    && difference <= 1e-6 + 1e-5 * expected.abs();
            }
            println!(
                "parallel delta-rule seq={seq}: max absolute difference {max_absolute_difference:e}, max relative difference {max_relative_difference:e} (denominator floored at 1e-6)"
            );
            assert!(is_close, "seq={seq}: exceeds atol=1e-6, rtol=1e-5");
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
