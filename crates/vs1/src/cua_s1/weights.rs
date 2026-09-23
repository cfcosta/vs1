use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
};

use candle_core::{DType, Device, Tensor, safetensors::MmapedSafetensors};
use serde::Deserialize;

use super::LoraAdapter;
use crate::{Result, SystemOneError};

#[derive(Deserialize)]
struct CheckpointIndex {
    weight_map: BTreeMap<String, String>,
}

/// Memory-mapped Qwen3.5 text weights with optional PEFT linear-weight merges.
pub struct TextWeights {
    base: MmapedSafetensors,
    weight_names: BTreeSet<String>,
    adapter: Option<LoraAdapter>,
    device: Device,
    dtype: DType,
}

impl TextWeights {
    /// Checkpoint files must not be modified while these weights are loaded.
    /// `adapter_directory` points directly to the text adapter directory.
    pub fn load(
        base_directory: &Path,
        adapter_directory: Option<&Path>,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        if !dtype.is_float() {
            return Err(SystemOneError::Config(
                "Qwen3.5 text weights require a floating-point dtype".into(),
            ));
        }
        let mut index: CheckpointIndex =
            serde_json::from_slice(&std::fs::read(
                base_directory.join("model.safetensors.index.json"),
            )?)?;
        index
            .weight_map
            .retain(|name, _| name.starts_with("model.language_model."));
        if index.weight_map.is_empty() {
            return Err(SystemOneError::Config(
                "checkpoint index contains no Qwen3.5 text weights".into(),
            ));
        }
        let shards: BTreeSet<_> = index
            .weight_map
            .values()
            .map(|filename| base_directory.join(filename))
            .collect();
        // SAFETY: checkpoint is mapped read-only and must not be modified while loaded.
        let base = unsafe {
            MmapedSafetensors::multi(&shards.into_iter().collect::<Vec<_>>())?
        };
        let adapter = adapter_directory
            .map(|directory| LoraAdapter::load(directory, device))
            .transpose()?;
        Ok(Self {
            base,
            weight_names: index.weight_map.into_keys().collect(),
            adapter,
            device: device.clone(),
            dtype,
        })
    }

    /// Fetches an unmerged tensor by its full checkpoint name.
    pub fn tensor(&self, name: &str) -> Result<Tensor> {
        Ok(self.load_base_tensor(name)?.to_dtype(self.dtype)?)
    }

    /// Returns the base weight with the LoRA delta merged in, unchanged when
    /// the module has no adapter.
    pub fn linear_weight(&mut self, name: &str) -> Result<Tensor> {
        let base = self.load_base_tensor(name)?;
        let merged = match &mut self.adapter {
            Some(adapter) => adapter.merge(name, &base)?,
            None => base,
        };
        Ok(merged.to_dtype(self.dtype)?)
    }

    /// Call after fetching model weights to report unused adapter modules.
    pub fn check_all_used(&self) -> Result<()> {
        if let Some(adapter) = &self.adapter {
            adapter.check_all_used()?;
        }
        Ok(())
    }

    fn load_base_tensor(&self, name: &str) -> Result<Tensor> {
        if !self.weight_names.contains(name) {
            return Err(SystemOneError::Config(format!(
                "weight is not in the text checkpoint index: {name}"
            )));
        }
        Ok(self.base.load(name, &self.device)?)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use candle_core::safetensors;

    use super::*;

    const NORM: &str = "model.language_model.norm.weight";
    const Q_PROJ: &str =
        "model.language_model.layers.3.self_attn.q_proj.weight";
    const IN_PROJ_QKV: &str =
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight";

    #[test]
    fn loads_only_indexed_text_weights_and_merges_before_casting() {
        let directory = std::env::temp_dir()
            .join(format!("vs1-cua-s1-weights-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let norm = Tensor::new(&[1.003f32, 2.], &Device::Cpu).unwrap();
        let base = Tensor::new(&[[1.003f32; 2]; 2], &Device::Cpu).unwrap();
        safetensors::save(
            &HashMap::from([
                (NORM, norm.clone()),
                ("model.visual.norm.weight", norm.clone()),
                ("mtp.norm.weight", norm.clone()),
                ("model.language_model.unindexed.weight", norm.clone()),
            ]),
            directory.join("first.safetensors"),
        )
        .unwrap();
        safetensors::save(
            &HashMap::from([
                (Q_PROJ, base.clone()),
                (IN_PROJ_QKV, base.clone()),
            ]),
            directory.join("second.safetensors"),
        )
        .unwrap();
        std::fs::write(
            directory.join("model.safetensors.index.json"),
            serde_json::to_vec(&serde_json::json!({"weight_map": {
                (NORM): "first.safetensors",
                (Q_PROJ): "second.safetensors",
                (IN_PROJ_QKV): "second.safetensors",
                "model.visual.norm.weight": "first.safetensors",
                "mtp.norm.weight": "first.safetensors",
                "model.visual.other.weight": "absent.safetensors",
                "mtp.other.weight": "absent.safetensors"
            }}))
            .unwrap(),
        )
        .unwrap();
        std::fs::write(
            directory.join("adapter_config.json"),
            br#"{"peft_type":"LORA","r":1,"lora_alpha":2,"bias":"none"}"#,
        )
        .unwrap();
        safetensors::save(
            &HashMap::from([
                (
                    "base_model.model.model.layers.3.self_attn.q_proj.lora_A.weight",
                    Tensor::new(&[[0.001f32; 2]], &Device::Cpu).unwrap(),
                ),
                (
                    "base_model.model.model.layers.3.self_attn.q_proj.lora_B.weight",
                    Tensor::new(&[[0.5f32]; 2], &Device::Cpu).unwrap(),
                ),
            ]),
            directory.join("adapter_model.safetensors"),
        )
        .unwrap();

        for dtype in [DType::F32, DType::BF16] {
            for adapter_directory in [None, Some(directory.as_path())] {
                let mut weights = TextWeights::load(
                    &directory,
                    adapter_directory,
                    &Device::Cpu,
                    dtype,
                )
                .unwrap();
                let loaded_norm = weights.tensor(NORM).unwrap();
                assert_eq!(loaded_norm.dtype(), dtype);
                assert!(loaded_norm.device().is_cpu());
                assert_eq!(
                    loaded_norm
                        .to_dtype(DType::F32)
                        .unwrap()
                        .to_vec1::<f32>()
                        .unwrap(),
                    norm.to_dtype(dtype)
                        .unwrap()
                        .to_dtype(DType::F32)
                        .unwrap()
                        .to_vec1::<f32>()
                        .unwrap()
                );
                for name in [
                    "model.visual.norm.weight",
                    "mtp.norm.weight",
                    "model.language_model.unindexed.weight",
                    "model.language_model.missing.weight",
                ] {
                    assert!(weights.tensor(name).is_err(), "{name}");
                    assert!(weights.linear_weight(name).is_err(), "{name}");
                }
                let unmerged = weights.tensor(Q_PROJ).unwrap();
                assert_eq!(
                    weights
                        .linear_weight(IN_PROJ_QKV)
                        .unwrap()
                        .to_dtype(DType::F32)
                        .unwrap()
                        .to_vec2::<f32>()
                        .unwrap(),
                    unmerged
                        .to_dtype(DType::F32)
                        .unwrap()
                        .to_vec2::<f32>()
                        .unwrap()
                );
                if adapter_directory.is_some() {
                    let error =
                        weights.check_all_used().unwrap_err().to_string();
                    assert!(error.contains(Q_PROJ), "{error}");
                } else {
                    weights.check_all_used().unwrap();
                }
                let merged = weights.linear_weight(Q_PROJ).unwrap();
                assert_eq!(merged.dtype(), dtype);
                assert!(merged.device().is_cpu());
                let expected = if adapter_directory.is_some() {
                    (&base + 0.001).unwrap()
                } else {
                    base.clone()
                };
                assert_eq!(
                    merged
                        .to_dtype(DType::F32)
                        .unwrap()
                        .to_vec2::<f32>()
                        .unwrap(),
                    expected
                        .to_dtype(dtype)
                        .unwrap()
                        .to_dtype(DType::F32)
                        .unwrap()
                        .to_vec2::<f32>()
                        .unwrap()
                );
                weights.check_all_used().unwrap();
            }
        }
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    #[ignore = "requires the pinned local base and text adapter in artifacts/cua-s1"]
    fn loads_pinned_text_weights_and_merges_q_proj() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../artifacts/cua-s1");
        let base_directory =
            root.join("base/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a");
        let adapter_directory =
            root.join("adapter/16818868b0cc7813808aae4e87b417657046ab79/text");
        let mut weights = TextWeights::load(
            &base_directory,
            Some(&adapter_directory),
            &Device::Cpu,
            DType::F32,
        )
        .unwrap();
        let embedding = weights
            .tensor("model.language_model.embed_tokens.weight")
            .unwrap();
        assert_eq!(embedding.dims(), [248320, 2560]);
        assert_eq!(embedding.dtype(), DType::F32);
        drop(embedding);
        let in_proj_qkv = weights.linear_weight(IN_PROJ_QKV).unwrap();
        assert_eq!(in_proj_qkv.dims(), [8192, 2560]);
        assert_eq!(in_proj_qkv.dtype(), DType::F32);
        drop(in_proj_qkv);
        let unmerged = weights.tensor(Q_PROJ).unwrap();
        let merged = weights.linear_weight(Q_PROJ).unwrap();
        assert_eq!(merged.dims(), [8192, 2560]);
        assert_eq!(merged.dtype(), DType::F32);
        let max_delta = (&merged - &unmerged)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(max_delta.is_finite() && max_delta > 0., "{max_delta}");
        let error = weights.check_all_used().unwrap_err().to_string();
        assert!(error.contains("unused LoRA modules (127)"), "{error}");
        assert!(!error.contains(Q_PROJ), "{error}");
        println!(
            "embedding [248320, 2560]; in_proj_qkv and q_proj [8192, 2560]; max q_proj delta {max_delta:e}"
        );
    }
}
