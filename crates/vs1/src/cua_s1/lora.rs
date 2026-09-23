use std::{
    collections::{BTreeMap, HashMap},
    path::Path,
};

use candle_core::{DType, Device, Tensor, safetensors};
use serde::Deserialize;

use crate::{Result, SystemOneError};

#[derive(Debug, Deserialize)]
struct Config {
    peft_type: String,
    r: usize,
    lora_alpha: f64,
    bias: String,
    #[serde(default)]
    use_dora: bool,
    #[serde(default)]
    use_rslora: bool,
    #[serde(default)]
    lora_bias: bool,
    #[serde(default)]
    fan_in_fan_out: bool,
    #[serde(default)]
    use_qalora: bool,
    #[serde(default)]
    rank_pattern: BTreeMap<String, serde_json::Value>,
    #[serde(default)]
    alpha_pattern: BTreeMap<String, serde_json::Value>,
    modules_to_save: Option<serde_json::Value>,
    layer_replication: Option<serde_json::Value>,
    alora_invocation_tokens: Option<serde_json::Value>,
    target_parameters: Option<serde_json::Value>,
    trainable_token_indices: Option<serde_json::Value>,
}

impl Config {
    fn from_slice(bytes: &[u8]) -> Result<Self> {
        let config: Self = serde_json::from_slice(bytes)?;
        for (setting, unsupported) in [
            ("peft_type", config.peft_type != "LORA"),
            ("bias", config.bias != "none"),
            ("use_dora", config.use_dora),
            ("use_rslora", config.use_rslora),
            ("lora_bias", config.lora_bias),
            ("fan_in_fan_out", config.fan_in_fan_out),
            ("use_qalora", config.use_qalora),
            ("rank_pattern", !config.rank_pattern.is_empty()),
            ("alpha_pattern", !config.alpha_pattern.is_empty()),
            ("modules_to_save", config.modules_to_save.is_some()),
            ("layer_replication", config.layer_replication.is_some()),
            (
                "alora_invocation_tokens",
                config.alora_invocation_tokens.is_some(),
            ),
            ("target_parameters", config.target_parameters.is_some()),
            (
                "trainable_token_indices",
                config.trainable_token_indices.is_some(),
            ),
        ] {
            if unsupported {
                return Err(SystemOneError::Config(format!(
                    "unsupported LoRA setting: {setting}"
                )));
            }
        }
        if config.r == 0 || !config.lora_alpha.is_finite() {
            return Err(SystemOneError::Config(
                "LoRA requires a positive r and finite lora_alpha".into(),
            ));
        }
        Ok(config)
    }
}

#[derive(Debug)]
struct Weights {
    a: Tensor,
    b: Tensor,
    used: bool,
}

/// PEFT text adapters keyed by the original Qwen3.5 checkpoint weight names.
#[derive(Debug)]
pub struct LoraAdapter {
    modules: BTreeMap<String, Weights>,
    scale: f64,
}

impl LoraAdapter {
    pub fn load(directory: &Path, device: &Device) -> Result<Self> {
        let config = Config::from_slice(&std::fs::read(
            directory.join("adapter_config.json"),
        )?)?;
        Self::from_tensors(
            &config,
            safetensors::load(
                directory.join("adapter_model.safetensors"),
                device,
            )?,
        )
    }

    fn from_tensors(
        config: &Config,
        tensors: HashMap<String, Tensor>,
    ) -> Result<Self> {
        let mut pairs =
            BTreeMap::<String, (Option<Tensor>, Option<Tensor>)>::new();
        for (name, tensor) in tensors {
            // PEFT's text model omits the checkpoint's language_model component.
            let (module, is_a) = name
                .strip_prefix("base_model.model.model.layers.")
                .and_then(|name| {
                    name.strip_suffix(".lora_A.weight")
                        .map(|module| (module, true))
                        .or_else(|| {
                            name.strip_suffix(".lora_B.weight")
                                .map(|module| (module, false))
                        })
                })
                .ok_or_else(|| {
                    SystemOneError::Config(format!(
                        "unsupported LoRA tensor name: {name}"
                    ))
                })?;
            if !tensor.dtype().is_float() {
                return Err(SystemOneError::Config(format!(
                    "LoRA tensor must be floating point: {name}"
                )));
            }
            let pair = pairs
                .entry(format!("model.language_model.layers.{module}.weight"))
                .or_default();
            let slot = if is_a { &mut pair.0 } else { &mut pair.1 };
            *slot = Some(tensor.to_dtype(DType::F32)?);
        }
        if pairs.is_empty() {
            return Err(SystemOneError::Config("empty LoRA adapter".into()));
        }
        let mut modules = BTreeMap::new();
        for (name, (a, b)) in pairs {
            let (Some(a), Some(b)) = (a, b) else {
                return Err(SystemOneError::Config(format!(
                    "LoRA module requires both A and B: {name}"
                )));
            };
            if a.rank() != 2
                || b.rank() != 2
                || a.dims()[0] != config.r
                || b.dims()[1] != config.r
                || a.dims()[1] == 0
                || b.dims()[0] == 0
            {
                return Err(SystemOneError::Config(format!(
                    "LoRA shape mismatch for {name}: A {:?}, B {:?}, r {}",
                    a.dims(),
                    b.dims(),
                    config.r
                )));
            }
            modules.insert(name, Weights { a, b, used: false });
        }
        Ok(Self {
            modules,
            scale: config.lora_alpha / config.r as f64,
        })
    }

    pub fn module_count(&self) -> usize {
        self.modules.len()
    }

    /// Returns W + (alpha / r) * B @ A in F32 on W's device.
    /// Weights without an adapter are returned unchanged, including their dtype.
    pub fn merge(
        &mut self,
        base_weight_name: &str,
        base_weight: &Tensor,
    ) -> Result<Tensor> {
        let Some(weights) = self.modules.get_mut(base_weight_name) else {
            return Ok(base_weight.clone());
        };
        let expected = [weights.b.dims()[0], weights.a.dims()[1]];
        if base_weight.dims() != expected {
            return Err(SystemOneError::Config(format!(
                "LoRA base shape mismatch for {base_weight_name}: expected {expected:?}, got {:?}",
                base_weight.dims()
            )));
        }
        if !base_weight.dtype().is_float() {
            return Err(SystemOneError::Config(format!(
                "LoRA base weight must be floating point: {base_weight_name}"
            )));
        }
        let a = weights.a.to_device(base_weight.device())?;
        let b = weights.b.to_device(base_weight.device())?;
        let delta = (b.matmul(&a)? * self.scale)?;
        let merged = (base_weight.to_dtype(DType::F32)? + delta)?;
        weights.used = true;
        Ok(merged)
    }

    /// Call after loading the base weights to catch missing or misnamed modules.
    pub fn check_all_used(&self) -> Result<()> {
        let unused: Vec<_> = self
            .modules
            .iter()
            .filter(|(_, weights)| !weights.used)
            .map(|(name, _)| name.as_str())
            .collect();
        if !unused.is_empty() {
            return Err(SystemOneError::Config(format!(
                "unused LoRA modules ({}): {}",
                unused.len(),
                unused.join(", ")
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MODULE: &str = "base_model.model.model.layers.3.self_attn.q_proj";
    const BASE: &str = "model.language_model.layers.3.self_attn.q_proj.weight";
    const CONFIG: &[u8] =
        br#"{"peft_type":"LORA","r":2,"lora_alpha":3,"bias":"none"}"#;

    fn tensors(a: Tensor, b: Tensor) -> HashMap<String, Tensor> {
        HashMap::from([
            (format!("{MODULE}.lora_A.weight"), a),
            (format!("{MODULE}.lora_B.weight"), b),
        ])
    }

    fn adapter(a: Tensor, b: Tensor) -> Result<LoraAdapter> {
        LoraAdapter::from_tensors(&Config::from_slice(CONFIG)?, tensors(a, b))
    }

    #[test]
    fn merge_matches_naive_f32() {
        let a = [[1f32, -2., 0.5], [3., 0.25, -1.]];
        let b = [[2f32, 1.], [-1., 4.], [0.5, -2.], [3., 0.25]];
        let w = [
            [0.5f32, 1., -1.],
            [2., -3., 4.],
            [1., 0., 2.],
            [-2., 1., 0.],
        ];
        for dtype in [DType::F32, DType::BF16] {
            let mut adapter = adapter(
                Tensor::new(&a, &Device::Cpu)
                    .unwrap()
                    .to_dtype(dtype)
                    .unwrap(),
                Tensor::new(&b, &Device::Cpu)
                    .unwrap()
                    .to_dtype(dtype)
                    .unwrap(),
            )
            .unwrap();
            let base = Tensor::new(&w, &Device::Cpu)
                .unwrap()
                .to_dtype(dtype)
                .unwrap();
            assert_eq!(adapter.module_count(), 1);
            let unchanged = adapter
                .merge("model.language_model.embed_tokens.weight", &base)
                .unwrap();
            assert_eq!(unchanged.id(), base.id());
            let merged = adapter.merge(BASE, &base).unwrap();
            assert_eq!(merged.dtype(), DType::F32);
            let actual = merged.to_vec2::<f32>().unwrap();
            for i in 0..4 {
                for j in 0..3 {
                    let delta: f32 = (0..2).map(|k| b[i][k] * a[k][j]).sum();
                    assert_eq!(actual[i][j], w[i][j] + 1.5 * delta);
                }
            }
            assert_eq!(
                base.to_dtype(DType::F32).unwrap().to_vec2::<f32>().unwrap(),
                w
            );
            adapter.check_all_used().unwrap();
        }
    }

    #[test]
    fn unused_modules_and_base_shape_mismatch() {
        let mut adapter = adapter(
            Tensor::ones((2, 3), DType::F32, &Device::Cpu).unwrap(),
            Tensor::ones((4, 2), DType::F32, &Device::Cpu).unwrap(),
        )
        .unwrap();
        let error = adapter.check_all_used().unwrap_err().to_string();
        assert!(error.contains("unused LoRA modules (1)"), "{error}");
        assert!(error.contains(BASE), "{error}");
        for shape in [vec![3, 4], vec![1, 4, 3]] {
            let base = Tensor::zeros(shape, DType::F32, &Device::Cpu).unwrap();
            let error = adapter.merge(BASE, &base).unwrap_err().to_string();
            assert!(error.contains("shape mismatch"), "{error}");
            assert!(adapter.check_all_used().is_err());
        }
        let base = Tensor::zeros((4, 3), DType::F32, &Device::Cpu).unwrap();
        adapter.merge(BASE, &base).unwrap();
        adapter.check_all_used().unwrap();
    }

    #[test]
    fn rejects_malformed_tensors() {
        for (a, b) in [
            (vec![1, 3], vec![4, 2]),
            (vec![2, 3], vec![4, 1]),
            (vec![2, 3], vec![4, 2, 1]),
            (vec![2], vec![4, 2]),
        ] {
            let error = adapter(
                Tensor::zeros(a, DType::F32, &Device::Cpu).unwrap(),
                Tensor::zeros(b, DType::F32, &Device::Cpu).unwrap(),
            )
            .unwrap_err()
            .to_string();
            assert!(error.contains("shape mismatch"), "{error}");
        }
        let config = Config::from_slice(CONFIG).unwrap();
        for suffix in ["lora_A.weight", "lora_B.weight", "lora_A.bias"] {
            let tensors = HashMap::from([(
                format!("{MODULE}.{suffix}"),
                Tensor::zeros((2, 3), DType::F32, &Device::Cpu).unwrap(),
            )]);
            assert!(LoraAdapter::from_tensors(&config, tensors).is_err());
        }
        assert!(LoraAdapter::from_tensors(&config, HashMap::new()).is_err());
    }

    #[test]
    fn rejects_unsupported_config() {
        for (setting, value) in [
            ("peft_type", r#""IA3""#),
            ("bias", r#""all""#),
            ("bias", r#""lora_only""#),
            ("use_dora", "true"),
            ("use_rslora", "true"),
            ("lora_bias", "true"),
            ("fan_in_fan_out", "true"),
            ("use_qalora", "true"),
            ("rank_pattern", r#"{"q_proj":4}"#),
            ("alpha_pattern", r#"{"q_proj":8}"#),
            ("modules_to_save", r#"["lm_head"]"#),
            ("layer_replication", "[[0, 1]]"),
            ("alora_invocation_tokens", "[1]"),
            ("target_parameters", r#"["weight"]"#),
            ("trainable_token_indices", "[1]"),
            ("r", "0"),
        ] {
            let mut raw: serde_json::Value =
                serde_json::from_slice(CONFIG).unwrap();
            raw[setting] = serde_json::from_str(value).unwrap();
            let error = Config::from_slice(&serde_json::to_vec(&raw).unwrap())
                .unwrap_err();
            assert!(matches!(error, SystemOneError::Config(_)), "{error}");
            assert!(error.to_string().contains(setting), "{setting}: {error}");
        }
    }

    #[test]
    #[ignore = "requires the pinned local adapter and base weights in artifacts/cua-s1"]
    fn pinned_adapter_merge() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../artifacts/cua-s1");
        let adapter_path =
            root.join("adapter/16818868b0cc7813808aae4e87b417657046ab79/text");
        let base_path =
            root.join("base/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a");
        let mut adapter =
            LoraAdapter::load(&adapter_path, &Device::Cpu).unwrap();
        assert_eq!(adapter.module_count(), 128);
        assert_eq!(adapter.scale, 2.);
        let index: serde_json::Value = serde_json::from_slice(
            &std::fs::read(base_path.join("model.safetensors.index.json"))
                .unwrap(),
        )
        .unwrap();
        let shards: std::collections::BTreeSet<_> = index["weight_map"]
            .as_object()
            .unwrap()
            .values()
            .map(|value| base_path.join(value.as_str().unwrap()))
            .collect();
        let base = unsafe {
            safetensors::MmapedSafetensors::multi(
                &shards.into_iter().collect::<Vec<_>>(),
            )
        }
        .unwrap();
        let mut expected = std::collections::BTreeSet::new();
        for layer in 0..32 {
            for projection in ["gate_proj", "up_proj", "down_proj"] {
                expected.insert(format!("model.language_model.layers.{layer}.mlp.{projection}.weight"));
            }
            if layer % 4 == 3 {
                for projection in ["q_proj", "k_proj", "v_proj", "o_proj"] {
                    expected.insert(format!("model.language_model.layers.{layer}.self_attn.{projection}.weight"));
                }
            }
        }
        assert_eq!(
            adapter
                .modules
                .keys()
                .cloned()
                .collect::<std::collections::BTreeSet<_>>(),
            expected
        );
        for (name, weights) in &adapter.modules {
            assert_eq!(
                base.get(name).unwrap().shape(),
                [weights.b.dims()[0], weights.a.dims()[1]],
                "{name}"
            );
            assert_eq!(weights.a.dims()[0], 16);
            assert_eq!(weights.b.dims()[1], 16);
        }
        let base = base.load(BASE, &Device::Cpu).unwrap();
        let raw = safetensors::load(
            adapter_path.join("adapter_model.safetensors"),
            &Device::Cpu,
        )
        .unwrap();
        let a = raw[&format!("{MODULE}.lora_A.weight")]
            .to_vec2::<f32>()
            .unwrap();
        let b = raw[&format!("{MODULE}.lora_B.weight")]
            .to_vec2::<f32>()
            .unwrap();
        let w = base.to_dtype(DType::F32).unwrap().to_vec2::<f32>().unwrap();
        let merged = adapter.merge(BASE, &base).unwrap();
        assert_eq!(merged.dtype(), DType::F32);
        let actual = merged.to_vec2::<f32>().unwrap();
        let mut max_error = 0f32;
        for i in 0..b.len() {
            for j in 0..a[0].len() {
                let delta: f32 = (0..16).map(|k| b[i][k] * a[k][j]).sum();
                let expected = w[i][j] + 2. * delta;
                let error = (actual[i][j] - expected).abs();
                assert!(
                    error <= 1e-6,
                    "({i}, {j}): {} != {expected}",
                    actual[i][j]
                );
                max_error = max_error.max(error);
            }
        }
        assert_eq!(
            adapter
                .modules
                .values()
                .filter(|weights| weights.used)
                .count(),
            1
        );
        assert!(adapter.check_all_used().is_err());
        println!(
            "128 modules; q_proj {:?}; max absolute error {max_error:e}",
            merged.dims()
        );
    }
}
