use std::{
    collections::{BTreeMap, BTreeSet},
    path::PathBuf,
};

use candle_core::{DType, Device};
use hf_hub::{Repo, RepoType, api::sync::Api};
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

use super::{CuaS1Input, CuaS1Option, TextConfig, TextModel, TextWeights};
use crate::{Result, SystemOneError};

pub const DEFAULT_REPO_ID: &str = "cua-ai/cua-s1-4b-0.2";
pub const ADAPTER_REVISION: &str = "16818868b0cc7813808aae4e87b417657046ab79";
pub const BASE_REPO_ID: &str = "Qwen/Qwen3.5-4B";
pub const BASE_REVISION: &str = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a";
pub const MODEL_NAME: &str = "Cua-S1-4B";

/// One option's final-position letter logit and option-only probability.
#[derive(Debug, Clone, Serialize)]
pub struct CuaS1OptionPrediction {
    pub letter: char,
    pub option: CuaS1Option,
    pub logit: f32,
    pub probability: f32,
}

/// Local Cua-S1 text decision model, supported on CPU/F32 and CUDA/BF16.
pub struct CuaS1 {
    model: TextModel,
    tokenizer: Tokenizer,
}

pub struct CuaS1Builder {
    adapter_repo: String,
    device: Device,
    dtype: Option<DType>,
    local_directories: Option<(PathBuf, PathBuf)>,
}

impl CuaS1Builder {
    pub fn with_device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }
    /// Defaults to BF16 on CUDA and F32 on CPU.
    pub fn with_dtype(mut self, dtype: DType) -> Self {
        self.dtype = Some(dtype);
        self
    }
    /// Loads both checkpoints locally without contacting the Hub.
    /// `adapter_directory` points directly to the `text` adapter directory.
    pub fn with_local_directories(
        mut self,
        base_directory: impl Into<PathBuf>,
        adapter_directory: impl Into<PathBuf>,
    ) -> Self {
        self.local_directories =
            Some((base_directory.into(), adapter_directory.into()));
        self
    }
}

impl TryFrom<CuaS1Builder> for CuaS1 {
    type Error = SystemOneError;

    fn try_from(builder: CuaS1Builder) -> Result<Self> {
        let dtype = builder.dtype.unwrap_or_else(|| {
            if builder.device.is_cuda() && cfg!(feature = "cuda") {
                DType::BF16
            } else {
                DType::F32
            }
        });
        if builder.device.is_cuda() && dtype == DType::F32 {
            return Err(SystemOneError::Config(
                "Cua-S1 F32 weights on CUDA do not fit in 12 GB of GPU memory; use BF16".into(),
            ));
        }
        if !(builder.device.is_cpu() && dtype == DType::F32
            || builder.device.is_cuda()
                && cfg!(feature = "cuda")
                && dtype == DType::BF16)
        {
            return Err(SystemOneError::Config(
                "Cua-S1 supports only CPU with F32 or CUDA with BF16 dtype (requires the cuda feature)".into(),
            ));
        }
        let (base_directory, adapter_directory) =
            match builder.local_directories {
                Some(directories) => directories,
                None => download_checkpoints(&builder.adapter_repo)?,
            };
        let config = TextConfig::from_slice(&std::fs::read(
            base_directory.join("config.json"),
        )?)?;
        let tokenizer =
            Tokenizer::from_file(base_directory.join("tokenizer.json"))?;
        let mut weights = TextWeights::load(
            &base_directory,
            Some(&adapter_directory),
            &builder.device,
            dtype,
        )?;
        Ok(Self {
            model: TextModel::load(&mut weights, &config)?,
            tokenizer,
        })
    }
}

#[derive(Deserialize)]
struct CheckpointIndex {
    weight_map: BTreeMap<String, String>,
}

fn download_checkpoints(adapter_repo: &str) -> Result<(PathBuf, PathBuf)> {
    let api = Api::new()?;
    let base = api.repo(Repo::with_revision(
        BASE_REPO_ID.into(),
        RepoType::Model,
        BASE_REVISION.into(),
    ));
    let mut base_directory = base.get("config.json")?;
    base_directory.pop();
    base.get("tokenizer.json")?;
    let index: CheckpointIndex = serde_json::from_slice(&std::fs::read(
        base.get("model.safetensors.index.json")?,
    )?)?;
    let shards: BTreeSet<_> = index
        .weight_map
        .iter()
        .filter(|(name, _)| name.starts_with("model.language_model."))
        .map(|(_, filename)| filename)
        .collect();
    for shard in shards {
        base.get(shard)?;
    }
    let adapter = api.repo(Repo::with_revision(
        adapter_repo.into(),
        RepoType::Model,
        ADAPTER_REVISION.into(),
    ));
    let mut adapter_directory = adapter.get("text/adapter_config.json")?;
    adapter_directory.pop();
    adapter.get("text/adapter_model.safetensors")?;
    Ok((base_directory, adapter_directory))
}

impl CuaS1 {
    /// Loads `adapter_repo` at [`ADAPTER_REVISION`] over the pinned base.
    /// Defaults to CPU/F32, or BF16 when selecting CUDA with the `cuda` feature.
    pub fn from(adapter_repo: &str) -> CuaS1Builder {
        CuaS1Builder {
            adapter_repo: adapter_repo.into(),
            device: Device::Cpu,
            dtype: None,
            local_directories: None,
        }
    }
    pub fn model_name(&self) -> &str {
        MODEL_NAME
    }

    /// Scores 1..=26 options in one forward pass, preserving option order.
    /// Probabilities are normalized over only the option-letter logits.
    pub fn score_options(
        &self,
        app: &str,
        task_family: &str,
        ax_tree: &str,
        goal: Option<&str>,
        options: &[CuaS1Option],
    ) -> Result<Vec<CuaS1OptionPrediction>> {
        let input = CuaS1Input::encode(
            &self.tokenizer,
            options,
            app,
            task_family,
            ax_tree,
            goal,
        )?;
        let prediction = self.model.forward(&input)?;
        Ok(options
            .iter()
            .zip(input.letters)
            .zip(prediction.logits)
            .zip(prediction.probabilities)
            .map(|(((option, letter), logit), probability)| {
                CuaS1OptionPrediction {
                    letter,
                    option: option.clone(),
                    logit,
                    probability,
                }
            })
            .collect())
    }
}
