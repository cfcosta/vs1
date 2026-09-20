//! Locating a checkpoint (Hub repo or local directory) and turning it
//! into a [`SystemOne`].

use std::path::{Path, PathBuf};

use candle_core::{DType, Device};
use hf_hub::{Repo, RepoType, api::sync::Api};

use crate::{
    error::{Result, SystemOneError},
    model::{CheckpointAssets, SystemOne},
};

/// Hub repo bundling every laya checkpoint.
pub const DEFAULT_REPO_ID: &str = "convaiinnovations/laya";

/// Subfolder of [`DEFAULT_REPO_ID`] holding the multilingual
/// (mmBERT-base) checkpoint.
pub const MULTILINGUAL_SUBFOLDER: &str = "multilingual";

/// Subfolder of [`DEFAULT_REPO_ID`] holding the checkpoint fine-tuned
/// on the typed-decisions workflows.
pub const TYPED_DECISIONS_SUBFOLDER: &str = "typed-decisions";

const AGENT_CONFIG_FILE: &str = "rl_agent_config.json";
const ENCODER_CONFIG_FILE: &str = "encoder/config.json";
const TOKENIZER_FILE: &str = "tokenizer/tokenizer.json";
const TOKENIZER_CONFIG_FILE: &str = "tokenizer/tokenizer_config.json";
const WEIGHTS_FILE: &str = "model.safetensors";

/// Configures and loads a [`SystemOne`].
///
/// ```no_run
/// use vs1::SystemOne;
///
/// let model: SystemOne = SystemOne::from("convaiinnovations/laya")
///     .with_batch_size(16)
///     .try_into()
///     .unwrap();
/// ```
pub struct SystemOneBuilder {
    repo_id: String,
    subfolder: Option<String>,
    model_name: Option<String>,
    device: Option<Device>,
    dtype: Option<DType>,
    batch_size: Option<usize>,
    max_len: Option<usize>,
    head_max_len: Option<usize>,
}

impl SystemOneBuilder {
    pub(crate) fn new(repo_id: &str) -> Self {
        Self {
            repo_id: repo_id.to_string(),
            subfolder: None,
            model_name: None,
            device: None,
            dtype: None,
            batch_size: None,
            max_len: None,
            head_max_len: None,
        }
    }

    /// Loads the checkpoint stored under `subfolder` of the repo or
    /// directory, e.g. [`MULTILINGUAL_SUBFOLDER`].
    pub fn with_subfolder(mut self, subfolder: impl Into<String>) -> Self {
        let subfolder = subfolder.into();
        self.subfolder = (!subfolder.is_empty()).then_some(subfolder);
        self
    }

    /// Name echoed in responses; defaults to the checkpoint's
    /// `model_name`.
    pub fn with_model_name(mut self, name: impl Into<String>) -> Self {
        self.model_name = Some(name.into());
        self
    }

    /// Device to run on; defaults to the CPU.
    pub fn with_device(mut self, device: Device) -> Self {
        self.device = Some(device);
        self
    }

    /// Dtype for the encoder and head; defaults to BF16 on CUDA and
    /// F32 elsewhere.
    pub fn with_dtype(mut self, dtype: DType) -> Self {
        self.dtype = Some(dtype);
        self
    }

    /// Question sequences per forward pass.
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = Some(batch_size);
        self
    }

    /// Overrides the checkpoint's `max_len`.
    pub fn with_max_len(mut self, max_len: usize) -> Self {
        self.max_len = Some(max_len);
        self
    }

    /// Overrides the checkpoint's `head_max_len`.
    pub fn with_head_max_len(mut self, head_max_len: usize) -> Self {
        self.head_max_len = Some(head_max_len);
        self
    }

    /// Where the checkpoint lives, for diagnostics.
    pub fn describe(&self) -> String {
        match &self.subfolder {
            Some(sub) => format!("{}/{sub}", self.repo_id),
            None => self.repo_id.clone(),
        }
    }
}

impl TryFrom<SystemOneBuilder> for SystemOne {
    type Error = SystemOneError;

    fn try_from(builder: SystemOneBuilder) -> Result<Self> {
        let device = builder.device.clone().unwrap_or(Device::Cpu);
        let local = PathBuf::from(&builder.repo_id);
        let assets = if local.is_dir() {
            load_local_assets(&local, builder.subfolder.as_deref())?
        } else {
            load_hub_assets(&builder.repo_id, builder.subfolder.as_deref())?
        };
        SystemOne::new(
            assets,
            builder.model_name,
            &device,
            builder.dtype,
            builder.batch_size,
            builder.max_len,
            builder.head_max_len,
        )
    }
}

fn with_subfolder(file: &str, subfolder: Option<&str>) -> String {
    match subfolder {
        Some(sub) => format!("{sub}/{file}"),
        None => file.to_string(),
    }
}

fn load_local_assets(
    root: &Path,
    subfolder: Option<&str>,
) -> Result<CheckpointAssets> {
    let root = match subfolder {
        Some(sub) => root.join(sub),
        None => root.to_path_buf(),
    };
    let read = |file: &str| -> Result<Vec<u8>> {
        let path = root.join(file);
        std::fs::read(&path).map_err(|err| {
            SystemOneError::Io(std::io::Error::new(
                err.kind(),
                format!("{}: {err}", path.display()),
            ))
        })
    };
    let weights_path = root.join(WEIGHTS_FILE);
    if !weights_path.is_file() {
        return Err(SystemOneError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("{} not found", weights_path.display()),
        )));
    }
    Ok(CheckpointAssets {
        agent_config: read(AGENT_CONFIG_FILE)?,
        encoder_config: read(ENCODER_CONFIG_FILE)?,
        tokenizer: read(TOKENIZER_FILE)?,
        tokenizer_config: read(TOKENIZER_CONFIG_FILE)?,
        weights_path,
    })
}

fn load_hub_assets(
    repo_id: &str,
    subfolder: Option<&str>,
) -> Result<CheckpointAssets> {
    let api = Api::new()?;
    let repo = api.repo(Repo::new(repo_id.to_string(), RepoType::Model));
    let fetch = |file: &str| -> Result<Vec<u8>> {
        Ok(std::fs::read(repo.get(&with_subfolder(file, subfolder))?)?)
    };
    Ok(CheckpointAssets {
        agent_config: fetch(AGENT_CONFIG_FILE)?,
        encoder_config: fetch(ENCODER_CONFIG_FILE)?,
        tokenizer: fetch(TOKENIZER_FILE)?,
        tokenizer_config: fetch(TOKENIZER_CONFIG_FILE)?,
        weights_path: repo.get(&with_subfolder(WEIGHTS_FILE, subfolder))?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subfolder_prefixes_every_file() {
        assert_eq!(
            with_subfolder("model.safetensors", None),
            "model.safetensors"
        );
        assert_eq!(
            with_subfolder("encoder/config.json", Some("multilingual")),
            "multilingual/encoder/config.json"
        );
    }

    #[test]
    fn builder_describe_includes_subfolder() {
        let b = SystemOneBuilder::new("convaiinnovations/laya")
            .with_subfolder(MULTILINGUAL_SUBFOLDER);
        assert_eq!(b.describe(), "convaiinnovations/laya/multilingual");
        let b = SystemOneBuilder::new("x").with_subfolder("");
        assert_eq!(b.describe(), "x");
    }

    #[test]
    fn missing_local_checkpoint_is_a_not_found_error() {
        let tmp = std::env::temp_dir().join("vs1-missing");
        std::fs::create_dir_all(&tmp).unwrap();
        let err = load_local_assets(&tmp, None).unwrap_err();
        assert!(err.to_string().contains("model.safetensors"));
    }
}
