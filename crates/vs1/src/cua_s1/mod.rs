//! Cua-S1 text configuration, prompts and weights for the Qwen3.5-4B decision model.

mod config;
mod lora;
mod ops;
mod prompt;
mod weights;

pub use config::{LayerType, TextConfig};
pub use lora::LoraAdapter;
pub use ops::{Mlp, normalize_l2, normalize_rms, normalize_rms_gated};
pub use prompt::{CuaS1Input, CuaS1Option};
pub use weights::TextWeights;
