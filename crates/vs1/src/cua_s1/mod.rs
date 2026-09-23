//! Cua-S1 text decoder, prompts and weights for the Qwen3.5-4B decision model.

mod attention;
mod config;
mod delta_net;
mod lora;
mod model;
mod ops;
mod prompt;
mod weights;

pub use attention::FullAttention;
pub use config::{LayerType, TextConfig};
pub use delta_net::GatedDeltaNet;
pub use lora::LoraAdapter;
pub use model::{CuaS1Prediction, TextModel};
pub use ops::{Mlp, normalize_l2, normalize_rms, normalize_rms_gated};
pub use prompt::{CuaS1Input, CuaS1Option};
pub use weights::TextWeights;
