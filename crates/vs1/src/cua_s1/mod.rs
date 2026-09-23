//! Cua-S1 text configuration, prompts and LoRA for the Qwen3.5-4B decision model.

mod config;
mod lora;
mod prompt;

pub use config::{LayerType, TextConfig};
pub use lora::LoraAdapter;
pub use prompt::{CuaS1Input, CuaS1Option};
