//! Cua-S1 text configuration and prompts for the Qwen3.5-4B decision model.

mod config;
mod prompt;

pub use config::{LayerType, TextConfig};
pub use prompt::{CuaS1Input, CuaS1Option};
