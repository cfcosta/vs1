use thiserror::Error;

/// Every failure a System One model can report.
#[derive(Error, Debug)]
pub enum SystemOneError {
    /// Hosted transport/protocol failure; deliberately excludes secrets and response bodies.
    #[error("hosted model error: {0}")]
    Remote(String),
    #[error("hosted model returned HTTP {0}")]
    HttpStatus(u16),
    /// Tensor or device failure from candle.
    #[error("candle error: {0}")]
    Candle(#[from] candle_core::Error),

    /// The tokenizer could not be built or could not encode a string.
    #[error("tokenizer error: {0}")]
    Tokenizer(String),

    /// A checkpoint file did not parse.
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    /// The Hugging Face Hub refused or failed a download.
    #[error("hugging face hub error: {0}")]
    Hub(#[from] hf_hub::api::sync::ApiError),

    /// A checkpoint file could not be read.
    #[error("i/o error: {0}")]
    Io(#[from] std::io::Error),

    /// The checkpoint is not a laya-style decision model, or a setting
    /// on it is out of range.
    #[error("model configuration error: {0}")]
    Config(String),

    /// A question in the request cannot be evaluated as written.
    #[error("question {id:?}: {reason}")]
    Question { id: String, reason: String },
}

impl From<Box<dyn std::error::Error + Send + Sync>> for SystemOneError {
    fn from(err: Box<dyn std::error::Error + Send + Sync>) -> Self {
        SystemOneError::Tokenizer(err.to_string())
    }
}

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, SystemOneError>;
