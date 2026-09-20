//! vs1: System One decision models on candle.
//!
//! This crate runs [laya](https://github.com/NandhaKishorM/laya)
//! checkpoints: a ModernBERT encoder with a small decision head that
//! answers typed questions about a piece of *state* in one
//! non-autoregressive forward pass. Nothing is generated, so there is
//! nothing to parse; every answer is a calibrated probability.
//!
//! The request and response follow the TypeSafe `systemone` shape that
//! laya is API-compatible with: a state, a map of named questions of
//! type `choice`, `score`, or `noul`, and one typed answer per
//! question.
//!
//! ```no_run
//! use vs1::{Question, SystemOne, SystemOneRequest};
//!
//! let model: SystemOne = SystemOne::from("convaiinnovations/laya")
//!     .try_into()
//!     .unwrap();
//! let request = SystemOneRequest::new("Please refund the duplicate charge.")
//!     .question("wants_refund", Question::noul("Does the sender ask for money back?"))
//!     .question(
//!         "department",
//!         Question::choice(
//!             "Which team handles this?",
//!             [("billing", "invoices and payments"), ("other", "anything else")],
//!         ),
//!     );
//! let response = model.system_one(&request).unwrap();
//! println!("{}", serde_json::to_string_pretty(&response).unwrap());
//! ```
//!
//! The crate started life inside docbert as a retrieval reranker:
//! fetch more chunks than asked for, ask the model whether each one
//! answers the query, keep the most probable. Measured zero-shot on
//! BeIR corpora, the shipped checkpoints separate relevant from
//! irrelevant passages only weakly (AUC 0.5 to 0.7); see the README.

pub mod builder;
pub mod config;
pub mod error;
pub mod head;
pub mod model;
pub mod modernbert;
pub mod pyjson;
pub mod sequence;
pub mod types;

#[cfg(feature = "cuda")]
mod geglu_cuda;

#[cfg(feature = "flash-attn")]
mod residual_norm_cuda;
#[cfg(feature = "flash-attn")]
mod rope_cuda;

#[cfg(all(test, feature = "flash-attn"))]
mod geglu_bench;

pub use builder::{
    DEFAULT_REPO_ID,
    MULTILINGUAL_SUBFOLDER,
    SystemOneBuilder,
    TYPED_DECISIONS_SUBFOLDER,
};
pub use config::AgentConfig;
pub use error::{Result, SystemOneError};
pub use model::{CheckpointAssets, SystemOne};
pub use types::{
    Action,
    Answer,
    ChoiceAnswer,
    ChoiceCriteria,
    ChoiceQuestion,
    Description,
    NoulAnswer,
    NoulCriteria,
    NoulQuestion,
    Question,
    QuestionKind,
    ScoreAnswer,
    ScoreQuestion,
    State,
    SystemOneRequest,
    SystemOneResponse,
    Usage,
};
