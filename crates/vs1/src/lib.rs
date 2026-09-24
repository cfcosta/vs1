//! vs1: local laya and caller-selected hosted Jev decision models.
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
//! Normal builds include `JevClient` (the default `jev` feature).
//! [`DecisionModel`] wraps either backend with the same typed methods;
//! selection and credentials belong to the caller.
//!
//! The crate started life inside docbert as a retrieval reranker:
//! fetch more chunks than asked for, ask the model whether each one
//! answers the query, keep the most probable. Measured zero-shot on
//! BeIR corpora, the shipped checkpoints separate relevant from
//! irrelevant passages only weakly (AUC 0.5 to 0.7); see the README.

pub mod builder;
pub mod config;
pub mod cua_s1;
pub mod error;
pub mod gliner_decide;
pub mod head;
pub mod model;
pub mod modernbert;
pub mod pyjson;
pub mod sequence;
pub mod types;

#[cfg(feature = "cuda")]
mod geglu_cuda;

#[cfg(feature = "cuda")]
mod bias_act_cuda;
#[cfg(feature = "cuda")]
mod residual_norm_cuda;
#[cfg(feature = "cuda")]
mod rope_cuda;

#[cfg(all(test, feature = "cuda"))]
mod geglu_bench;

#[cfg(feature = "cuda")]
mod gemm_cuda;

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

#[cfg(feature = "jev")]
pub mod jev;
#[cfg(feature = "jev")]
pub use jev::{JevClient, JevStats};

pub mod backend;
pub use backend::DecisionModel;

pub mod openjev;
pub use cua_s1::{CuaS1, CuaS1Builder, CuaS1Option, CuaS1OptionPrediction};
pub use gliner_decide::{
    GlinerDecide,
    GlinerDecideBuilder,
    GlinerDecideInput,
    GlinerTask,
    GlinerTaskPrediction,
};
pub use openjev::{OpenJev, OpenJevBuilder, OpenJevInput, OpenJevPrediction};
pub use types::AbstentionAnswer;

#[cfg(feature = "cuda")]
mod parallel_cuda;

#[cfg(feature = "cuda")]
mod cutlass_geglu;
