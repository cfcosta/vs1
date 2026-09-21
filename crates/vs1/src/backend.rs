//! Caller-selected backends sharing the typed decision interface.
use crate::{Result, SystemOne, SystemOneRequest, SystemOneResponse};
/// Explicit runtime selection. Existing `SystemOne` remains the local laya API.
pub enum DecisionModel {
    Laya(Box<SystemOne>),
    #[cfg(feature = "jev")]
    Jev(Box<crate::JevClient>),
}
impl From<SystemOne> for DecisionModel {
    fn from(model: SystemOne) -> Self {
        Self::Laya(Box::new(model))
    }
}
#[cfg(feature = "jev")]
impl From<crate::JevClient> for DecisionModel {
    fn from(model: crate::JevClient) -> Self {
        Self::Jev(Box::new(model))
    }
}
impl DecisionModel {
    pub fn model_name(&self) -> &str {
        match self {
            Self::Laya(m) => m.model_name(),
            #[cfg(feature = "jev")]
            Self::Jev(m) => m.model_name(),
        }
    }
    pub fn system_one(
        &self,
        request: &SystemOneRequest,
    ) -> Result<SystemOneResponse> {
        match self {
            Self::Laya(m) => m.system_one(request),
            #[cfg(feature = "jev")]
            Self::Jev(m) => m.system_one(request),
        }
    }
    pub fn system_one_batch(
        &self,
        requests: &[SystemOneRequest],
    ) -> Result<Vec<SystemOneResponse>> {
        match self {
            Self::Laya(m) => m.system_one_batch(requests),
            #[cfg(feature = "jev")]
            Self::Jev(m) => m.system_one_batch(requests),
        }
    }
    /// Access tokenizer/device facilities only when the caller selected local inference.
    pub fn local(&self) -> Option<&SystemOne> {
        match self {
            Self::Laya(m) => Some(m),
            #[cfg(feature = "jev")]
            Self::Jev(_) => None,
        }
    }
}
