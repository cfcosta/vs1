//! Caller-selected backends sharing the typed decision interface.
use crate::{Result, SystemOne, SystemOneRequest, SystemOneResponse};
/// Explicit runtime selection. Existing `SystemOne` remains the local laya API.
pub enum DecisionModel {
    Laya(Box<SystemOne>),
    OpenJev(Box<crate::OpenJev>),
    #[cfg(feature = "jev")]
    Jev(Box<crate::JevClient>),
}
impl From<SystemOne> for DecisionModel {
    fn from(model: SystemOne) -> Self {
        Self::Laya(Box::new(model))
    }
}
impl From<crate::OpenJev> for DecisionModel {
    fn from(model: crate::OpenJev) -> Self {
        Self::OpenJev(Box::new(model))
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
            Self::OpenJev(m) => m.model_name(),
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
            Self::OpenJev(m) => m.system_one(request),
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
            Self::OpenJev(m) => m.system_one_batch(requests),
            #[cfg(feature = "jev")]
            Self::Jev(m) => m.system_one_batch(requests),
        }
    }
    /// Access Laya-specific tokenizer/device facilities. OpenJev is also local.
    pub fn local(&self) -> Option<&SystemOne> {
        match self {
            Self::Laya(m) => Some(m),
            Self::OpenJev(_) => None,
            #[cfg(feature = "jev")]
            Self::Jev(_) => None,
        }
    }
}
