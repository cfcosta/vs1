//! Caller-selected backends sharing the typed decision interface.
use crate::{Result, SystemOne, SystemOneRequest, SystemOneResponse};
/// Explicit runtime selection. Existing `SystemOne` remains the local laya API.
pub enum DecisionModel {
    Laya(Box<SystemOne>),
    OpenJev(Box<crate::OpenJev>),
    CuaS1(Box<crate::CuaS1>),
    GlinerDecide(Box<crate::GlinerDecide>),
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
impl From<crate::GlinerDecide> for DecisionModel {
    fn from(model: crate::GlinerDecide) -> Self {
        Self::GlinerDecide(Box::new(model))
    }
}
impl From<crate::CuaS1> for DecisionModel {
    fn from(model: crate::CuaS1) -> Self {
        Self::CuaS1(Box::new(model))
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
            Self::CuaS1(m) => m.model_name(),
            Self::GlinerDecide(m) => m.model_name(),
            #[cfg(feature = "jev")]
            Self::Jev(m) => m.model_name(),
        }
    }
    /// Configured maximum prompt size in tokens.
    /// Hosted Jev uses a vs1-declared budget, not a provider limit.
    pub fn context_tokens(&self) -> usize {
        match self {
            Self::Laya(m) => m.context_tokens(),
            Self::OpenJev(m) => m.context_tokens(),
            Self::CuaS1(m) => m.context_tokens(),
            Self::GlinerDecide(m) => m.context_tokens(),
            #[cfg(feature = "jev")]
            Self::Jev(m) => m.context_tokens(),
        }
    }
    /// Whether every question fits without truncating its input.
    /// Laya reserves its tournament header budget; hosted Jev is approximate.
    /// Cua-S1 checks first-round prompts; finalist prompts are limited at scoring.
    pub fn request_fits(&self, request: &SystemOneRequest) -> Result<bool> {
        match self {
            Self::Laya(m) => m.request_fits(request),
            Self::OpenJev(m) => m.request_fits(request),
            Self::CuaS1(m) => m.request_fits(request),
            Self::GlinerDecide(m) => m.request_fits(request),
            #[cfg(feature = "jev")]
            Self::Jev(m) => m.request_fits(request),
        }
    }
    pub fn system_one(
        &self,
        request: &SystemOneRequest,
    ) -> Result<SystemOneResponse> {
        match self {
            Self::Laya(m) => m.system_one(request),
            Self::OpenJev(m) => m.system_one(request),
            Self::CuaS1(m) => m.system_one(request),
            Self::GlinerDecide(m) => m.system_one(request),
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
            Self::CuaS1(m) => m.system_one_batch(requests),
            Self::GlinerDecide(m) => m.system_one_batch(requests),
            #[cfg(feature = "jev")]
            Self::Jev(m) => m.system_one_batch(requests),
        }
    }
    /// Access Laya-specific tokenizer/device facilities.
    pub fn local(&self) -> Option<&SystemOne> {
        match self {
            Self::Laya(m) => Some(m),
            Self::OpenJev(_) => None,
            Self::CuaS1(_) => None,
            Self::GlinerDecide(_) => None,
            #[cfg(feature = "jev")]
            Self::Jev(_) => None,
        }
    }
}
