//! Mailbox categorization using local laya decision models.

mod config;
pub use config::{Config, Rule};

mod classification;
pub use classification::{
    Classification,
    Email,
    classification_request,
    classify,
};
