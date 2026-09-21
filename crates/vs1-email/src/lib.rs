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

mod mailbox;
pub use mailbox::{Mailbox, parse_email, read_mailbox};

mod report;
pub use report::{DryRunReport, dry_run};
