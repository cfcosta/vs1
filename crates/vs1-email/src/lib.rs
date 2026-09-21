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
pub use mailbox::{Mailbox, MessageFailure, parse_email, read_maildir};

mod report;
pub use report::{DryRunReport, dry_run, dry_run_with_progress};

mod chunks;
pub use chunks::{request_fits, split_body};

mod cleanup;
pub use cleanup::clean_body;
