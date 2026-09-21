use anyhow::{Context, Result};
use serde::Serialize;
use vs1::{SystemOneRequest, SystemOneResponse};

use crate::{Classification, Config, Mailbox, classify};

#[derive(Serialize)]
pub struct DryRunReport {
    pub dry_run: bool,
    pub host: String,
    pub username: String,
    pub mailbox: String,
    pub uid_validity: u32,
    pub classifications: Vec<Classification>,
}

pub fn dry_run(
    config: &Config,
    host: &str,
    username: &str,
    mailbox_name: &str,
    mailbox: &Mailbox,
    decide: &mut impl FnMut(&SystemOneRequest) -> Result<SystemOneResponse>,
) -> Result<DryRunReport> {
    let classifications = mailbox
        .emails
        .iter()
        .map(|email| {
            classify(config, email, decide)
                .with_context(|| format!("cannot classify UID {}", email.uid))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(DryRunReport {
        dry_run: true,
        host: host.into(),
        username: username.into(),
        mailbox: mailbox_name.into(),
        uid_validity: mailbox.uid_validity,
        classifications,
    })
}
