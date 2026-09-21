use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, ensure};
use mailparse::{DispositionType, MailHeaderMap, ParsedMail};
use serde::Serialize;

use crate::Email;

/// Snapshot identity and decoded messages from one read-only mailbox.
pub struct Mailbox {
    pub path: PathBuf,
    pub emails: Vec<Email>,
    pub failures: Vec<MessageFailure>,
}

/// A message that could not be read or MIME-decoded.
#[derive(Clone, Debug, Serialize)]
pub struct MessageFailure {
    pub path: PathBuf,
    pub error: String,
}

pub fn parse_email(path: impl Into<PathBuf>, raw: &[u8]) -> Result<Email> {
    let parsed = mailparse::parse_mail(raw).context("invalid MIME message")?;
    let header =
        |name| parsed.headers.get_first_value(name).unwrap_or_default();
    Ok(Email {
        path: path.into(),
        message_id: header("Message-ID"),
        from: header("From"),
        to: header("To"),
        subject: header("Subject"),
        date: header("Date"),
        body: body_text(&parsed)?,
    })
}

fn body_text(mail: &ParsedMail<'_>) -> Result<String> {
    if mail.get_content_disposition().disposition == DispositionType::Attachment
    {
        return Ok(String::new());
    }
    if mail.ctype.mimetype == "multipart/alternative" {
        let preferred = mail.subparts.iter().find(|part| {
            part.ctype.mimetype == "text/plain"
                && part.get_content_disposition().disposition
                    != DispositionType::Attachment
        });
        if let Some(plain) = preferred {
            return body_text(plain);
        }
        // Last supported alternative is the sender's preferred representation.
        for part in mail.subparts.iter().rev() {
            let text = body_text(part)?;
            if !text.is_empty() {
                return Ok(text);
            }
        }
        return Ok(String::new());
    }
    if !mail.subparts.is_empty() {
        let parts = mail
            .subparts
            .iter()
            .map(body_text)
            .collect::<Result<Vec<_>>>()?;
        return Ok(parts
            .into_iter()
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join("\n"));
    }
    match mail.ctype.mimetype.as_str() {
        "text/plain" => Ok(mail.get_body()?),
        "text/html" => {
            let html = mail.get_body()?;
            Ok(html2text::config::plain()
                .link_footnotes(false)
                .string_from_read(html.as_bytes(), 120)?)
        }
        _ => Ok(String::new()),
    }
}

/// Reads a Maildir or sync root, including all descendant Maildir folders.
/// The limit applies globally in sorted absolute file-path order.
pub fn read_maildir(path: &Path, limit: usize) -> Result<Mailbox> {
    ensure!(limit > 0, "limit must be greater than zero");
    let path = path.canonicalize().context("cannot open Maildir")?;
    let mut paths = Vec::new();
    let mut pending = vec![path.clone()];
    let mut folders = 0;
    while let Some(directory) = pending.pop() {
        let entries = fs::read_dir(&directory)
            .with_context(|| format!("cannot inspect {}", directory.display()))?
            .collect::<std::io::Result<Vec<_>>>()?;
        let mut storage = Vec::new();
        for entry in entries {
            // Do not follow symlinks, including directory links and cycles.
            if !entry.file_type()?.is_dir() {
                continue;
            }
            if ["cur", "new", "tmp"]
                .iter()
                .any(|name| entry.file_name() == *name)
            {
                storage.push(entry.path());
            } else {
                pending.push(entry.path());
            }
        }
        if storage.is_empty() {
            continue;
        }
        ensure!(
            storage.len() == 3,
            "incomplete Maildir {}: expected cur/, new/, and tmp/",
            directory.display()
        );
        folders += 1;
        for name in ["cur", "new"] {
            for entry in fs::read_dir(directory.join(name))? {
                let entry = entry?;
                if entry.file_type()?.is_file() {
                    paths.push(entry.path());
                }
            }
        }
    }
    ensure!(
        folders > 0,
        "no Maildir folders found under {}",
        path.display()
    );
    paths.sort();
    paths.truncate(limit);
    let mut emails = Vec::new();
    let mut failures = Vec::new();
    for file in paths {
        let parsed = fs::read(&file)
            .map_err(anyhow::Error::from)
            .and_then(|raw| parse_email(file.clone(), &raw));
        match parsed {
            Ok(email) => emails.push(email),
            Err(error) => failures.push(MessageFailure {
                path: file,
                error: format!("{error:#}"),
            }),
        }
    }
    Ok(Mailbox {
        path,
        emails,
        failures,
    })
}
