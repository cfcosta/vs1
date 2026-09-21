use std::io::{Read, Write};

use anyhow::{Context, Result, ensure};
use mailparse::{DispositionType, MailHeaderMap, ParsedMail};

use crate::Email;

/// Snapshot identity and decoded messages from one read-only mailbox.
pub struct Mailbox {
    pub uid_validity: u32,
    pub emails: Vec<Email>,
}

pub fn parse_email(uid: u32, raw: &[u8]) -> Result<Email> {
    let parsed = mailparse::parse_mail(raw).context("invalid MIME message")?;
    let header =
        |name| parsed.headers.get_first_value(name).unwrap_or_default();
    Ok(Email {
        uid,
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
        "text/plain" | "text/html" => Ok(mail.get_body()?),
        _ => Ok(String::new()),
    }
}

/// Uses EXAMINE and UID BODY.PEEK exclusively; never changes flags or folders.
/// Messages are processed in ascending UID order, up to `limit`.
pub fn read_mailbox<T: Read + Write>(
    session: &mut imap::Session<T>,
    mailbox: &str,
    search: &str,
    limit: usize,
) -> Result<Mailbox> {
    ensure!(limit > 0, "limit must be greater than zero");
    for (name, value) in [("mailbox", mailbox), ("search", search)] {
        ensure!(
            !value.trim().is_empty() && !value.chars().any(char::is_control),
            "{name} must be nonempty and contain no control characters"
        );
    }
    let selected =
        session.examine(mailbox).context("cannot examine mailbox")?;
    let uid_validity = selected
        .uid_validity
        .filter(|n| *n > 0)
        .context("mailbox has no valid UIDVALIDITY")?;
    let mut uids = session.uid_search(search)?.into_iter().collect::<Vec<_>>();
    uids.sort_unstable();
    uids.truncate(limit);
    let mut emails = Vec::with_capacity(uids.len());
    for uid in uids {
        let fetched =
            session.uid_fetch(uid.to_string(), "(UID BODY.PEEK[])")?;
        let message = fetched
            .iter()
            .find(|message| message.uid == Some(uid))
            .with_context(|| {
            format!("UID {uid} disappeared during fetch")
        })?;
        let body = message
            .body()
            .with_context(|| format!("UID {uid} has no fetched body"))?;
        emails.push(
            parse_email(uid, body)
                .with_context(|| format!("cannot parse UID {uid}"))?,
        );
    }
    Ok(Mailbox {
        uid_validity,
        emails,
    })
}
