# vs1-email

Categorize an IMAP mailbox with local laya inference through `vs1`.
Only `--dry-run` is implemented. Running without it returns an error before
reading files, credentials, the mailbox, or model weights.

Copy [`examples/email-rules.toml`](../../examples/email-rules.toml) and fill in
`[owner]`. Each `[[rules]]` entry has a unique `category`, required `what`, and
optional `not_for` and `examples`. At least two categories are required. The
model selects one category per message; array order is preserved for the model,
not interpreted as first-match priority. Include an `other` rule for unmatched
mail. Owner values provide context for references such as `owner.fixed_bills`.

```bash
export VS1_EMAIL_HOST=imap.example.com
export VS1_EMAIL_USERNAME=you@example.com
# Set VS1_EMAIL_PASSWORD to your IMAP password or app password.
cargo run -p vs1-email -- \
  --config examples/email-rules.toml --dry-run --mailbox INBOX \
  --search UNSEEN --limit 20

nix run .#vs1-email -- --config examples/email-rules.toml --dry-run
```

TLS with certificate and hostname verification is mandatory (port 993 by
default). This version uses IMAP LOGIN; OAuth-only accounts are not supported.
Mailbox access uses `EXAMINE`, `UID SEARCH`, and `UID FETCH ... BODY.PEEK[]`,
followed by `LOGOUT`. It does not mark mail as read, label it, move it, or expunge
it. Search defaults to `ALL`; the limit defaults to 100 messages in ascending
UID order. A message disappearing during the read is an error rather than a
silently incomplete report. Attachments are excluded; MIME headers and transfer
encodings are decoded, plain text is preferred, and HTML-only mail retains its
HTML. Attachment-only messages are classified using their headers.

Stdout is one JSON report with `dry_run`, host, username, mailbox, UIDVALIDITY,
and `classifications`. Each classification includes UID, Message-ID, subject,
category, confidence, per-category probabilities, model name, and token usage.
No success report is emitted if reading or inference fails. Inference runs
locally; the first model load may download weights from Hugging Face. Empty
mailboxes skip model loading.

The model defaults to `convaiinnovations/laya`. Use `--model` for a local
checkpoint directory or another Hugging Face checkpoint, `--subfolder
multilingual` for the multilingual variant, and `--device cpu|cuda|metal` with
optional `--dtype f32|bf16|f16` and `--batch-size`. Each message currently makes
one model request; batch size controls the underlying model engine, not mailbox
batching.

`--max-len` defaults to 4096 and `--head-max-len` to 2048 to accommodate the
17-category example. Complete criteria are placed first in the model state,
followed by owner context and email; category names are the choice options.
This avoids laya's 48-token per-option description cap. The complete state is
still subject to the sequence budget and truncation on the right. Keep rules
and owner context concise enough to leave room for email, and review results;
this preview does not establish classification accuracy for your mailbox.

Cargo forwards the same acceleration features as `vs1`: `cuda`, `flash-attn`,
`metal`, `mkl`, and `accelerate`. CPU works without features. The flake exposes
`vs1-email`, `vs1-email-cuda`, `vs1-email-flash-attn`, and `vs1-email-metal`,
following the existing package conventions and platform requirements.

```bash
cargo test -p vs1-email
cargo clippy -p vs1-email --all-targets -- -D warnings
nix build .#vs1-email
```

Tests use synthetic messages and a scripted IMAP transport; they need neither
mailbox credentials nor model downloads.
