# vs1-email

Categorize a local Maildir mailbox with local laya inference through `vs1`.
Only `--dry-run` is implemented. Running without it returns an error before
reading files, the mailbox, or model weights.

Copy [`examples/email-rules.toml`](../../examples/email-rules.toml) and fill in
`[owner]`. Each `[[rules]]` entry has a unique `category`, required `what`, and
optional `not_for` and `examples`. At least two categories are required. The
model selects one category per message; array order is preserved for the model,
not interpreted as first-match priority. Include an `other` rule for unmatched
mail. Owner values provide context for references such as `owner.fixed_bills`.

```bash
cargo run -p vs1-email -- --config email.toml --dry-run --mailbox ~/Mail/account --limit 20
nix run .#vs1-email -- --config email.toml --dry-run --mailbox ~/Mail/account
```

`--mailbox` selects a local Maildir or an OfflineIMAP/mbsync sync root.
All descendant Maildir folders are discovered recursively, including nested
folders and hidden Maildir++ folders such as `.Sent`. If the root is itself
a Maildir, its messages are included too. Each Maildir must contain `cur/`,
`new/`, and `tmp/`. Incomplete Maildirs and roots with no Maildirs are errors.
No mail server, credentials, or IMAP connection is needed.
This version reads Maildir, not mbox files.

Dry-run reads regular message files in `cur/` and `new/`, skipping `tmp/`,
metadata files and symlink entries. It never descends into the `cur/`, `new/`,
or `tmp/` message storage directories when discovering folders. It never moves messages,
renames files, changes flags, or writes mailbox contents. Filesystem access
times may update when files are read. The limit defaults to 100 messages total across all folders in
sorted absolute path order. Unreadable or malformed messages are listed in `failures` with their paths
and errors; the remaining messages are classified. Attachments are excluded;
MIME headers and transfer encodings are decoded, plain text is preferred, and
HTML-only mail is converted to visible text, excluding scripts and styles. Attachment-only messages use their headers.

Stdout is one JSON report with `dry_run`, the absolute selected root path in
`mailbox`, `failures`, and `classifications`. Each classification includes its per-chunk evidence and absolute
source `path`, Message-ID, subject, category, confidence, per-category
probabilities, model name, and token usage. Folder discovery and inference failures abort the run; per-message read
and MIME failures are included in the report. Inference runs locally; the first model load may
download weights from Hugging Face. Use `--model /path/to/checkpoint` for
local model assets. Empty mailboxes skip model loading.

The model defaults to `convaiinnovations/laya`. Use `--model` for a local
checkpoint directory or another Hugging Face checkpoint, `--subfolder
multilingual` for the multilingual variant, and `--device cpu|cuda|metal` with
optional `--dtype f32|bf16|f16` and `--batch-size`. `--batch-size` defaults to 16 and controls how many messages are submitted
together to the model. The model prepares requests in parallel on CUDA and
packs their sequences into batched forward passes. Reduce it if GPU memory
is insufficient. The final partial batch is included and results retain file order.

Emails are split into nonoverlapping UTF-8 body chunks, preferring whitespace
boundaries. Every chunk repeats the email headers and owner context. The actual
model tokenizer checks the complete request against the checkpoint budget, so
body text is never silently dropped. Oversized metadata that leaves no body
capacity produces an explicit error. Empty bodies still produce one request.

`--max-len` and `--head-max-len` default to the checkpoint's own configuration
(512/192 for the base model, 1024/256 for multilingual). Overriding these is
experimental: a longer sequence can degrade decisions, not just increase cost.
Rule descriptions are choice criteria, not email content. Laya still caps
individual descriptions at 48 tokens and can shorten them further to fit the
question-header budget, particularly with many categories.

Chunk category probabilities are averaged using each chunk's Unicode character
count as its weight (an empty body has weight one). The highest aggregate score
wins, with ties resolved in rule order. `confidence` is normalized entropy of
those aggregate scores, not a newly calibrated email-level probability. Each
`chunks` entry reports its body character count, category, scores, confidence,
and token usage; full body text is not copied into the report. Email token usage
is summed across chunks. Disagreement between chunks is retained for inspection.

Chunking ensures coverage; it does not establish classification accuracy. Review
results on representative, labeled messages before trusting folder assignments.

Cargo forwards the same acceleration features as `vs1`: `cuda`, `flash-attn`,
`metal`, `mkl`, and `accelerate`. CPU works without features. The flake exposes
`vs1-email`, `vs1-email-cuda`, `vs1-email-flash-attn`, and `vs1-email-metal`,
following the existing package conventions and platform requirements.

```bash
cargo test -p vs1-email
cargo clippy -p vs1-email --all-targets -- -D warnings
nix build .#vs1-email
```

Tests use synthetic messages and temporary Maildirs; they need neither
mailbox credentials nor model downloads.
