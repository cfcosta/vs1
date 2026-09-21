# vs1-email

Categorize a local Maildir mailbox through `vs1`, with caller-selected local
laya (the default) or hosted Jev inference.
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

Select hosted inference at runtime in the normal build:

```bash
# TYPESAFE_API_KEY must already be set; sampled email content is sent to TypeSafe.
cargo run --release -p vs1-email -- \
  --backend jev --model jev-1.13.0 --dry-run --config email.toml \
  --mailbox ~/Mail/account --limit 100
# Equivalent packaged command:
nix run .#vs1-email -- --backend jev --model jev-1.13.0 \
  --dry-run --config email.toml --mailbox ~/Mail/account --limit 100
```

Jev uses one whole cleaned email and one Choice containing all categories per
request (at most 255 categories). It preserves the same subject, sender, date,
owner context and recipient exclusion. Bodies are neither split nor truncated;
provider context-limit errors abort inference. No checkpoint is loaded.
`--model` defaults to `jev-latest` in this mode. Local-only device, dtype,
subfolder and token-budget options are rejected. `--batch-size` controls hosted
concurrency, defaulting to 16.

The email policy selects probability argmax and normalizes provider rounding
only when the probability sum is within 0.025 of one. Invalid distributions
remain errors. Original answers are retained in each chunk's `decisions.provider`;
`decisions.evaluated` records the normalized decision. This adaptation belongs
to the email classifier; `vs1::JevClient` returns unmodified provider answers.

Stderr includes a JSON `Jev calls:` summary with logical calls, actual HTTP
attempts, retries, successes and questions, even when an inference request fails.
For 100 successfully parsed emails, a successful no-retry Jev run makes 100 calls
and asks 100 questions. Empty mailboxes need no API key. Both backends preserve
dry-run-only behavior and support `--progress-jsonl`.

`--mailbox` selects a local Maildir or an OfflineIMAP/mbsync sync root.
All descendant Maildir folders are discovered recursively, including nested
folders and hidden Maildir++ folders such as `.Sent`. If the root is itself
a Maildir, its messages are included too. Each Maildir must contain `cur/`,
`new/`, and `tmp/`. Incomplete Maildirs and roots with no Maildirs are errors.
No mail server credentials or IMAP connection are needed to read the mailbox.
Hosted Jev inference separately requires a TypeSafe API key.
This version reads Maildir, not mbox files.

Dry-run reads regular message files in `cur/` and `new/`, skipping `tmp/`,
metadata files and symlink entries. It never descends into the `cur/`, `new/`,
or `tmp/` message storage directories when discovering folders. It never moves messages,
renames files, changes flags, or writes mailbox contents. Filesystem access
times may update when files are read. The limit defaults to 100 messages total across all folders in
sorted absolute path order. Unreadable or malformed messages are listed in `failures` with their paths
and errors; the remaining messages are classified. Attachments are excluded;
MIME headers and transfer encodings are decoded, plain text is preferred, and
HTML-only mail and complete HTML documents mislabeled as plain text are converted
to visible text, excluding scripts and styles. Isolated HTML/CSS examples in
ordinary plain text are preserved. Before
chunking, text cleanup collapses horizontal whitespace and repeated blank lines,
removes decorative table borders, and retains table cell values and paragraph
breaks. Recognized tracking redirects (SendGrid click links, Mailchimp click
links, and compressed `/c/` links on email tracking hosts) are removed entirely.
Known tracking query parameters are removed from other URLs. Long opaque
values are removed only for recognized token parameters such as `sparams` and
`access_token`; domains, paths, fragments and ordinary query fields remain.
This preprocessing is for model input: resulting links may no longer be usable
for authentication. Quoted history and boilerplate prose are retained. Attachment-only messages use their headers.

Stdout is one JSON report with `dry_run`, the absolute selected root path in
`mailbox`, `failures`, and `classifications`. Each classification includes its per-chunk evidence and absolute
source `path`, Message-ID, subject, category, confidence, per-category
probabilities, model name, and token usage. Folder discovery and inference failures abort the run; per-message read,
MIME and chunk-preparation failures are included in the report. With the default laya backend, inference runs locally; the first model load may
download weights from Hugging Face. Use `--model /path/to/checkpoint` for
local model assets. Empty mailboxes skip model loading.

For `--backend laya`, the model defaults to `convaiinnovations/laya`. Use `--model` for a local
checkpoint directory or another Hugging Face checkpoint, `--subfolder
multilingual` or `--subfolder typed-decisions` for alternate checkpoints, and `--device cpu|cuda|metal` with
optional `--dtype f32|bf16|f16` and `--batch-size`. `--batch-size` defaults to 16 and controls how many messages are submitted
together to the model. Each chunk may require multiple tournament rounds. The model prepares requests in parallel on CUDA and
packs their sequences into batched forward passes. Reduce it if GPU memory
is insufficient. The final partial batch is included and results retain file order.

For laya, emails are split into nonoverlapping UTF-8 body chunks, preferring whitespace
boundaries. Every chunk repeats the subject, sender, date and owner context. Recipient
lists are excluded from model input. The actual
model tokenizer checks the complete request against the checkpoint budget, so
body text is never silently dropped. Oversized metadata that leaves no body
capacity is recorded in `failures`; other messages continue. Empty bodies still produce one request.

`--max-len` and `--head-max-len` default to the checkpoint's own configuration
(512/192 for the base model, 1024/256 for multilingual). Overriding these is
experimental: a longer sequence can degrade decisions, not just increase cost.
Each decision compares at most five categories. Larger rule sets use balanced
contests in configuration order: their winners advance to another small contest
until a final choice is reached. All contests in a round run in batches; for 17
rules this is four preliminary questions and one final question per body chunk.
When four winners leave one spare slot in the five-way final, the closest
runner-up also advances. Closeness is the runner's probability divided by its
own group's winning probability; ties and finalist order follow configuration
order. This uses the existing preliminary answers and adds no model question.
An early elimination can lose the correct category, so this is an approximation.

Descriptions use plain text, followed by exclusions and examples. Keep them
short: laya caps each option at 48 tokens and can shorten it further to fit the
header budget. The root `email.toml` uses concise descriptions. Body chunks
reserve the full question-header budget so later finalists cannot truncate them.

Final-round scores are conditional on the surviving shortlist. Eliminated
categories receive zero in that chunk's `probabilities`; zero does not mean
impossibility. Each chunk's `decisions` retains every round's candidates and raw
answers, including eliminated choices. Usage includes all rounds.

Chunk category probabilities are averaged using each chunk's Unicode character
count as its weight (an empty body has weight one). The highest aggregate score
wins, with ties resolved in rule order. `confidence` is normalized entropy of
those aggregate scores, not a newly calibrated email-level probability. Each
`chunks` entry reports its body character count, category, scores, confidence,
and token usage; full body text is not copied into the report. Email token usage
is summed across chunks. Disagreement between chunks is retained for inspection.

Chunking ensures coverage; it does not establish classification accuracy. Review
results on representative, labeled messages before trusting folder assignments.

Cargo forwards `jev` and the same acceleration features as `vs1`: `cuda`, `flash-attn`,
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

For long runs, add `--progress-jsonl /path/to/new-results.jsonl`. Each completed
email is flushed as one JSON line while inference continues. The file must not
already exist. These partial results survive later inference errors; they are
not an automatic resume mechanism. The final stdout report includes failures.
