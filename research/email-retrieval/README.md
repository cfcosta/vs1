# Local labeled-example retrieval experiment

The retained experiment adds two similar labeled messages to laya's state as
`labeled_examples`, separate from the target `email`. Category rules, the
five-choice tournament, and length-weighted chunk pooling remain unchanged.
The normal `vs1-email` command does not enable this experiment.

`prepare.py` runs a frozen multilingual MiniLM encoder locally on CUDA. It
splits subject plus body into nonoverlapping 120-token chunks, normalizes each
chunk vector, takes their token-count-weighted sum, and normalizes that sum.
Cosine similarity selects two training messages. Each example contains its
subject, the first 160 body characters, and its reviewed reference category.
The model revision is pinned in the script. Python is an offline research
utility; the Rust inference path does not launch it.

Training JSON is an array of records with `subject`, `sender` (normalized email
address), `body`, and `label`. Targets have `subject`, `sender`, `date`, and
`body`; target labels are never read. Subject/date must uniquely identify each
target. The evaluation helper rejects targets sharing a sender or a normalized
body template with training data (five-word shingle Jaccard similarity >= 0.5).
Keep evaluation labels frozen before inference and separate from training.

A tested research environment used Python 3.13, PyTorch 2.11.0+cu128,
sentence-transformers 5.7.0, transformers 4.57.6, and NumPy 2.5.3. Install those
in an isolated environment with CUDA support. All email data remains local.

```sh
python3 -m unittest discover -s research/email-retrieval
python3 research/email-retrieval/prepare.py training.json targets.json examples.json
nix develop -c cargo build --release -p vs1-email --features cuda,flash-attn --example retrieval_audit
LD_LIBRARY_PATH=/run/opengl-driver/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} \
  target/release/examples/retrieval_audit email.toml SAMPLE_MAILDIR NEW_OUTPUT_DIR typed-decisions examples.json
```

The audit expects a frozen, clean 100-message Maildir sample and selects only
messages named by the examples map. It performs baseline/retrieval/retrieval/
baseline passes on exactly the same messages, with CUDA BF16, flash attention,
and batch size 16. Both fit checks and inference include the added examples;
chunk boundaries may therefore differ. It writes raw reports and timing
summaries to a new directory, and never changes mailbox files. Its timings
exclude model loading and example preparation. Treat outputs as private: the
audit also exports decoded selected messages for reproducibility.

See [experiment findings](../../docs/email-experiments.md#10-checkpoint-retrieval-embedding-and-larger-model-experiments)
for results and limitations. The private reproduction inputs and rejected
prototypes are under `~/.local/state/vs1-email/next-four-20260921/`.
