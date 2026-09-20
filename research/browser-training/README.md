# Browser training research artifacts

The [training specification](../../docs/browser-decision-training-spec.md) defines
the proposed model/data/training/evaluation pipeline and a bounded Vast.ai pilot.
The [sample audit](sample-audit.md) records what was actually downloaded and
inspected. Only the sampling and inspection utilities are implemented here.

These are offline research tools. They add no Python dependency or service to
the Rust browser-agent CLI. They do not train a model, call Jev, visit sampled
websites or rent a GPU.

## Reproduce the samples

From the repository root, with Python and `uv` available:

```sh
uv venv /tmp/vs1-browser-data-audit
uv pip install --python /tmp/vs1-browser-data-audit/bin/python \
  -r research/browser-training/requirements.txt
/tmp/vs1-browser-data-audit/bin/python \
  research/browser-training/download_samples.py
```

On NixOS, `direnv exec .` exposes `uv`; select a Nix-provided Python interpreter
if needed, and make the matching GCC runtime library directory available through
`LD_LIBRARY_PATH` for binary Python wheels. No CUDA library is needed for this
sampling/inspection process. The observed audit used CPython 3.14.7.

The downloader pins three dataset commits and selects training data only:

- First two complete tasks in Mind2Web train shards 0, 5 and 10. A bounded range
  grows from 8 MiB to at most 32 MiB to find complete JSON array objects.
- Complete 598,824-byte `typed-decisions/all/train` Parquet file, selecting the
  first, middle and last record of each workflow for the 12-case sample.
- Four complete WebWorldData JSONL records near each of three fixed byte offsets.
  Mid-line fragments are discarded. It transfers three 8 MiB ranges, not the full
  52 GB file. HTTP range support is required; a full-file response is rejected.

Output: `artifacts/browser-training/samples/`, excluded from version control.
The three `*-sample.jsonl` files wrap each unchanged source record in
`{"source": {...}, "record": {...}}`. The source identifies the original shard
and task index, Parquet row index, or exact JSONL byte offset.

`download-manifest.json` includes source revisions, URLs, transferred ranges,
byte counts, response hashes and sample hashes. It includes unsuccessful-size
prefix attempts that were retried with a larger cap. Rerunning uses the same
pinned source and deterministic selection and overwrites only these audit files.

## Inspect the samples

Use the same tokenizer as the English checkpoint being evaluated:

```sh
/tmp/vs1-browser-data-audit/bin/python \
  research/browser-training/inspect_samples.py \
  --tokenizer /path/to/laya/tokenizer/tokenizer.json
```

For this machine, the measured tokenizer came from checkpoint revision
`c5d78730f3493e4fe16d61507ef4b78eef7318cf` in the Hugging Face cache. Its SHA-256
is recorded in `inspection.json`. The script disables token truncation/padding,
counts source-text tokens, checks label support/probabilities, parses inert HTML,
and parses action syntax without execution. It does not implement the proposed
production converters or estimate candidate-retrieval recall.

The inspection outputs are `inspection.json`, plus
`webworld-transition-audit.json` for transition-by-transition checks. The committed
[inspection.json](inspection.json) and [download-manifest.json](download-manifest.json)
are small snapshots from the observed run. Raw data, downloaded source cards and
metadata stay in the local artifact directory.

All counts describe a small convenience sample. Task/byte-offset sampling is not
uniform random sampling over websites, actions or trajectories. No full-corpus
quality or training-time claim follows from it.

## Provenance and licenses

See the immutable dataset revisions in the manifest and specification.
Mind2Web declares CC BY 4.0; typed-decisions and WebWorldData declare Apache 2.0.
No raw dataset samples are committed. Preserve dataset attribution and relevant
notices in future derived releases. The downloaded typed-decisions data is the
subset documented in Laya's fine-tuning notebook, not its complete original
training corpus.
