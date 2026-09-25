# GLiNER2.5-Decide parity

The oracle is upstream `gliner2` at commit
`55656fbfa01d3d4a77485e1a1eeeaf682990ccdf`, with checkpoint revision
`7ee5da4c2415e32259bcdc0b1a7367c32ce8d6f6` on CPU, PyTorch 2.14.0 and
transformers 4.57.6. `reference.py` applies vs1's question-to-task mapping,
calls upstream `classify_text`, and records the token IDs, the `[L]` positions
and every label's logit and softmax probability. The script asserts that its
manual encoder → `[L]` → classifier path chooses the same label as
`classify_text`, with confidence equal to within 1e-5.

`request.json` has 6 requests and 9 questions. They cover single and multi-task
requests, bare and described choices, a score scale, noul with and without
criteria, null instructions, a JSON state, non-ASCII text and a state without
final punctuation.

`request-extended.json` has 18 requests and 24 questions: 15 examples from the
model card, two requests from the OpenJev validation, and one 742-token email
thread with three questions. The long request checks inputs beyond DeBERTa's
512-entry absolute position table, which this model does not use.

| Run                                        | Set      | Token IDs and markers | Max probability error | Labels agree |
| ------------------------------------------ | -------- | --------------------- | --------------------- | ------------ |
| Rust CPU F32, candle encoder (2026-09-24)  | base     | identical (6/6)       | 4.8e-7                | 9/9          |
| Rust CUDA F32, candle encoder (2026-09-24) | base     | identical (6/6)       | 5.4e-7                | 9/9          |
| Rust CUDA F32, vs1 encoder (2026-09-25)    | base     | identical (6/6)       | 3.6e-7                | 9/9          |
| Rust CUDA F32, vs1 encoder (2026-09-25)    | extended | identical (18/18)     | 6.6e-7                | 24/24        |
| Rust CUDA BF16, vs1 encoder (2026-09-25)   | base     | identical (6/6)       | 0.0055                | 9/9          |
| Rust CUDA BF16, vs1 encoder (2026-09-25)   | extended | identical (18/18)     | 0.0094                | 24/24        |

All Rust runs used batch 8 with `--max-len 1024`; the first two used the
default 512. Rust batched the requests with padding, and the reference ran each
request alone. The results therefore also cover padding and masking. BF16 runs
were repeated after length-sorted batching was added, with identical errors.
The CUDA runs used an RTX 3080 Ti.

### Encoder changes

The first version used `candle-transformers` 0.11.0 unchanged. It was limited
to F32, because its masked softmax mixes hard-coded F32 tensors with the
attention scores. It also failed on inputs over 512 tokens. It sliced the
absolute position table even though `position_biased_input` is false; PyTorch
never reads that slice. vs1's copy fixes both. It sums the content and position
scores and runs the mask and softmax in F32. It reads absolute positions only
when they bias the input.

### Throughput (single runs, 2026-09-25)

These are whole-CLI measurements after loading. They include tokenization and
first-call setup. They are not paired AB/BA benchmarks.

| Input                                    | Batching         | BF16             | F32              |
| ---------------------------------------- | ---------------- | ---------------- | ---------------- |
| 17 short requests × 16 (272, ≤99 tokens) | input order, 8   | 4.9 ms/question  | 7.3 ms/question  |
| 17 short requests × 16 (272, ≤99 tokens) | input order, 32  | 4.3 ms/question  | 6.4 ms/question  |
| extended × 16 (288, one 742 per 18)      | input order, 8   | 40.2 ms/question | 47.7 ms/question |
| extended × 16 (288, one 742 per 18)      | length-sorted, 8 | 7.9 ms/question  | 10.0 ms/question |

Batches in input order padded every row to the longest request in the batch.
Sorting by length cut the mixed run by 5.1× in BF16. With input-order batches of
32, the mixed set ran out of GPU memory in both precisions.

The unit test for the word splitter uses expected words produced by upstream
`WhitespaceTokenSplitter`. Rust's `\w` and Python's `\w` differ on some Unicode
marks. That difference is not covered here.

Upstream answers the model card's treaty example ("Did the treaty enter into
force in 1992?") with `yes` at 0.996. The card lists `no` as a potential
output. vs1 reproduces upstream, not the card.

## Reproduce

```sh
uv venv /tmp/gliner2-ref
uv pip install --python /tmp/gliner2-ref/bin/python torch \
  "gliner2[local] @ git+https://github.com/fastino-ai/GLiNER2@55656fbfa01d3d4a77485e1a1eeeaf682990ccdf"
/tmp/gliner2-ref/bin/python research/gliner-decide/reference.py \
  CHECKPOINT research/gliner-decide/request.json artifacts/gliner-decide/reference.json

target/release/vs1 --backend gliner-decide --dump-ids \
  research/gliner-decide/request.json > artifacts/gliner-decide/rust-cpu.json
python3 research/gliner-decide/compare.py \
  artifacts/gliner-decide/reference.json artifacts/gliner-decide/rust-cpu.json

# CUDA BF16 (the default there) on the extended set.
target/release/vs1 --backend gliner-decide --device cuda --max-len 1024 \
  --dump-ids research/gliner-decide/request-extended.json \
  > artifacts/gliner-decide/cuda-bf16-extended.json
```

On NixOS, binary wheels need the GCC runtime and a 64-bit zlib on
`LD_LIBRARY_PATH`. CUDA executables need `/run/opengl-driver/lib`.
`compare.py` checks that `ids.ids` and `ids.markers` equal the reference's
`ids` and `markers[1:]`. The first marker of each reference task is its `[P]`
token. It then reports the largest probability difference and any label that
differs from upstream.
