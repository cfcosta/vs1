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

| Run (2026-09-24)       | Token IDs and markers | Max probability error | Labels agree |
| ---------------------- | --------------------- | --------------------- | ------------ |
| Rust CPU F32, batch 8  | identical (6/6)       | 4.8e-7                | 9/9          |
| Rust CUDA F32, batch 8 | identical (6/6)       | 5.4e-7                | 9/9          |

Rust ran all 6 requests as one padded batch, and the reference ran each request
alone. The results therefore also cover padding and masking in candle's DeBERTa.
The CUDA run used an RTX 3080 Ti and took 168 ms for the batch after loading.
This is a single run, not a benchmark.

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
```

On NixOS, binary wheels need the GCC runtime and a 64-bit zlib on
`LD_LIBRARY_PATH`. CUDA executables need `/run/opengl-driver/lib`. Compare
`ids.ids` and `ids.markers` with the reference's `ids` and `markers[1:]`. The
first marker of each reference task is its `[P]` token. Then compare each
answer's probabilities with the reference's `answers`.
