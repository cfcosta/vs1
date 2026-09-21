# OpenJev (Verdict) integration and validation

Date: 2026-09-21. This adds a separate local model; it does not replace, prune,
distill or retrain Laya or OpenJev.

## Architecture and implementation

The [checkpoint](https://huggingface.co/heman10x/rlcd-modernbert-151m/tree/8af2496eb63c7fa66d7d234e1f62629380030eb4)
contains 143 F32 tensors: a ModernBERT-base encoder (22 layers, width 768,
12 attention heads, intermediate width 1152), two 768→768→768 projection heads,
and an unused logit-scale parameter. The encoder uses GeGLU, global attention
every third layer, local window 128 and the original rotary embeddings.

The integration reuses `ModernBert::load` and `ModernBert::forward` without
kernel changes. OpenJev's GLiClass head differs from Laya's head: project the
CLS embedding and each `<<LABEL>>` embedding with exact GELU, then take their
dot products. This checkpoint disables feature normalization, so its
`logit_scale` must not be applied. Each instance owns its tokenizer and heads.

Formatting follows the [Verdict engine](https://github.com/Heman10x-NGU/Verdict-open-jev)
and scoring follows [GLiClass](https://github.com/Knowledgator/GLiClass).
The oracle in `reference.py` uses GLiClass 0.1.20, Transformers 5.17.0 and
PyTorch 2.13.0+cu130, with the checkpoint's actual tokenizer and model forward.
It independently formats requests instead of taking prompts from the Rust port.
It performs no training. Python is an offline validation dependency only.

Calibration comes from the pinned `calibrator.json`: global temperature
**2.8039**, not the older value 1.0716 in the model card. Exact `per_k` entries
override the global value, counting the abstention candidate in k; missing
entries use the global temperature without interpolation. The artifact reports
hash `b4742a033bce3fb4707e78d910dc89d53c30656d028b1836a992ba87e0766324`.

## Retained: F32 native inference

Thirteen cases cover choice, score, noul, missing evidence, Portuguese text,
24 substantive options, a missing per-k temperature entry, and right truncation
to 512 tokens. Comparison checks exact prompts, token IDs and marker positions,
raw logits, the full calibrated probability vector and selected candidate.
Mixed-length/cardinality batches are also compared with individual inference.

| Measurement                                    |     CPU F32 |    CUDA F32 |
| ---------------------------------------------- | ----------: | ----------: |
| Decisions matching PyTorch                     |       13/13 |       13/13 |
| Maximum absolute logit error                   |  0.00002003 |  0.00001991 |
| Maximum absolute probability error             | 0.000000954 | 0.000000775 |
| Maximum batch/singleton probability difference |           0 | 0.000000537 |
| Batch/singleton selected candidates matching   |       13/13 |       13/13 |
| Warm short-case median, 20 iterations          |    43.14 ms |     4.79 ms |

CUDA measurements used an RTX 3080 Ti; CPU used four Rayon threads. The short
case is 41 tokens with two substantive options plus abstention. Timings exclude
loading and tokenization and include native input validation, GPU transfer and
prediction readback. The desktop shares this GPU, and clocks were not locked;
an earlier CUDA F32 run measured 5.37 ms. These are smoke timings, not a paired
speedup claim against Laya or a workload benchmark.

The final CUDA executable passed the same parity check after removing the
rejected BF16 branch. CLI execution with batch size two returned all three
normal answer types and explicit abstention, preserving two-request ordering.
The native distribution retains abstention mass; the shared adapter conditions
non-abstained answers on sufficient evidence as documented in
[the API guide](../../docs/openjev.md).

Laya (the local reference checkpoint reporting `rl-agent`) and OpenJev were
loaded together on one CUDA device through `DecisionModel`. OpenJev's output
was unchanged after loading Laya, and three alternating calls per model were
identical to each model's initial response. This verifies simultaneous residency
and alternating use; it does not establish concurrent-stream throughput.

## Rejected: BF16 weights plus packed FlashAttention

The temporary BF16 implementation converted encoder and projection weights and
called the existing `forward_varlen_packed` path. It measured 2.71 ms on the
short case. The mixed batch matched all 13 F32-reference selected candidates,
but singleton/batch agreement was only **12/13**. For “The package arrived late
and damaged,” the top satisfaction level changed between 0 and 1.

Maximum logit error versus F32 was 0.10950; maximum probability error was
0.00605; maximum batch/singleton probability difference was 0.00511. Score's
expected value is continuous, so this was a top-category flip, not an abstention
flip. It still fails our stricter selected-candidate stability gate. This result
does not isolate weight rounding, attention arithmetic and shape-dependent GEMM
rounding from each other.

The BF16 implementation was removed. The builder rejects reduced precision
before downloading a checkpoint. No changes to Laya's precision policy or
shared kernels were retained. A future lower-precision attempt needs new
evidence; these measurements are not permission to silently enable it.

## Reproduce

Download `config.json`, `tokenizer.json`, `tokenizer_config.json`,
`calibrator.json` and `model.safetensors` from the pinned revision into
`artifacts/openjev/checkpoint`. In a separate Python environment with the
versions above installed:

```bash
python research/openjev/reference.py \
  artifacts/openjev/checkpoint artifacts/openjev/reference.json cpu

cargo build --release -p vs1 --features flash-attn \
  --example openjev_parity --example models_together --bin vs1

RAYON_NUM_THREADS=4 target/release/examples/openjev_parity \
  artifacts/openjev/checkpoint artifacts/openjev/reference.json cpu

target/release/examples/openjev_parity \
  artifacts/openjev/checkpoint artifacts/openjev/reference.json cuda

target/release/examples/models_together \
  /path/to/laya artifacts/openjev/checkpoint cuda

target/release/vs1 --backend openjev --device cuda \
  --model artifacts/openjev/checkpoint --batch-size 2 --dump-ids \
  research/openjev/request.json
```

On NixOS, expose the NVIDIA driver with
`LD_LIBRARY_PATH=/run/opengl-driver/lib` when running the CUDA executables.
The CPU-only build needs no CUDA feature. Generated reference outputs and
logs stay under the ignored `artifacts/openjev/` directory.

Validation also includes workspace tests, CUDA-feature Clippy, core library
tests with FlashAttention compiled, and a no-default-features check. Existing
ignored hardware microbenchmarks are not part of this integration's test run.
Unit tests cover prompt formatting, candidate limits, reserved markers,
temperature fallback, conditional answer semantics, abstention serialization,
unsupported noul criteria and rejected precision. The parity executable checks
empty batches, malformed prepared tokens and truncated candidate headers.

This small suite validates implementation parity, not OpenJev's accuracy on
the user's workloads or its quality relative to Laya.
