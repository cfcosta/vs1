# OpenJev (Verdict) integration and validation

Date: 2026-09-21. This adds a separate local model; it does not replace, prune,
distill or retrain Laya or OpenJev.

## Architecture and implementation

The [checkpoint](https://huggingface.co/heman10x/rlcd-modernbert-151m/tree/8af2496eb63c7fa66d7d234e1f62629380030eb4)
contains 143 F32 tensors: a ModernBERT-base encoder (22 layers, width 768,
12 attention heads, intermediate width 1152), two 768→768→768 projection heads,
and an unused logit-scale parameter. The encoder uses GeGLU, global attention
every third layer, local window 128 and the original rotary embeddings.

The integration reuses `ModernBert::load`, `ModernBert::forward` and the BF16
`forward_varlen_packed` path without kernel changes. OpenJev's GLiClass head differs from Laya's head: project the
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

## Initial rejection: BF16 weights plus packed FlashAttention

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

At that point, the BF16 implementation was removed and the builder rejected
reduced precision before downloading a checkpoint. No changes to Laya's precision policy or
shared kernels were retained. A future lower-precision attempt needs new
evidence. The re-evaluation below supersedes that initial decision.

## BF16 re-evaluation: the original gate tested the wrong score output

The rejection conflated the native rubric argmax with the public `score`,
which is an expected value conditional on sufficient evidence. The problematic
input's F32 score was **1.857550** on a 0–4 scale. BF16 returned **1.863468**
alone and **1.879906** in the original mixed batch. The two leading F32 rubric
probabilities were only **0.198621 versus 0.195799**. Their order changed, but
the actual batch/singleton score difference was **0.016439**, not a one-level
change. No choice or abstention outcome changed in those 13 cases.

Experiments were run sequentially:

1. Reproduce the original all-BF16 encoder/head with packed FlashAttention.
   It reproduced the original probability errors and score-argmax flip.
2. Keep the encoder in BF16 but load both projection heads in F32. This did
   not fix the flip or improve the maximum reference probability error
   (0.00613 versus 0.00605). Reverted. The effect persists before final-head
   rounding; this test does not isolate the individual encoder operations.
3. Expand the independent PyTorch F32 oracle to **95 cases**, including
   cardinalities 2–24, reversed option order, absent evidence, more rubrics and
   512-token contexts. Duplicate prompts at different batch positions also
   exercise batch-composition effects. All-BF16 packed attention preserved
   every choice and abstention outcome in singleton and mixed-batch execution.
4. Test BF16 without FlashAttention. No choice/abstention changed, but its
   maximum numeric-answer error was **0.014905**, above the 0.01 gate. This
   path is not enabled. BF16 now requires CUDA plus `flash-attn`.

The revised validation gates retain exact choice and abstention outcomes.
For BF16, they allow <0.01 normalized expected-score/noul error, <0.02 absolute
probability error against the F32 reference, and <0.03 between batch and
singleton probability vectors. F32 retains its strict 0.0002 probability and
numeric-answer tolerance and exact native argmax checks. Native score argmax
differences remain in reports rather than disappearing from the evidence.
These are engineering tolerances for this suite, not a downstream quality
benchmark or a guarantee that every near-tied choice stays unchanged.

| 95-case measurement                                      | BF16 packed |
| -------------------------------------------------------- | ----------: |
| Choice or abstention changes vs F32 / batch vs singleton |       0 / 0 |
| Maximum absolute probability error vs F32                |    0.011773 |
| Maximum absolute batch/singleton probability difference  |    0.022722 |
| Maximum expected-score error vs F32 (0–4 scale)          |    0.023385 |
| Maximum normalized expected-score error vs F32           |    0.005847 |
| Maximum noul error vs F32                                |    0.009585 |

The expanded F32 control retained all 95 native argmaxes and stayed within
0.000001431 of the reference probabilities. Both F32 and BF16 passed the
revised executable validation. Regression tests distinguish a harmless score
argmax flip from an actual choice or abstention change.

BF16 is restored and is the default on CUDA when compiled with `flash-attn`.
The encoder and heads use BF16; logits and calibration use F32. CPU and plain
CUDA builds retain F32 defaults. Explicit F32 remains available. The large
matrix weights use two bytes per element instead of four; total GPU memory also includes
activations, workspaces and runtime allocations.

To reproduce the extended audit, add `extended` to the Python reference
command and `bf16` to the Rust parity command. `audit_precision.py` summarizes
captured JSON in terms of the public answers. `openjev_precision_bench`
alternates F32/BF16 execution order for 40 pairs per scenario after warm-up,
excluding loading and tokenization.

Two 40-pair timing runs on the shared RTX 3080 Ti gave:

| Scenario                | First run F32 → BF16 | Repeat F32 → BF16 | Speedup range |
| ----------------------- | -------------------: | ----------------: | ------------: |
| 41-token singleton      |       6.01 → 5.34 ms |    5.33 → 3.00 ms |    1.12–1.78× |
| Mixed 13-question batch |    400.12 → 54.14 ms | 205.11 → 15.07 ms |   7.39–13.61× |

The repeat followed completion of validation/build work; neither run isolates
the GPU from the desktop. The spread makes a single universal latency claim
inappropriate. The batch improvement includes packed execution eliminating
padding and FlashAttention replacing masked dense attention, not just changing
the weight dtype. Both cases use the same original requests and checkpoint.

Workspace tests, CUDA-feature tests and Clippy pass. CLI smoke verifies the
CUDA default loads as BF16; alternating Laya/OpenJev GPU execution stays stable.

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
