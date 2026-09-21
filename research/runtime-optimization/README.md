# Unchanged-model runtime experiments

Sequential experiments on 2026-09-21, RTX 3080 Ti, CUDA 12.9, BF16 Laya,
FlashAttention. Weights, model architecture, batching, precision and numerical
acceptance are unchanged. Require exact serialized responses and explicit action
probabilities, alternate content and changing shapes. A performance gain needs
repeated paired AB/BA timings, not a favorable standalone median. The GPU also
drives the desktop; small differences remain uncertain.

Baseline is the runtime at parent of this work (email research commits after
`7b27ad254db6` do not change inference). `baseline.json` records the six-workload
30-iteration baseline, input lengths, outputs, alternates and shape cycles.
Local logs and larger artifacts are in ignored `artifacts/runtime-optimization/`.

## 1. Deferred bounded CUDA Graph cache: rejected

Extend the earlier single-entry graph prototype to capture only on a shape's
third occurrence, retain at most four graphs, remember at most 32 candidate keys,
and capture only up to 1536 tokens. Larger or excess shapes keep eager execution.
The key contains lengths, primitive kinds and marker positions; input tokens
are updated before replay. Explicit position buffers are retained per graph.

Repeated identical inputs ran, but the alternate-content/revisit check failed.
Moving mutable input allocation outside Candle's immutable host-upload cache
was insufficient: returning to the original input produced NaN probabilities.
`01-graphs-failure.txt` preserves the observed failure. The underlying lifetime
or replay defect was not localized; this is not a verdict against CUDA Graphs
in general. No timing result is accepted. All source changes were reverted;
`01-deferred-graphs.patch` archives the failed prototype for diagnosis.

## 2. Pack token IDs before embedding: rejected

Avoid building a padded floating-point embedding tensor only to discard padding
and concatenate the live rows. Pack IDs first, embed only live tokens, and retain
the exact downstream matrix shapes. No token truncation or batch regrouping.
The standalone six-workload run passed all output, alternate and churn checks.
Its timing variation is too large to interpret without paired repeats.

The 60-pair AB/BA check passed all 600 output comparisons, but paired median
changes were +0.03%, -0.44%, -0.20%, +2.21%, -0.39% for one, thirty,
mixed lengths and the two browser cases. Candidate won only 27–35 of 60 pairs.
This does not establish a useful improvement; reverted. See `02-pack-paired.json`
and the archived `02-pack-ids.patch`. Standalone timings were especially variable
and are not used for the acceptance decision.

## 3. Residual addition plus following normalization: accepted

Delay each encoder block's final addition until the next block's attention
normalization; use the already validated residual/LayerNorm kernel to produce
both the rounded residual and normalized tensor. Fuse the last addition with
the final encoder norm as well. BF16 width 1024 only, with original fallback.
No change to the addition order, BF16 rounding boundaries or GEMM shapes.

Two independent 60-pair AB/BA runs passed all 1,200 timed response/action checks.
Standalone original/alternate/shape-cycle checks also passed. Paired median changes:

| Workload       | First run | Repeat |
| -------------- | --------: | -----: |
| One            |    -2.29% | -1.59% |
| Thirty         |    -0.12% | -0.06% |
| Mixed lengths  |    -0.39% | -0.55% |
| Browser call 3 |    -2.35% | -0.57% |
| Browser call 5 |    -2.03% | -1.73% |

This is a modest gain, strongest in browser call 5, not a massive improvement.
Thirty-question performance is effectively unchanged. See `03-deferred-paired.json`
and `03-deferred-repeat.json`. The standalone timing shifts were much larger
than these paired effects, reinforcing the need for adjacent comparisons.

Fusion validation: CPU `cargo test -p vs1 --all-targets` passed (39 unit and five
conformance tests), release FlashAttention Clippy passed with warnings denied,
the CUDA residual/normalization bit-equivalence test passed, and the cleaned-up
implementation passed another six-workload original/alternate/shape-cycle check
against the original baseline. `nix fmt` passed. Debug Clippy was interrupted
during redundant CUDA dependency compilation; release Clippy completed instead.
The finite corpus supports this hardware-specific result, not universal bit
identity across arbitrary inputs or devices.

## 4. Retain graph dependencies and capture completion events: rejected as default

The earlier graph failure was addressed by retaining the gathered cos/sin tables,
which normally live in single-entry RoPE caches and are replaced when lengths
change. The retry holds those tensors for every graph, permits eight cache entries,
and copies pooled output before releasing the graph mutex. It passed all six
workloads, alternate content and five shape cycles, but initially panicked during
GEMM workspace teardown. Ordinary captured completion events were unsuitable for
host synchronization. Recording an external completion event during capture
fixed teardown; the flag must not be used outside capture.

The corrected prototype passed two 60-pair checks (1,200 timed output comparisons)
and the standalone expanded check. Nevertheless, its useful region is narrow:

| Workload                | First paired change | Repeat |
| ----------------------- | ------------------: | -----: |
| One                     |              -6.02% | -4.35% |
| Thirty (eager fallback) |              +0.15% | -0.18% |
| Mixed (eager fallback)  |              -0.09% | -0.04% |
| Browser call 3          |              +0.06% | -1.81% |
| Browser call 5          |              -0.12% | -1.52% |

Warm single-shape reuse helps, but browser gains were inconsistent and capture
cost is real. The third single-question candidate call took 21.84 ms versus
6.75/7.31 ms for neighboring eager calls; browser call 3 capture took 45.42 ms
versus 18.46/20.76 ms nearby. Capture is excluded from warm paired numbers.
The experimental eager reference also includes the added capture-status query
in the specialized GEMM path, so the warm one-question figures are not an exact
comparison with the untouched production implementation.

No general default improvement was established. Reverted all graph changes while
retaining the accepted fusion. Fixed-shape, long-lived use remains a possible
opt-in direction; no such API is shipped here. Reports include startup samples
and the corrected prototype patch. Graph handles were externally serialized,
consistent with [NVIDIA's graph thread-safety requirements](https://docs.nvidia.com/cuda/cuda-driver-api/graphs-thread-safety.html).

## 5. Larger-workspace cuBLASLt search: rejected

Increase the search workspace from 64 MiB to 256 MiB and evaluate up to 64
heuristics per projection shape. Preserve the already shipped 129–136-token
specialization. Search actual weights and activations for 1, 8 and 32 questions
and both browser fixtures, including alternate content. Validate surviving
algorithms on subsequent layer inputs before selecting them.

The search considered 101 algorithm/shape candidates over 21 shapes and selected
nine shape-specific candidates based on isolated timing and exact BF16 results.
The 40-pair whole-model check remained exact (400 timed comparisons), but paired
changes for 8, 32 and the browser cases were only -0.12%, -0.56%, -0.25%, -0.79%,
with 20–24 of 40 pairs faster. The one-question control (unchanged dispatch)
varied +4.59%, illustrating the timing noise. These results do not justify the
extra workspace or dispatch changes. Reverted; full search and paired summaries
are retained. This is a bounded heuristic search, not exhaustive CUDA autotuning.

## 6. Fixed-length FlashAttention dispatch: rejected

When all packed sequences have the same length, view Q/K/V as an ordinary
four-dimensional batch and dispatch fixed-length windowed FlashAttention instead
of varlen FlashAttention. Preserve token membership, local/global windows,
attention scale, precision and all projection shapes; mixed lengths retain varlen.

The 20-iteration standalone regression corpus passed, as did 600 timed exact
comparisons in the 60-pair benchmark. Paired changes were -0.23%, -0.23%, +0.09%,
-0.57%, +0.16% for one, thirty, mixed and browser calls 3/5. Candidate won only
27–32 of 60 pairs. No useful improvement established; reverted.

## 7. TensorRT exactness gate: rejected for this engine

Reuse the unchanged-weight BF16 TensorRT 10.13.3.9 engine built in the earlier
[compression study](../model-compression/README.md#9-tensorrt-feasibility).
Evaluate all three validation examples compatible with its fixed token count,
primitive kind and marker layout. All three retained their decisions and action
probabilities, but none matched the Rust logits/probabilities exactly. Maximum
probability errors were 0.0017993, 0.0061722 and 0.0009894.

That fails this round's unchanged-output requirement, so no latency result or
application backend is accepted. The path includes the earlier PyTorch/ONNX port;
these differences cannot be attributed exclusively to TensorRT. This rejects the
existing engine, not every precision, export or TensorRT configuration. The
fixed-shape test does not cover variable shapes or all primitives. See
`07-tensorrt-exactness.json` and `trt_check.py`. The original ONNX export was
removed to recover disk space; its hash and export recipe remain in the earlier
study, and the serialized engine remains a local artifact.

## Reproduction and retained outcome

Only experiment 3 changes production code, in Jujutsu commit
`5ce2b72a56c9e84ed4d61aff573fc0867a02d4cf`. Model weights, model architecture,
precision and batching are unchanged. No massive speedup was demonstrated.

Apply archived prototypes in disposable checkouts, one at a time. Patches 01/02
use the parent of that commit; patches 04/05/06 use that commit. Do not combine
failed prototypes. The original baseline source hashes are recorded separately.

The committed fusion can be rechecked with:

```sh
nix develop -c bash -c '
  export LD_LIBRARY_PATH=/run/opengl-driver/lib:$LD_LIBRARY_PATH
  cargo test --release -p vs1 --features flash-attn --lib paired_deferred \
    -- --ignored --nocapture --test-threads=1
'
```

Use the same CUDA shell for archived experiments:

| Patch | Environment                   | Ignored test / example                         |
| ----- | ----------------------------- | ---------------------------------------------- |
| 01    | `VS1_EXPERIMENT_GRAPHS=1`     | `cuda_bench` example (expected replay failure) |
| 02    | `VS1_EXPERIMENT_PACK_IDS=1`   | `paired_pack_ids`                              |
| 04    | `VS1_EXPERIMENT_GRAPHS=1`     | `paired_graphs`                                |
| 05    | none                          | `tune_encoder_gemms`                           |
| 06    | `VS1_EXPERIMENT_FIXED_ATTN=1` | `paired_fixed_attn`                            |

`cuda_bench` takes an output JSON path, iteration count, and optional baseline
JSON path. It verifies response/action outputs, alternates and shape cycles.
Paired tests print `PAIRED_REPORT=` JSON; the GEMM test writes its reports under
`artifacts/runtime-optimization/`. Run GPU experiments without competing compute
jobs. Desktop activity can still affect timings.

For TensorRT, use the Python/Torch/TensorRT environment described in the earlier
study and run `trt_check.py --engine /path/to/engine.plan --data
/path/to/distillation-data.json --out /path/to/result.json`. No Python or TensorRT
dependency was added to the application.

Final verification: all five archived patches apply to their documented bases.
Production inference files exactly match the accepted fusion commit after the
remaining prototypes were reverted. Release FlashAttention library tests passed
(39 passed, 16 opt-in tests ignored), and release all-targets FlashAttention Clippy
passed with warnings denied. The exercised opt-in experiments are documented
above; ignored tests were not reported as executed. JSON parsing, Python syntax,
Ruff and canonical formatting checks passed before the research commit.
