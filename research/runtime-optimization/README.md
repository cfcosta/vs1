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
