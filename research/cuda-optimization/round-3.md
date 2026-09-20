# Round 3: elementwise CUDA kernels

This round starts from `681e418257ee`, which already includes the fused BF16
GELU/gate operation from round 2. Each experiment gets a fresh standalone
baseline before implementation. Numerical acceptance includes full response and
action-probability equality, alternate inputs and changing shapes, plus focused
kernel tests. Performance decisions use adjacent baseline/candidate calls on a
single loaded model when differences are small, reversing order each pair.

Hardware remains the RTX 3080 Ti, CUDA 12.9, BF16 Laya with FlashAttention. The
default checkpoint has 28 encoder layers and hidden width 1024.
The desktop shares the GPU. Model loading and response serialization are excluded
from latency; tokenization, transfers and inference are included. Standalone
baselines use five warmups and 60 measured calls per workload. Paired runs use
60 pairs for each of five workloads. Finite regression corpora do not prove
arbitrary-input or cross-hardware bit equivalence.

## 9. Pairwise BF16 GeGLU — accepted

The candidate loads two adjacent BF16 values together, retains scalar
`normcdff` evaluation for each, and uses native BF16x2 products with the same
three rounding boundaries. A 256-thread launch replaces the 1024-thread scalar
launch on aligned inputs. Odd storage offsets retain the original scalar
kernel; a scalar tail handles odd lengths. The existing exhaustive finite-BF16
sweep passes, including the additional aligned odd-length case.

First paired measurements are in [`09-paired.json`](09-paired.json). Raw runs
are saved under `artifacts/cuda-optimization/09-*`. The generic paired harness
is shared with the earlier GeGLU comparison; switches exist only in test builds.

| Workload      | First paired change | Repeat paired change |
| ------------- | ------------------: | -------------------: |
| one           |              +0.08% |               -1.67% |
| thirty        |              -1.24% |               -1.01% |
| mixed_lengths |              -1.55% |               -1.08% |
| browser_call3 |              -1.77% |               -1.49% |
| browser_call5 |              -0.87% |               -0.59% |

The larger workloads show a repeatable, modest improvement; single-question
latency is less convincing. Repeat details are in `09-paired-repeat.json`.
All 600 paired comparisons match exactly. A separate 30-iteration run against
the saved pre-change baseline also passes across all six workloads, alternate
content and changing shapes. The finite-BF16 sweep, 44 default tests, formatting
and FlashAttention Clippy with warnings denied pass before committing.

## 10. Pair Q/K rotary application — accepted

Baseline includes experiment 9. Apply rotary position embeddings to Q and K in
one launch and one output allocation, reusing each cosine/sine lookup. Keep
the separate Q/K/V GEMMs and preserve every BF16 multiply/add/subtract rounding
step. F32/F16 and unsupported layouts continue through Candle's existing path.

The isolated check compares both outputs against Candle at head widths 16, 64
and 80, multiple token/head counts, nonzero input/table offsets, and a case
containing every finite BF16 activation value. All bit patterns match. Two
whole-model paired runs and a standalone six-workload comparison also pass
exact response/action equality, including alternate inputs and shape changes.

| Workload      | First paired change | Repeat paired change |
| ------------- | ------------------: | -------------------: |
| one           |              -1.15% |               -1.25% |
| thirty        |              -1.21% |               -1.14% |
| mixed_lengths |              -1.25% |               -1.11% |
| browser_call3 |              -1.40% |               -1.40% |
| browser_call5 |              -1.12% |               -1.07% |

This is a small repeatable gain across all five paired workloads; all 600
paired output checks pass. The standalone timings vary more with GPU conditions
and are not used to claim larger speedups. Detailed numbers are in
`10-paired.json` and `10-paired-repeat.json`; raw runs use the `10-` prefix.
44 default tests, formatting and FlashAttention Clippy pass before committing.

## 11. Fuse residual addition and MLP normalization — accepted

Baseline includes experiments 9 and 10. Combine the attention residual addition
with the following zero-bias LayerNorm in the packed BF16 encoder path. Return
both the rounded residual and normalized values from one allocation and launch.
Preserve the BF16 addition boundary, F32 accumulation, XOR shuffle order,
variance formula, reciprocal square root and affine FMA of Candle's kernel.
Widths below 1024 use 32 threads; width 1024 uses the same 1024-thread, two-stage
reduction as Candle. Other widths/dtypes/layouts retain the existing path.

The first prototype covered only widths below 1024 and therefore never ran on
the default checkpoint. Its near-zero timing changes were a measurement of the
unchanged path and are excluded. The corrected benchmark asserts that the new
kernel executes. Widths 1, 7, 31, 32, 33, 64, 128, 768, 1023 and 1024 pass exact
kernel comparisons, including constant rows, signed zeros, varied scales,
three epsilon values, storage offsets and every finite BF16 input value at
widths 768 and 1024 (including sums that overflow). Both the residual and norm
outputs are compared bit for bit against the separate Candle operations.

| Workload      | First paired change | Repeat paired change |
| ------------- | ------------------: | -------------------: |
| one           |              -1.15% |               -1.39% |
| thirty        |              -1.18% |               -1.22% |
| mixed_lengths |              -1.37% |               -1.24% |
| browser_call3 |              -1.04% |               -1.16% |
| browser_call5 |              -1.34% |               -1.21% |

All 600 paired output comparisons pass. The separate 30-iteration comparison
against the fresh pre-change baseline passes for all six workloads, alternate
content and repeated shape changes. Details are in `11-paired.json` and
`11-paired-repeat.json`; raw standalone runs use the `11-` prefix.

## Combined result and final validation

Toggle all three round-three changes together in the same loaded model, keeping
the round-two fused scalar GeGLU as the reference. These are measured combined
results, not a sum of the incremental percentages. Each percentage is the
median of 60 adjacent candidate/reference latency ratios. The reference is the
runtime behavior at `681e418257ee`.

| Workload      | First paired change | Repeat paired change |
| ------------- | ------------------: | -------------------: |
| one           |              -3.14% |               -3.21% |
| thirty        |              -3.17% |               -3.87% |
| mixed_lengths |              -3.94% |               -3.66% |
| browser_call3 |              -3.84% |               -3.81% |
| browser_call5 |              -3.37% |               -3.51% |

Both runs pass all 300 paired response/action comparisons, bringing the round
to 2,400 exact paired comparisons including the individual experiments.
`round-3-paired.json` and `round-3-paired-repeat.json` retain the medians and
number of faster pairs. The repeat wins 57–58 of 60 pairs for every workload.
The final BF16 standalone snapshots (original and alternate inputs) also match
the saved pre-round `09-baseline.json` exactly across all six workloads.

The saved pre-round binary and final binary additionally pass the six-workload
F32 comparison with 10 measured calls per workload, alternate content and shape
cycling (`round3-f32-baseline.json` / `round3-f32-final.json`). These F32 timings
are regression checks, not claimed improvements. All 44 default tests pass;
FlashAttention all-target Clippy with warnings denied and `nix fmt` pass.
The `vs1-flash-attn` Nix derivation evaluates and includes the new kernel sources;
a full Nix package rebuild was not performed. Kernel compilation and GPU
execution were validated through Cargo in the pinned development shell.

To reproduce the combined paired measurement, run alone on the GPU:

```sh
direnv exec . bash -c 'export LD_LIBRARY_PATH=/run/opengl-driver/lib:$LD_LIBRARY_PATH; cargo test --release -p vs1 --features flash-attn paired_round3_latency -- --ignored --nocapture --test-threads=1'
```

Replace the test filter with `paired_vector_latency`, `paired_rope_latency` or
`paired_norm_latency` for the individual changes. Kernel comparisons use
`all_finite_bf16_activations_match_candle_exactly`, `rotary_matches_candle_bits`
and `residual_norm_matches_candle_bits`. Run these tests individually or with
one test thread; the test-only reference switches are process-global.
