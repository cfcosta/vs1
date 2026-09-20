# Removing CUDA runtime overhead

Follow-up to the first four experiments in [README.md](README.md). Each task
starts with a fresh runtime baseline, then compares full responses and Rust
action probabilities exactly using `cuda_bench`. Original inputs, alternate
content, repeated calls, and changing shapes are included. Hardware and timing
boundaries are unchanged: RTX 3080 Ti, BF16 Laya with FlashAttention, loading
excluded, five warmups and 60 measured iterations per workload.

The desktop shares this GPU. Paired repeats are used when apparent gains are
small. No tiny single-run change is accepted as a demonstrated speedup. Exact
equality on this corpus is a regression check, not a proof for all inputs.

## 5. Pack token IDs before embedding — rejected as inconclusive

Baseline runtime: `8f2b25d043ec`. The original path embeds padded IDs and then
copies valid hidden rows into packed storage. The candidate packs integer IDs on
the host and embeds only actual tokens, keeping every subsequent tensor shape
and operation unchanged. In the mixed fixture that removes 6,570 of 11,970
embedding lookups and the hidden-state packing copy.

All exact-output checks passed. Initial median timings (ms):

| Workload      | Baseline | Candidate |
| ------------- | -------: | --------: |
| one           |    8.396 |     8.402 |
| five          |   17.020 |    16.806 |
| thirty        |   99.119 |    98.059 |
| mixed_lengths |  123.096 |   126.000 |
| browser_call3 |   23.700 |    24.084 |
| browser_call5 |   43.271 |    43.121 |

A reverse-order repeat changed absolute timings considerably across both
binaries. It did not establish a repeatable gain attributable to packing.
The initial mixed case regressed 2.4% despite doing less embedding work.
Embedding and a single packing copy are only a small part of the full forward
pass; desktop load masks differences at this scale. This is not evidence that
packing is intrinsically slower. Under the measured-improvement acceptance rule,
the production change was reverted and archived as `05-packed-ids.patch`.

Raw runs: `artifacts/cuda-optimization/05-{baseline,candidate}.json` and
`05-{baseline,candidate}-repeat.json`. Saved binaries allow repeats without
rebuilding between runs. Run order was baseline, candidate, candidate, baseline.

## 6. Share sequence metadata — rejected as inconclusive

Baseline runtime: `8f2b25d043ec`. Prepare and upload token-kind and marker metadata
before queuing the encoder. Reuse one cumulative-sequence tensor in both encoder
and decision-head attention, and use its prefix for CLS indices instead of
uploading another copy. No cache or arithmetic changes are involved.

Exact outputs passed in both pairs. Initial timings and a reverse-order repeat
(ms) show why the larger apparent small-case gains were not accepted:

| Workload      | Baseline | Candidate | Repeat baseline | Repeat candidate |
| ------------- | -------: | --------: | --------------: | ---------------: |
| one           |    8.025 |     7.078 |           7.074 |            7.013 |
| five          |   17.593 |    14.656 |          14.755 |           14.681 |
| thirty        |   83.575 |    85.829 |          86.321 |           85.919 |
| mixed_lengths |  105.540 |   108.468 |         108.574 |          108.627 |
| browser_call3 |   20.033 |    20.020 |          20.328 |           20.099 |
| browser_call5 |   37.416 |    37.376 |          37.572 |           37.492 |

The repeat is essentially tied, with most deltas under 1%. Removing two small
uploads did not produce a clear end-to-end gain beyond run-to-run variation.
This is an inconclusive performance result, not a numerical failure or proof
that sharing metadata is slower. The production change was reverted and saved
as `06-shared-metadata.patch`. Raw files use the `06-` prefix, with the same
baseline, candidate, candidate-repeat, baseline-repeat order as experiment 5.

## 7. Queue pooled states before readback — rejected as inconclusive

Baseline runtime: `8f2b25d043ec`. Defer the logits host read until after queuing
the pooled CLS gather/cast, in both packed and padded paths. Arithmetic and
CPU confidence features are unchanged. All exact-output checks passed.

| Workload      | Baseline | Candidate |
| ------------- | -------: | --------: |
| one           |    7.231 |     7.056 |
| five          |   14.594 |    14.617 |
| thirty        |   85.968 |    86.130 |
| mixed_lengths |  108.194 |   108.423 |
| browser_call3 |   20.198 |    20.096 |
| browser_call5 |   37.451 |    37.515 |

Four of six median timings were slightly worse, and every multi-question case
was within about 0.5% of baseline. The 0.175 ms single-question difference is
small relative to the variation seen across prior runs. No meaningful overall
gain was established. CLS gathering is tiny; reordering it around a host read
does not remove the necessary CPU confidence computation or its transfers.
The production change was reverted and archived as
`07-pooled-before-readback.patch`. Raw runs are `07-baseline.json` and
`07-candidate.json` in the same artifact directory.

## 8. Fuse BF16 GELU and gate multiplication — accepted

Baseline runtime: `8f2b25d043ec`. A single CUDA kernel replaces the encoder MLP's
separate GELU and gate-multiply kernels. It removes one launch and one intermediate
tensor per encoder layer. Projection matrices, GEMM dimensions, batching, and
attention are unchanged. CPU, Metal, F16 and F32 keep the original path and order.

Numerical equivalence requires **three** BF16 rounding steps: round the normal
CDF, round its product with the activation, then round the product with the gate.
The kernel preserves each step. Native BF16 FMA with a negative-zero addend
matches CUDA's Ampere BF16 multiplication, including signed zero. It uses the
same `normcdff` function as Candle rather than substituting another GELU formula.

The first prototype used NVRTC. Both standalone 60-iteration A/B pairs improved
all six medians and passed exact outputs, but the size of the apparent gains
varied with GPU conditions. The final implementation compiles PTX with nvcc at
build time; it adds no runtime NVRTC requirement. `build.rs` is included in the
Nix source fileset. CPU builds do not invoke nvcc.

To separate the gain from clock/load variation, an opt-in test alternates the
original and fused MLP **within one loaded model**, reversing order every pair.
This uses the final build-time PTX. It warms both paths, brackets each complete
`system_one_batch` call with device synchronization, and checks full responses
and action probabilities outside timing. The reference branch is test-only.

| Workload      | Baseline p50 ms | Fused p50 ms | Median paired latency change | Faster pairs |
| ------------- | --------------: | -----------: | ---------------------------: | -----------: |
| one           |           8.006 |        7.829 |                       -1.62% |        37/60 |
| thirty        |          96.721 |       93.254 |                       -3.25% |        60/60 |
| mixed_lengths |         124.005 |      119.507 |                       -3.60% |        60/60 |
| browser_call3 |          23.502 |       22.855 |                       -2.72% |        43/60 |
| browser_call5 |          43.770 |       42.301 |                       -3.39% |        55/60 |

The change column is the median of adjacent candidate/baseline ratios, not the
ratio of the two independently calculated medians. Full figures are in
[`08-paired.json`](08-paired.json). The larger batches support a consistent
roughly 3–4% latency reduction (about 3–4% more question throughput), while the
single-question improvement remains less convincing. These are warm local-model
measurements, not browser-task wall times or remote Jev performance.

Validation:

- Exact elementwise comparison for all 65,280 finite BF16 activation values:
  five shifted gate pairings also cover every finite BF16 gate value, and eight
  additional scalar gates cover signs, zeros, infinities and NaNs. Separate
  cases cover nonfinite activations, nonzero storage offsets, non-block-aligned
  lengths, and strided/F32 fallbacks. Comparison uses float bit patterns.
- All 300 paired whole-model comparisons match responses and action
  probabilities exactly.
- Final standalone BF16 run: 30 measured iterations per workload, exact match
  against the pre-change baseline across all six workloads, alternate inputs
  and shape cycling (`08-final.json`).
- Final F32 run: ten iterations per workload against a saved pre-change F32
  binary run, with the same exact checks (`08-f32-{baseline,final}.json`).
- 39 normal unit tests and five wire-conformance tests pass. CUDA and
  FlashAttention configurations compile; FlashAttention Clippy passes with
  warnings denied. The Nix FlashAttention derivation evaluates with the new
  build script included; a full Nix package rebuild was not performed.

The exhaustive activation sweep is not a test of every possible activation/gate
pair, checkpoint, input, GPU or compiler version. Both source-level preservation
of rounding and regression checks are needed; no arbitrary-input test guarantee
is claimed. The validated hardware remains the RTX 3080 Ti and CUDA 12.9.

Reproduce the additional checks (run GPU commands sequentially):

```sh
direnv exec . bash -c 'export LD_LIBRARY_PATH=/run/opengl-driver/lib:$LD_LIBRARY_PATH; cargo test --release -p vs1 --features flash-attn --lib geglu_cuda::tests -- --ignored --nocapture'
direnv exec . bash -c 'export LD_LIBRARY_PATH=/run/opengl-driver/lib:$LD_LIBRARY_PATH; cargo test --release -p vs1 --features flash-attn --lib paired_model_latency -- --ignored --nocapture --test-threads=1'
```

Raw standalone runs use the `08-` prefix in `artifacts/cuda-optimization/`:
`baseline`, `candidate`, `candidate-repeat`, `baseline-repeat` (prototype),
then `final` and `f32-*` (build-time PTX). All three rejected production changes
remain reverted. The only retained runtime optimization is the fused BF16 MLP
activation/gate operation.
