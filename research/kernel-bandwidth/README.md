# Memory-bound kernel experiments, 2026-09-23

Starting revision: `0357a90` (rounded CUTLASS GeGLU epilogue). RTX 3080 Ti,
driver 595.45.04, CUDA 12.9 from the repository Nix shell, BF16 default Laya
with FlashAttention. The GPU also drives the desktop; no game or other
compute workload ran during measurements (a small docbert process was
resident). Raw paired reports are retained here; scratch output lives in the
ignored `artifacts/residual-norm-wide/` and sibling directories.

## Motivation

The round-5 CUPTI trace (`../cuda-optimization/14-kernels-128.tsv`) put
GEMMs at ~68% and FlashAttention at ~14% of kernel time. Neither can change
under the exact-output rule without new accumulation orders. The
elementwise/normalization kernels were well below DRAM bandwidth at 4128
rows x 1024 columns (~900 GB/s peak):

| Kernel                  |   Mean | Bandwidth floor |  Gap |
| ----------------------- | -----: | --------------: | ---: |
| `residual_norm_bf16`    | 117 µs |          ~37 µs |  ~3x |
| Candle `layernorm_bf16` | 111 µs |          ~19 µs |  ~6x |
| Candle `badd_bf16`      |  89 µs |          ~28 µs |  ~3x |
| `urelu_bf16` (head)     | 230 µs |          ~74 µs |  ~3x |
| `geglu_bf16_pair`       |  93 µs |          ~72 µs | near |
| `rope_pair_bf16`        |  47 µs |          ~37 µs | near |

The 1024-thread-per-row normalization launches fit one block per SM on
GA102, issue one 2-byte load per thread and synchronize the whole block,
so too few bytes are in flight to saturate memory. That profile predates
`5ce2b72`, so absolute shares have moved; ranking was done from it and
code reading, not from a new profile.

Acceptance is unchanged from earlier rounds: exact public responses and
action probabilities on every timed call, two independent runs of 40
adjacent AB/BA pairs over the nine-workload `batch_bench` corpus, and a
repeatable gain. Otherwise revert and keep only this record.

## 1. Warp-per-row residual LayerNorm — accepted

Width-1024 BF16 residual+LayerNorm (every encoder attention and MLP residual
plus the final norm, 57 calls per batch) now uses one warp per row, four
rows per 128-thread block, 16-byte loads/stores, and keeps the row in
registers instead of re-reading its own output from global memory.

Exactness is by construction. Candle reduces a 1024-wide row with element
`t = 32w + l` in lane `l` of warp `w`: an XOR butterfly over lanes, then the
32 warp sums XOR-folded again. The new kernel gives lane `w` Candle's warp
`w` as 32 contiguous values, performs the first butterfly in registers with
the same operand pairs (float addition is commutative), then runs the same
shuffle sequence. Widths other than 1024 and views not aligned to eight BF16
values keep the original kernel.

The first attempt failed exactness on cancellation-heavy and overflowing
rows. Candle's PTX leaves `mean_var.y / ncols - mean * mean` as unrounded
`mul.f32`/`sub.f32`, which ptxas turns into `FFMA(-mean, mean, y / n)` in
Candle's SASS (verified with `cuobjdump`); in the new kernel ptxas instead
fused the 2^-10 scale into the subtraction. The kernel now spells out that
arithmetic with explicit `__fmul_rn`/`__fmaf_rn`/`__fadd_rn`/`__fsub_rn` so
the backend cannot choose a different fusion. Dividing by 1024 and
multiplying by 2^-10 round the same real value and are identical. The
existing kernel continues to rely on matching Candle's compiled shape.

The second attempt was exact but **~6% slower**: a nested fold loop with a
variable bound left both 32-float scratch arrays in a 256-byte local-memory
stack frame. Constant-bound folds remove the stack (64 registers, no spills).

Checks: `residual_norm_matches_candle_bits` (all widths, signed zeros,
offsets, every finite BF16 value at 768/1024 including overflowing sums) and
the new `wide_residual_norm_matches_candle_bits` (4133 rows including a
partial block, five magnitude bands, offset-50 cancellation rows, two eps
values, offset views, and an unaligned fallback view) compare every bit
against Candle's add plus LayerNorm. The kernel dispatch count is asserted.

| Workload      | First paired change | Faster | Repeat | Faster |
| ------------- | ------------------: | -----: | -----: | -----: |
| 1             |              -2.18% |  35/40 | -2.04% |  40/40 |
| 8             |              -3.67% |  32/40 | -3.52% |  39/40 |
| 32            |              -3.34% |  32/40 | -3.87% |  40/40 |
| 64            |              -4.43% |  40/40 | -3.55% |  40/40 |
| 128           |              -3.27% |  40/40 | -3.73% |  40/40 |
| mixed128      |              -4.04% |  40/40 | -3.92% |  40/40 |
| shared128     |              -4.14% |  39/40 | -3.61% |  40/40 |
| browser_call3 |              -3.56% |  39/40 | -3.59% |  40/40 |
| browser_call5 |              -2.96% |  37/40 | -3.27% |  37/40 |

All 720 timed calls match the reference path exactly. Reports:
`01-paired.json`, `01-paired-repeat.json`. Release FlashAttention library
tests (52 passed), all-target Clippy with warnings denied and `nix fmt` pass.

```sh
nix develop -c bash -c '
  export LD_LIBRARY_PATH=/run/opengl-driver/lib:$LD_LIBRARY_PATH
  cargo test --release -p vs1 --features flash-attn --lib \
    residual_norm_matches_candle_bits -- --ignored --test-threads=1
  cargo test --release -p vs1 --features flash-attn --lib \
    paired_wide_norm -- --ignored --nocapture --test-threads=1
'
```

## 2. Warp-per-row norms for the head and embeddings — rejected

The experiment-1 kernel was templated for an optional residual input and
an optional real bias (weight and bias stacked as one `2 x 1024` tensor).
It replaced the decision head's `x + attn -> norm2` and `x + ffn -> norm1`
pairs (the head layer output was deferred, like the encoder's MLP
residual, so the next layer's `norm1` absorbs its add; the type-embedding
add fuses into layer 0's `norm1`) and the encoder's embedding LayerNorm.
Candle computes `FMA(lhs, alpha, beta)` for biased norms, reproduced with
`__fmaf_rn`.

`biased_and_plain_wide_norms_match_candle_bits` passed: 1031 random rows
across five magnitude bands and cancellation rows, plus every finite BF16
value as both operands, for biased residual and plain +0-bias norms at two
epsilons. A dispatch count confirmed five extra fused calls per batch
(61 vs 56 at 32 questions).

| Workload      | First paired change | Faster | Repeat | Faster |
| ------------- | ------------------: | -----: | -----: | -----: |
| 1             |              -0.58% |  35/40 | -0.28% |  27/40 |
| 8             |              -0.44% |  29/40 | -0.59% |  26/40 |
| 32            |              +0.56% |  18/40 | +0.98% |  15/40 |
| 64            |              +0.96% |  15/40 | +0.45% |  19/40 |
| 128           |              -0.45% |  22/40 | +0.34% |  18/40 |
| mixed128      |              -0.11% |  20/40 | +0.23% |  18/40 |
| shared128     |              -0.50% |  23/40 | -0.43% |  24/40 |
| browser_call3 |              -1.07% |  32/40 | -0.44% |  27/40 |
| browser_call5 |              -0.55% |  23/40 | -0.43% |  25/40 |

All outputs were exact, but the changes straddle zero and flip sign between
runs. Five norm/add pairs per batch are too little work relative to 28
encoder layers to show above noise. Reverted; the prototype and tests are
archived in `02-head-norms.patch` (applies on top of experiment 1), with
reports `02-paired.json` and `02-paired-repeat.json`.

## 3. Fused head bias add and ReLU — accepted

Each decision-head `Linear` (Q, K, V, `out_proj`, `linear1`,
`linear2`; two layers) ran Candle's GEMM followed by a broadcast
`badd_bf16`, and `linear1` was followed by a separate `urelu_bf16` over
the 4096-wide activation. `bias_act.cu` performs the bias add, and the
ReLU where needed, in one pass with 16-byte loads and BF16x2 arithmetic:
`fma.rn.bf16x2(x, 1, bias)` is the Ampere lowering of Candle's BF16 add,
and `max.NaN.bf16x2(x, 0)` is the lowering of its `__hmax_nan` ReLU. The
GEMM is the same `x.matmul(&w.t())` call `candle_nn::Linear` makes, so
matrix shapes and rounding are unchanged. Unaligned or non-BF16 inputs
fall back to Candle.

`bias_relu_matches_candle_bits` checks every BF16 bit pattern (NaNs,
infinities, signed zeros, subnormals) against 64 reversed full-sweep
biases and nine special bias values, with and without ReLU, plus a
4096-wide row-offset view and the unaligned fallback.

| Workload      | First paired change | Faster | Repeat | Faster |
| ------------- | ------------------: | -----: | -----: | -----: |
| 1             |              -1.41% |  39/40 | -1.38% |  34/40 |
| 8             |              -2.02% |  38/40 | -1.90% |  28/40 |
| 32            |              -3.48% |  40/40 | -2.94% |  33/40 |
| 64            |              -2.95% |  40/40 | -1.79% |  29/40 |
| 128           |              -3.13% |  38/40 | -2.97% |  40/40 |
| mixed128      |              -3.12% |  40/40 | -3.01% |  38/40 |
| shared128     |              -2.81% |  36/40 | -2.55% |  36/40 |
| browser_call3 |              -2.78% |  35/40 | -2.50% |  29/40 |
| browser_call5 |              -2.26% |  28/40 | -3.39% |  31/40 |

All 720 timed calls match exactly. The gain is larger than the old trace's
kernel averages suggested; this run did not profile how much came from the
broadcast adds versus the ReLU pass. Reports: `03-paired.json`,
`03-paired-repeat.json`. Release FlashAttention library tests (52 passed),
all-target Clippy with warnings denied, CPU-only and plain-CUDA `cargo
check`, and `nix fmt` pass.

```sh
cargo test --release -p vs1 --features flash-attn --lib \
  bias_relu_matches_candle_bits -- --ignored --test-threads=1
cargo test --release -p vs1 --features flash-attn --lib \
  paired_bias_act -- --ignored --nocapture --test-threads=1
```
