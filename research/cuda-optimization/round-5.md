# Round 5: cuBLASLt encoder matrix multiplication

Starting runtime: `25f1368fdeb6` (production code from `996c60a68551`).
Hardware: RTX 3080 Ti, CUDA 12.9, driver 595.45.04; Ryzen 9 7950X3D.
The GPU also drives the desktop. Laya uses BF16, FlashAttention, and the
existing stable length ordering and 32-question batches.

The target is 30% more throughput or 30% less latency while preserving exact
outputs. These targets differ: 30% more throughput requires 23.08% less
execution time at a fixed batch size. No approximate math, quantization,
projection fusion, or batch reshaping is used in this experiment.

## Fresh baseline

`artifacts/cuda-optimization/14-baseline.json` contains 20 measured calls per
workload, after five warmups, with all response fields and explicit Rust action
probabilities. Alternate content and five cycles of changing shapes are also
checked. Loading and output serialization are excluded from timing.

| Workload      | Median milliseconds |
| ------------- | ------------------: |
| one           |               7.705 |
| five          |              16.819 |
| thirty        |              90.637 |
| mixed_lengths |             118.450 |
| browser_call3 |              22.380 |
| browser_call5 |              40.964 |
| eight         |              27.177 |
| 32            |              99.007 |
| 64            |             199.399 |
| 128           |             378.989 |
| mixed128      |             512.059 |
| shared128     |             386.421 |

The baseline has noticeable desktop variation. Acceptance therefore requires
adjacent reference/candidate calls with alternating AB/BA order, rather than
comparing independent timing runs alone.

## Kernel tracing

`kernel_trace.cpp` uses the installed CUDA 12.9 CUPTI activity API to collect
actual kernel execution durations. `kernel_profile` loads and warms the model
before tracing ten calls, and synchronizes before stopping capture. The
collector rejects empty traces and dropped records; both captures report zero
dropped records. No performance counter or SM occupancy measurement is implied.

| Workload      | Kernel time per call | GEMM share | FlashAttention share | Kernels per call |
| ------------- | -------------------: | ---------: | -------------------: | ---------------: |
| 128           |           369.778 ms |     68.47% |               14.25% |             1824 |
| browser_call5 |            37.368 ms |     74.64% |                9.98% |              425 |

These are sums of kernel durations across the whole model, including its
decision head. They exclude host work and gaps between kernels, and must not
be interpreted as uninstrumented end-to-end latency. Full names and counts
are in `14-kernels-128.tsv` and `14-kernels-browser.tsv`.

Nsight Systems **2025.1.3.140** was then built from the repository's pinned
Nixpkgs `cudaPackages.nsight_systems`. The daemon's download stalled; the exact
archive was downloaded over IPv4 and imported with its pinned SHA-256 before
completing the normal Nix build. The resulting `nsys` CLI captured the same
warmed workloads with CUDA/cuBLAS tracing and a `cudaProfilerApi` capture range.
CPU sampling and context-switch tracing were disabled. An initial 128-question
capture had device event tracing enabled; it was repeated with that setting
disabled, and the committed CSV reports use the repeat. Browser tracing also
disables device event tracing.

| Workload      | Nsight kernel time/call | GEMM share | FlashAttention share | GPU time in use |
| ------------- | ----------------------: | ---------: | -------------------: | --------------: |
| 128           |              355.492 ms |     68.33% |               14.11% |           97.6% |
| browser_call5 |               37.138 ms |     74.67% |               10.60% |           92.8% |

The last column is Nsight's `gpu_time_util` rule over one chunk spanning the
first through last GPU operation. It counts GPU operations and profiler
overheads, not SM occupancy. It supports the conclusion that scheduling gaps
are too small to account for the requested 30% gain in these workloads; it
does not prove that concurrent streams cannot improve resource utilization.
The `gpu_gaps` reports retain gaps of at least 1 ms.

The CUDA API summary again shows that host readback waits for outstanding
compute: the 128-question trace spends roughly 349 ms/call inside the DtoH
API, while the actual GPU copy takes only microseconds. Treating that API wait
as transfer overhead would misidentify the bottleneck.

Raw `.nsys-rep` and SQLite captures remain under `artifacts/cuda-optimization/`;
the `14-nsys-*` CSV and JSON reports are retained here.

## Algorithm search

The test-only hook intercepts the encoder's seven separate linear projections
per layer. cuBLASLt receives the same matrix dimensions, transpositions,
BF16 inputs/outputs and FP32 compute type as Candle. A persistent 64 MiB
workspace and descriptors are reused. Up to 64 heuristic candidates are
requested per shape; the actual returned count is recorded, not assumed to
be 64. Algorithms may have different reduction orders despite using the same
precision, so equality is checked explicitly.

Every returned algorithm is timed on a real encoder input, including those
whose outputs differ. Exact candidates are then checked against every
subsequent encoder projection with that shape, including changed text and
all layers. Comparisons use the bits of every output element after exact
BF16-to-F32 conversion, including signed zero. Discovery always propagates
the baseline activations. Candidates that differ at any checked input are
removed from consideration. A GPU counter compares raw BF16 storage bits; it
is checked against host comparisons for every finite BF16 encoding, signed
zero, offsets, launch tails, and a real projection of every measured shape.
This avoids copying every large intermediate matrix back to the CPU during
the full sweep. This finite corpus is not a proof for arbitrary
inputs or other GPUs/toolkit versions.

The first host-comparison sweep completed in 484 seconds but failed to write
its report because Cargo uses the crate directory as its test working
directory. No paired results were produced by that run. The report paths were
fixed to derive from `CARGO_MANIFEST_DIR`, and the entire sweep was repeated
with the independently checked GPU counter.

The final paired experiment only selects surviving candidates with at least
2% lower isolated timing; other shapes use Candle. It compares the complete
response and action probabilities on every call. Tuning and validation work
are excluded from the warmed paired measurement and must be accounted for
before considering any production autotuning design.

Reference: [cuBLASLt heuristic selection and caching](https://docs.nvidia.com/cuda/archive/12.9.1/cublas/index.html#using-the-cublaslt-api).

### Broad search result

The completed search covers **51 matrix shapes, 246 returned algorithms and
7,448 baseline encoder projections**. Of the algorithm/shape combinations,
125 fail the first exact comparison. The 121 remaining combinations survive
subsequent checked projections; two shapes have no exact candidate at all.
Twenty-six shapes select a candidate under the exploratory isolated timing
threshold, and the paired run actually executes 37,324 candidate GEMMs.

All 360 end-to-end pairs match complete baseline responses and action
probabilities exactly:

| Workload      | Paired latency change | Faster pairs |
| ------------- | --------------------: | -----------: |
| 1             |               -10.28% |        39/40 |
| 8             |                +0.16% |        17/40 |
| 32            |                -0.01% |        21/40 |
| 64            |                +0.06% |        19/40 |
| 128           |                +0.11% |        18/40 |
| mixed128      |                -1.22% |        35/40 |
| shared128     |                -0.10% |        22/40 |
| browser_call3 |                +0.89% |        17/40 |
| browser_call5 |                -0.82% |        28/40 |

This does **not** achieve the 30% target. The broad autotuner is not a production
optimization: isolated timings are noisy enough to affect selection, the
browser_call3 control selects no new GEMMs, uniform large batches are tied,
and discovering exact algorithms on one input alone is not a correctness
guarantee. Raw search and paired data are in `14-gemm-search.json` and
`14-paired.json`. The corrected GPU-counter search and paired run together
took 155 seconds.

The consistent single-question gain is a narrower lead: the exact candidate
changes a 128x64 output tile to 64x64 while retaining algorithm 21, split-K=4,
reduction scheme=4 and the same staging/options. Experiment 15 starts with a
fresh 20-iteration baseline (`artifacts/cuda-optimization/15-baseline.json`)
before testing this tile-only rule on additional shapes and adversarial
products. That baseline's single-question median is 8.659 ms, browser_call5
is 39.867 ms, and 128 questions take 367.544 ms. Paired comparisons below
control for the substantial variation between standalone runs.

## Narrow output-tile rule

The second experiment restricts the rule to the measured RTX 3080 Ti and
cuBLASLt 12.9.1 (`cublasLtGetVersion() == 120901`). Only packed BF16 products
with 129–136 rows, 1,024 output columns and 1,024 or 2,624 input columns are
eligible. The first returned heuristic must match every checked field of
`[algorithm=21, tile=18, splitK=4, reduction=4, stages=12, swizzle=0, custom=0]`;
the replacement differs only in its output tile (`15`). Other configurations
use the existing Candle call. Input alignment, contiguity, dtype and absence
of bias are checked as well. No calibration or timing occurs during inference.
The final guard also requires equal inner and cluster shapes; both are zero
on all eligible shapes. Those two configuration fields use 16-bit storage,
unlike the preceding seven. An initial diagnostic query with a 32-bit buffer
returned `CUBLAS_STATUS_INVALID_VALUE`; correcting the documented buffer type
restored the check. The same 48 products pass with the stricter guard, using
the same algorithms as the timed implementation. See NVIDIA's
[configuration attribute definitions](https://docs.nvidia.com/cuda/archive/12.9.1/cublas/index.html#cublasltmatmulalgoconfigattributes-t).

The implementation caches at most 16 descriptors and shares a 64 MiB scratch
buffer per encoder. A mutex protects host access, and a CUDA event orders
scratch use between host threads and before destruction. This extra device
memory and the first-use descriptor/heuristic work are costs; warmed timing
does not include initialization. The specialization deliberately does not
extend a single GPU's measured result to other hardware or toolkit releases.

The product test sweeps row counts 1–512 for both input widths to discover
eligible algorithms, then checks all 16 eligible shapes using uniform random,
signed powers of two, and cancellation-heavy products. All **6,512,640 BF16
output elements across 48 products** match Candle exactly, including the
actual runtime dispatch. The cuBLASLt version and nonzero runtime dispatch
count are asserted, so a silent fallback cannot pass this test. Results are
in `15-retile-adversarial.json`. This is finite input coverage, not a formal
proof of equivalence for every possible matrix.

### Full-model result

Before timing, the model is checked at each length from 128 through 137
tokens with `noul`, `choice` and `score` questions (30 inputs), including
both boundaries outside the specialization. Six concurrent calls through
one model also reproduce the reference, exercising scratch reuse across
host threads. Each timed call checks its entire response and action
probabilities against the original Candle result.

Two runs of 40 AB/BA pairs over 15 workloads pass all **1,200 paired
comparisons**. The final repeat places the common shape rejection before
the more expensive layout/device checks; arithmetic and selection are
unchanged. The measured latency changes are:

| Workload           | First run |  Repeat | Faster pairs, repeat |
| ------------------ | --------: | ------: | -------------------: |
| 1                  |   -10.25% | -10.20% |                40/40 |
| 8                  |    +0.25% |  +0.48% |                17/40 |
| 32                 |    +0.11% |  +0.18% |                18/40 |
| 64                 |    +0.23% |  -0.14% |                23/40 |
| 128                |    +0.21% |  -0.26% |                24/40 |
| mixed128           |    -0.01% |  -0.28% |                24/40 |
| shared128          |    +0.10% |  -0.30% |                26/40 |
| browser_call3      |    -0.13% |  +0.43% |                18/40 |
| browser_call5      |    -0.08% |  -0.25% |                22/40 |
| 128 tokens, noul   |    +0.03% |  -0.16% |                22/40 |
| 130 tokens, noul   |   -10.09% | -10.77% |                40/40 |
| 131 tokens, choice |   -10.17% | -10.95% |                36/40 |
| 132 tokens, score  |   -10.21% | -11.70% |                39/40 |
| 136 tokens, noul   |   -10.13% | -10.16% |                39/40 |
| 137 tokens, noul   |    -0.00% |  +0.08% |                19/40 |

The ordinary single-question median is **6.626 → 5.948 ms** in the repeat.
The consistent gain warrants retaining this guarded specialization. No
general browser or large-batch speedup is claimed: their changes span
-0.30% to +0.48%, with several controls reversing direction between runs.
The 30% target is unmet. The broad autotuner remains a test-only experiment;
none of its timing-based selections run in production. Full measurements are
in `15-paired.json` and `15-paired-repeat.json`.

### Regression checks

The standalone final BF16 run (`15-candidate.json`) matches all response
fields and action probabilities in the fresh baseline across 12 workloads,
alternate content and five cycles of shape changes. The F32 run also matches
a fresh execution of the saved pre-change binary on its six workloads,
including alternate content and shape changes (`15-f32-baseline.json` and
`15-f32-candidate.json`). These files remain in `artifacts/cuda-optimization/`.
Their independent timing variation is not used to claim an improvement.
After adding the inner/cluster-shape guard, the full-model boundary and
concurrent-call checks passed again with `VS1_RETILE_CHECK_ONLY=1`, which skips
the timing loop without skipping these correctness checks.

Default and FlashAttention builds each pass 44 ordinary tests and two
doctests. The FlashAttention build passes Clippy for all targets with
warnings denied. Nix formatting and package derivation evaluation pass;
the Nix package itself was not rebuilt. The archived patch is excluded from
formatting and whitespace checks because its context lines are significant.

## Reproduction

All commands use the repository's pinned development environment. Benchmark
and profiler runs must execute serially on the GPU. Build before timing.

```sh
nix build --impure --no-link --print-out-paths --expr \
  'let f = builtins.getFlake (toString ./.); p = import f.inputs.nixpkgs { system = "x86_64-linux"; config.allowUnfree = true; }; in p.cudaPackages.nsight_systems'

direnv exec . cargo build --release -p vs1 --features flash-attn --example kernel_profile --example cuda_bench

# Replace NSYS below with the returned package path plus /bin/nsys.
direnv exec . bash -c 'export LD_LIBRARY_PATH=/run/opengl-driver/lib:$LD_LIBRARY_PATH; VS1_PROFILE_NSYS=1 NSYS profile --trace=cuda,cublas --sample=none --cpuctxsw=none --cuda-event-trace=false --capture-range=cudaProfilerApi --capture-range-end=stop -o artifacts/cuda-optimization/profile-128 target/release/examples/kernel_profile 128 ignored'

direnv exec . bash -c 'export LD_LIBRARY_PATH=/run/opengl-driver/lib:$LD_LIBRARY_PATH; VS1_BENCH_LARGE=1 target/release/examples/cuda_bench artifacts/cuda-optimization/retile.json 20 artifacts/cuda-optimization/15-baseline.json'

direnv exec . bash -c 'export LD_LIBRARY_PATH=/run/opengl-driver/lib:$LD_LIBRARY_PATH; cargo test --release -p vs1 --features flash-attn --lib retile_preserves_adversarial_products -- --ignored --nocapture --test-threads=1'

direnv exec . bash -c 'export LD_LIBRARY_PATH=/run/opengl-driver/lib:$LD_LIBRARY_PATH; VS1_PAIRED_REPORT=/tmp/retile-paired.json cargo test --release -p vs1 --features flash-attn --lib retile_model_outputs_and_latency -- --ignored --nocapture --test-threads=1'
```

For the exact broad experiment, `14-cublaslt-experiment.patch` applies to
`25f1368fdeb6`; `git apply --check` passed in an isolated Jujutsu workspace at
that revision. Its ignored `tune_encoder_gemms` test reproduces the search and
paired comparison. The current test module also retains that diagnostic, but
the archive preserves the precise version that produced experiment 14.
