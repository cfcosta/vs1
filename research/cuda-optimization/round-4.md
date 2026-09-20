# Round 4: preparation and batch scheduling

Starting runtime: `0b6b4e9c7cb3` (all three round-three kernel improvements).
Hardware: RTX 3080 Ti, CUDA 12.9, driver 595.45.04; Ryzen 9 7950X3D. The GPU
also drives the desktop. Laya uses BF16 and FlashAttention, with the existing
32-question batch size and stable descending-length ordering throughout.

The expanded standalone corpus adds 8, 32, 64 and 128 separate requests,
128 mixed-length requests and 128 questions sharing one state to the earlier
six workloads. Each optimization starts with a fresh standalone baseline;
acceptance uses adjacent reference/candidate calls on one loaded model, with
AB/BA order reversed each pair. The larger paired corpus has nine workloads,
six alternating warmups and 40 pairs per workload. Timings include tokenization,
transfers, inference and response construction; loading, snapshot serialization
and exact equality checks are excluded. All response fields and Rust action
probabilities are checked. Finite regression inputs do not prove universal
or cross-hardware equivalence.

## Baseline stage profile

Nsight was unavailable in the development shell. An opt-in Rust test instead
records host wall time and CUDA events around the original packed forward's
stages, over 10 repetitions after five warmups. Events are resolved only after
the profiled operations have completed. The mirrored diagnostic path checks
every logit and action probability against the normal forward. This is stage
instrumentation, not a sampled kernel/occupancy trace. Event spans include
dispatch gaps, and host/GPU durations overlap; they must not be summed.

Selected mean durations in milliseconds, from `round-4-profile.json`:

| Workload      | CPU preparation | Encoder CUDA span | Decision-head CUDA span | CPU confidence |
| ------------- | --------------: | ----------------: | ----------------------: | -------------: |
| 1             |           0.132 |             7.597 |                   0.701 |       0.000389 |
| 32            |           3.213 |            82.097 |                   7.583 |       0.002983 |
| 128           |          11.722 |           326.718 |                  30.192 |       0.013165 |
| mixed128      |          16.422 |           433.214 |                  43.351 |       0.012229 |
| shared128     |           3.673 |           330.435 |                  31.648 |       0.011430 |
| browser_call3 |           0.986 |            18.763 |                   1.985 |       0.002019 |
| browser_call5 |           2.451 |            34.825 |                   3.249 |       0.004119 |

For 128 questions, the logits host read blocks for 349.447 ms while its CUDA
event span is 0.142 ms: most of the host wait is outstanding computation, not
copying logits. Moving the tiny confidence calculation alone has little room
to help. A coarse `nvidia-smi` sample during profiling reported 100% GPU activity,
32% memory activity and 1800 MHz; this does not establish SM occupancy or prove
that concurrent streams cannot help.

## 12. Parallel CPU preparation — accepted

Fresh baseline: `artifacts/cuda-optimization/12-baseline.json`, 20 measured calls
per workload before implementation. A bounded, lazily initialized pool of up
to four threads prepares CUDA requests with two or more questions. State
tokenization remains shared within each request. Indexed parallel collection
preserves request/question ordering, and errors are collected in order before
propagation to preserve the original first failure. Single-question and
non-CUDA calls use the original serial preparation. Pool creation failure also
falls back to serial execution. No GPU batch membership or matrix shape changes.

The preparation-only paired check (`12-preparation-only.json`) verifies exact
encoded sequences and measures 64–73% less preparation time for multi-request
workloads and shared128. Browser preparation improves 26–30%. For example,
128-request preparation falls from 11.014 to 3.127 ms; mixed128 falls from
16.247 to 4.418 ms. These are CPU-stage gains, not whole-model speedups.

| Workload      | First paired change | Repeat paired change |
| ------------- | ------------------: | -------------------: |
| 1             |              +2.39% |               +0.44% |
| 8             |              -2.12% |               -1.98% |
| 32            |              -2.02% |               -2.53% |
| 64            |              -2.30% |               -2.32% |
| 128           |              -1.31% |               -1.97% |
| mixed128      |              -2.45% |               -2.35% |
| shared128     |              -0.33% |               -0.44% |
| browser_call3 |              -1.83% |               -0.45% |
| browser_call5 |              -2.01% |               -1.65% |

Both 360-pair runs pass exact full-response/action checks. The multi-request
gain repeats at roughly 1–2.5%; browser_call5 also improves consistently.
Shared128 and browser_call3 have smaller/less convincing total-latency gains.
The single-question control keeps serial execution; its inconsistent timing
deltas illustrate background variation rather than a parallel-path speedup.
The implementation adds a small question-count/dispatch check to that path.
Detailed measurements are in `12-paired.json` and `12-paired-repeat.json`.

The sequence regression also compares every encoded item before GPU execution
and verifies first-error selection with multiple invalid questions, both within
one request and across separate requests. Pool creation is asserted in that
test, so a silent fallback cannot make the comparison pass without exercising
parallel preparation. Pool initialization is excluded from warmed paired
measurements; standalone results retain first-call latency.

The expanded 12-workload standalone candidate (10 measured calls each) passes
against the pre-change BF16 baseline, including alternate content and five
shape-change cycles. A separate six-workload F32 comparison against the saved
pre-change binary passes too (`12-f32-baseline.json` / `12-f32-candidate.json`,
10 measured calls each). All 44 default tests, FlashAttention all-target Clippy
with warnings denied, formatting and Nix derivation evaluation pass before
committing. No full Nix package rebuild is claimed.

## 13. Pipeline existing GPU batches — rejected as inconclusive

Fresh baseline includes experiment 12 (`996c60a68551`):
`artifacts/cuda-optimization/13-baseline.json`, 10 measured calls per workload.
The prototype splits packed scoring from response finalization. It reads and
pools batch A, queues batch B's encoder and decision head, then computes A's
CPU confidence features and queues A's action head. Only two batches are in
flight; their membership, token order and matrix shapes stay unchanged. It
applies only to CUDA/FlashAttention calls exceeding one batch.

Forty adjacent pairs per workload give these median latency ratios:

| Workload  | Paired change | Faster pairs |
| --------- | ------------: | -----------: |
| 64        |        +0.08% |        19/40 |
| 128       |        +0.18% |        17/40 |
| mixed128  |        +0.11% |        15/40 |
| shared128 |        -0.12% |        22/40 |

The target cases are essentially tied, with three slightly worse. Smaller
calls do not use the pipeline and serve as noisy controls; their deltas range
from -0.73% to +0.35%. There is no demonstrated throughput/latency improvement
worth retaining the scheduling complexity. All 360 paired full-response/action
checks pass, as does a separate comparison of partial batches and mixed question
kinds. No claim of an isolated regression of 0.1% is made.

The profile explains the limited opportunity: CPU confidence is tiny compared
with encoder/head execution, and this prototype preserves a single sequential
GPU stream. Reordering work does not reduce GPU arithmetic or eliminate the
necessary readbacks. This is an explanation from stage measurements and code,
not an occupancy trace or proof that all pipelining designs would fail.

The implementation and opt-in tests are archived in `13-batch-pipeline.patch`
against `996c60a68551`; their source changes were restored with Jujutsu.
`13-paired.json` retains every measured workload. The production runtime remains
the already validated parallel-preparation commit.

## Scope and reproduction

Concurrent CUDA streams were conditional on finding spare GPU capacity. The
available profile establishes that GPU computation dominates but does not
resolve kernel occupancy or stream-overlap potential. No concurrent-stream
optimization was implemented or benchmarked, and no speedup/rejection is
claimed for it. A kernel timeline/occupancy trace is still needed to assess that
separate experiment. The changes above improve local inference; they do not
measure Jev service or complete browser-task latency.

Fresh expanded standalone baseline:

```sh
direnv exec . bash -c 'export LD_LIBRARY_PATH=/run/opengl-driver/lib:$LD_LIBRARY_PATH; VS1_BENCH_LARGE=1 cargo run --release -p vs1 --features flash-attn --example cuda_bench -- artifacts/cuda-optimization/baseline.json 20'
```

After applying a candidate, pass the baseline file as the third example argument
to check full response/action equality, alternate content and changing shapes.
Opt-in paired preparation measurement:

```sh
direnv exec . bash -c 'export LD_LIBRARY_PATH=/run/opengl-driver/lib:$LD_LIBRARY_PATH; cargo test --release -p vs1 --features flash-attn paired_preparation_latency -- --ignored --nocapture --test-threads=1'
```

Other test filters are `preparation_cpu_latency`,
`parallel_preparation_preserves_sequences_and_errors` and `profile_batch_stages`.
The stage profiler intentionally retains serial preparation to reproduce the
pre-change diagnostic. To reproduce experiment 13, apply its archived patch in
an isolated checkout and run `paired_pipeline_latency` and
`pipeline_preserves_partial_and_mixed_batches`. Never overlap GPU benchmark runs.
