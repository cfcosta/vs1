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
