# FluidInference-inspired runtime experiments

Run on 2026-09-21 against `efdfdf9913fcd32045f8027d1e51968b0ebbd5e8`.
RTX 3080 Ti, driver 595.45.04, CUDA toolkit from the repository Nix shell,
BF16 default Laya checkpoint with FlashAttention and F32 action head.
This is a CUDA implementation study, not a Core ML / M5 speed comparison.

## Rules

Keep only repeated, useful end-to-end improvements with exact public responses
and explicit action probabilities. Preserve weights, precision, attention
membership, input token budgets and question batching. Use adjacent AB/BA timing
pairs, alternate content and shape revisits. Revert failed or inconclusive
runtime changes, archive the experiments, and commit each accepted improvement
separately with Jujutsu. The desktop shares the GPU; small timing shifts can be
noise. No competing GPU compute process was visible at the initial check.

The baseline uses the existing six-case `cuda_bench` harness with 30 iterations,
five warmups, original/alternate inputs and five shape cycles. Loading is
excluded; tokenization, transfers, inference, CPU postprocessing and explicit
CUDA synchronization are included. Raw logs and larger snapshots live under
ignored `artifacts/fluid-inference/`.

## 1. GPU action features with double transcendental math: rejected

The earlier `research/cuda-optimization/01-gpu-features.patch` failed exact F32
feature equality using CUDA `expf`/`logf`. This retry uses double `exp`/`log`
rounded to F32, keeping every other operation and accumulation in F32, with
FMA contraction disabled. It queues the action head before the first readback.

The original 256-feature test now passes. A broader deterministic test (seed
713, seven widths 2/3/5/17/32/128/255, 512 rows each, five logit scales) finds
9 differing values among 14,336 features, maximum absolute error
`1.1920928955078125e-7`. Widths 17, 128 and 255 fail. This fails the numerical
gate before performance acceptance; no speedup is claimed. Reverted.

See `01-feature-parity.json` for per-width counts and reproducible first
counterexamples; `01-gpu-features-double.patch` contains the prototype and both
parity tests. This test is stricter than checking final decisions alone.

## 2. Fixed-shape CUDA Graph for only the action head: rejected

Capture only the F32 action head, keyed by batch size, after a shape's third
call. Unlike the previous full-encoder graph experiments, token lengths,
question types and marker positions do not enter this graph's key. Keep up to
eight graphs and 32 candidate batch sizes, with eager fallback. Copy each new
concatenated pooled-state/feature input into persistent storage before replay;
read output under the cache mutex. The CPU action features remain unchanged.

An initial multi-row capture hit Candle's host-parameter cache-miss error.
Enabling its parameter cache for warmup as well as capture fixed that error.
The corrected prototype passed all original/alternate outputs and five shape
cycles, then two independent 60-pair AB/BA runs: 1,200 timed response/action
comparisons, all exact.

| Workload       | First paired latency change | Repeat |
| -------------- | --------------------------: | -----: |
| One            |                      +2.33% | +2.26% |
| Thirty         |                      +6.80% | +6.71% |
| Mixed lengths  |                      +5.70% | +5.88% |
| Browser call 3 |                      +2.05% | +3.54% |
| Browser call 5 |                      +1.31% | +1.87% |

Positive means slower. Reverted. Capture startup is excluded from these warm
pairs, so amortizing capture would not rescue this implementation's result.
The extra copying, cache management and replay are plausible costs, but this
run did not profile their individual contributions. These results reject this
small-head graph implementation, not all CUDA Graph strategies.

See `02-paired.json`, `02-paired-repeat.json`, `02-standalone.json` and
`02-action-graph.patch`. The much larger standalone median shifts were misleading
and were not used to accept the candidate. `fused_p50_ms` in the existing paired
harness output means the graph candidate here.

## 3. Fixed token buckets with isolated padding: rejected at prerequisite gate

Try 128/256/512/1024 packed-token buckets while retaining variable-length
FlashAttention. Add one independent dummy sequence to fill the bucket; real
sequences cannot attend to it. Keep original real question batches, markers,
token usage and truncation unchanged; discard the dummy output. Inputs above
1024 packed tokens retain the existing path. This tests the numerical
prerequisite for bucketed execution, not a compiled bucket backend: graph
capture was not added after the prerequisite failed.

A fresh eager run matched the original baseline. The bucketed run was internally
stable across repeated/alternate inputs and shape cycles, but differed from the
baseline on all three workloads that actually gained padding:

| Workload       | Original packed tokens | Bucket | Differing scalar leaves, original + alternate |
| -------------- | ---------------------: | -----: | --------------------------------------------: |
| One            |                    129 |    256 |                                             2 |
| Five           |                    597 |   1024 |                                            24 |
| Browser call 3 |                    832 |   1024 |                                            20 |

Thirty, mixed lengths and browser call 5 fell back unchanged and had no output
differences. The one-question `noul` changed from `0.733471691608429` to
`0.7346251606941223`; browser call 3's CLICK probability changed from
`0.4113747477531433` to `0.40535619854927063`.

The implementation changes projection and scorer row counts; the arithmetic
path/rounding can therefore differ despite unchanged attention membership.
This explanation is consistent with prior shape experiments, but this run did
not isolate which individual operation caused each difference. No performance
claim is accepted from its standalone timings. Reverted. See
`03-comparison.json` and `03-token-buckets.patch`.

## Reproduction

All three patches apply independently to the baseline revision above. Do not
stack them. Use a disposable checkout for archived prototypes. The repository
Nix shell supplies CUDA 12.9.86; the driver advertises CUDA 13.2 compatibility.

Baseline / restored verification:

```sh
nix develop -c bash -c '
  export LD_LIBRARY_PATH=/run/opengl-driver/lib:$LD_LIBRARY_PATH
  cargo run --release -p vs1 --features flash-attn --example cuda_bench -- \
    artifacts/fluid-inference/baseline.json 30
'
```

After applying one patch, run under the same shell/library-path setup:

- Experiment 1: `cargo test --release -p vs1 --features flash-attn --lib
confidence_features -- --ignored --nocapture --test-threads=1`.
  The broad test is expected to fail and writes its JSON under
  `research/fluid-inference/`.
- Experiment 2: set `VS1_EXPERIMENT_ACTION_GRAPH=1`; run the `cuda_bench`
  example with `OUT.json 20 BASELINE.json`, then run the ignored
  `paired_action_graph` library test twice with one test thread. The log's
  `PAIRED_REPORT=` line contains the comparison JSON.
- Experiment 3: set `VS1_EXPERIMENT_BUCKETS=1`; run `cuda_bench` with
  `OUT.json 10 BASELINE.json`. The final baseline comparison is expected
  to fail after it writes the output snapshots.

## Retained outcome

No production change or speedup is retained. Only documentation, measurement
summaries and reproducible prototype patches remain. The experiment rules were
also saved to the user's memory-update inbox at their explicit request.

Final verification: the restored six-workload run (10 iterations plus alternate
inputs and five shape cycles) exactly matches the initial response/action
snapshots. Inference source hashes match the initial baseline; the Jujutsu diff
contains only this research directory. Release FlashAttention library tests
passed (50 passed, 16 opt-in tests ignored), and release all-targets FlashAttention
Clippy passed with warnings denied. `nix fmt` passed, all retained JSON parses,
and each archived patch passes `patch --dry-run -p1` independently against the
restored tree. The ignored tests exercised during these experiments are reported
above; the final ordinary test run did not execute every opt-in experiment.
