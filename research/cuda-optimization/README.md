# CUDA optimization experiments, 2026-09-20

None of these first four prototypes met the acceptance gates. Their production source
changes were reverted. The retained Rust benchmark, comparisons, and patches
record the experiments so that failures can be reproduced without shipping them.

The [second round](round-2.md) subsequently found a measured improvement from
fusing BF16 GELU and gate multiplication while preserving intermediate rounding.

## Method and acceptance gates

Runtime baseline: `f1e8031a80dd`. Hardware: RTX 3080 Ti, 12 GiB, NVIDIA driver
595.45.04. Model: cached `convaiinnovations/laya`, BF16 encoder, F32 action head,
CUDA with FlashAttention. Each experiment started with a fresh baseline run on
the unmodified runtime. Model loading is excluded from the timings.

The Rust example runs six workloads, five warmups and normally 30 measured
iterations each. Timings cover the entire `system_one_batch` call, including
tokenization, packing, transfers, inference and CPU postprocessing. Explicit
device synchronization brackets each measured call; snapshot serialization and
equality checks are outside that interval. Reported throughput is questions per
second calculated from median latency, not concurrent server throughput.

| Workload      | Questions | Packed tokens | Sequence lengths |
| ------------- | --------: | ------------: | ---------------- |
| one           |         1 |           129 | 129              |
| five          |         5 |           597 | 113–123          |
| thirty        |        30 |          3870 | 129              |
| mixed_lengths |        30 |          5400 | 57–399           |
| browser_call3 |         2 |           832 | 416              |
| browser_call5 |         3 |          1536 | 512              |

Keep a change only if it improves relevant performance and preserves exact
outputs. Snapshots include serialized responses **and** `Answer::action()`
probabilities, which ordinary response serialization omits. Repeated runs check
determinism; expanded checks alternate content and cycle through shapes to catch
stale graph inputs. Finite fixtures cannot guarantee equivalence for every input.
Restoring all runtime changes preserves the existing implementation outright.

The GPU also drives the desktop. Clocks and background graphics load were not
locked or removed, and absolute timings varied substantially across runs. Small
differences are inconclusive; neither favorable single runs nor selected cases
are treated as a reliable overall speedup.

## Results

All numbers below are median milliseconds, baseline → candidate. Each column
uses its own pre-change baseline. Negative changes mean faster.

| Workload      | GPU features      | CUDA graph, expanded run | Fused QKV         | Token budget 2048 | Token budget 4096 |
| ------------- | ----------------- | ------------------------ | ----------------- | ----------------- | ----------------- |
| one           | 8.423 → 8.052     | 7.991 → 7.147            | 7.615 → 8.150     | 8.719 → 9.006     | 8.719 → 8.319     |
| five          | 17.053 → 16.865   | 14.446 → 14.172          | 14.428 → 14.489   | 16.825 → 16.687   | 16.825 → 16.757   |
| thirty        | 97.656 → 97.087   | 96.903 → 81.430          | 82.993 → 87.713   | 96.538 → 102.633  | 96.538 → 98.405   |
| mixed_lengths | 124.785 → 123.261 | 105.801 → 102.579        | 104.888 → 109.103 | 123.123 → 141.814 | 123.123 → 138.180 |
| browser_call3 | 23.531 → 23.177   | 23.438 → 23.495          | 20.028 → 18.917   | 23.411 → 23.682   | 23.411 → 23.964   |
| browser_call5 | 43.292 → 42.370   | 37.348 → 36.689          | 37.089 → 35.812   | 42.176 → 43.263   | 42.176 → 43.197   |

### 1. GPU confidence features: rejected

Moved softmax-derived top probability, probability gap, normalized entropy and
candidate count onto CUDA, avoiding the intermediate host feature computation
and upload. The patch includes a deterministic synthetic feature parity test.

The initial run looked modestly faster, but repeats were inconsistent. A later
restored-runtime check put the single-question baseline at 7.021 ms versus
7.961 ms for the candidate, with most larger cases essentially tied. This later
check is retrospective, not a replacement for the original pre-change baseline.

Final response outputs, including action probabilities in the expanded check,
matched on this corpus. However, the isolated feature test found **32 of 256 F32
values bit-different**, maximum absolute error `1.1920929e-7`. CUDA and host
floating-point math did not reproduce exactly, even with fused multiply-add
disabled. We rejected that numerical risk alongside the unstable performance;
we did not observe final-response drift in this experiment.

The initial and 60-iteration repeat harness serialized only the public wire
response. `01-candidate-full-output.json` and `01-full-output-recheck.json`
close that coverage gap by including the Rust action probability accessor.

### 2. CUDA Graphs: rejected for changing shapes

Captured the encoder and decision head with persistent packed input storage,
updated token contents before replay, and cached one exact metadata/shape key.
Confidence features and the action head remained on the ordinary execution path.
The expanded corpus passed exact response and action-probability comparisons,
including alternate content and shape changes.

Warm recurring shapes sometimes improved. But cycling through all six shapes
forced repeated capture: median cycle time rose from **274.790 to 544.884 ms
(98.3% slower)**. Those cycle timings include snapshot serialization/equality on
both sides. The five cycle measurements were:

- Baseline: 274.244453, 274.789820, 272.139575, 279.520922, 289.491793 ms.
- Candidate: 542.976458, 555.432224, 557.101724, 544.884315, 539.473626 ms.

First calls for the thirty-question and mixed workloads increased from
82.5/121.9 ms to 164.9/205.9 ms. Browser state lengths change, so this single-entry
cache was a poor default. This rejects this prototype, not CUDA Graphs generally.
A bounded multi-shape cache with capture deferred until a shape recurs remains
a possible separate experiment.

### 3. Fused QKV projection/layout/RoPE: rejected

Combined separate Q/K/V GEMMs and added a fused CUDA layout/rotary kernel,
preserving the existing pre-scaled Q weights. Exact final outputs failed:
one `noul` probability changed from `0.733471691608429` to
`0.7342410087585449`; a browser CLICK probability changed from
`0.4113747477531433` to `0.40338966250419617`.

The isolated BF16 layout/rotary checks passed for all three components, pointing
to the changed combined GEMM as the source of BF16 end-to-end drift. The isolated
F32 rotary test separately failed. Changing matrix shapes can change floating
point accumulation and rounding. Two browser fixtures got faster, but other
workloads regressed, and exactness already ruled out keeping the change.

### 4. Token-budget batching: rejected at both budgets

Kept descending-length ordering and the existing maximum question count, adding
a greedy token limit. FlashAttention uses actual packed token counts; the
non-flash path uses padded token cost. CPU grouping was unchanged. A single
oversized request is allowed to ensure progress.

Both 2048 and 4096 token budgets failed exact output comparison. The 2048 limit
changed thirty-question and mixed outputs; 4096 changed mixed outputs. At 2048,
the comparison found 58 and 56 differing scalar leaves respectively, including
alternate-input snapshots. Mixed latency regressed **15.2% / 12.2%** respectively.
At 4096, mixed inputs had 54 differing scalar leaves; for example, the first
`noul` probability changed from `0.9121146202087402` to `0.9111628532409668`.

The likely explanation is that packed attention already avoids most padding;
splitting adds launches and per-batch synchronization while changing GEMM shapes
and rounding. This is an implementation-based explanation, not profiler proof.
The archived patch uses 2048; the second run changed that constant to 4096.

## Reproduction and evidence

The raw JSON runs and binaries are local, ignored artifacts under
`artifacts/cuda-optimization/`. Committed comparison JSON files retain medians,
p95, throughput and representative exact differences. Patch files are archived
failed implementations, not active source changes. Apply an individual patch to
the baseline runtime in a disposable checkout when reproducing it; do not stack
the experiments.

Fresh baseline and candidate commands (run sequentially):

```sh
direnv exec . bash -c 'export LD_LIBRARY_PATH=/run/opengl-driver/lib:$LD_LIBRARY_PATH; cargo run --release -p vs1 --features flash-attn --example cuda_bench -- artifacts/cuda-optimization/baseline.json 30'
# Apply one experiment using your patch tool, then:
direnv exec . bash -c 'export LD_LIBRARY_PATH=/run/opengl-driver/lib:$LD_LIBRARY_PATH; cargo run --release -p vs1 --features flash-attn --example cuda_bench -- artifacts/cuda-optimization/candidate.json 30 artifacts/cuda-optimization/baseline.json'
```

An output mismatch exits nonzero after saving the measurements. The standalone
`compare.py` reads two existing JSON files and prints a comparison; it does not
edit source files. Older committed comparisons for experiments 1–3 counted
original-output differences only; the current comparator also counts alternates.
This does not change their acceptance decisions.

Raw run pairs:

- GPU features: `01-baseline.json` / `01-candidate.json`, plus corresponding
  `*-repeat.json` (60 iterations). Full-output retrospective check:
  `02-baseline.json` / `01-candidate-full-output.json` (30 / 10 iterations).
- Graphs: `02-baseline.json` / `02-candidate.json`, followed by
  `02-baseline-expanded.json` / `02-candidate-expanded.json` with alternate
  content, first-call and shape-churn checks (30 iterations each).
- QKV: `03-baseline.json` / `03-candidate.json`; isolated kernel failure output
  is in `03-kernel-test.log` (30 benchmark iterations).
- Batching: `04-baseline.json` / `04-candidate-2048.json` and
  `04-candidate-4096.json` (30 iterations each).

No production speedup is claimed or shipped from these experiments.

After restoring the runtime, the final harness ran another 10 measured iterations
per workload against `04-baseline.json`. All original and alternate response
snapshots matched exactly, including action probabilities and shape churn.
That verification is saved locally as `final-restored.json`; its timings are
validation measurements, not evidence of an optimization.

Final checks passed: `cargo test -p vs1 --all-targets` (39 unit tests and five
conformance tests), `cargo clippy -p vs1 --all-targets --features flash-attn --
-D warnings`, and formatting. The opt-in Criterion model benchmarks were skipped
by the test command; GPU measurements came from the example commands above.
