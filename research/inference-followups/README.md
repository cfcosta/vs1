# Remaining exact-output inference experiments

RTX 3080 Ti, CUDA/cuBLAS 12.9, driver 595.45.04, BF16 default Laya with
FlashAttention. Preserve weights, precision, real batch membership and exact
response/action outputs. Accept repeated end-to-end paired improvements only;
revert unsuccessful code, retain evidence, and commit each real improvement.
The GPU drives the desktop. Raw logs/snapshots are under ignored
`artifacts/inference-followups/`.

## Hardware-counter profile

Built the pinned Nsight Compute 2025.2.1.3 package and attempted SpeedOfLight,
Occupancy and MemoryWorkloadAnalysis on the warmed kernel-profile example.
The run failed with `ERR_NVGPUCTRPERM`: `/proc/driver/nvidia/params` reports
`RmProfilingAdminOnly: 1`, and `sudo -n true` requires interactive authentication.
No driver settings were changed. `00-ncu-permission.txt` records the actual
failure. No occupancy, saturation or memory-throughput conclusion is claimed.
The remaining experiments use direct output checks and adjacent AB/BA timings.

## 1. Independent projection streams

The initial prototype uses three reusable non-default CUDA streams, each with
its own cuBLAS handle. Q/K/V projections and the FFN activation/gate projections
keep their original matrix shapes and BF16/F32 compute settings. Events order
producer inputs before each group and join all outputs before their consumer.
The existing short-row retile remains on its original path.

The initial six-case regression passes original/alternate content and shape
revisits. The 60-pair exploratory whole-model check finds larger-workload gains
but a single-question regression. Separate QKV-only and FFN-only runs identify
where the overlap helps. The exploratory code and its environment switch are
archived in `01-exploratory-streams.patch`, and `01-{both,qkv,ffn}.json` retain
the three paired runs. These exploratory results alone are not the final
acceptance result.

### Accepted dispatch and verification

The retained implementation limits Q/K/V overlap to at least 1536 packed rows,
and FFN overlap to at least 2048 rows. It requires BF16 contiguous matrices with
1024 input columns, the measured RTX 3080 Ti and cuBLAS 12.9.1; other cases use
Candle unchanged. Three streams/handles are reused per calling thread and CUDA
context, rather than created for every layer. The default small-row retile is
unaffected. No experiment environment variable is needed in production.

Two independent 40-pair acceptance runs on the final dispatch:

| Workload                        | First paired latency change |     Repeat |
| ------------------------------- | --------------------------: | ---------: |
| One (unchanged path)            |                      -0.19% |     -0.41% |
| Eight (unchanged path)          |                      -0.51% |     +0.18% |
| Thirty-two                      |                  **-3.55%** | **-2.90%** |
| Browser call 3 (unchanged path) |                      +0.87% |     +0.31% |
| Browser call 5                  |                  **-5.93%** | **-6.66%** |

Every 32-question and browser-call-5 pair favored the candidate in both runs.
The small control differences are treated as timing noise, not gains.
The preceding expanded 40-pair runs found roughly 2–3% gains for 64, 128,
mixed128 and shared128; their large-matrix dispatch is unchanged by the final
minimum-row adjustment. The eight-question result was inconclusive and its
optimization was removed. Expanded and final paired reports are retained
separately so the dispatch revision is explicit.

Validation includes the 12-workload, 10-iteration regression against the initial
20-iteration baseline, original/alternate content and five shape cycles.
All response and action outputs match exactly. A separate product test checks
84 products across seven row counts, two projection widths, three independent
weights and positive/negated inputs; every BF16 result bit matches Candle.
Finite tests do not prove equality for arbitrary inputs or other configurations.

Nsight Systems confirms actual overlapping kernels on three projection streams:
40.898 ms of 341.711 ms of kernel-active time across ten warmed browser-call-5
calls has kernels on multiple streams concurrently (11.97%). This is timestamp
evidence of overlap, not occupancy or hardware-counter evidence. The summary
is `01-overlap.json`; full `.nsys-rep` and SQLite files remain in artifacts.

Release FlashAttention tests passed (50 passed, 20 opt-in tests ignored), including
separate execution of the projection product and paired tests above. Release
all-targets FlashAttention Clippy and canonical `nix fmt` passed. Model loading,
stream/handle initialization and snapshot comparison are excluded from warm
paired timings; standalone logs retain first-call samples. The additional
streams/handles and cuBLAS workspaces are a resource cost, not a free speedup.

Starting revision: `4309c5ed6751ca23064c6603cff968bc6e4ed4ed`.

Recheck the retained optimization:

```sh
nix develop -c bash -c '
  export LD_LIBRARY_PATH=/run/opengl-driver/lib:$LD_LIBRARY_PATH
  cargo test --release -p vs1 --features flash-attn --lib \
    paired_projection_acceptance -- --ignored --nocapture --test-threads=1
  cargo test --release -p vs1 --features flash-attn --lib \
    independent_products_match_candle -- --ignored --nocapture --test-threads=1
'
```

`paired_projection_large` exercises the expanded nine-workload timing corpus.
The archived exploratory patch applies to the starting revision above and uses
`VS1_EXPERIMENT_PROJECTIONS=both|qkv|ffn` with `paired_projection_streams`.

## 2. Concurrent whole batches

Accepted as an opt-in builder setting, `.with_parallel_cuda_batches(true)`
with `flash-attn`, CUDA and BF16. It loads two independent model copies on two
non-default CUDA streams and reuses a two-thread Rayon pool. It preserves the
existing stable length sort, batch size and batch membership. A mutex serializes
concurrent callers of this model; paired batches run concurrently inside a call.
Single or final unpaired batches keep the ordinary path. Defaults are unchanged.

This costs another full checkpoint in GPU memory plus concurrent activation
buffers and workspaces. Loading and warmup are excluded from the warm latency
comparison. It is inappropriate when memory is constrained or calls fit in one
batch. The evidence is for the measured GPU/checkpoint, not a portable speedup
claim.

Two exploratory runs of 30 adjacent AB/BA pairs found 6.1–8.2% improvements for
64/128 uniform questions, 5.7–7.8% for shared-state 128 questions and 2.6–4.0%
for mixed-length 128 questions. The 32-question single-batch control was within
0.3% noise. Reports: `02-batch-streams-{first,repeat}.json`.

The final public builder implementation was checked against the original default
CUDA model on all nine workloads plus 0, 31, 33, 63, 65 and 97 questions,
including partial batches, original response order, action probabilities and
usage. Invalid-input errors match. Four concurrent callers also return exact
outputs. Paired timings additionally check changed-content and original-content
revisits. All checks passed.

Final implementation timings, 30 AB/BA pairs per run:

| Workload  | First paired change | Repeat |
| --------- | ------------------: | -----: |
| 32        |              +0.37% | +0.43% |
| 64        |              -7.32% | -7.35% |
| 128       |              -7.86% | -7.73% |
| mixed128  |              -1.67% | -3.82% |
| shared128 |              -8.10% | -8.01% |

`02-batch-final.json` retains samples and exactness results. Run the opt-in
`batch_workers_preserve_default_outputs` and `paired_batch_streams` tests in
`model::followup_bench` with one test thread and no other GPU workload.

The final repeat is `02-batch-final-repeat.json`. Release FlashAttention unit
tests (50 passed), all-target release Clippy with warnings denied, and the
CPU-only build check passed. Canonical `nix fmt` passed.

## 3. Rounded CUTLASS GeGLU epilogue

The unrestricted pilot passed checkpoint-output checks but slowed single-question
calls by 22.4%. That dispatch was rejected. The retained candidate is limited to
BF16 contiguous matrices with 1024 input columns, 2624 output columns, 2048–32768
packed rows, default BF16 reduction precision, the RTX 3080 Ti and cuBLAS 12.9.1.
Other cases retain the preceding implementation. It reuses the same pinned
CUTLASS commit/cache as Candle FlashAttention and builds a static CUDA library;
there is no runtime NVRTC compilation or external shared-library path in production.

The activation GEMM accumulates in F32, then its epilogue rounds to BF16 before
GELU, rounds the normal CDF to BF16, and performs both multiplications with BF16
rounding. The gate projection remains a separate unchanged GEMM. This removes
the activation-buffer write/read and the standalone GeGLU launch. On eligible
shapes it supersedes the FFN projection-stream path from experiment 1; Q/K/V
stream overlap remains active. The ordinary one/eight/browser cases are below
the row cutoff and keep their prior path.

The initial actual-checkpoint first-FFN screen covered four shapes. The extended
seeded test covers 24 products (positive/negative inputs at 12 row counts, mixed
input magnitudes), including 2047/2048/2049 boundaries and rows up to 32768.
Across **417,651,584** BF16 outputs, the plain CUTLASS GEMM matches Candle, the
rounded epilogue matches the separate GeGLU kernel, and the fused result matches
Candle plus GeGLU exactly. These finite checks do not prove arbitrary-input
bitwise equivalence. `03-final-seeded-exactness.json` contains the comparisons.

The initial all-row and narrowed pilot reports are `03-pilot-paired.json` and
`03-narrow-paired.json`. `03-pilot.patch` plus `03-pilot.cu` preserve the offline
prototype (apply to `da52a505b732`, compile with CUDA 12.9 and the pinned CUTLASS
headers, then set `VS1_CUTLASS_LIB` for its opt-in tests). The archived prototype
contains the narrowed cutoff; remove its row cutoff only to reproduce the
rejected unrestricted dispatch.

Integrated implementation, 40 AB/BA pairs per workload:

| Workload      | First paired latency change | Repeat |
| ------------- | --------------------------: | -----: |
| 1             |                      -0.05% | -0.25% |
| 8             |                      -0.14% | -0.03% |
| 32            |                      -2.37% | -2.63% |
| 64            |                      -1.49% | -1.66% |
| 128           |                      -1.55% | -1.58% |
| mixed128      |                      -0.54% | -0.37% |
| shared128     |                      -1.84% | -1.51% |
| browser_call3 |                      +0.03% | -0.00% |
| browser_call5 |                      +0.09% | +0.07% |

Small-path fluctuations are controls, not improvements. The mixed-length gain
is small and variable; the strongest repeatable evidence is the uniform and
shared-state larger workloads. `03-final-paired.json` and `03-final-repeat.json` record both runs.
The opt-in tests are `rounded_products_match_candle` and
`paired_rounded_epilogue`, run with release FlashAttention and one test thread.

The full integrated 12-workload regression (10 iterations, alternate inputs,
five shape cycles) matches the initial `4309c5ed6751` response/action baseline
exactly; see `03-final-regression-summary.json`. Worker ordering/error/concurrent
caller checks and the cache-plus-workers check also passed with fusion enabled.
A combined-worker run (`03-combined-batches.json`) still improves uniform/shared
multi-batch latency by 4.5–6.1% over fused serial inference, and mixed128 by 3.7%.
Those figures use a newer baseline than experiment 2 and should not be added to
its earlier percentages.

To isolate the interaction, a further 40-pair comparison keeps parallel workers
on and changes only fusion. Fusion improves 64 questions by 1.90% and shared128
by 1.17%, with 39/40 pairs faster in each case and exact outputs. The report is
`03-worker-interaction.json`; the earlier worker implementation is its baseline.
This directly checks that fusion helps the opted-in worker path as well.

The independent worker-path repeat improves 64 by 1.54% (29/40 pairs faster), shared128 by 2.15% (33/40 pairs faster).
`03-worker-interaction-repeat.json` records it. Final release tests (52 passed),
all-target release Clippy with warnings denied, CPU-only tests (46 passed),
plain-CUDA compilation, and canonical `nix fmt` pass. The widened seeded
product check also passes again with explicit dispatch-boundary assertions.

## 4. Exact prepared-batch result cache

Accepted as the opt-in builder setting `.with_result_cache_capacity(32)`; the
default capacity is zero. The cache has an additional 4 MiB retained-payload
budget (allocator/deque bookkeeping is extra), evicts least-recently-used
entries, and stores only host token/marker metadata, raw logits and action
probabilities. It does not retain GPU tensors or complete responses.

A key contains the **whole ordered batch**, token IDs, marker positions, question
kinds and Candle's three CUDA reduction flags. Equality compares actual vectors,
not a hash alone. Cache scope is one immutable model/configuration, shared by its
two workers if enabled. Inference runs outside the cache mutex; errors are never
cached. Answer labels, question IDs and usage are assembled for the current
request. Option labels are themselves part of tokenization and changing them
changes the key; changing only a question ID can reuse the raw output safely.

The saved local corpus contains 44 sessions and 610 requests. 288 requests are
invalid under the local model's input contract; those errors remain identical.
The remaining 322 prepared batches yield 130 exact reuse hits (40.4%) with a
32-entry, 4 MiB cache cleared at each session boundary. This corpus includes
failed/repetitive browser loops and hosted-model requests. It is **not** evidence
of a 40% hit rate in successful production browser tasks.

Each replay run uses five AB/BA pairs, compares every response/action/error, and
sums synchronized call latency, including tokenization and invalid calls.
Snapshot comparisons and session-boundary cache clearing are outside timing.
The first run reduced paired aggregate latency by **35.66%**, from roughly
9.8–10.4 seconds to 6.4–6.6 seconds per replay. `04-cache-replay-first.json`
contains all five pairs; `04-replay-audit.json` contains per-session counts.

Unique-input miss controls use 40 AB/BA pairs and assert zero cache hits. The
first run's changes were +0.44% (one), -0.05% (eight), +0.05% (browser call 5).
The repeat adds explicit warmup before timing to exclude initial kernel loading.
These small differences are overhead/noise, not a cache benefit on unique inputs.

Validation covers key order/content/markers/type/precision, LRU eviction and
oversized-entry rejection; exact renamed-label/question-ID responses and usage;
shape/batch revisits; and the combination with parallel batch workers.
The raw corpus stays in ignored artifacts. `collect_replays.py` reconstructs it
from saved `artifacts/**/trace.json` decisions without modifying their requests.
It preserves request order within each session; session enumeration follows the
local filesystem. The `audit_exact_replay_keys`, `paired_cache_replay`,
`paired_cache_misses`, and `cache_preserves_labels_and_batch_context` tests are
opt-in and must run alone with one test thread.

The independent repeat reduced paired replay latency by **34.64%**.
Each of its five candidate replays recorded exactly 130 cache hits.
Repeat miss-control changes: 1 +0.13%, 8 -0.03%, browser_call5 +0.39%.
Reports are `04-cache-{replay,misses}-repeat.json`. Release FlashAttention
tests passed (52 passed), as did all-target release Clippy with warnings denied,
CPU-only tests (46 passed), and canonical `nix fmt`.
