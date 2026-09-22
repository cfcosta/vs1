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
