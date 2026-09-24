# Cua-S1 large-page and GEMM speed experiments

Date: 2026-09-23. Text-only scoring. Results and decisions come from the
[dispatcher notes](experiments/dispatcher-notes.txt), archived patches and GEMM
search JSON below; retained commit IDs were checked with `jj log`.

## Method

Each experiment was implemented, then measured against the previous build or
with an in-build switch in alternating A/B pairs. The desktop shares the GPU,
so absolute timings drift between sessions. Speed changes were kept only when
faster; behavior changes also had to be no worse on live tasks. Other changes
were reverted, with their patches archived.

The scoring benchmarks were the captured
[56-option Wikipedia decision](wikipedia-56-options.json) and the call5 replay
request (`call5x4` and `call5x8` in the notes). Behavior checks used live hotel
and Wikipedia tasks. Compare paired measurements within each experiment, not
absolute timings across experiments. The G3 microbenchmarks used an RTX 3080 Ti,
cuBLASLt 120901, BF16 inputs/outputs, F32 compute and CUDA-event timing.

## Summary

Times are milliseconds unless stated otherwise; arrows run from control to
experiment. Paired timings retain the order recorded in the notes.

| Experiment | Change                                                    | Result                                                                                     | Key numbers                                                                 |
| ---------- | --------------------------------------------------------- | ------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------- |
| A4         | Share the tournament prompt prefix                        | Kept: `8d234cf1b77c`                                                                       | 56 options: 1207 → 480 ms; 6400 → 2182 tokens                               |
| B1         | Batch group suffixes                                      | Rejected: [patch](experiments/b1-batched-suffixes.patch)                                   | Off 480.3/481.8 vs on 478.1/479.8 ms; ~0.4%, noise                          |
| C1         | Suppress clicks that leave the page fingerprint unchanged | Rejected: [patch](experiments/c1-fingerprint-suppression.patch)                            | Hotel 0/3, Wikipedia 0/3; rule never triggered                              |
| D1         | Dedupe identical native options                           | Rejected: [patch](experiments/d1-dedupe-options.patch)                                     | 56 → 55 options, still 4 passes; on 493.5/482.0 vs off 491.1/481.0 ms       |
| E1         | Trim visible-text lines equal to control labels           | Kept: `284809880e60`                                                                       | 33 lines removed; prefix 1406 → 1268 tokens; 480 → 452 ms                   |
| G1         | Compute SwiGLU with a CUTLASS dual GEMM                   | Kept: `5c864f91f9f5`                                                                       | call5x4: faster in 5/7 pairs, 2 ties; median ~6% less time                  |
| G2         | Concatenate projections sharing an input                  | Rejected: [patch](experiments/g2-fused-projections.patch)                                  | Fused replay ran out of memory; separate mode 347–355 ms/question           |
| G3         | Search cuBLASLt algorithms by shape and M                 | Measurement: [search](experiments/g3-gemm-search.json), [sweep](experiments/g3-sweep.json) | Notes summarize summed GEMM time as 176.4 → 160.3 ms; about 10% opportunity |
| G3c        | Autotune cuBLASLt at runtime                              | Rejected: [patch](experiments/g3-cublaslt-search-and-autotune.patch)                       | On 362.0/365.0 vs off 350.5/351.9 ms/question; tolerance ratio 4.72         |

## A4: prefix sharing

Encode the shared prompt prefix once and continue tournament groups from the
saved state. Two pairs on the 56-option fixture measured 1207 → 480 ms.
Processed tokens fell from 6400 to 2182 (1406 prefix + 776 suffix). The choice
remained `e1`, with probability 0.1681 → 0.1732; the top three stayed the same
and ranks four/five swapped. Kept in `8d234cf1b77c`.

Live controls after A4 recorded hotel 0/3 successes (3 actions, 148 ms median)
and Wikipedia 0/3 (4 actions, DONE after clicking the logo, 408 ms median).
Disabling sharing in the same binary with `VS1_CUA_S1_DISABLE_PREFIX_SHARING=1`
still gave Wikipedia 0/2 with the same decision sequence, at 841 ms median. Prefix sharing
was not the cause of these failures. Earlier Wikipedia passes (1–2/2) were not
a stable control: the live main page and search change during the day.

## B1: batched suffixes

Batch group suffixes after prefix encoding. Off measured 480.3/481.8 ms versus
478.1/479.8 ms on, about 0.4% and within noise. The choice stayed `e1`, but its
probability changed from 0.1732 to 0.1824. The prefix pass dominates and group
suffix passes are few and short, so the change was rejected.

## C1: fingerprint suppression

Suppress repeated no-op clicks using unchanged page fingerprints. In the same
session as the live control, hotel had 0/3 successes with the identical trajectory:
Free cancellation, Design, View Casa Flora, DONE. Wikipedia also had 0/3; runs
1–2 matched the control (search, article, logo, search, DONE), while run 3 had
an unrelated text-helper error, “returned no valid text.” No same-fingerprint
repeats occurred, so the rule never triggered. Rejected without a demonstrated
behavior improvement.

## D1: option dedupe

Dedupe native options with identical `(role, label, action)`. Only one duplicate,
“axioms,” was removed from the fixture: 55 options still required 4 passes.
On measured 493.5/482.0 ms versus 491.1/481.0 ms off. The choice and probability
stayed `e1` and 0.1732. Rejected; no live run was needed because this page could
not reduce its pass count.

## E1: visible-text trimming

Remove visible-text lines already represented by control labels in native
policy state. Removing 33 lines shortened the prefix from 1406 to 1268 tokens
and measured 480 → 452 ms over two pairs. The choice stayed `e1` with
probability 0.1732 → 0.1856, but the ranking changed: `e30` left the top five.
Kept in `284809880e60`.

Hotel now typed Destination first, like Jev, then selected Free cancellation,
Design, Casa Flora and DONE. It still failed the filters: neither cua-s1 nor
Jev submitted “Find stays.” Hotel run 3 errored on text-helper invalid JSON.
Wikipedia was inconclusive: two of three runs hit text-helper invalid JSON;
the clean run matched the control. Control run 2 also hit CUDA out-of-memory
mid-run on a long page with a ~2.4 GB desktop baseline, exposing memory pressure
from prefix state alongside Chromium.

## G1: SwiGLU dual GEMM

Use a CUTLASS dual GEMM with a 128×64×32 tile and an epilogue matching Candle's
BF16 SiLU rounding. Intermediate and MLP outputs were bit-identical at
M=129/1860/4096. At M=1, cuBLAS uses a GEMV path: 3666/9216 elements differed,
with tolerance ratio 0.69, within tolerance. Model parity was unchanged
(maximum difference 6.7e-5; all top choices preserved).

The call5x4 control/experiment pairs were 359.8/354.7, 392.6/354.1,
402.2/351.8, 375.1/352.0, 378.9/352.5, 359.0/360.7 and 365.5/365.7 ms.
The notes classify five as faster and two as ties, with a median reduction
of about 6%. Kept in `5c864f91f9f5`.

## G2: fused projections

Concatenate DeltaNet qkv+z+a+b and attention q+k+v projections on CUDA. Fused
mode ran out of GPU memory on call5 with a ~2.7 GB desktop baseline; separate
mode measured 347–355 ms/question. Splitting required contiguous copies for
DeltaNet qkv/z and attention q/k/v; only a/b stayed views, reducing the potential
GEMM saving. No latency gain was demonstrated, so this was rejected.

`q_proj`, `in_proj_qkv` and `in_proj_z` were identical. K/V differed in ~27%
of elements (absolute difference ≤0.031, tolerance ratio ≤0.47), and a/b in
~0.1%, because cuBLAS selected a different algorithm for the wider matrix.

## G3: cuBLASLt search

The [initial search](experiments/g3-gemm-search.json) used 21 samples per
candidate. Best algorithms were only 0–5% faster than Candle's default at
M=200/1860, but 13–52% faster at M=1400. The
[sweep](experiments/g3-sweep.json) covered M=128 through 4096 in steps of 128,
with seven samples. At scattered M values the default was 15–73% slower:
down_proj at 1152/1280/2176 had speedups of 1.73×/1.63×/1.47×; out_proj at
1152 had 1.59×; k_proj at 1408 had 1.55×.

The notes summarize summed projection GEMM time as 176.4 → 160.3 ms and call
this “10.1% less.” Summing the JSON medians gives about 9.15% less time
(about 1.10× speedup), so the opportunity is approximately 10%, not an
end-to-end scoring gain. Many winning algorithms were not bit-identical.
This was a measurement only; `measure_cua_s1_gemms` is archived with the G3c
patch.

## G3c: runtime autotune

Tune per `(n, k, m)` using up to eight candidates and a ≥3% improvement threshold.
Rejected for accuracy, overhead and reproducibility:

- A selected candidate returned -0.421875 versus the default -0.53515625,
  failing the combined tolerance with ratio 4.72. Some fast candidates,
  including split-K algorithms, were not accurate enough.
- call5x8 measured 362.0/365.0 ms/question with tuning versus 350.5/351.9
  without. Tuning cost 580–589 ms for 18 keys (13 non-default). Excluding
  tuning gave about 3.5% warm improvement, but browser pages continually change
  M, causing tuning to recur.
- Live timing changed algorithm selection and winner probability between runs
  (0.1853 versus 0.1998), where the model had been deterministic.

The warm 56-option fixture improved 478 → ~464 ms (~3%) with the same choice.
For down_proj at M=1152, default 1.452 ms became 0.929 ms (1.56×), but tuning
cost 31 ms. Those warm gains did not justify retaining the tuner.

## Open follow-ups

- The M-dependent cuBLAS default leaves a real but small opportunity: selecting
  the best algorithm per shape and M could cut projection GEMM time by about
  10% in the measured sweep. Restrict candidates to accurate, reproducible
  algorithms and avoid runtime tuning; the unrestricted search does not
  establish how much gain survives those requirements.
- Improve long-page GPU memory headroom, including saved prefix state and the
  shared desktop/Chromium footprint. E1's control OOM and G2's fused replay OOM
  show that this remains unresolved.
- Fix the missing “Find stays” step in the hotel task for both cua-s1 and Jev;
  trimming made Destination entry more consistent but did not complete the task.
