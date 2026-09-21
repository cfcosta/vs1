# Model compression experiments

Status: bounded experiments complete. No candidate qualified as an improvement.
All runtime experiment hooks were reverted; the original checkpoint is unchanged.
Experiments ran sequentially, with each candidate starting from original weights.

Baseline runtime: `c74de8f3ffd6d990675fb0ddf02393e1860a8044`.
Checkpoint: default `convaiinnovations/laya` (ModernBERT-large, 28 encoder
blocks, two decision-head blocks, GeGLU intermediate width 2624).
Hardware: RTX 3080 Ti, BF16, packed FlashAttention.

## Protocol

Each candidate starts from the original encoder. Weight changes are in memory
only. Paired AB/BA timing measures synchronized `system_one_batch` calls,
including input preparation and postprocessing; model loading, pruning,
serialization and comparisons are outside the timed interval. Five warmups per
variant precede 20 pairs per workload. The desktop shares the GPU. Small timing
changes are not sufficient evidence of improvement.

The screening corpus contains 144 synthetic triage answers spanning `choice`,
`noul`, and `score`, plus five browser-fixture answers. Synthetic states include
negation, cancellations, refunds, routine messages, technical incidents, and
long distractor prefixes/suffixes. Decisions compare choice winners, noul at
0.5, and score modal levels; numeric checks include probability distributions,
confidence, expected score, and the separately accessed action probabilities.
This measures teacher agreement, not ground-truth task accuracy or calibration.
It is a rejection screen, not sufficient acceptance evidence. A promising
candidate requires independent held-out task evaluation and repeated timing.

Every run checks deterministic outputs for both variants and exact restoration
of the original encoder on the entire screening corpus. The conservative approximation screen requires unchanged decisions, at most
0.01 absolute probability/confidence/action drift and 0.02 expected-score drift
on the 0–2 scale. It does not establish ground-truth quality. Exact output
agreement is reported separately; large decision regressions reject a candidate.

## 1. Magnitude-ranked FFN channel pruning

Rank channels by the product of squared L2 norms of the activation input row,
gate input row, and output column. Keep the highest-scoring channels, preserving
their original relative order. Slice the same indices in both GeGLU input
projections and the output projection. This physically reduces dense matrix
sizes; it does not merely zero weights. All 28 encoder FFNs are pruned, and the
decision head remains intact.

### Width 2496 (4.9% removed): rejected

35/149 decisions changed. Maximum output error was approximately 1.0: a browser
click switched from target 3 to target 1. Mean numeric output error was 0.1114.
Action probabilities did not change on this corpus (not evidence of general
invariance).

| Workload         | Paired latency change |
| ---------------- | --------------------: |
| One question     |                +6.65% |
| Thirty questions |                -6.55% |
| Mixed lengths    |                -3.82% |
| Browser call 3   |                -0.98% |
| Browser call 5   |                -0.90% |

Full outputs, input requests, raw timing samples and restore result are in
`artifacts/model-compression/ffn-2496.json` (local, ignored). No checkpoint or production execution was changed by this run.

### Other widths: rejected

| Width | Channels removed | Decisions changed | Maximum numeric error |    One | Thirty |  Mixed | Browser 3 | Browser 5 |
| ----- | ---------------: | ----------------: | --------------------: | -----: | -----: | -----: | --------: | --------: |
| 2560  |             2.4% |            19/149 |                 1.000 | +3.94% | -6.04% | -2.82% |    +1.59% |    -1.32% |
| 2304  |            12.2% |            50/149 |                 1.276 | +1.53% | -7.35% | -6.44% |    -2.10% |    -1.62% |

Maximum numeric error includes the 0–2 expected score, so it may exceed 1.
Neither candidate meets the decision agreement requirement. Single-question
performance can regress when pruning changes GEMM dispatch; the existing narrow
retile specialization only supports the original input widths.

The width-2624 control applies the same gather/rebuild operation without removing
any channel and reproduces every screened output exactly. All four runs restore
the original model exactly after the paired comparisons.

## 2. Individual block ablation

Remove one whole encoder block, retaining the stored global/local attention and
RoPE configuration of every other block. Test all 28 blocks separately, deepest
first. Each candidate starts from the original checkpoint, not the preceding
candidate. Three timing pairs per workload are exploratory only: quality is the
first rejection gate; any surviving candidate requires full timing repeats.

All 28 individual block removals failed the numeric screen. Blocks 22, 24,
26 and 27 preserved all 149 screened decisions, but their maximum numeric errors
were 0.1173, 0.0574, 0.0513 and 0.1413 respectively. Every other block removal
changed at least one decision. Individual reports use zero-based block indices.

## 3. Individual attention-branch ablation

Skip just the attention residual branch and its pre-normalization, retaining
that block's FFN and the original settings of all other blocks. All 28 individual
branches were tested independently. Layers 22 through 27 preserved the screened
decisions, but all exceeded the numeric screen. The least disruptive was layer
27: maximum noul error 0.01865, option probability error 0.01506, confidence error
0.01720 and score error 0.00958. This is still not a passing candidate.

## Timing interference

An independent `multi_question_audit` GPU process was detected during the INT8
experiment. It had been running for about two minutes at detection, overlapping
the end of the attention sweep and the INT8 timings. Those timings are
provisional and cannot establish speedups or slowdowns. No candidate had passed
the quality screen, so no improvement decision depends on them. Further timed
runs were paused; quality-only tests may share the GPU. A subsequent INT8 repeat
started with no competing compute process, but monitoring caught a new
`next_four` process during that run too. Both INT8 timing sets are therefore
unqualified. The process log is retained with the raw artifacts. The desktop also uses
the GPU throughout, as in earlier repository experiments.

## 4. INT8 / INT4 FFN kernels

Use Candle 0.11's `QMatMul` with Q8_0 or Q4_0 block weights for all three encoder
FFN projections. The CUDA MMQ path also quantizes activations internally to
Q8_1; this is not a test of every possible W8A16/W4A16 kernel. Convert projection
outputs back to BF16 before the original fused GeGLU. The head stays unchanged.

Q8_0 preserved all 149 decisions but exceeded the numeric screen: maximum
numeric error 0.02963 and mean 0.00418. Its original latency samples overlapped
another GPU job and are provisional. Q4_0 changed five decisions, with maximum
numeric error 0.27845 and mean 0.02920. Both were rejected on quality. INT4 was
screened without timing while the GPU was shared.

## 5. Magnitude 2:4 sparsity

Within each consecutive group of four weights along the GEMM reduction axis,
zero the two smallest magnitudes, independently in all three encoder FFN
projections. This is structured 2:4 pruning, not unstructured sparsity. It changes
35/149 decisions; maximum numeric error 0.75171, mean 0.14958. Rejected.

This is a quality screen of the zero-shot transformation. Dense GEMMs remain in
use, so it is not a benchmark of sparse Tensor Cores. A retrained checkpoint and
sparse kernel integration would be a distinct experiment.

## 6. FP8 execution feasibility

Attempt per-output-row scaled E4M3 weight storage, with explicit F32
dequantization/scaling and BF16 dense projections. The CUDA conversion fails
with `DriverError(CUDA_ERROR_NOT_FOUND, "named symbol not found")` before any
candidate result. No full-model FP8 precision or latency result is claimed. Source inspection
identified a naming mismatch: Candle 0.11 builds `cast_f32_f8_e4m3`, while its
dtype-based dispatch requests `cast_f32_f8e4m3`. This is a runtime integration
failure, not evidence that FP8 conversion is impossible on this GPU.

A separate PyTorch probe uses the actual first encoder FFN weights and seeded
random BF16 activations at 129, 832, 1536 and 3870 token rows. Scaled E4M3 storage
plus explicit F32 dequantization and BF16 GEMMs gives output cosine
0.99959–0.99961. This is isolated-FFN numerical evidence, not final decision
quality. Timing was disabled because the GPU was shared; see
`fp8-ffn-probe.json`. No FP8 implementation is retained in the application.

## 7. RoPE removal and distillation

Removing RoPE from all 28 encoder attention blocks, without training, changes
100/149 decisions. Maximum numeric error 0.98351; mean 0.24771. Original attention
projections, head dimension, and global/local windows remain intact.

The bounded distillation trial exports 288 training examples and 149 validation
examples with exact Rust-built token sequences, marker positions, primitive
ids, temperatures, raw teacher logits and action probabilities. Training uses
separate synthetic message templates; the screening inputs are excluded.
Teacher targets are collected one question at a time, while final Rust screening
uses its original batching. This matters because BF16 results can depend on
matrix shape.

The offline PyTorch BF16 port preserves all validation decisions before surgery,
but differs numerically from Candle: maximum probability error 0.06618, mean
0.00319. Thus its validation is diagnostic, not the final acceptance check.
Trained checkpoints must be re-evaluated through the actual Rust runtime.

An initial AdamW attempt ran out of GPU memory on its first update. The run was
restarted from original weights with Adafactor (lr 0.001, no weight decay), F32
master parameters, BF16 autocast, gradient checkpointing, batch size one, seed
17, gradient norm clipping at 1, and one epoch. The loss matches centered final
marker logits plus the action probability, with gradients through the network.
There is no Python dependency added to the application.

### Distillation outcomes

| Candidate                            | Decisions changed before training (Rust) | After one epoch (Rust) | Result                                  |
| ------------------------------------ | ---------------------------------------: | ---------------------: | --------------------------------------- |
| Remove all encoder RoPE              |                                  100/149 |                 80/149 | Rejected                                |
| Remove encoder block 26              |                                    0/149 |                 17/149 | Rejected; training made agreement worse |
| Remove RoPE only in layers 0, 12, 15 |                                   26/149 |                 29/149 | Rejected                                |

The selective RoPE layers are the three highest decision-flip counts in the
attention-ablation sweep. The all-RoPE trial used lr 0.001; the other two used
0.0001. Each trial restarted from original weights. These are bounded pilots on
a small synthetic training set, not evidence that other data, optimizers,
learning rates or longer training cannot succeed. Final acceptance was based on
Rust outputs, not PyTorch diagnostics. See `training-results.json` and
`results.json`.

## 8. Gated short convolution pilot

Replace attention in all 28 encoder blocks **and both decision-head blocks**
with a seven-tap, symmetric depthwise convolution over the V projection,
multiplied by a sigmoid gate from Q, followed by the original output projection.
Initialize the depthwise filters to a uniform local average. Retain norms, FFNs,
embeddings and scorer; K is unused. No global-attention layer remains.

After one epoch (288 examples, Adafactor lr 0.001), the offline model still
changes 69/149 validation decisions, with maximum probability error 0.99495.
It fails the offline quality screen; no Rust convolution backend or latency
benchmark was retained. This is a bounded failure of this specific prototype,
not a reproduction of the other project's 40-epoch study or a claim that every
convolutional architecture must fail.

The initial optimizer attempt hit a NixOS Triton library-discovery error for
three-dimensional optimizer state. Storing each depthwise filter as a 2D
parameter and viewing it as 3D for convolution avoided that dispatch. The
successful run restarted from original weights and filter initialization.

## 9. TensorRT feasibility

A fixed-shape ONNX export of the offline baseline port built successfully with
TensorRT 10.13.3.9, BF16 enabled, TF32 disabled, a 1 GiB workspace and builder
optimization level 1. The serialized engine is approximately 806 MiB. A single
46-token, two-marker noul example agreed on the decision, with maximum
probability error 0.0017993 and action error zero against the Rust teacher.
See `tensorrt-probe.json` and artifact hashes in `tensorrt-artifacts.json`.

The legacy Torch exporter needed two narrowly checked repairs: remove 60
multiply-by-one chains incorrectly exported through COMPLEX128 casts, and wrap
the two final vector softmaxes in singleton dimensions. The repaired graph
passes ONNX checker. Export constant folding was disabled after a CPU/CUDA
constant-folding failure. `trt_probe.py` records the actual export/build/check
path and assertions.

This establishes fixed-shape feasibility only. It does not validate packed
batches, variable sequence lengths, all primitives, end-to-end Rust integration,
or latency. No TensorRT backend is shipped and no claim of a speedup or slowdown
is made. The other project's TensorRT result cannot be transferred to this
architecture from this smoke test.

## Artifacts and reproduction

- `results.json`: 68 Rust screens, including the unpruned control. Only that
  control passes the numeric screen; every run restores the original model exactly.
- `screen-corpus.json`: shared requests and baseline outputs. These are synthetic
  fidelity checks, not labeled task-accuracy evidence.
- `training-results.json`: baseline-port diagnostics and four bounded training trials.
- `experiment.patch`: archived Rust-only experiment hooks, intentionally absent
  from production source. Apply to baseline `c74de8f3ffd6d990675fb0ddf02393e1860a8044`
  in a disposable checkout. These hooks require the `flash-attn` feature.
- Full raw outputs, logs, exported teacher data and temporary dependencies live
  under ignored `artifacts/model-compression/` in the experiment workspace.
  Rejected training checkpoints were removed to recover disk space; their hashes
  are retained in `rejected-checkpoints.json`.

After applying the archived patch, run one candidate at a time:

```sh
nix develop -c bash -c '
  export LD_LIBRARY_PATH=/run/opengl-driver/lib:$LD_LIBRARY_PATH
  VS1_COMPRESSION=ffn-2560 VS1_COMPRESSION_QUALITY_ONLY=1 \
    cargo test --release -p vs1 --features flash-attn --lib compression_screen \
    -- --ignored --nocapture --test-threads=1
'
```

Other variant names include `block-26`, `attention-27`, `rope-all`,
`rope-0,12,15`, `quant-8`, `quant-4`, `sparse-2of4` and `fp8`. Omit
`VS1_COMPRESSION_QUALITY_ONLY` only for an idle-GPU timing run. Use
`VS1_COMPRESSION=ffn-2624` as the gather/rebuild control. Export teacher data with
`cargo test --release -p vs1 --features flash-attn --lib export_compression_training -- --ignored --nocapture`
in the same CUDA environment.

The offline scripts use Python 3.12, Torch 2.13.0+cu130 and the checkpoint's
`model.safetensors`. For example, with a working CUDA Torch environment:

```sh
python research/model-compression/distill.py \
  --checkpoint /path/to/laya --data artifacts/model-compression/distillation-data.json \
  --out artifacts/model-compression/distilled-rope-critical \
  --variant rope-critical --epochs 1 --lr 0.0001
```

Re-evaluate trained attention variants in Rust using
`VS1_DISTILLED_CHECKPOINT=/path/to/output` and `VS1_COMPRESSION=trained-rope-0,12,15`
(or `trained-rope-all` / `trained-block-26`). The convolution pilot has only an
offline evaluator. Run `summarize.py` to rebuild the compact reports from local
raw artifacts. FP8 uses `fp8_probe.py --checkpoint /path/to/laya --out /path/to/result.json --quality-only`.
TensorRT additionally uses ONNX 1.20.0 and TensorRT 10.13.3.9; run `trt_probe.py`
with modes `export`, `repair`, `build`, `check` sequentially, each with the same
`--checkpoint`, `--data` and `--out` arguments. The ONNX/engine files are large
local artifacts, not committed model dependencies.

## Outcome

There is no validated performance improvement to retain or make a performance
commit for. Pruning, ablation, quantization, sparsity and the bounded distillation
pilots failed the stated quality gate. FP8 and TensorRT remain incomplete
application-level evaluations, with the precise feasibility evidence above.
Contended timing samples are retained for audit but cannot support performance
conclusions. Runtime source and the shipped checkpoint remain at the baseline;
only research evidence, scripts and the archived experimental patch are retained.

## Final verification

After reverting the runtime hooks, both `cargo test -p vs1 --lib` and
`cargo test --release -p vs1 --features flash-attn --lib` passed (39 passed in each;
15 GPU/model-dependent tests ignored in the FlashAttention suite). `nix fmt`, Ruff checks/formatting and Python syntax checks
passed. A dry-run application of `experiment.patch` against the recorded baseline
succeeded. The two modified runtime files were verified identical to the baseline;
the temporary test module is absent. Unrelated concurrent research was excluded
from this change.
