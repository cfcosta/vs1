# Round 3: elementwise CUDA kernels

This round starts from `681e418257ee`, which already includes the fused BF16
GELU/gate operation from round 2. Each experiment gets a fresh standalone
baseline before implementation. Numerical acceptance includes full response and
action-probability equality, alternate inputs and changing shapes, plus focused
kernel tests. Performance decisions use adjacent baseline/candidate calls on a
single loaded model when differences are small, reversing order each pair.

Hardware remains the RTX 3080 Ti, CUDA 12.9, BF16 Laya with FlashAttention. The
desktop shares the GPU. Model loading and response serialization are excluded
from latency; tokenization, transfers and inference are included. Standalone
baselines use five warmups and 60 measured calls per workload. Paired runs use
60 pairs for each of five workloads. Finite regression corpora do not prove
arbitrary-input or cross-hardware bit equivalence.

## 9. Pairwise BF16 GeGLU — accepted

The candidate loads two adjacent BF16 values together, retains scalar
`normcdff` evaluation for each, and uses native BF16x2 products with the same
three rounding boundaries. A 256-thread launch replaces the 1024-thread scalar
launch on aligned inputs. Odd storage offsets retain the original scalar
kernel; a scalar tail handles odd lengths. The existing exhaustive finite-BF16
sweep passes, including the additional aligned odd-length case.

First paired measurements are in [`09-paired.json`](09-paired.json). Raw runs
are saved under `artifacts/cuda-optimization/09-*`. The generic paired harness
is shared with the earlier GeGLU comparison; switches exist only in test builds.

| Workload      | First paired change | Repeat paired change |
| ------------- | ------------------: | -------------------: |
| one           |              +0.08% |               -1.67% |
| thirty        |              -1.24% |               -1.01% |
| mixed_lengths |              -1.55% |               -1.08% |
| browser_call3 |              -1.77% |               -1.49% |
| browser_call5 |              -0.87% |               -0.59% |

The larger workloads show a repeatable, modest improvement; single-question
latency is less convincing. Repeat details are in `09-paired-repeat.json`.
All 600 paired comparisons match exactly. A separate 30-iteration run against
the saved pre-change baseline also passes across all six workloads, alternate
content and changing shapes. The finite-BF16 sweep, 44 default tests, formatting
and FlashAttention Clippy with warnings denied pass before committing.
