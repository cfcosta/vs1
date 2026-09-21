"""Isolated FFN FP8 storage/dequantization probe, not a full-model benchmark."""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from distill import read_weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--quality-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(17)
    weights = read_weights(args.checkpoint / "model.safetensors")
    wi = weights["encoder.layers.0.mlp.Wi.weight"].cuda().bfloat16()
    wo = weights["encoder.layers.0.mlp.Wo.weight"].cuda().bfloat16()
    del weights
    dense = [t.contiguous() for t in (*wi.chunk(2), wo)]
    scales = [
        w.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 448 for w in dense
    ]
    quantized = [
        (w.float() / s).to(torch.float8_e4m3fn)
        for w, s in zip(dense, scales, strict=True)
    ]

    @torch.no_grad()
    def run(x, fp8):
        w = (
            [(q.float() * s).bfloat16() for q, s in zip(quantized, scales, strict=True)]
            if fp8
            else dense
        )
        return F.linear(
            F.gelu(F.linear(x, w[0]), approximate="none") * F.linear(x, w[1]), w[2]
        )

    results = []
    for tokens in [129, 832, 1536, 3870]:
        x = torch.randn(tokens, 1024, device="cuda", dtype=torch.bfloat16)
        expected, actual = run(x, False), run(x, True)
        cosine = F.cosine_similarity(
            expected.float().flatten(), actual.float().flatten(), dim=0
        ).item()
        if args.quality_only:
            results.append(
                {
                    "tokens": tokens,
                    "cosine": cosine,
                    "timing": "not measured while GPU is shared",
                }
            )
            continue
        for _ in range(10):
            run(x, False)
            run(x, True)
        baseline, candidate, ratios = [], [], []
        for i in range(40):
            pair = [0.0, 0.0]
            for side in [0, 1] if i % 2 == 0 else [1, 0]:
                torch.cuda.synchronize()
                start = time.perf_counter()
                run(x, bool(side))
                torch.cuda.synchronize()
                pair[side] = (time.perf_counter() - start) * 1000
            baseline.append(pair[0])
            candidate.append(pair[1])
            ratios.append(pair[1] / pair[0])
        results.append(
            {
                "tokens": tokens,
                "baseline_ms": statistics.median(baseline),
                "candidate_ms": statistics.median(candidate),
                "paired_change_percent": 100 * (statistics.median(ratios) - 1),
                "cosine": cosine,
                "baseline_samples_ms": baseline,
                "candidate_samples_ms": candidate,
            }
        )
    args.out.write_text(
        json.dumps(
            {
                "scope": "isolated first encoder FFN; real weights and seeded random activations; PyTorch BF16 versus per-row E4M3 storage, explicit F32 dequantization/scaling and BF16 GEMMs",
                "torch": torch.__version__,
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            [
                {k: v for k, v in r.items() if not k.endswith("samples_ms")}
                for r in results
            ]
        )
    )


if __name__ == "__main__":
    main()
