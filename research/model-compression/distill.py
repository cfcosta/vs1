"""Offline bounded compression/distillation experiment; no runtime dependency.

Requires PyTorch with CUDA. Reads/writes safetensors directly to avoid changing
an existing environment. Training examples/teacher logits come from Rust.
"""

import argparse
import json
import math
import random
import struct
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def read_weights(path):
    types = {"F32": torch.float32, "BF16": torch.bfloat16, "F16": torch.float16}
    with open(path, "rb") as f:
        header = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
        offset = f.tell()
        weights = {}
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            start, end = meta["data_offsets"]
            f.seek(offset + start)
            weights[name] = (
                torch.frombuffer(
                    bytearray(f.read(end - start)), dtype=types[meta["dtype"]]
                )
                .reshape(meta["shape"])
                .clone()
            )
    return weights


def write_weights(path, weights):
    header, offset = {}, 0
    for name, value in weights.items():
        size = value.numel() * 4
        header[name] = {
            "dtype": "F32",
            "shape": list(value.shape),
            "data_offsets": [offset, offset + size],
        }
        offset += size
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)
        for value in weights.values():
            f.write(value.detach().float().cpu().contiguous().numpy().tobytes())


class Student(torch.nn.Module):
    def __init__(self, weights, variant):
        super().__init__()
        if variant == "conv-all":
            for prefix in [f"encoder.layers.{i}" for i in range(28)] + [
                f"head.layers.{i}" for i in range(2)
            ]:
                weights[prefix + ".shortconv.weight"] = torch.full((1024, 7), 1.0 / 7.0)
        self.names = list(weights)
        self.weights = torch.nn.ParameterDict(
            {
                k.replace(".", "___"): torch.nn.Parameter(
                    v.float().cuda(), requires_grad=k != "temperature"
                )
                for k, v in weights.items()
            }
        )
        self.variant = variant
        self.cache = {}

    def w(self, name):
        return self.weights[name.replace(".", "___")]

    def linear(self, x, prefix, bias=False):
        result = F.linear(x, self.w(prefix + ".weight"))
        return result + self.w(prefix + ".bias").to(result.dtype) if bias else result

    def norm(self, x, prefix, bias=False):
        return F.layer_norm(
            x,
            (1024,),
            self.w(prefix + ".weight").to(x.dtype),
            self.w(prefix + ".bias").to(x.dtype) if bias else None,
            1e-5,
        )

    def rotary(self, x, layer):
        n = x.shape[-2]
        theta = 160000.0 if layer % 3 == 0 else 10000.0
        key = (n, theta, x.dtype)
        if key not in self.cache:
            freq = (
                torch.arange(n, device=x.device).float()[:, None]
                * (
                    1.0
                    / theta ** (torch.arange(0, 64, 2, device=x.device).float() / 64)
                )[None, :]
            )
            self.cache[key] = (freq.cos().to(x.dtype), freq.sin().to(x.dtype))
        c, s = self.cache[key]
        a, b = x[..., :32], x[..., 32:]
        return torch.cat((a * c - b * s, a * s + b * c), dim=-1)

    def attention(self, x, prefix, layer=None):
        n = x.shape[0]
        encoder = layer is not None
        weight = self.w(prefix + (".Wqkv.weight" if encoder else ".in_proj_weight"))
        bias = None if encoder else self.w(prefix + ".in_proj_bias")
        # Match the runtime's pre-scaled Q projection and separate GEMMs.
        qw, kw, vw = weight.chunk(3, dim=0)
        qb, kb, vb = (None, None, None) if bias is None else bias.chunk(3)
        q = F.linear(x, qw * 0.125)
        k = F.linear(x, kw) if self.variant != "conv-all" else None
        v = F.linear(x, vw)
        if bias is not None:
            q = q + (qb * 0.125).to(q.dtype)
            if k is not None:
                k = k + kb.to(k.dtype)
            v = v + vb.to(v.dtype)
        if self.variant == "conv-all":
            block_prefix = prefix.rsplit(".", 1)[0]
            mixed = F.conv1d(
                v.T[None],
                self.w(block_prefix + ".shortconv.weight").view(1024, 1, 7),
                padding=3,
                groups=1024,
            )[0].T
            gated = torch.sigmoid(q) * mixed
            return self.linear(
                gated, prefix + (".Wo" if encoder else ".out_proj"), not encoder
            )
        q, k, v = [t.view(n, 16, 64).transpose(0, 1) for t in (q, k, v)]
        if encoder and not (
            self.variant == "rope-all"
            or (self.variant == "rope-critical" and layer in [0, 12, 15])
        ):
            q, k = self.rotary(q, layer), self.rotary(k, layer)
        mask = None
        if encoder and layer % 3 != 0:
            key = ("mask", n)
            if key not in self.cache:
                pos = torch.arange(n, device=x.device)
                self.cache[key] = (pos[:, None] - pos[None, :]).abs() <= 64
            mask = self.cache[key]
        attended = (
            F.scaled_dot_product_attention(
                q[None], k[None], v[None], attn_mask=mask, scale=1.0
            )[0]
            .transpose(0, 1)
            .reshape(n, 1024)
        )
        return self.linear(
            attended, prefix + (".Wo" if encoder else ".out_proj"), not encoder
        )

    def block(self, x, i):
        if self.variant == "block-26" and i == 26:
            return x
        p = f"encoder.layers.{i}"
        if not (self.variant == "attention-27" and i == 27):
            h = x if i == 0 else self.norm(x, p + ".attn_norm")
            x = x + self.attention(h, p + ".attn", i)
        h = self.norm(x, p + ".mlp_norm")
        wi = self.w(p + ".mlp.Wi.weight")
        a, g = wi.chunk(2)
        h = F.gelu(F.linear(h, a), approximate="none") * F.linear(h, g)
        return x + self.linear(h, p + ".mlp.Wo")

    def forward(self, row):
        ids = (
            row["ids"]
            if isinstance(row["ids"], torch.Tensor)
            else torch.tensor(row["ids"], device="cuda", dtype=torch.long)
        )
        x = F.embedding(ids, self.w("encoder.embeddings.tok_embeddings.weight"))
        # Match BF16 runtime activations under AMP.
        x = self.norm(x.to(torch.bfloat16), "encoder.embeddings.norm")
        for i in range(28):
            if self.training:
                x = checkpoint(self.block, x, i, use_reentrant=False)
            else:
                x = self.block(x, i)
        x = self.norm(x, "encoder.final_norm") + self.w("type_emb.weight")[
            row["kind"]
        ].to(x.dtype)
        for i in range(2):
            p = f"head.layers.{i}"
            x = x + self.attention(self.norm(x, p + ".norm1", True), p + ".self_attn")
            x = x + self.linear(
                F.relu(
                    self.linear(self.norm(x, p + ".norm2", True), p + ".linear1", True)
                ),
                p + ".linear2",
                True,
            )
        pooled = x[0].float()
        x = x[row["markers"]]
        logits = (
            self.linear(
                F.gelu(
                    self.linear(self.norm(x, "scorer.0", True), "scorer.1", True),
                    approximate="none",
                ),
                "scorer.3",
                True,
            )
            .flatten()
            .float()
        )
        with torch.autocast("cuda", enabled=False):
            probs = logits.softmax(-1)
            top = probs.topk(2).values
            entropy = -(probs * probs.clamp_min(1e-9).log()).sum() / math.log(
                len(probs)
            )
            features = torch.stack(
                (top[0], top[0] - top[1], entropy, probs.new_tensor(len(probs) / 255.0))
            )
            action_logits = self.linear(
                F.gelu(
                    self.linear(torch.cat((pooled, features)), "act_head.0", True),
                    approximate="none",
                ),
                "act_head.2",
                True,
            )
        return logits, action_logits.softmax(-1)[0]


@torch.no_grad()
def evaluate(model, rows):
    model.eval()
    errors, flips, losses, action_errors = [], 0, [], []
    for row in rows:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, action = model(row)
        target = torch.tensor(row["logits"], device="cuda")
        p = (logits / row["temperature"]).softmax(-1)
        t = (target / row["temperature"]).softmax(-1)
        errors.extend((p - t).abs().tolist())
        action_errors.append(abs(action.item() - row["action"]))
        flips += int(p.argmax() != t.argmax())
        losses.append(F.mse_loss(logits - logits.mean(), target - target.mean()).item())
    return {
        "answers": len(rows),
        "decision_flips": flips,
        "max_probability_error": max(errors),
        "mean_probability_error": sum(errors) / len(errors),
        "centered_logit_mse": sum(losses) / len(losses),
        "max_action_error": max(action_errors),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--variant",
        choices=[
            "baseline",
            "rope-all",
            "block-26",
            "attention-27",
            "conv-all",
            "rope-critical",
        ],
        required=True,
    )
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()
    torch.manual_seed(17)
    random.seed(17)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data = json.loads(args.data.read_text())
    model = Student(read_weights(args.checkpoint / "model.safetensors"), args.variant)
    args.out.mkdir(parents=True, exist_ok=True)
    before = evaluate(model, data["validation"])
    print("BEFORE", json.dumps(before), flush=True)
    report = {
        "variant": args.variant,
        "seed": 17,
        "train_examples": len(data["train"]),
        "before": before,
        "epochs": [],
        "torch": torch.__version__,
        "optimizer": "Adafactor",
        "lr": args.lr,
    }
    if args.variant != "baseline":
        optimizer = torch.optim.Adafactor(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr,
            weight_decay=0.0,
            foreach=False,
        )
        for epoch in range(args.epochs):
            model.train()
            rows = list(data["train"])
            random.shuffle(rows)
            start, losses = time.monotonic(), []
            for i, row in enumerate(rows):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits, action = model(row)
                target = torch.tensor(row["logits"], device="cuda")
                loss = (
                    F.mse_loss(logits - logits.mean(), target - target.mean())
                    + (action - row["action"]).square()
                )
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(loss.item())
                if i % 16 == 0:
                    print("STEP", epoch, i, loss.item(), flush=True)
            result = evaluate(model, data["validation"])
            report["epochs"].append(
                {
                    "epoch": epoch + 1,
                    "seconds": time.monotonic() - start,
                    "mean_train_loss": sum(losses) / len(losses),
                    "validation": result,
                }
            )
            (args.out / "training.json").write_text(json.dumps(report, indent=2))
            print("EPOCH", json.dumps(report["epochs"][-1]), flush=True)
        write_weights(
            args.out / "model.safetensors",
            {name: model.w(name) for name in model.names},
        )
    (args.out / "training.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
