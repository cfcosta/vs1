"""Check all compatible examples against the existing fixed-shape TRT engine."""

import argparse
import json
from pathlib import Path

import torch


def main():
    import tensorrt as trt

    p = argparse.ArgumentParser()
    p.add_argument("--engine", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    rows = json.loads(args.data.read_text())["validation"]
    first = rows[0]
    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, "")
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(args.engine.read_bytes())
    context = engine.create_execution_context()
    stream = torch.cuda.Stream()
    results = []
    with torch.cuda.stream(stream):
        ids = torch.empty(len(first["ids"]), device="cuda", dtype=torch.int64)
        logits = torch.empty(len(first["markers"]), device="cuda", dtype=torch.float32)
        action = torch.empty((), device="cuda", dtype=torch.float32)
        for name, tensor in [("ids", ids), ("logits", logits), ("action", action)]:
            assert tuple(engine.get_tensor_shape(name)) == tuple(tensor.shape)
            expected = trt.DataType.INT64 if name == "ids" else trt.DataType.FLOAT
            assert engine.get_tensor_dtype(name) == expected
            context.set_tensor_address(name, tensor.data_ptr())
        for i, row in enumerate(rows):
            if (len(row["ids"]), row["markers"], row["kind"]) != (
                len(first["ids"]),
                first["markers"],
                first["kind"],
            ):
                continue
            ids.copy_(torch.tensor(row["ids"], device="cuda"))
            assert context.execute_async_v3(stream.cuda_stream)
            stream.synchronize()
            got = (logits / row["temperature"]).softmax(-1)
            reference = (
                torch.tensor(row["logits"], device="cuda") / row["temperature"]
            ).softmax(-1)
            results.append(
                {
                    "validation_index": i,
                    "logits": logits.tolist(),
                    "teacher_logits": row["logits"],
                    "logits_exact": logits.tolist() == row["logits"],
                    "max_probability_error": (got - reference).abs().max().item(),
                    "decision_equal": got.argmax().item() == reference.argmax().item(),
                    "action_error": abs(action.item() - row["action"]),
                }
            )
    report = {
        "tensorrt": trt.__version__,
        "cases": results,
        "scope": "Existing BF16 fixed-shape engine; exact-output rejection gate, no latency claim.",
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
