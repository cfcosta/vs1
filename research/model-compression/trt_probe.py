"""Fixed-shape ONNX/TensorRT feasibility probe; not an application backend."""

import argparse
import json
from pathlib import Path

import torch
from distill import Student, read_weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["export", "repair", "build", "check"])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    row = json.loads(args.data.read_text())["validation"][0]
    if args.mode == "repair":
        import onnx
        from onnx import numpy_helper

        model = onnx.load(args.out / "model.onnx")
        producers = {o: n for n in model.graph.node for o in n.output}
        aliases, removed = {}, set()
        for node in model.graph.node:
            if node.op_type != "Cast" or not any(
                a.name == "to" and a.i == onnx.TensorProto.COMPLEX128
                for a in node.attribute
            ):
                continue
            sqrt = producers[node.input[0]]
            assert sqrt.op_type == "Sqrt"
            constant = producers[sqrt.input[0]]
            assert constant.op_type == "Constant"
            value = next(a.t for a in constant.attribute if a.name == "value")
            assert numpy_helper.to_array(value).item() == 1.0
            users = [n for n in model.graph.node if node.output[0] in n.input]
            assert users
            for user in users:
                assert user.op_type == "Mul" and len(user.input) == 2
                other = next(i for i in user.input if i != node.output[0])
                aliases[user.output[0]] = other
                removed.add(user.name)
            removed.add(node.name)
        assert len(aliases) in (0, 60), len(aliases)
        kept = [n for n in model.graph.node if n.name not in removed]
        for node in kept:
            for i, name in enumerate(node.input):
                while name in aliases:
                    name = aliases[name]
                node.input[i] = name
        del model.graph.node[:]
        model.graph.node.extend(kept)
        del model.graph.value_info[:]
        # TensorRT requires rank >= 2 for Softmax. Adding/removing a leading
        # singleton dimension preserves the two final vector softmaxes.
        if not any(
            n.name == "trt_vector_softmax_axes" for n in model.graph.initializer
        ):
            axes = "trt_vector_softmax_axes"
            model.graph.initializer.append(
                onnx.helper.make_tensor(axes, onnx.TensorProto.INT64, [1], [0])
            )
            softmaxes = [n.name for n in model.graph.node if n.op_type == "Softmax"]
            assert len(softmaxes) == 32
            targets = set(softmaxes[-2:])
            nodes = []
            for node in model.graph.node:
                if node.name in targets:
                    source, dest = node.input[0], node.output[0]
                    expanded, reduced = dest + "_rank2_in", dest + "_rank2_out"
                    nodes.append(
                        onnx.helper.make_node(
                            "Unsqueeze",
                            [source, axes],
                            [expanded],
                            name=node.name + "_expand",
                        )
                    )
                    node.input[0], node.output[0] = expanded, reduced
                    for attr in node.attribute:
                        if attr.name == "axis" and attr.i >= 0:
                            attr.i += 1
                    nodes.append(node)
                    nodes.append(
                        onnx.helper.make_node(
                            "Squeeze",
                            [reduced, axes],
                            [dest],
                            name=node.name + "_squeeze",
                        )
                    )
                else:
                    nodes.append(node)
            del model.graph.node[:]
            model.graph.node.extend(nodes)
        onnx.save(model, args.out / "model.onnx")
        onnx.checker.check_model(str(args.out / "model.onnx"))
        print(
            "Removed",
            len(aliases),
            "verified multiply-by-one artifacts from legacy SDPA export",
            flush=True,
        )
        return
    ids = torch.tensor(row["ids"], device="cuda", dtype=torch.long)
    if args.mode == "export":
        student = Student(
            read_weights(args.checkpoint / "model.safetensors"), "baseline"
        ).eval()

        class Wrapper(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.student = student

            def forward(self, ids):
                return self.student(row | {"ids": ids})

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            torch.onnx.export(
                Wrapper().eval(),
                (ids,),
                str(args.out / "model.onnx"),
                opset_version=18,
                dynamo=False,
                do_constant_folding=False,
                input_names=["ids"],
                output_names=["logits", "action"],
            )
        print("Exported", args.out / "model.onnx", flush=True)
        return
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, "")
    if args.mode == "build":
        builder = trt.Builder(logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        onnx = trt.OnnxParser(network, logger)
        if not onnx.parse_from_file(str(args.out / "model.onnx")):
            raise RuntimeError(
                "\n".join(str(onnx.get_error(i)) for i in range(onnx.num_errors))
            )
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
        config.set_flag(trt.BuilderFlag.BF16)
        config.clear_flag(trt.BuilderFlag.TF32)
        config.builder_optimization_level = 1
        engine = builder.build_serialized_network(network, config)
        if engine is None:
            raise RuntimeError("TensorRT build returned no engine")
        (args.out / "engine.plan").write_bytes(bytes(engine))
        print("Built", (args.out / "engine.plan").stat().st_size, "bytes", flush=True)
        return
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine((args.out / "engine.plan").read_bytes())
    context = engine.create_execution_context()
    logits = torch.empty(len(row["markers"]), device="cuda", dtype=torch.float32)
    action = torch.empty((), device="cuda", dtype=torch.float32)
    for name, tensor in [("ids", ids), ("logits", logits), ("action", action)]:
        assert tuple(engine.get_tensor_shape(name)) == tuple(tensor.shape)
        assert engine.get_tensor_dtype(name) == (
            trt.DataType.INT64 if name == "ids" else trt.DataType.FLOAT
        )
        context.set_tensor_address(name, tensor.data_ptr())
    assert context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    teacher = torch.tensor(row["logits"], device="cuda")
    candidate = (logits / row["temperature"]).softmax(-1)
    reference = (teacher / row["temperature"]).softmax(-1)
    result = {
        "tensorrt": trt.__version__,
        "tokens": len(row["ids"]),
        "probabilities": candidate.tolist(),
        "teacher": reference.tolist(),
        "max_probability_error": (candidate - reference).abs().max().item(),
        "action_error": abs(action.item() - row["action"]),
        "scope": "one fixed-shape example only; no latency or application-quality claim",
    }
    (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
