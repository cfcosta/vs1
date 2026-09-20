"""Download bounded, revision-pinned training samples; never fetch test splits.

Research-only utility. Requires pyarrow for the small typed-decisions train file.
Run from the repository root. Raw samples stay in the ignored artifacts directory.
"""

import argparse
import concurrent.futures
import hashlib
import io
import json
from pathlib import Path
import urllib.request

SOURCES = {
    "mind2web": ("osunlp/Mind2Web", "17ece8eb89862368edc0cc806acee6fca5163474"),
    "typed-decisions": ("LocalLLaMA/typed-decisions", "ea9306458d6e9563628369a3d1e72e362fb381d2"),
    "webworld": ("Qwen/WebWorldData", "e108c5f8e35445c9ddff71cde2a5b1fc4db4020c"),
}
MIB = 1024 * 1024


def fetch(name, path, start=None, size=8 * MIB):
    repo, revision = SOURCES[name]
    url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{path}"
    headers = {"User-Agent": "vs1-dataset-audit/1.0"}
    if start is not None:
        headers["Range"] = f"bytes={start}-{start + size - 1}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=90) as response:
        if start is not None:
            content_range = response.headers.get("Content-Range", "")
            if response.status != 206 or not content_range.startswith(f"bytes {start}-"):
                raise RuntimeError(f"Server did not honor bounded range: {name}, {content_range}")
        data = response.read(size + 1)
        if len(data) > size:
            raise RuntimeError(f"Download exceeded cap: {name}/{path}")
        metadata = {
            "dataset": repo,
            "revision": revision,
            "path": path,
            "url": url,
            "http_status": response.status,
            "content_range": response.headers.get("Content-Range"),
            "bytes_received": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    return data, metadata


def write_jsonl(path, records):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    return {"file": str(path), "records": len(records), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def mind2web(out):
    records, manifest = [], []
    for shard in [0, 5, 10]:
        # Array boundaries require parsing complete objects from a prefix.
        for cap in [8 * MIB, 16 * MIB, 32 * MIB]:
            raw, metadata = fetch("mind2web", f"data/train/train_{shard}.json", 0, cap)
            manifest.append(metadata)
            text = raw.decode("utf-8", errors="ignore")
            decoder = json.JSONDecoder()
            position = text.index("[") + 1
            selected = []
            while len(selected) < 2:
                while position < len(text) and text[position] in " \t\r\n,":
                    position += 1
                try:
                    value, position = decoder.raw_decode(text, position)
                except json.JSONDecodeError:
                    break
                selected.append(value)
            if len(selected) == 2:
                break
        if len(selected) != 2:
            raise RuntimeError(f"Could not read two complete tasks in shard {shard} within cap")
        for index, row in enumerate(selected):
            records.append({"source": {"shard": shard, "task_index_in_shard": index}, "record": row})
    return {"downloads": manifest, "sample": write_jsonl(out / "mind2web-sample.jsonl", records)}


def typed_decisions(out):
    import pyarrow.parquet as pq

    raw, metadata = fetch("typed-decisions", "all/train-00000-of-00001.parquet", size=2 * MIB)
    (out / "typed-decisions-train.parquet").write_bytes(raw)
    rows = pq.read_table(io.BytesIO(raw)).to_pylist()
    groups = {}
    for index, row in enumerate(rows):
        groups.setdefault(row["workflow"], []).append((index, row))
    selected = []
    for workflow, group in sorted(groups.items()):
        for offset in [0, len(group) // 2, len(group) - 1]:
            index, row = group[offset]
            selected.append({"source": {"workflow": workflow, "row_index": index}, "record": row})
    return {"downloads": [metadata], "full_train_rows": len(rows),
            "sample": write_jsonl(out / "typed-decisions-sample.jsonl", selected)}


def webworld(out):
    records, manifest = [], []
    # Byte-stratified convenience sample, not a random sample of trajectories.
    for offset in [0, 17_414_449_242, 34_828_898_485]:
        raw, metadata = fetch("webworld", "WebWorld_Training.jsonl", offset)
        lines = raw.split(b"\n")
        first = 0 if offset == 0 else 1  # discard partial first/last records
        byte_position = offset + (len(lines[0]) + 1 if first else 0)
        count = 0
        for line in lines[first:-1]:
            if line.strip():
                row = json.loads(line)
                records.append({"source": {"byte_offset": byte_position}, "record": row})
                count += 1
                if count == 4:
                    break
            byte_position += len(line) + 1
        if count != 4:
            raise RuntimeError(f"Insufficient complete JSONL records near byte {offset}")
        manifest.append(metadata)
    return {"downloads": manifest, "sample": write_jsonl(out / "webworld-sample.jsonl", records)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/browser-training/samples"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = {"scope": "Training splits only. Convenience samples, not population estimates.", "sources": SOURCES}
    jobs = {"mind2web": mind2web, "typed-decisions": typed_decisions, "webworld": webworld}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = {name: pool.submit(fn, args.output) for name, fn in jobs.items()}
        for name, future in futures.items():
            results[name] = future.result()
            print(name, results[name]["sample"], flush=True)
    (args.output / "download-manifest.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
