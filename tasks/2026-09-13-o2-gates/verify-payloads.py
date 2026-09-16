#!/usr/bin/env python3
"""Run before deleting the Q2 base/donor; compare every O2 tensor payload."""
import hashlib
import json
import sys
import time
from contextlib import ExitStack
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "gguf-tools/mixed"))
from splice_mixed_expert_layers_gguf import parse_gguf, should_take_donor

G = Path.home() / "models/gguf"
PREFIX = "DeepSeek-V4-Flash-Vision-Uncensored-orcarouter-"
SUFFIX = "Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-AProjQ8-SExpQ8-OutQ8.gguf"
paths = {
    "base": G / f"{PREFIX}IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8.gguf",
    "donor": G / "tmp-vision-uncen-o2-donor-KEEP.gguf",
    "o1": G / f"{PREFIX}Layers37-42{SUFFIX}",
    "o2": G / f"{PREFIX}Layers33-42{SUFFIX}",
}
info = {k: parse_gguf(p) for k, p in paths.items()}
before = {k: (p.stat().st_ino, p.stat().st_size, p.stat().st_mtime_ns) for k, p in paths.items()}
assert len(info["o2"].tensors) == len(info["o2"].tensor_by_name) == 1328
assert info["o2"].kv_blob == info["base"].kv_blob
assert set(info["o2"].tensor_by_name) == set(info["base"].tensor_by_name)
rows = []
start = time.monotonic()
with ExitStack() as stack:
    streams = {k: stack.enter_context(p.open("rb")) for k, p in paths.items()}
    for t in info["o2"].tensors:
        selected = should_take_donor(t.name, set(range(33, 43)))
        source = "donor" if selected else "base"
        peers = [source] + (["o1"] if should_take_donor(t.name, set(range(37, 43))) else [])
        for peer in peers:
            other = info[peer].tensor_by_name[t.name]
            assert (t.dims, t.ggml_type, t.n_bytes) == (other.dims, other.ggml_type, other.n_bytes), t.name
            streams[peer].seek(other.data_offset)
        if selected:
            assert t.ggml_type == 12, t.name
        streams["o2"].seek(t.data_offset)
        remaining = t.n_bytes
        digest = hashlib.sha256()
        while remaining:
            size = min(4 * 1024 * 1024, remaining)
            data = streams["o2"].read(size)
            assert len(data) == size, f"short O2 read: {t.name}"
            for peer in peers:
                assert data == streams[peer].read(size), f"payload mismatch: {t.name} vs {peer}"
            digest.update(data)
            remaining -= size
        rows.append({"tensor": t.name, "bytes": t.n_bytes, "source": source, "sha256": digest.hexdigest()})
        if len(rows) % 200 == 0:
            print(f"verified {len(rows)}/1328 tensors", flush=True)
assert sum(r["source"] == "donor" for r in rows) == 30
assert before == {k: (p.stat().st_ino, p.stat().st_size, p.stat().st_mtime_ns) for k, p in paths.items()}
result = {"pass": True, "files": {k: {"path": str(paths[k]), "identity": v} for k, v in before.items()},
          "tensors": rows, "seconds": round(time.monotonic() - start, 2)}
Path(__file__).with_name("payload-verification.json").write_text(json.dumps(result, indent=2) + "\n")
print(f"PAYLOAD_PASS tensors=1328 donor=30 shared_o1=18 seconds={result['seconds']}", flush=True)
