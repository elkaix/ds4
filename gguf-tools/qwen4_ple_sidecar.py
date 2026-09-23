#!/usr/bin/env python3
"""Build the external Q4_1 PLE sidecar GGUF for Qwen3.8-Flash-Next.

ds4 loads the per-layer n-gram table through `--ple FILE` so the ~30 GiB table
stays out of the Metal working set (it is gathered on the CPU and demand-paged).
The sidecar holds `ple.weight` plus the three hash-constant tensors; the main
model GGUF is converted with `qwen4_exp_convert.py --no-ple`.

Rows are the 128 `ngram_embedding.shard_N` tensors concatenated in index order,
matching llama.cpp's `_place_ple_shard`.
"""
from __future__ import annotations

import argparse
import json
import mmap
import os
import struct
import sys
from pathlib import Path

import numpy as np

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
ALIGNMENT = 32
T_UINT32, T_INT32, T_FLOAT32, T_STRING, T_ARRAY, T_UINT64 = 4, 5, 6, 8, 9, 10
QTYPE_Q4_1, QTYPE_I64 = 3, 27
PLE_PREFIX = "ple.ple_embedding."
SHARD_MARK = PLE_PREFIX + "ngram_embedding.shard_"
AUX = ("layer_multipliers", "ngram_heads_offsets", "ngram_heads_vocab_sizes")


def w_str(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def kv(key: str, vtype: int, payload: bytes) -> bytes:
    return w_str(key) + struct.pack("<I", vtype) + payload


def kv_u32(key, value):
    return kv(key, T_UINT32, struct.pack("<I", int(value)))


def kv_u64(key, value):
    return kv(key, T_UINT64, struct.pack("<Q", int(value)))


def kv_str(key, value):
    return kv(key, T_STRING, w_str(value))


class Safetensors:
    """Read-only mmap view over a sharded safetensors checkpoint."""

    def __init__(self, root: Path):
        self.root = root
        index = json.loads((root / "model.safetensors.index.json").read_text())
        self.weight_map = index["weight_map"]
        self._maps: dict[str, mmap.mmap] = {}
        self._headers: dict[str, dict] = {}

    def _shard(self, name: str):
        shard = self.weight_map[name]
        if shard not in self._maps:
            fh = open(self.root / shard, "rb")
            n = struct.unpack("<Q", fh.read(8))[0]
            self._headers[shard] = json.loads(fh.read(n))
            fh.seek(0)
            self._maps[shard] = mmap.mmap(fh.fileno(), 0, prot=mmap.PROT_READ)
            self._headers[shard]["__data_start__"] = 8 + n
            fh.close()
        return shard

    def info(self, name: str):
        shard = self._shard(name)
        meta = self._headers[shard][name]
        return meta["shape"], meta["dtype"]

    def rows(self, name: str, lo: int, hi: int) -> np.ndarray:
        """float32 rows [lo, hi) of a 2-D BF16/F32/F16 tensor."""
        shard = self._shard(name)
        hdr = self._headers[shard]
        meta = hdr[name]
        base = hdr["__data_start__"] + meta["data_offsets"][0]
        shape = meta["shape"]
        dtype = meta["dtype"]
        width = shape[1]
        item = {"BF16": 2, "F16": 2, "F32": 4}[dtype]
        start = base + lo * width * item
        count = (hi - lo) * width
        if dtype == "BF16":
            raw = np.frombuffer(self._maps[shard], dtype="<u2", count=count,
                                offset=start)
            return (raw.astype("<u4") << 16).view("<f4").reshape(hi - lo, width)
        np_dtype = "<f2" if dtype == "F16" else "<f4"
        raw = np.frombuffer(self._maps[shard], dtype=np_dtype, count=count,
                            offset=start)
        return raw.astype("<f4").reshape(hi - lo, width)


def _gguf_quants(llama_cpp: str):
    """gguf-py ships the reference Q4_1 packer; use it rather than a second
    implementation, so the sidecar is byte-identical to llama.cpp's encoder."""
    path = os.path.join(os.path.expanduser(llama_cpp), "gguf-py")
    if path not in sys.path:
        sys.path.insert(0, path)
    try:
        from gguf import quants
    except ImportError:
        sys.exit(f"gguf-py not importable from {path}: pass --llama-cpp")
    return quants


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path, help="HF checkpoint directory")
    ap.add_argument("--out", required=True, type=Path, help="output sidecar GGUF")
    ap.add_argument("--name", default="Qwen3.8-Flash-Next ple")
    ap.add_argument("--source-revision", default="")
    ap.add_argument("--llama-cpp", default=os.environ.get("LLAMA_CPP", "~/bin/llama.cpp"),
                    help="llama.cpp checkout supplying gguf-py's reference Q4_1 packer")
    ap.add_argument("--chunk-rows", type=int, default=262144)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    quants = _gguf_quants(args.llama_cpp)
    db = Safetensors(args.src)
    config = json.loads((args.src / "config.json").read_text())
    hp = config.get("text_config", config)

    shards = sorted((name for name in db.weight_map if SHARD_MARK in name),
                    key=lambda n: int(n.rpartition(".shard_")[2].partition(".")[0]))
    if not shards:
        sys.exit("no ngram_embedding shards found in the checkpoint")
    expected = hp.get("split_ngram_parts", len(shards))
    if len(shards) != expected:
        sys.exit(f"found {len(shards)} shards, config declares {expected}")

    rows = 0
    row_dim = None
    for name in shards:
        shape, dtype = db.info(name)
        if len(shape) != 2:
            sys.exit(f"{name} is not 2-D")
        if row_dim is None:
            row_dim = shape[1]
        elif shape[1] != row_dim:
            sys.exit(f"{name} row dim {shape[1]} != {row_dim}")
        rows += shape[0]
    if row_dim % 32:
        sys.exit(f"PLE row dim {row_dim} is not a multiple of the Q4_1 block")

    aux = {}
    for suffix in AUX:
        hits = [n for n in db.weight_map if n.endswith(PLE_PREFIX + suffix)]
        if len(hits) != 1:
            sys.exit(f"expected exactly one {suffix} tensor, found {len(hits)}")
        shape, _ = db.info(hits[0])
        shard = db._shard(hits[0])
        meta = db._headers[shard][hits[0]]
        base = db._headers[shard]["__data_start__"] + meta["data_offsets"][0]
        span = meta["data_offsets"][1] - meta["data_offsets"][0]
        count = int(np.prod(shape))
        itemsize = span // count
        dt = {8: "<i8", 4: "<i4", 2: "<i2"}[itemsize]
        aux[suffix] = np.frombuffer(db._maps[shard], dtype=dt, count=count,
                                    offset=base).astype("<i8")

    row_bytes = (row_dim // 32) * 20
    ple_bytes = rows * row_bytes
    ple_layers = [i - 1 for i in hp["ple_layer_ids"]]

    meta = b"".join([
        kv_str("general.architecture", "qwen4-exp-ple"),
        kv_str("general.name", args.name),
        kv_u32("general.alignment", ALIGNMENT),
        kv_str("general.source.revision", args.source_revision),
        kv_str("ds4.pack.quant.ple", "Q4_1"),
        kv_u32("qwen4-exp.ple.layer", ple_layers[0] if ple_layers else 1),
        kv_u32("qwen4-exp.ple.embedding_length", hp["ple_embed_dim"]),
        kv_u64("qwen4-exp.ple.row_count", rows),
        kv_u32("qwen4-exp.ple.row_dimension", row_dim),
        kv_u32("qwen4-exp.ple.conv_kernel", hp["ple_conv_kernel_size"]),
        kv_u32("qwen4-exp.ple.ngram_size", hp["ngram_size"]),
        kv_u32("qwen4-exp.ple.heads_per_ngram", hp["heads_per_ngram"]),
        kv_u64("qwen4-exp.ple.vocab_base", hp["ngram_vocab_size_base"]),
        kv_u64("qwen4-exp.ple.vocab_divisor", hp["make_ngram_vocab_size_divisible_by"]),
        kv_u32("qwen4-exp.ple.shard_count", len(shards)),
    ])
    n_kv = 15

    tensors = [("ple.weight", (row_dim, rows), QTYPE_Q4_1, ple_bytes)]
    for suffix in AUX:
        data = aux[suffix]
        tensors.append((f"ple.{suffix}", (data.size,), QTYPE_I64, data.nbytes))

    infos = b""
    offset = 0
    for name, dims, qtype, nbytes in tensors:
        infos += w_str(name) + struct.pack("<I", len(dims))
        infos += b"".join(struct.pack("<Q", d) for d in dims)
        infos += struct.pack("<I", qtype) + struct.pack("<Q", offset)
        offset += (nbytes + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT

    header = (GGUF_MAGIC + struct.pack("<I", GGUF_VERSION)
              + struct.pack("<QQ", len(tensors), n_kv) + meta + infos)
    pad = (-len(header)) % ALIGNMENT
    total = len(header) + pad + offset

    print(f"shards      : {len(shards)}")
    print(f"ple.weight  : Q4_1 [{row_dim}, {rows}]  {ple_bytes / 2**30:.2f} GiB")
    for suffix in AUX:
        print(f"ple.{suffix:<24}: I64 {list(aux[suffix][:4])}{'...' if aux[suffix].size > 4 else ''}")
    print(f"total file  : {total / 2**30:.2f} GiB -> {args.out}")
    if args.dry_run:
        return

    tmp = args.out.with_name(args.out.name + ".incomplete")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    written_rows = 0
    check = None
    with open(tmp, "wb") as out:
        out.write(header + b"\x00" * pad)
        for name in shards:
            shape, _ = db.info(name)
            for lo in range(0, shape[0], args.chunk_rows):
                hi = min(lo + args.chunk_rows, shape[0])
                block = db.rows(name, lo, hi)
                packed = quants.Q4_1.quantize_rows(block)
                if check is None:
                    got = quants.Q4_1.dequantize_rows(packed[:1])
                    check = float(np.sqrt(((got.reshape(1, -1) - block[:1]) ** 2).mean()))
                out.write(packed.tobytes())
                written_rows += hi - lo
            print(f"  {name.rpartition('.shard_')[2].partition('.')[0]:>3}/"
                  f"{len(shards)}  rows {written_rows}/{rows}", flush=True)
        body = out.tell() - (len(header) + pad)
        if body != ple_bytes:
            sys.exit(f"wrote {body} PLE bytes, expected {ple_bytes}")
        out.write(b"\x00" * ((-body) % ALIGNMENT))
        for suffix in AUX:
            data = aux[suffix]
            out.write(data.tobytes())
            out.write(b"\x00" * ((-data.nbytes) % ALIGNMENT))
        size = out.tell()
    if size != total:
        sys.exit(f"wrote {size} bytes, expected {total}")
    os.replace(tmp, args.out)
    print(f"first-row Q4_1 roundtrip RMSE: {check:.3e}")
    print(f"wrote {args.out} ({size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
