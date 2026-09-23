#!/usr/bin/env python3
"""Sanity-check Hybrid v2 GGUF against the immutable RTN control.

Confirms: control mtime/size untouched; B same size/types; Q4_K gate/up
payloads nonzero; dequant RMS in the same ballpark as control (~0.0136),
not zero. Does not overwrite either GGUF.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / "bin/llama.cpp/gguf-py"))
from gguf import GGMLQuantizationType, GGUFReader, dequantize  # noqa: E402

CONTROL = Path.home() / "models/gguf/Qwen3.8-Flash-Next-Uncensored-DS4-Q4KGateUp-Q8_0Down-MTP.gguf"
CAND = Path.home() / "models/gguf/Qwen3.8-Flash-Next-Uncensored-DS4-Q4KGateUp-Q8_0Down-MTP-imatrix.gguf"
CONTROL_BYTES = 96_114_378_688
CONTROL_MTIME = 1_789_208_337
SAMPLE = (
    "blk.0.ffn_gate_exps.weight",
    "blk.0.ffn_up_exps.weight",
    "blk.24.ffn_gate_exps.weight",
    "blk.47.ffn_gate_exps.weight",
    "blk.48.ffn_gate_exps.weight",
)
OUT = Path(__file__).resolve().parent / "p3-v2-reconvert-dequant.json"


def fail(msg: str) -> None:
    raise SystemExit(msg)


def packed_stats(t) -> dict:
    import hashlib

    u8 = np.asarray(t.data).reshape(-1).view(np.uint8)
    head = bytes(u8[: min(u8.size, 1 << 20)])
    return {
        "nbytes": int(u8.nbytes),
        "nonzero": int(np.count_nonzero(u8)),
        "u8_sum_head4k": int(u8[:4096].sum()) if u8.size else 0,
        "sha256_head1m": hashlib.sha256(head).hexdigest() if head else "",
    }


def dequant_rms(t) -> dict:
    arr = dequantize(t.data, t.tensor_type)
    arr = np.asarray(arr, dtype=np.float32)
    rms = float(np.sqrt(np.mean(np.square(arr, dtype=np.float64))))
    return {
        "shape": list(arr.shape),
        "rms": rms,
        "finite": bool(np.isfinite(arr).all()),
        "min": float(arr.min()) if arr.size else 0.0,
        "max": float(arr.max()) if arr.size else 0.0,
    }


def qtype_name(t) -> str:
    # IntEnum str() is "12", not "Q4_K".
    tt = t.tensor_type
    if isinstance(tt, GGMLQuantizationType):
        return tt.name
    return GGMLQuantizationType(int(tt)).name


def inventory(reader: GGUFReader) -> dict[str, str]:
    out = {}
    for t in reader.tensors:
        n = t.name
        if "ffn_gate_exps.weight" in n or "ffn_up_exps.weight" in n or "ffn_down_exps.weight" in n:
            out[n] = qtype_name(t)
    return out


def main() -> None:
    cst = CONTROL.stat()
    if cst.st_size != CONTROL_BYTES:
        fail(f"control size changed: {cst.st_size} != {CONTROL_BYTES}")
    if int(cst.st_mtime) != CONTROL_MTIME:
        fail(f"control mtime changed: {int(cst.st_mtime)} != {CONTROL_MTIME}")
    if not CAND.is_file():
        fail(f"missing candidate {CAND}")
    bst = CAND.stat()
    if bst.st_size != CONTROL_BYTES:
        fail(f"candidate size {bst.st_size} != control {CONTROL_BYTES}")
    if os.path.samefile(CONTROL, CAND):
        fail("candidate path resolved to control")

    a = GGUFReader(str(CONTROL))
    b = GGUFReader(str(CAND))
    inv_a = inventory(a)
    inv_b = inventory(b)
    if inv_a != inv_b:
        fail(f"expert type inventory mismatch: a={len(inv_a)} b={len(inv_b)}")
    gate = {k: v for k, v in inv_b.items() if "ffn_gate_exps.weight" in k}
    up = {k: v for k, v in inv_b.items() if "ffn_up_exps.weight" in k}
    down = {k: v for k, v in inv_b.items() if "ffn_down_exps.weight" in k}
    if len(gate) != 49 or len(up) != 49 or len(down) != 49:
        fail(f"expected 49 gate/up/down, got {len(gate)}/{len(up)}/{len(down)}")
    if set(gate.values()) != {"Q4_K"} or set(up.values()) != {"Q4_K"}:
        fail(f"gate/up not all Q4_K: {set(gate.values())} {set(up.values())}")
    if set(down.values()) != {"Q8_0"}:
        fail(f"down not all Q8_0: {set(down.values())}")

    ta = {t.name: t for t in a.tensors}
    tb = {t.name: t for t in b.tensors}
    samples = []
    for name in SAMPLE:
        sa = packed_stats(ta[name])
        sb = packed_stats(tb[name])
        if sa["nonzero"] == 0:
            fail(f"control packed zeros: {name}")
        if sb["nonzero"] == 0:
            fail(f"candidate packed zeros: {name}")
        ra = dequant_rms(ta[name])
        rb = dequant_rms(tb[name])
        if not ra["finite"] or not rb["finite"]:
            fail(f"non-finite dequant {name}")
        if rb["rms"] <= 0.0:
            fail(f"candidate dequant rms=0 {name}")
        if abs(rb["rms"] - ra["rms"]) / max(ra["rms"], 1e-12) > 0.5:
            fail(f"{name} rms far from control: B={rb['rms']:.6g} A={ra['rms']:.6g}")
        samples.append({
            "name": name,
            "control_packed": sa,
            "cand_packed": sb,
            "packed_identical": sa["sha256_head1m"] == sb["sha256_head1m"],
            "control_dequant": ra,
            "cand_dequant": rb,
        })

    if all(s["packed_identical"] for s in samples):
        fail("candidate Q4_K head bytes identical to RTN control — imatrix did not change weights")

    payload = {
        "control": {"path": str(CONTROL), "bytes": cst.st_size, "mtime": int(cst.st_mtime)},
        "candidate": {"path": str(CAND), "bytes": bst.st_size, "mtime": int(bst.st_mtime)},
        "n_gate_q4k": len(gate),
        "n_up_q4k": len(up),
        "n_down_q8": len(down),
        "samples": samples,
        "ok": True,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print("VERIFY_V2_OK", OUT)
    for s in samples:
        print(
            f"{s['name']} A_rms={s['control_dequant']['rms']:.6g} "
            f"B_rms={s['cand_dequant']['rms']:.6g} "
            f"B_nonzero={s['cand_packed']['nonzero']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
