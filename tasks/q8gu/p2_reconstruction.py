#!/usr/bin/env python3
"""P2 weight-reconstruction sensitivity proxy.

For each trunk layer L=0..47, split HF fused gate_up_proj [512,1280,2560]
the same way llama.cpp conversion/qwen.py does (gate = [:, :640], up = [:, 640:]),
quantize with the production DS4 encoders, dequantize with ggml-py, score Q4_K
vs Q8_0 against the exact uncensored BF16.

Not activation sensitivity. This pack has no imatrix (F6).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

QSA_LAYERS = frozenset({3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47})
N_TRUNK = 48
N_EXPERTS = 512
N_FF = 640
N_EMBD = 2560
HF_KEY = "model.language_model.layers.{L}.mlp.experts.gate_up_proj"
DOCUMENTED_REV = "8336e613ea508b13c2159bd0f68965d97a606b95"
P99_EDGES = np.geomspace(1e-12, 10.0, 2049)
Q8_ROW_BYTES = 2720  # 2560/32 * 34
Q4_ROW_BYTES = 1440  # 2560/256 * 144


def layer_kind(layer: int) -> str:
    return "QSA" if layer in QSA_LAYERS else "GDN"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_hf_revision(hf_dir: Path) -> dict:
    refs = Path.home() / (
        ".cache/huggingface/hub/models--orcarouter--Qwen3.8-Flash-Next-Uncensored/refs/main"
    )
    rev = refs.read_text().strip() if refs.is_file() else None
    if rev != DOCUMENTED_REV:
        raise SystemExit(f"HF revision {rev!r} != documented {DOCUMENTED_REV}")
    return {
        "path": str(hf_dir),
        "revision": rev,
        "documented_rev": DOCUMENTED_REV,
        "config_sha256": sha256_file(hf_dir / "config.json"),
        "index_sha256": sha256_file(hf_dir / "model.safetensors.index.json"),
    }


def install_native_q4k(gguf, library_path: str) -> str:
    """Same libds4quants hook as qwen4_exp_convert._install_native_kquants, Q4_K only."""
    import ctypes
    from concurrent.futures import ThreadPoolExecutor

    lib = ctypes.CDLL(library_path)
    lib.ds4q_can_quantize.argtypes = [ctypes.c_int]
    lib.ds4q_can_quantize.restype = ctypes.c_bool
    lib.ds4q_block_size.argtypes = [ctypes.c_int]
    lib.ds4q_block_size.restype = ctypes.c_int64
    lib.ds4q_quantize_init.argtypes = [ctypes.c_int]
    lib.ds4q_quantize_init.restype = None
    lib.ds4q_quantize_chunk.argtypes = [
        ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_void_p,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.ds4q_quantize_chunk.restype = ctypes.c_size_t
    if not lib.ds4q_can_quantize(12):
        raise SystemExit("libds4quants cannot encode Q4_K")
    cls = gguf.quants.Q4_K
    if lib.ds4q_block_size(12) != cls.block_size:
        raise SystemExit("Q4_K block size mismatch")
    type_size = cls.type_size
    workers = max(1, min(16, (os.cpu_count() or 4)))
    pool = ThreadPoolExecutor(max_workers=workers)

    def quantize_blocks(_cls, blocks):
        src = np.ascontiguousarray(blocks, dtype="<f4")
        n, ncols = src.shape
        out = np.empty(n * type_size, dtype=np.uint8)
        lib.ds4q_quantize_init(12)
        src_p = src.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        out_p = ctypes.cast(out.ctypes.data, ctypes.c_void_p).value

        def run(lo, hi):
            return lib.ds4q_quantize_chunk(
                12,
                ctypes.cast(
                    ctypes.addressof(src_p.contents) + lo * ncols * 4,
                    ctypes.POINTER(ctypes.c_float),
                ),
                ctypes.c_void_p(out_p + lo * type_size),
                0, hi - lo, ncols, None,
            )

        if n < 4 * workers:
            written = run(0, n)
        else:
            step = (n + workers - 1) // workers
            bounds = [(i, min(i + step, n)) for i in range(0, n, step)]
            written = sum(pool.map(lambda b: run(*b), bounds))
        if written != out.nbytes:
            raise RuntimeError(f"native Q4_K wrote {written}, expected {out.nbytes}")
        return out.reshape(n, type_size)

    cls.quantize_blocks = classmethod(quantize_blocks)
    return library_path


class ErrAcc:
    def __init__(self) -> None:
        self.n = 0
        self.sum_sq_x = 0.0
        self.sum_sq_e = 0.0
        self.sum_dot = 0.0
        self.sum_sq_q = 0.0
        self.max_abs = 0.0
        self.hist = np.zeros(len(P99_EDGES) - 1, dtype=np.int64)
        self.finite = True

    def add(self, x: np.ndarray, xq: np.ndarray) -> None:
        if not (np.isfinite(x).all() and np.isfinite(xq).all()):
            self.finite = False
        err = xq - x
        abs_e = np.abs(err)
        self.n += int(x.size)
        self.sum_sq_x += float(np.square(x, dtype=np.float64).sum())
        self.sum_sq_e += float(np.square(err, dtype=np.float64).sum())
        self.sum_dot += float((x.astype(np.float64) * xq.astype(np.float64)).sum())
        self.sum_sq_q += float(np.square(xq, dtype=np.float64).sum())
        self.max_abs = max(self.max_abs, float(abs_e.max(initial=0.0)))
        counts, _ = np.histogram(abs_e.ravel(), bins=P99_EDGES)
        self.hist += counts

    def finish(self) -> dict:
        nmse = self.sum_sq_e / self.sum_sq_x if self.sum_sq_x else float("nan")
        rel_f = float(np.sqrt(nmse)) if nmse == nmse else float("nan")
        denom = float(np.sqrt(self.sum_sq_x * self.sum_sq_q))
        cos = self.sum_dot / denom if denom else float("nan")
        return {
            "n": self.n,
            "nmse": nmse,
            "rel_frobenius": rel_f,
            "cosine": cos,
            "cos_error": (1.0 - cos) if cos == cos else float("nan"),
            "max_abs": self.max_abs,
            "p99_abs": self._p99(),
            "finite": self.finite,
            "energy": self.sum_sq_x,
        }

    def _p99(self) -> float:
        target = 0.99 * self.n
        c = 0
        for i, k in enumerate(self.hist):
            c += int(k)
            if c >= target:
                return float(P99_EDGES[i + 1])
        return self.max_abs


def score_pair(q4: dict, q8: dict) -> dict:
    return {
        "q4_nmse": q4["nmse"],
        "q8_nmse": q8["nmse"],
        "delta_nmse": q4["nmse"] - q8["nmse"],
        "q4_rel_frobenius": q4["rel_frobenius"],
        "q8_rel_frobenius": q8["rel_frobenius"],
        "q4_cos_error": q4["cos_error"],
        "q8_cos_error": q8["cos_error"],
        "q4_max_abs": q4["max_abs"],
        "q8_max_abs": q8["max_abs"],
        "q4_p99_abs": q4["p99_abs"],
        "q8_p99_abs": q8["p99_abs"],
        "energy": q4["energy"],
        "n": q4["n"],
        "finite": q4["finite"] and q8["finite"],
    }


def packed_mismatch(a: np.ndarray, b: np.ndarray) -> dict:
    a = np.ascontiguousarray(a).view(np.uint8).reshape(-1)
    b = np.ascontiguousarray(b).view(np.uint8).reshape(-1)
    if a.shape != b.shape:
        return {"match": False, "reason": f"shape {a.shape} vs {b.shape}"}
    n = int(a.size)
    n_eq = int(np.equal(a, b).sum())
    return {
        "match": n_eq == n,
        "nbytes": n,
        "n_equal": n_eq,
        "n_mismatch": n - n_eq,
        "frac_equal": (n_eq / n) if n else 0.0,
    }


def gguf_packed(gguf, path: Path, name: str):
    r = gguf.GGUFReader(str(path))
    for t in r.tensors:
        if t.name == name:
            return np.ascontiguousarray(t.data).view(np.uint8), str(t.tensor_type)
    raise KeyError(name)


def load_layer_batches(hf_dir: Path, index: dict, layer: int, batch: int):
    import torch
    from safetensors import safe_open

    key = HF_KEY.format(L=layer)
    shard = hf_dir / index["weight_map"][key]
    with safe_open(str(shard), framework="pt") as f:
        sl = f.get_slice(key)
        shape = tuple(sl.get_shape())
        if shape != (N_EXPERTS, 2 * N_FF, N_EMBD):
            raise SystemExit(f"{key} shape {shape}, expected {(N_EXPERTS, 2 * N_FF, N_EMBD)}")
        for e0 in range(0, N_EXPERTS, batch):
            e1 = min(N_EXPERTS, e0 + batch)
            chunk = sl[e0:e1].to(dtype=torch.float32).contiguous().numpy()
            yield chunk[:, :N_FF, :], chunk[:, N_FF:, :]


def run_layer(gguf, hf_dir: Path, index: dict, layer: int, batch: int, keep_q8: bool = False) -> dict:
    g4, g8, u4, u8 = ErrAcc(), ErrAcc(), ErrAcc(), ErrAcc()
    q8_gate: list[np.ndarray] = []
    q8_up: list[np.ndarray] = []
    t0 = time.perf_counter()
    n_batches = 0
    for gate, up in load_layer_batches(hf_dir, index, layer, batch):
        n_batches += 1
        g_rows = np.ascontiguousarray(gate.reshape(-1, N_EMBD))
        u_rows = np.ascontiguousarray(up.reshape(-1, N_EMBD))
        g4p = gguf.quants.Q4_K.quantize_rows(g_rows)
        g8p = gguf.quants.Q8_0.quantize_rows(g_rows)
        u4p = gguf.quants.Q4_K.quantize_rows(u_rows)
        u8p = gguf.quants.Q8_0.quantize_rows(u_rows)
        g4.add(g_rows, gguf.quants.Q4_K.dequantize_rows(g4p).reshape(g_rows.shape))
        g8.add(g_rows, gguf.quants.Q8_0.dequantize_rows(g8p).reshape(g_rows.shape))
        u4.add(u_rows, gguf.quants.Q4_K.dequantize_rows(u4p).reshape(u_rows.shape))
        u8.add(u_rows, gguf.quants.Q8_0.dequantize_rows(u8p).reshape(u_rows.shape))
        if keep_q8:
            q8_gate.append(np.ascontiguousarray(g8p).reshape(-1, Q8_ROW_BYTES))
            q8_up.append(np.ascontiguousarray(u8p).reshape(-1, Q8_ROW_BYTES))
        del g_rows, u_rows, g4p, g8p, u4p, u8p, gate, up
    gate = score_pair(g4.finish(), g8.finish())
    up = score_pair(u4.finish(), u8.finish())
    out = {
        "layer": layer,
        "type": layer_kind(layer),
        "gate": gate,
        "up": up,
        "combined_delta_nmse": gate["delta_nmse"] + up["delta_nmse"],
        "wall_s": time.perf_counter() - t0,
        "batches": n_batches,
        "finite": gate["finite"] and up["finite"],
    }
    if keep_q8:
        out["_q8_gate"] = np.concatenate(q8_gate, axis=0).reshape(N_EXPERTS, N_FF, Q8_ROW_BYTES)
        out["_q8_up"] = np.concatenate(q8_up, axis=0).reshape(N_EXPERTS, N_FF, Q8_ROW_BYTES)
    return out


def l24_cross_check(gguf, hf_dir, index, l24_gguf: Path, batch: int) -> dict:
    rec = run_layer(gguf, hf_dir, index, 24, batch, keep_q8=True)
    g_gguf, g_ty = gguf_packed(gguf, l24_gguf, "blk.24.ffn_gate_exps.weight")
    u_gguf, u_ty = gguf_packed(gguf, l24_gguf, "blk.24.ffn_up_exps.weight")
    gate_cmp = packed_mismatch(rec.pop("_q8_gate"), g_gguf)
    up_cmp = packed_mismatch(rec.pop("_q8_up"), u_gguf)
    rec["l24_cross_check"] = {
        "gguf": str(l24_gguf),
        "gate_gguf_type": g_ty,
        "up_gguf_type": u_ty,
        "gate_bytes": gate_cmp,
        "up_bytes": up_cmp,
        "pass": bool(gate_cmp["match"] and up_cmp["match"]),
    }
    return rec


def q4_control_check(gguf, hf_dir, index, ctrl_gguf: Path, batch: int, layer: int = 0) -> dict:
    parts = []
    for gate, _up in load_layer_batches(hf_dir, index, layer, batch):
        rows = np.ascontiguousarray(gate.reshape(-1, N_EMBD))
        packed = gguf.quants.Q4_K.quantize_rows(rows)
        parts.append(np.ascontiguousarray(packed).reshape(-1, Q4_ROW_BYTES))
        del rows, packed, gate
    ours = np.concatenate(parts, axis=0).reshape(N_EXPERTS, N_FF, Q4_ROW_BYTES)
    theirs, ty = gguf_packed(gguf, ctrl_gguf, f"blk.{layer}.ffn_gate_exps.weight")
    cmp = packed_mismatch(ours, theirs)
    return {
        "layer": layer,
        "gguf_type": ty,
        "bytes": cmp,
        "pass": bool(cmp["match"]),
        "gguf": str(ctrl_gguf),
    }


def write_outputs(out_dir: Path, manifest: dict, layers: list[dict]) -> None:
    ranked = sorted(layers, key=lambda r: r["combined_delta_nmse"], reverse=True)
    for i, r in enumerate(ranked, 1):
        r["rank"] = i
    recon = {
        "label": "weight-reconstruction sensitivity proxy — not activation sensitivity",
        "manifest": manifest,
        "layers": layers,
        "rankings": {
            "overall_top16": [r["layer"] for r in ranked[:16]],
            "gdn_top": [r["layer"] for r in ranked if r["type"] == "GDN"],
            "qsa_top": [r["layer"] for r in ranked if r["type"] == "QSA"],
        },
    }
    (out_dir / "p2-reconstruction.json").write_text(json.dumps(recon, indent=2) + "\n")

    cols = [
        "rank", "layer", "type", "combined_delta_nmse",
        "gate_delta_nmse", "up_delta_nmse",
        "gate_q4_nmse", "gate_q8_nmse", "up_q4_nmse", "up_q8_nmse",
        "gate_q4_cos_error", "gate_q8_cos_error",
        "up_q4_cos_error", "up_q8_cos_error",
        "gate_q4_rel_frobenius", "gate_q8_rel_frobenius",
        "finite",
    ]
    lines = [",".join(cols)]
    for r in ranked:
        g, u = r["gate"], r["up"]
        lines.append(",".join([
            str(r["rank"]), str(r["layer"]), r["type"],
            f"{r['combined_delta_nmse']:.10g}",
            f"{g['delta_nmse']:.10g}", f"{u['delta_nmse']:.10g}",
            f"{g['q4_nmse']:.10g}", f"{g['q8_nmse']:.10g}",
            f"{u['q4_nmse']:.10g}", f"{u['q8_nmse']:.10g}",
            f"{g['q4_cos_error']:.10g}", f"{g['q8_cos_error']:.10g}",
            f"{u['q4_cos_error']:.10g}", f"{u['q8_cos_error']:.10g}",
            f"{g['q4_rel_frobenius']:.10g}", f"{g['q8_rel_frobenius']:.10g}",
            str(r["finite"]).lower(),
        ]))
    (out_dir / "p2-ranking.csv").write_text("\n".join(lines) + "\n")

    gdn = [r for r in ranked if r["type"] == "GDN"]
    qsa = [r for r in ranked if r["type"] == "QSA"]
    deltas = [r["combined_delta_nmse"] for r in ranked]
    md = [
        "# P2 weight-reconstruction sensitivity proxy",
        "",
        "Not activation sensitivity. No imatrix. Ranking is ΔNMSE of Q4_K vs Q8_0",
        "on the uncensored BF16 gate/up tensors; gate and up are normalized separately, then summed.",
        "",
        f"- source: `{manifest['source']['path']}`",
        f"- revision: `{manifest['source']['revision']}`",
        f"- Q4_K encoder: `{manifest['encoders']['q4_k']}`",
        f"- Q8_0 encoder: `{manifest['encoders']['q8_0']}`",
        f"- dequant: `{manifest['encoders']['dequant']}`",
        f"- L24 Q8 byte match: **{manifest['l24_cross_check']['pass']}**",
        f"- Q4_K control byte match (blk.{manifest['q4_control_check']['layer']}): **{manifest['q4_control_check']['pass']}**",
        f"- layers: {len(layers)}/48  tensors: {2 * len(layers)}/96  finite: {all(r['finite'] for r in layers)}",
        "",
        "## Rankings",
        "",
        "### Overall top 16",
        "",
        "| rank | layer | type | combined ΔNMSE | gate ΔNMSE | up ΔNMSE |",
        "| ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for r in ranked[:16]:
        md.append(
            f"| {r['rank']} | {r['layer']} | {r['type']} | "
            f"{r['combined_delta_nmse']:.6g} | {r['gate']['delta_nmse']:.6g} | {r['up']['delta_nmse']:.6g} |"
        )
    md += [
        "",
        "### Top GDN",
        "",
        "| rank | layer | combined ΔNMSE |",
        "| ---: | ---: | ---: |",
    ]
    for r in gdn[:16]:
        md.append(f"| {r['rank']} | {r['layer']} | {r['combined_delta_nmse']:.6g} |")
    md += [
        "",
        "### Top QSA",
        "",
        "| rank | layer | combined ΔNMSE |",
        "| ---: | ---: | ---: |",
    ]
    for r in qsa:
        md.append(f"| {r['rank']} | {r['layer']} | {r['combined_delta_nmse']:.6g} |")
    md += ["", "## Pattern", ""]
    if gdn and qsa:
        md.append(f"- GDN n={len(gdn)} median ΔNMSE {float(np.median([r['combined_delta_nmse'] for r in gdn])):.6g}")
        md.append(f"- QSA n={len(qsa)} median ΔNMSE {float(np.median([r['combined_delta_nmse'] for r in qsa])):.6g}")
        md.append(
            f"- overall min/median/max {min(deltas):.6g} / {float(np.median(deltas)):.6g} / {max(deltas):.6g}"
        )
        md.append(f"- top-16 composition: {dict(Counter(r['type'] for r in ranked[:16]))}")
    md += [
        "",
        "P3 should inspect this curve before building full GGUFs. Prefer a knee plus",
        "controls (median, bottom, matched GDN/QSA), not automatic top-16 converts.",
        "",
    ]
    (out_dir / "p2-summary.md").write_text("\n".join(md))


def main() -> None:
    here = Path(__file__).resolve().parent
    tools = here.parents[1] / "gguf-tools"
    sys.path.insert(0, str(tools))
    from qwen4_exp_convert import parse_layer_spec

    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", default=os.path.expanduser("~/models/hf/Qwen3.8-Flash-Next-Uncensored"))
    ap.add_argument("--llama-cpp", default=os.path.expanduser("~/bin/llama.cpp"))
    ap.add_argument("--quants-library", default=str(tools / "libds4quants.dylib"))
    ap.add_argument(
        "--l24-gguf",
        default=os.path.expanduser(
            "~/models/gguf/Qwen3.8-Flash-Next-Uncensored-DS4-Q4KGateUp-Q8_0Down-MTP-L24Q8GU.gguf"
        ),
    )
    ap.add_argument(
        "--control-gguf",
        default=os.path.expanduser(
            "~/models/gguf/Qwen3.8-Flash-Next-Uncensored-DS4-Q4KGateUp-Q8_0Down-MTP.gguf"
        ),
    )
    ap.add_argument("--out-dir", default=str(here))
    ap.add_argument("--batch-experts", type=int, default=8)
    ap.add_argument("--layers", default="0-47")
    args = ap.parse_args()

    hf_dir = Path(args.hf)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(Path(args.llama_cpp) / "gguf-py"))
    sys.path.insert(0, args.llama_cpp)
    import gguf

    index = json.loads((hf_dir / "model.safetensors.index.json").read_text())
    source = resolve_hf_revision(hf_dir)
    layers_wanted = sorted(parse_layer_spec(args.layers))
    lib_path = install_native_q4k(gguf, args.quants_library)

    print("L24 Q8 cross-check...", flush=True)
    l24 = l24_cross_check(gguf, hf_dir, index, Path(args.l24_gguf), args.batch_experts)
    print("Q4_K control blk.0 cross-check...", flush=True)
    q4_ctrl = q4_control_check(gguf, hf_dir, index, Path(args.control_gguf), args.batch_experts, 0)
    print(
        json.dumps(
            {
                "l24_pass": l24["l24_cross_check"]["pass"],
                "l24_gate": l24["l24_cross_check"]["gate_bytes"],
                "l24_up": l24["l24_cross_check"]["up_bytes"],
                "q4_ctrl_pass": q4_ctrl["pass"],
                "q4_ctrl": q4_ctrl["bytes"],
            },
            indent=2,
        ),
        flush=True,
    )
    if not l24["l24_cross_check"]["pass"]:
        raise SystemExit("P2 L24 Q8 byte cross-check FAILED")
    if not q4_ctrl["pass"]:
        raise SystemExit("P2 Q4_K native encoder does not match control GGUF blk.0")

    manifest = {
        "source": source,
        "encoders": {
            "q4_k": f"libds4quants Q4_K ({lib_path}) — production convert path",
            "q8_0": "gguf-py Q8_0 (bit-exact ggml-quants.c; production convert path — convert does not patch Q8_0)",
            "dequant": "llama.cpp gguf-py dequantize_blocks (ggml decoder)",
        },
        "l24_cross_check": l24["l24_cross_check"],
        "q4_control_check": q4_ctrl,
        "batch_experts": args.batch_experts,
        "metric": "combined_delta_nmse = (NMSE_Q4-NMSE_Q8)_gate + (NMSE_Q4-NMSE_Q8)_up",
        "label": "weight-reconstruction sensitivity proxy",
    }

    results: list[dict] = []
    partial = out_dir / "p2-reconstruction.partial.jsonl"
    partial.write_text("")
    if 24 in layers_wanted:
        slim = {k: v for k, v in l24.items() if k != "l24_cross_check"}
        results.append(slim)
        partial.write_text(json.dumps(slim) + "\n")
        print(
            f"layer 24 GDN ΔNMSE={slim['combined_delta_nmse']:.8g} (from cross-check)",
            flush=True,
        )

    for layer in layers_wanted:
        if layer == 24:
            continue
        rec = run_layer(gguf, hf_dir, index, layer, args.batch_experts)
        results.append(rec)
        with partial.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(
            f"layer {layer:02d} {rec['type']} ΔNMSE={rec['combined_delta_nmse']:.8g} "
            f"gate={rec['gate']['delta_nmse']:.8g} up={rec['up']['delta_nmse']:.8g} "
            f"{rec['wall_s']:.1f}s finite={rec['finite']}",
            flush=True,
        )

    results.sort(key=lambda r: r["layer"])
    if len(results) != len(layers_wanted):
        raise SystemExit(f"expected {len(layers_wanted)} layers, got {len(results)}")
    if not all(r["finite"] for r in results):
        raise SystemExit("NaN/Inf in reconstruction stats")
    if any(r["gate"]["n"] != N_EXPERTS * N_FF * N_EMBD for r in results):
        raise SystemExit("gate element count mismatch")
    if any(r["up"]["n"] != N_EXPERTS * N_FF * N_EMBD for r in results):
        raise SystemExit("up element count mismatch")

    write_outputs(out_dir, manifest, results)
    print("WROTE", out_dir / "p2-reconstruction.json")
    print("WROTE", out_dir / "p2-ranking.csv")
    print("WROTE", out_dir / "p2-summary.md")


if __name__ == "__main__":
    main()
