#!/usr/bin/env python3
"""Compare two DS4_BENCH_DUMP_GEN_LOGITS_DIR dumps (tokens.txt + step_*.f32)."""
from __future__ import annotations
import array
import math
import sys
from pathlib import Path


def load_tokens(d: Path) -> list[int]:
    return [int(x) for x in (d / "tokens.txt").read_text().splitlines() if x.strip()]


def load_f32(path: Path) -> array.array:
    data = array.array("f")
    with path.open("rb") as f:
        data.fromfile(f, path.stat().st_size // 4)
    return data


def main() -> int:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} dirA dirB", file=sys.stderr)
        return 2
    a_dir, b_dir = Path(sys.argv[1]), Path(sys.argv[2])
    ta, tb = load_tokens(a_dir), load_tokens(b_dir)
    print(f"tokens A={len(ta)} B={len(tb)}")
    if ta != tb:
        n = min(len(ta), len(tb))
        first = next((i for i in range(n) if ta[i] != tb[i]), None)
        print(f"TOKEN_MISMATCH first_index={first} A={ta[first] if first is not None else None} "
              f"B={tb[first] if first is not None else None}")
        print("DECODE_TRACE_FAIL tokens")
        return 1
    print("tokens identical")

    steps = sorted(a_dir.glob("step_*.f32"))
    if not steps:
        print("DECODE_TRACE_FAIL no step_*.f32 in A")
        return 1
    if len(steps) != len(ta):
        print(f"WARN step files {len(steps)} != tokens {len(ta)}")

    worst_abs = 0.0
    worst_step = -1
    n_diff = 0
    n_nan = 0
    for i, pa in enumerate(steps):
        pb = b_dir / pa.name
        if not pb.exists():
            print(f"DECODE_TRACE_FAIL missing {pb.name} in B")
            return 1
        fa, fb = load_f32(pa), load_f32(pb)
        if len(fa) != len(fb):
            print(f"DECODE_TRACE_FAIL vocab_len step={i} A={len(fa)} B={len(fb)}")
            return 1
        for j, (x, y) in enumerate(zip(fa, fb)):
            if not (math.isfinite(x) and math.isfinite(y)):
                if math.isfinite(x) != math.isfinite(y) or x != y:
                    n_nan += 1
                    n_diff += 1
                continue
            d = abs(x - y)
            if d != 0.0:
                n_diff += 1
                if d > worst_abs:
                    worst_abs = d
                    worst_step = i
        if (i + 1) % 64 == 0:
            print(f"  compared {i+1}/{len(steps)} steps; worst_abs={worst_abs:g} at step {worst_step}")

    print(f"frames={len(steps)} vocab={len(load_f32(steps[0]))}")
    print(f"n_diff_elements={n_diff} n_nan_mismatches={n_nan}")
    print(f"worst_abs={worst_abs:.9g} worst_step={worst_step}")
    if n_diff == 0 and worst_abs == 0.0:
        print("DECODE_TRACE_PASS byte-identical F32 frames + tokens")
        return 0
    print("DECODE_TRACE_FAIL logits differ")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
