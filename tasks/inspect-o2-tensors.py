#!/usr/bin/env python3
"""Gate 1 for O2b: every tensor matches the Q2 base in name, shape and type, except the
30 routed-expert tensors of layers 33-42, which must be q4_K. Also confirms O1's layer
37-42 expert tensors and O2b's are byte-identical in size/type (same Q4_K recipe).

usage: tasks/inspect-o2-tensors.py BASE O1 O2
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gguf-tools" / "mixed"))
from splice_mixed_expert_layers_gguf import EXPERT_TENSOR_RE, parse_gguf  # noqa: E402

GGML_Q4_K = 12
Q4_LAYERS = set(range(33, 43))


def main() -> int:
    base, o1, o2 = (parse_gguf(Path(p)) for p in sys.argv[1:4])
    errors: list[str] = []
    if len(base.tensors) != len(o2.tensors):
        errors.append(f"tensor count base={len(base.tensors)} o2={len(o2.tensors)}")
    upgraded = 0
    for bt in base.tensors:
        t = o2.tensor_by_name.get(bt.name)
        if t is None:
            errors.append(f"missing in o2: {bt.name}")
            continue
        if t.dims != bt.dims:
            errors.append(f"shape {bt.name}: base={bt.dims} o2={t.dims}")
        m = EXPERT_TENSOR_RE.match(bt.name)
        if m and int(m.group(1)) in Q4_LAYERS:
            upgraded += 1
            if t.ggml_type != GGML_Q4_K:
                errors.append(f"{bt.name}: expected q4_K (12), got {t.ggml_type}")
            o1t = o1.tensor_by_name[bt.name]
            if int(m.group(1)) >= 37 and o1t.ggml_type != t.ggml_type:
                errors.append(f"{bt.name}: o1 type {o1t.ggml_type} != o2 type {t.ggml_type}")
        elif t.ggml_type != bt.ggml_type:
            errors.append(f"unintended type change {bt.name}: base={bt.ggml_type} o2={t.ggml_type}")
    print(f"tensors={len(o2.tensors)} upgraded_q4k={upgraded} errors={len(errors)}")
    for e in errors[:50]:
        print("!!", e)
    ok = not errors and upgraded == 30
    print("GATE1_PASS" if ok else "GATE1_FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
