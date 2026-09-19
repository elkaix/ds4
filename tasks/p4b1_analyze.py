#!/usr/bin/env python3
"""Summarize tasks/p4b1_bench.sh output: per-frontier paired A/B gains with a
paired bootstrap CI (unit = run pair), plus frontier-logit equivalence across
arms and across repeated runs.  Usage: tasks/p4b1_analyze.py [bench-dir]"""
import csv, glob, json, os, random, statistics as st, sys
d = sys.argv[1] if len(sys.argv) > 1 else "tasks/data/p4b1/bench"
rows = {}
for f in sorted(glob.glob(f"{d}/*.csv")):
    ctx, arm, run = os.path.basename(f)[:-4].split("-")
    r = list(csv.DictReader(open(f)))
    if not r: continue
    rows.setdefault(int(ctx), {}).setdefault(arm, {})[int(run)] = {k: float(v) for k, v in r[-1].items()}
random.seed(1)
def ci(diffs, n=20000):
    if len(diffs) < 2: return (float("nan"), float("nan"))
    bs = sorted(st.mean(random.choices(diffs, k=len(diffs))) for _ in range(n))
    return bs[int(0.025 * n)], bs[int(0.975 * n)]
print(f"{'ctx':>7} {'metric':>16} {'A med':>9} {'B med':>9} {'gain%':>7} {'95% CI (pp)':>18} n")
for ctx in sorted(rows):
    A, B = rows[ctx].get("A", {}), rows[ctx].get("B", {})
    pairs = sorted(set(A) & set(B))
    for m in ("gen_steady_tps", "prefill_tps", "gen_first_ms"):
        a = [A[i][m] for i in pairs]; b = [B[i][m] for i in pairs]
        if not a: continue
        diffs = [100.0 * (y - x) / x for x, y in zip(a, b)]
        lo, hi = ci(diffs)
        print(f"{ctx:>7} {m:>16} {st.median(a):9.2f} {st.median(b):9.2f} {st.mean(diffs):7.2f} [{lo:7.2f},{hi:7.2f}] {len(pairs)}")
# logits
import math
def load(p): 
    j = json.load(open(p)); return j["logits"], j["argmax_id"]
for ctx in sorted(rows):
    files = {}
    for dd in glob.glob(f"{d}/logits-{ctx}-*"):
        fs = sorted(glob.glob(f"{dd}/frontier_*.json"))
        if fs: files[os.path.basename(dd)[len("logits-"):]] = fs[-1]
    if not files: continue
    ref_tag = sorted(t for t in files if "-A-" in t)[:1]
    if not ref_tag: continue
    ref, ref_arg = load(files[ref_tag[0]])
    worst = {"A": 0.0, "B": 0.0}; argm = {"A": True, "B": True}
    for t, p in files.items():
        if t == ref_tag[0]: continue
        l, arg = load(p); arm = t.split("-")[1]
        worst[arm] = max(worst[arm], max(abs(x - y) for x, y in zip(ref, l)))
        argm[arm] &= (arg == ref_arg)
    print(f"logits ctx={ctx}: vs {ref_tag[0]}: A-runs max_abs={worst['A']:.3g} argmax_same={argm['A']}; "
          f"B-runs max_abs={worst['B']:.3g} argmax_same={argm['B']}")
