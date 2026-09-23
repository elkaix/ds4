#!/usr/bin/env python3
"""Summarize tasks/glm_1029_arms.sh output: per point, arm means and paired
per-cycle ratios vs arm A with a bootstrap CI95."""
import csv, random, statistics, sys
from pathlib import Path

out = Path(sys.argv[1])
POINTS = ["cold8k", "cold16k", "cold24k", "cont14k"]
rows = {}  # (point, arm, cycle) -> list of (prefill_tps, steady_tps)
for f in sorted(out.glob("*.csv")):
    point, arm, cycle, _slot = f.stem.split("-")
    data = list(csv.DictReader(f.open()))
    if not data:
        continue
    r = data[-1]  # cold: the only row; cont: the 2K continued row
    rows.setdefault((point, arm, int(cycle)), []).append(
        (float(r["prefill_tps"]), float(r["gen_steady_tps"])))

def boot(xs, n=10000):
    rnd = random.Random(0)
    ms = sorted(statistics.fmean(rnd.choices(xs, k=len(xs))) for _ in range(n))
    return ms[int(0.025 * n)], ms[int(0.975 * n)]

n_cycles = len({c for (_p, a, c) in rows if a == "A"})
if n_cycles < 3:
    print(f"WARNING: {n_cycles} cycle(s); bootstrap CIs over n<3 ratios are not "
          "confidence intervals. Read the raw runs.")

for metric, idx in (("prefill t/s", 0), ("decode steady t/s", 1)):
    print(f"\n{metric}")
    print(f"{'point':8} {'A base':>14} {'B dense':>14} {'C valid':>14}  {'B/A [CI95]':>22}  {'C/A [CI95]':>22}")
    for p in POINTS:
        cycles = sorted({c for (q, a, c) in rows if q == p and a == "A"})
        cell, ratio = {}, {}
        for a in "ABC":
            vals = [v[idx] for c in cycles for v in rows.get((p, a, c), [])]
            cell[a] = (f"{statistics.fmean(vals):7.2f}±{statistics.stdev(vals):4.2f}"
                       if len(vals) > 1 else "-")
        for a in "BC":
            rs = [statistics.fmean(v[idx] for v in rows[(p, a, c)]) /
                  statistics.fmean(v[idx] for v in rows[(p, "A", c)])
                  for c in cycles if (p, a, c) in rows]
            if rs:
                lo, hi = boot(rs)
                ratio[a] = f"x{statistics.fmean(rs):.4f} [{lo:.3f},{hi:.3f}] n={len(rs)}"
            else:
                ratio[a] = "-"
        print(f"{p:8} {cell['A']:>14} {cell['B']:>14} {cell['C']:>14}  {ratio['B']:>22}  {ratio['C']:>22}")
