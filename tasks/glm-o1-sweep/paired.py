#!/usr/bin/env python3
"""Paired per-case comparison of sweep arms on the shared 100-case fixture.

usage: paired.py [--boot N] [--expect-n 100] [--json OUT] REF ARM [ARM...]
Names are tsv basenames in this dir (e.g. base-q2 t14-16-19). Rows are joined by case id.
delta_i = avg_nll(ARM,i) - avg_nll(REF,i):  positive = regression, negative = improvement.
Reports mean delta with a paired bootstrap 95% CI, median, improved/regressed/unchanged
counts, signed P90/P95 of the deltas, worst regression (max delta), first_match, avg_lcp.
Fails loudly on id-set mismatch, duplicates, wrong case count, or non-finite values."""
import argparse, csv, json, math, os, random, statistics as st, sys
D = os.path.dirname(os.path.abspath(__file__))

def load(name, expect_n):
    path = os.path.join(D, name + ".tsv")
    rows = {}
    with open(path) as f:
        for x in csv.DictReader(f, delimiter="\t"):
            cid = x["id"]
            if cid in rows: sys.exit(f"{name}: duplicate case id {cid}")
            v = float(x["avg_nll"])
            if not math.isfinite(v): sys.exit(f"{name}: non-finite avg_nll for {cid}")
            rows[cid] = x
    if len(rows) != expect_n: sys.exit(f"{name}: {len(rows)} cases, expected {expect_n}")
    return rows

def pct(sorted_vals, p):
    k = (len(sorted_vals) - 1) * p; lo = math.floor(k); hi = math.ceil(k)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)

def compare(ref, arm, boot, rng):
    if set(ref) != set(arm): sys.exit(f"case id sets differ: {sorted(set(ref) ^ set(arm))[:5]}")
    ids = sorted(ref)
    d = [float(arm[i]["avg_nll"]) - float(ref[i]["avg_nll"]) for i in ids]
    n = len(d)
    means = sorted(st.mean(rng.choices(d, k=n)) for _ in range(boot))
    sd = sorted(d)
    return dict(
        n=n, mean=st.mean(d), ci_lo=pct(means, 0.025), ci_hi=pct(means, 0.975),
        median=st.median(d),
        improved=sum(x < 0 for x in d), regressed=sum(x > 0 for x in d), unchanged=sum(x == 0 for x in d),
        p90=pct(sd, 0.90), p95=pct(sd, 0.95), worst=max(d), worst5=st.mean(sd[-5:]),
        first_match=sum(int(arm[i]["first_match"]) for i in ids),
        avg_lcp=st.mean(int(arm[i]["greedy_lcp"]) for i in ids),
    )

ap = argparse.ArgumentParser()
ap.add_argument("--boot", type=int, default=10000)
ap.add_argument("--expect-n", type=int, default=100)
ap.add_argument("--seed", type=int, default=20260913)
ap.add_argument("--json", help="also write results as JSON")
ap.add_argument("ref"); ap.add_argument("arms", nargs="+")
a = ap.parse_args()
rng = random.Random(a.seed)
ref = load(a.ref, a.expect_n)
out = {}
print(f"ref={a.ref}  boot={a.boot}  (delta>0 = regression vs ref)")
print(f"{'arm':16} {'mean':>8} {'ci95_lo':>8} {'ci95_hi':>8} {'median':>8} {'imp':>3} {'reg':>3} {'unc':>3} "
      f"{'p90':>8} {'p95':>8} {'worst5':>8} {'worst':>8} {'fm':>3} {'lcp':>5}")
for name in a.arms:
    s = compare(ref, load(name, a.expect_n), a.boot, rng)
    out[name] = s
    print(f"{name:16} {s['mean']:+8.4f} {s['ci_lo']:+8.4f} {s['ci_hi']:+8.4f} {s['median']:+8.4f} "
          f"{s['improved']:3d} {s['regressed']:3d} {s['unchanged']:3d} "
          f"{s['p90']:+8.4f} {s['p95']:+8.4f} {s['worst5']:+8.4f} {s['worst']:+8.4f} {s['first_match']:3d} {s['avg_lcp']:5.2f}")
if a.json:
    with open(a.json, "w") as f: json.dump({"ref": a.ref, "arms": out}, f, indent=1)
