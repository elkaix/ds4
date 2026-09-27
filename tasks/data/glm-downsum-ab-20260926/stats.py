#!/usr/bin/env python3
"""Paired B/A analysis for the DOWNSUM8 A/B: per-cycle ratio of arm means
(B=down+sum, A=generic), steady t/s. ABBA order, drift absorbed by pairing."""
import csv, glob, re, statistics as st, sys

D = sys.argv[1] if len(sys.argv) > 1 else '.'

def steady(f):
    vals = []
    for r in csv.DictReader(open(f)):
        v = r.get('gen_steady_tps')
        try: vals.append(float(v))
        except (ValueError, TypeError): pass
    return vals[0] if vals else None

def first_ms(f):
    vals = []
    for r in csv.DictReader(open(f)):
        v = r.get('gen_first_ms')
        try: vals.append(float(v))
        except (ValueError, TypeError): pass
    return vals[0] if vals else None

def cycles(pat):
    out = {}
    for f in glob.glob(f'{D}/{pat}'):
        m = re.search(r'-(\d+)-(\d+)\.csv$', f)
        if m: out[(int(m.group(1)), int(m.group(2)))] = f
    return out

A, B = cycles('A-*'), cycles('B-*')
ratios, firsts = [], []
for c in sorted({k[0] for k in A} & {k[0] for k in B}):
    a = [steady(A[(c, i)]) for i in (1, 2) if (c, i) in A]
    b = [steady(B[(c, i)]) for i in (1, 2) if (c, i) in B]
    if a and b:
        am, bm = st.mean(a), st.mean(b)
        ratios.append(bm / am)
        firsts.append((c, [first_ms(A[(c, i)]) for i in (1, 2)], [first_ms(B[(c, i)]) for i in (1, 2)]))
        print(f'cycle {c}: A={am:.2f} B={bm:.2f} ratio={bm/am:.4f}')
if ratios:
    m = st.mean(ratios)
    print(f'\npaired mean B/A = x{m:.4f}  (n={len(ratios)} cycles)')
    print(f'verdict guide: >1.03 candidate, 0.97-1.03 NULL band (drift spread ~8%)')
for c, fa, fb in firsts:
    print(f'gen_first_ms c{c}: A={fa} B={fb}')
