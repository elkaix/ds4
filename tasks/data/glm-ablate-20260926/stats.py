#!/usr/bin/env python3
"""Stage-2 2K attribution stats: base vs ablated (routed/shared), plus MoE
stage profiler summaries from prof-l5f/prof-l20f logs."""
import csv, glob, re, statistics as st, sys

D = sys.argv[1] if len(sys.argv) > 1 else '.'

def col(f, name):
    out = []
    try:
        for r in csv.DictReader(open(f)):
            v = r.get(name)
            if v is None:  # repeated header row
                continue
            try: out.append(float(v))
            except ValueError: pass
    except FileNotFoundError: pass
    return out

def arm(pat):
    return [x for f in sorted(glob.glob(f'{D}/{pat}')) for x in col(f, 'gen_steady_tps')]

base, rout, shar = arm('base-*.csv'), arm('abl-routed-*.csv'), arm('abl-shared-*.csv')
def ms(x): return 1000.0 / x
if base and rout:
    b, r = st.mean(base), st.mean(rout)
    print(f'base    n={len(base)} {base} mean {b:.2f} t/s ({ms(b):.2f} ms/tok)')
    print(f'routed  n={len(rout)} {rout} mean {r:.2f} t/s ({ms(r):.2f} ms/tok)')
    print(f'ROUTED stage = {ms(b)-ms(r):.2f} ms/token ({(ms(b)-ms(r))/ms(b)*100:.1f}%)')
    print(f'drift check: base spread {min(base):.2f}-{max(base):.2f} ({(max(base)-min(base))/b*100:.1f}%)')
if base and shar:
    b, s = st.mean(base), st.mean(shar)
    print(f'shared  n={len(shar)} {shar} mean {s:.2f} t/s')
    print(f'SHARED stage = {ms(b)-ms(s):.2f} ms/token ({(ms(b)-ms(s))/ms(b)*100:.1f}%)')
for tag in ('prof-l5f', 'prof-l20f'):
    stages = {}
    for line in open(f'{D}/{tag}.log', errors='replace'):
        m = re.search(r'gate=(\S+) down=(\S+) path=(\S+) (\w+)=(\d+\.\d+) ms', line)
        if m:
            stages.setdefault((m.group(1), m.group(2), m.group(3), m.group(4)), []).append(float(m.group(5)))
    if stages:
        print(f'--- {tag} ---')
        for k, v in sorted(stages.items()):
            print(f'  {k}: n={len(v)} mean {st.mean(v):.3f} ms')
