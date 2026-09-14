#!/usr/bin/env python3
"""Same-condition bracket: 2K, 60K, 2K in one process.

The first F14 comparison put 60K rep0 on a fresh post-restart machine
(free=89 MB, the worst reading of that series) against a 2K rep0 taken after
that series with free climbing. That compares two different points on the
pressure trajectory, so the 1.5% it produced is not a clean context contrast.
Flanking the 60K block with 2K blocks makes the drift estimable instead:
the two 2K blocks bound whatever moved during the 60K block.
"""
import os, sys, json, re, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sweep as S, warm_repeat as W

def decode_from_log(off):
    """The server's own decoding line. Wall clock at 60K is 95% re-prefill."""
    with open(W.__dict__.get("HERE", HERE) + "/server.out", "rb") as f:
        f.seek(off); seg = f.read().decode(errors="replace")
    tps = re.findall(r"decoding chunk=[\d.]+ t/s avg=([\d.]+) t/s ([\d.]+)s", seg)
    return (float(tps[-1][0]), float(tps[-1][1])) if tps else (None, None)

def block(label, target, reps, cpt):
    corpus = open(S.CORPUS, encoding="utf-8", errors="replace").read()
    nchars = int(target * cpt)
    r, _, _ = S.chat(S.INSTRUCTION + corpus[:nchars], max_tokens=1)
    ptok = r["usage"]["prompt_tokens"]
    prompt = S.INSTRUCTION + corpus[:nchars]
    out = []
    for i in range(reps):
        v0, off = W.vmstat(), S.server_log_size()
        r, t0, t1 = S.chat(prompt)
        tps, ds = decode_from_log(off)
        v1 = W.vmstat()
        rec = {"label": label, "ptok": ptok, "rep": i, "decode_tps": tps,
               "decode_s": ds, "wall_s": round(t1 - t0, 2),
               "free_mb": v1.get("free", 0) * W.PAGE // 1048576,
               "d_pageins": v1.get("pageins", 0) - v0.get("pageins", 0)}
        out.append(rec)
        print(f"  {label:>6} rep{i} decode={tps:6.2f} t/s ({ds:5.2f}s) "
              f"wall={rec['wall_s']:7.2f}s free={rec['free_mb']:>5}MB "
              f"pagein={rec['d_pageins']}", flush=True)
    return ptok, out

res = []
for label, target, cpt in [("2k-pre", 2048, 3.0), ("60k", 60000, 3.647), ("2k-post", 2048, 3.0)]:
    ptok, o = block(label, target, 3, cpt)
    print(f"# {label}: {ptok} tok", flush=True)
    res += o
json.dump(res, open(HERE + "/bracket.json", "w"), indent=1)

def med(l): 
    s = sorted(l); return s[len(s)//2]
pre  = med([x["decode_tps"] for x in res if x["label"] == "2k-pre"])
mid  = med([x["decode_tps"] for x in res if x["label"] == "60k"])
post = med([x["decode_tps"] for x in res if x["label"] == "2k-post"])
ref = (pre + post) / 2.0
print(f"\n2K before {pre:.2f} | 60K {mid:.2f} | 2K after {post:.2f}")
print(f"drift across the 60K block (2K after / 2K before): {100*(post-pre)/pre:+.1f}%")
print(f"60K vs drift-corrected 2K reference {ref:.2f}: {100*(mid-ref)/ref:+.1f}%")
