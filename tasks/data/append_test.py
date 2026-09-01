#!/usr/bin/env python3
"""Does ordinary multi-turn chat hit the F16 re-prefill, or only re-sends?

F16's scope claim -- "normal chat appends, extends the live frontier, and never
triggers the failed rewind" -- was read off live_prefix_rewind_target, not
observed. It decides whether F16 is a P0 or a benchmark-harness footnote, so
test it directly.

Turn 1: [user P]                       establishes a 60K live context
Turn 2: [user P, assistant C, user Q]  a strict EXTENSION of turn 1's frontier

If turn 2 prefills only its delta, the scope claim holds. If turn 2 also
re-prefills ~60K, F16 is far larger than currently framed.
"""
import os, sys, json, re, subprocess, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sweep as S

def post(messages, max_tokens=64):
    body = {"model": S.MODEL, "messages": messages, "temperature": 0,
            "ignore_eos": True, "max_tokens": max_tokens,
            "reasoning_effort": "none", "stream": False}
    t0 = time.time()
    out = subprocess.run(["curl", "-s", "--max-time", "3600",
                          "-H", "Content-Type: application/json",
                          "-d", json.dumps(body),
                          S.BASE + "/v1/chat/completions"],
                         capture_output=True, text=True).stdout
    return json.loads(out), t0, time.time()

def logseg(off):
    with open(HERE + "/server.out", "rb") as f:
        f.seek(off); return f.read().decode(errors="replace")

def report(name, seg, wall, usage):
    tps = re.findall(r"decoding chunk=[\d.]+ t/s avg=([\d.]+) t/s ([\d.]+)s", seg)
    dec_tps, dec_s = (float(tps[-1][0]), float(tps[-1][1])) if tps else (None, 0.0)
    rew = re.findall(r"rewound GLM live prefix from (\d+) to (\d+)", seg)
    chunks = re.findall(r"prefill chunk (\d+)/(\d+) \([\d.]+%\) chunk=([\d.]+) t/s avg=[\d.]+ t/s ([\d.]+)s", seg)
    pf_s = float(chunks[-1][3]) if chunks else 0.0
    print(f"\n--- {name} ---")
    print(f"  prompt_tokens={usage['prompt_tokens']} completion={usage['completion_tokens']}")
    print(f"  wall={wall:.1f}s  decode={dec_s:.2f}s @ {dec_tps} t/s  prefill_phase={pf_s:.1f}s")
    print(f"  non-decode = {wall - dec_s:.1f}s ({100*(wall-dec_s)/wall:.1f}% of wall)")
    print(f"  rewind lines: {rew if rew else 'none'}")
    print(f"  last prefill chunk line: {chunks[-1] if chunks else 'none'}")
    return wall - dec_s

corpus = open(S.CORPUS, encoding="utf-8", errors="replace").read()
prompt = S.INSTRUCTION + corpus[:int(60000 * 3.647)]

off = S.server_log_size()
r1, t0, t1 = post([{"role": "user", "content": prompt}], max_tokens=256)
nd1 = report("turn 1 (establish 60K live context)", logseg(off), t1 - t0, r1["usage"])
reply = r1["choices"][0]["message"]["content"]

off = S.server_log_size()
r2, t0, t1 = post([{"role": "user", "content": prompt},
                   {"role": "assistant", "content": reply},
                   {"role": "user", "content": "Summarise that in one sentence."}],
                  max_tokens=64)
nd2 = report("turn 2 (APPEND: extends the frontier)", logseg(off), t1 - t0, r2["usage"])

off = S.server_log_size()
r3, t0, t1 = post([{"role": "user", "content": prompt}], max_tokens=256)
nd3 = report("turn 3 (RE-SEND the original prompt: truncates)", logseg(off), t1 - t0, r3["usage"])

print(f"\n=== VERDICT ===")
print(f"append non-decode   {nd2:7.1f}s")
print(f"re-send non-decode  {nd3:7.1f}s")
if nd2 < nd3 * 0.25:
    print("SCOPE CLAIM HOLDS: appending extends the frontier and skips the re-prefill.")
else:
    print("SCOPE CLAIM FAILS: ordinary appending ALSO re-prefills. F16 is larger than framed.")
