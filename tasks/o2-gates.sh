#!/usr/bin/env bash
# O2b (layers 33-42 Q4_K experts) promotion gates. Gate 2 (perplexity) is chained by
# tasks/vision-uncen-o2-build.sh. The launcher switch (gate 8) is manual, after review.
#
#   tasks/o2-gates.sh [1] [3] [4] [5] [6]      default: all, in that order
#
#   1  tensor inspection: only blk.33-42 ffn_{gate,up,down}_exps changed, all q4_K
#   3  official 100-case continuation scorer, O1 vs O2b (relative A/B: the fixture is the 0731
#      checkpoint, not orcarouter Vision-Uncensored, so absolute numbers are not release bands).
#      score_official has no --vision flag; the encoder is optional in ds4_engine_open, so both
#      arms load text-only, which is what a text continuation scorer measures anyway.
#   4  representative prompts, greedy, O1 vs O2b side by side + corruption heuristics
#   5  decode/prefill A/B via tasks/decode-ab.sh, interleaved o1 o2 o2 o1
#   6  O2b at a ~258K-token prompt on a 262144 ctx server, sampling swap + memory pressure
#      (gate 7). Run it with your normal Pi/browser/background apps open.
#
# The prod server must be stopped. Nothing else heavy may run while gates 5/6 run.
set -uo pipefail
D="$(cd "$(dirname "$0")/.." && pwd)"; cd "$D"
G="$HOME/models/gguf"
BASE="$G/DeepSeek-V4-Flash-Vision-Uncensored-orcarouter-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8.gguf"
O1="$G/DeepSeek-V4-Flash-Vision-Uncensored-orcarouter-Layers37-42Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-AProjQ8-SExpQ8-OutQ8.gguf"
O2="$G/DeepSeek-V4-Flash-Vision-Uncensored-orcarouter-Layers33-42Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-AProjQ8-SExpQ8-OutQ8.gguf"
V="$G/DeepSeek-V4-Flash-Vision-Encoder.gguf"
OUT="$D/tasks/2026-09-13-o2-gates"
KVROOT="$HOME/.ds4/o2-gates"
PORT=8009
mkdir -p "$OUT" "$KVROOT"
GATES=("$@"); [[ ${#GATES[@]} -eq 0 ]] && GATES=(1 3 4 5 6)

noserver() {
  if pgrep -x ds4-server >/dev/null || pgrep -x ds4 >/dev/null; then
    echo "!! another ds4 process is running; stop it first"; exit 1
  fi
}

SRV=""
start_server() { # $1=label $2=model
  local kv="$KVROOT/kv-$1"; rm -rf "$kv"; mkdir -p "$kv"
  ./ds4-server --chdir "$D" --metal --model "$2" --vision "$V" \
    --ctx 262144 --tokens 32768 --warm-weights --power 100 \
    --host 127.0.0.1 --port "$PORT" \
    --kv-disk-dir "$kv" --kv-disk-space-mb 8192 --kv-cache-min-tokens 2048 \
    > "$OUT/$1.server.log" 2>&1 &
  SRV=$!
  for _ in $(seq 1 200); do
    curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { grep -m1 'memory:.*planned' "$OUT/$1.server.log"; return 0; }
    kill -0 "$SRV" 2>/dev/null || { echo "!! $1 server died, see $OUT/$1.server.log"; return 1; }
    sleep 3
  done
  echo "!! $1 server never became healthy"; return 1
}
stop_server() { [[ -n "$SRV" ]] && { kill -INT "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null; SRV=""; }; }
trap stop_server EXIT

gate1() {
  echo "== $(date +%T) gate1 tensor inspection"
  python3 tasks/inspect-o2-tensors.py "$BASE" "$O1" "$O2" | tee "$OUT/gate1.log"
}

gate3() {
  noserver
  for arm in o1 o2; do
    local m; [[ $arm == o1 ]] && m=$O1 || m=$O2
    echo "== $(date +%T) gate3 official continuation $arm"
    gguf-tools/quality-testing/score_official "$m" gguf-tools/quality-testing/data/flash/manifest.tsv \
      "$OUT/official-$arm.tsv" 4096 2> "$OUT/official-$arm.log"
    grep -E '^(summary|api_summary)' "$OUT/official-$arm.log" || echo "!! $arm scorer produced no summary"
  done
}

gate4() {
  noserver
  for arm in o1 o2; do
    local m; [[ $arm == o1 ]] && m=$O1 || m=$O2
    echo "== $(date +%T) gate4 prompts $arm"
    start_server "prompts-$arm" "$m" || return 1
    python3 - "$PORT" "$OUT/prompts-$arm.jsonl" <<'PY'
import json, sys, time, urllib.request
port, out = sys.argv[1], sys.argv[2]
P = [
  ("code", "none", "Write a Python function that merges overlapping intervals, with type hints and three doctest examples. Code only."),
  ("debug", "none", "This Go code deadlocks. Explain why in two sentences and give the fix:\n\nfunc main() {\n  ch := make(chan int)\n  ch <- 1\n  fmt.Println(<-ch)\n}"),
  ("reason", "none", "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost? Show the algebra briefly."),
  ("json", "none", "Return only JSON: an array of 3 objects with keys name (string), port (integer), healthy (boolean) describing made-up local services."),
  ("zh", "none", "用中文简要解释什么是混合专家模型（MoE），不超过100字。"),
  ("summarize", "none", "Summarize in 3 bullet points: Rust's ownership model enforces that each value has exactly one owner; when the owner goes out of scope the value is dropped. Borrowing lets code reference a value without taking ownership, and the borrow checker ensures references never outlive the data or alias mutably."),
  ("think", "high", "How many positive integers below 1000 are divisible by 3 or 5 but not by 15? Give the final number."),
]
with open(out, "w") as f:
    for name, effort, content in P:
        body = {"model": "deepseek-v4-flash", "temperature": 0, "max_tokens": 1536,
                "reasoning_effort": effort, "messages": [{"role": "user", "content": content}]}
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                     data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        t0 = time.time()
        r = json.load(urllib.request.urlopen(req, timeout=1800))
        ch = r["choices"][0]
        f.write(json.dumps({"name": name, "wall": round(time.time() - t0, 2), "finish": ch.get("finish_reason"),
                            "usage": r.get("usage"), "content": ch["message"].get("content") or "",
                            "reasoning": ch["message"].get("reasoning_content") or ""}, ensure_ascii=False) + "\n")
        print(name, ch.get("finish_reason"), r.get("usage", {}).get("completion_tokens"), flush=True)
PY
    stop_server
  done
  python3 - "$OUT/prompts-o1.jsonl" "$OUT/prompts-o2.jsonl" <<'PY' | tee "$OUT/gate4.log"
import json, re, sys
a = [json.loads(l) for l in open(sys.argv[1])]; b = [json.loads(l) for l in open(sys.argv[2])]
def flags(r):
    t = r["content"] + r["reasoning"]; out = []
    if not r["content"].strip(): out.append("EMPTY")
    if "�" in t: out.append("REPLACEMENT_CHAR")
    if re.search(r"(.{12,}?)\1{6,}", t, re.S): out.append("REPEAT_LOOP")
    if r["finish"] != "stop": out.append(f"finish={r['finish']}")
    return out
bad = 0
for x, y in zip(a, b):
    fx, fy = flags(x), flags(y); bad += len(fy)
    print(f"--- {x['name']}  o1={fx or 'ok'} tok={x['usage'].get('completion_tokens')}  o2={fy or 'ok'} tok={y['usage'].get('completion_tokens')}  identical={x['content']==y['content']}")
    print("o1:", x["content"][:400].replace("\n", " / "))
    print("o2:", y["content"][:400].replace("\n", " / "))
print("GATE4_FLAGS_O2", bad)
print("note: 'think' uses reasoning_effort=high, which discards temperature=0, so identical=False there is expected")
PY
}

gate5() {
  noserver
  # decode-ab.sh header: ABAB then BABA on a cooled machine; this Mac drifts 20-40%, so only
  # believe a delta when every window of one arm beats every window of the other.
  echo "== $(date +%T) gate5 decode A/B (o1 o2 o1 o2 o2 o1 o2 o1)"
  local n=0
  for arm in o1-1 o2-1 o1-2 o2-2 o2-3 o1-3 o2-4 o1-4; do
    local m; [[ $arm == o1-* ]] && m=$O1 || m=$O2
    (( n++ > 0 )) && sleep 120   # cool between arms
    DECODE_AB_OUT="$KVROOT/decode-ab" DECODE_AB_MODEL="$m" tasks/decode-ab.sh "$arm" 2>&1 | tail -12
  done
  cp "$KVROOT/decode-ab/results.tsv" "$OUT/gate5-results.tsv"
}

gate6() {
  noserver
  echo "== $(date +%T) gate6/7 O2b at ~258K tokens"
  start_server "fill-o2" "$O2" || return 1
  python3 - "$SRV" "$OUT/gate7-memory.tsv" <<'PY' &
import subprocess, sys, time, os
pid, out = int(sys.argv[1]), sys.argv[2]
def swapouts():
    for l in subprocess.run(["vm_stat"], capture_output=True, text=True).stdout.splitlines():
        if l.startswith("Swapouts:"): return int(l.split()[1].rstrip("."))
with open(out, "w") as f:
    f.write("time\tswapouts\tswapusage\tfree_pct\tserver_rss_gib\n")
    while True:
        try: os.kill(pid, 0)
        except ProcessLookupError: break
        su = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout.strip()
        mp = subprocess.run(["memory_pressure", "-Q"], capture_output=True, text=True).stdout.strip().split()[-1]
        rss = subprocess.run(["ps", "-p", str(pid), "-o", "rss="], capture_output=True, text=True).stdout.strip() or "0"
        f.write(f"{time.strftime('%T')}\t{swapouts()}\t{su}\t{mp}\t{int(rss)/1048576:.1f}\n"); f.flush()
        time.sleep(5)
PY
  local SAMPLER=$!
  python3 - "$PORT" "$D/speed-bench/promessi_sposi.txt" <<'PY' | tee "$OUT/gate6.log"
import json, sys, time, urllib.request, urllib.error
port, src = sys.argv[1], open(sys.argv[2], encoding="utf-8").read()
def ask(nbytes):
    text = src.encode()[:nbytes].decode("utf-8", "ignore")
    body = {"model": "deepseek-v4-flash", "temperature": 0, "max_tokens": 256, "ignore_eos": True, "reasoning_effort": "none",
            "messages": [{"role": "user", "content": text + "\n\nRiassumi in italiano l'ultimo paragrafo qui sopra."}]}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        r = json.load(urllib.request.urlopen(req, timeout=7200))
    except urllib.error.HTTPError as e:
        return None, e.read().decode()[:300]
    st = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/stats"))
    u = r.get("usage", {})
    print(f"bytes={nbytes} prompt={u.get('prompt_tokens')} gen={u.get('completion_tokens')} "
          f"prefill_tps={st.get('last_prefill_tps'):.1f} decode_tps={st.get('last_decode_tps'):.2f} wall={time.time()-t0:.0f}s "
          f"tail={r['choices'][0]['message'].get('content','')[:160]!r}", flush=True)
    return u.get("prompt_tokens"), None
tok, err = ask(200_000)
if not tok: sys.exit(f"!! calibration failed: {err}")
nbytes = int(200_000 * 258_000 / tok)
nbytes = min(nbytes, len(src.encode()))
for _ in range(5):
    tok, err = ask(nbytes)
    if tok: break
    print("retry smaller:", err, flush=True); nbytes = int(nbytes * 0.97)
print("GATE6_PROMPT_TOKENS", tok)
PY
  stop_server
  wait "$SAMPLER" 2>/dev/null
  python3 - "$OUT/gate7-memory.tsv" <<'PY' | tee -a "$OUT/gate6.log"
import sys
rows = [l.rstrip("\n").split("\t") for l in open(sys.argv[1])][1:]
so = [int(r[1]) for r in rows]; free = [int(r[3].rstrip("%")) for r in rows if r[3].rstrip("%").isdigit()]
rss = [float(r[4]) for r in rows]
print(f"GATE7 swapouts_delta={so[-1]-so[0] if so else 'n/a'} min_free_pct={min(free) if free else 'n/a'} max_rss_gib={max(rss) if rss else 'n/a'} samples={len(rows)}")
PY
}

for g in "${GATES[@]}"; do "gate$g" || echo "!! gate $g returned nonzero"; done
echo "== $(date +%T) O2_GATES_COMPLETE"
