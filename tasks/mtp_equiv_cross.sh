#!/bin/bash
# Boundary-crossing companion to mtp_equiv.sh: calibrates a prompt to ~1990
# tokens (measured via usage.prompt_tokens) so 96 greedy tokens cross the 2051
# dense boundary during generation, then compares MTP-on vs plain greedy.
# Each arm uses a fresh, private KV disk dir so both prompts are cold.
# Usage: tasks/mtp_equiv_cross.sh [out-dir]   (server must NOT be running)
set -Eeuo pipefail
cd "$(dirname "$0")/.."
OUT=${1:-tasks/data/p4a/mtp-equiv-cross}
mkdir -p "$OUT"
PORT=8000
TARGET=${TARGET:-1990}

start_server() {
    rm -rf "$OUT/kv-$1"; mkdir -p "$OUT/kv-$1"
    GLM_DS4_KV_DIR="$PWD/$OUT/kv-$1" GLM_DS4_MTP_MAX_CTX=0 GLM_DS4_MTP_TIMING=1 \
        nohup ./run-glm-ds4.sh > "$OUT/server-$1.log" 2>&1 &
    for _ in $(seq 1 60); do
        curl -s -m 2 "127.0.0.1:$PORT/health" | grep -q '"ok"' && return 0
        sleep 3
    done
    echo "server $1 did not come up" >&2; return 1
}
stop_server() {
    pkill -INT -f run-glm-ds4.sh || true
    pkill -INT -x ds4-server || true
    for _ in $(seq 1 30); do pgrep -x ds4-server >/dev/null || return 0; sleep 1; done
    pkill -KILL -x ds4-server || true
}
mkbody() {  # $1 nunits $2 max_tokens -> stdout json
    python3 - "$1" "$2" <<'PY'
import sys, json
n, mt = int(sys.argv[1]), int(sys.argv[2])
unit = ("Section %d. The reservoir gauge read %d units at dawn and the valve "
        "log noted a %d percent duty cycle before the night crew rotated. ")
text = "".join(unit % (i, (i * 37) % 1000, (i * 13) % 100) for i in range(n))
prompt = text + "\n\nSummarize the pattern in the gauge readings across all sections:"
print(json.dumps({"model": "glm", "prompt": prompt, "max_tokens": mt,
                  "temperature": 0, "stream": False}))
PY
}
ntok() {  # $1 nunits -> prompt_tokens
    mkbody "$1" 1 | curl -s -m 600 "127.0.0.1:$PORT/v1/completions" -H 'Content-Type: application/json' --data-binary @- \
      | python3 -c 'import json,sys;print(json.load(sys.stdin)["usage"]["prompt_tokens"])'
}

echo "== $(date +%T) start mtp"
export GLM_DS4_MTP=1
start_server mtp
# calibrate: units are ~31 tokens each; search n so that prompt_tokens in [TARGET-6, TARGET]
lo=40; hi=120; best=""
while (( lo <= hi )); do
    mid=$(( (lo + hi) / 2 )); t=$(ntok "$mid")
    echo "calib units=$mid tokens=$t"
    if (( t > TARGET )); then hi=$((mid-1)); else best=$mid; besttok=$t; lo=$((mid+1)); fi
done
echo "calibrated units=$best prompt_tokens=$besttok (gen crosses 2051 at token $((2051-besttok)))"
mkbody "$best" 96 > "$OUT/cross.body.json"
# rerun cold: calibration requests below kv-cache-min-tokens(2048) are not stored, so this prefill is fresh
curl -s -m 3600 "127.0.0.1:$PORT/v1/completions" -H 'Content-Type: application/json' --data-binary @"$OUT/cross.body.json" > "$OUT/cross.mtp.json"
grep -E 'prompt done' "$OUT/server-mtp.log" | tail -1 | cut -c1-160
stop_server
echo "== $(date +%T) start plain"
export GLM_DS4_MTP=0
start_server plain
curl -s -m 3600 "127.0.0.1:$PORT/v1/completions" -H 'Content-Type: application/json' --data-binary @"$OUT/cross.body.json" > "$OUT/cross.plain.json"
grep -E 'prompt done' "$OUT/server-plain.log" | tail -1 | cut -c1-160
stop_server
echo "== $(date +%T) compare"
python3 - "$OUT" "$besttok" <<'PY'
import json, sys, re
out, ptok = sys.argv[1], int(sys.argv[2])
a = json.load(open(f"{out}/cross.mtp.json"))["choices"][0]["text"]
b = json.load(open(f"{out}/cross.plain.json"))["choices"][0]["text"]
same = a == b
print(f"cross: {'IDENTICAL' if same else 'MISMATCH'} ({len(a)} vs {len(b)} chars) prompt_tokens={ptok}")
if not same:
    i = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), min(len(a), len(b)))
    print("  first divergence at char", i, repr(a[i:i+40]), "vs", repr(b[i:i+40]))
log = open(f"{out}/server-mtp.log").read()
cyc = re.findall(r"glm mtp utility: width=2 pos=(\d+) .*?verify\[([^\]]+)\].*?result=(\w+)", log)
cyc = [(int(p), v, r) for p, v, r in cyc if int(p) >= ptok]
rej = [p for p, v, r in cyc if r.lower().startswith("reject")]
print(f"cycles={len(cyc)} accept={len(cyc)-len(rej)} reject={len(rej)}")
print("verify path by pos:", [(p, v) for p, v, r in cyc if 2040 <= p <= 2060])
print("rejections 2040-2060:", [p for p in rej if 2040 <= p <= 2060], "above 2051:", sum(p > 2051 for p in rej), "at/below 2051:", sum(p <= 2051 for p in rej))
ok = same and bool(cyc) and any(p > 2051 for p in rej) and any(p <= 2051 for p in rej)
if not ok and same: print("NOTE: identical text but rejections not observed on both sides of 2051")
print("RESULT:", "PASS" if ok else "FAIL")
PY
