#!/bin/bash
# Live rejected-speculation equivalence: greedy generation with GLM MTP on
# (ceiling removed, timing on so rejections are visible) must produce the same
# tokens as plain greedy decode.  Runs one prompt that crosses the 2051 dense
# boundary during generation and one at ~100K context.
#
# Usage: tasks/mtp_equiv.sh [out-dir]   (server must NOT be running on :8000)
set -Eeuo pipefail
cd "$(dirname "$0")/.."
OUT=${1:-tasks/data/p4a/mtp-equiv}
mkdir -p "$OUT"
GEN=96
PORT=8000

prompt_file() {  # $1 name $2 target tokens (approx 1 token per 4 chars of prose)
    python3 - "$1" "$2" "$OUT" <<'EOF'
import sys, json
name, target, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
unit = ("Section %d. The reservoir gauge read %d units at dawn and the valve "
        "log noted a %d percent duty cycle before the night crew rotated. ")
text = "".join(unit % (i, (i * 37) % 1000, (i * 13) % 100) for i in range(target // 22 + 1))
prompt = text + "\n\nSummarize the pattern in the gauge readings across all sections:"
json.dump({"model": "glm", "prompt": prompt, "max_tokens": 96, "temperature": 0,
           "stream": False}, open(f"{out}/{name}.body.json", "w"))
print(len(prompt))
EOF
}

start_server() {  # $1 arm name, env already exported
    GLM_DS4_MTP_MAX_CTX=0 GLM_DS4_MTP_TIMING=1 nohup ./run-glm-ds4.sh > "$OUT/server-$1.log" 2>&1 &
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

run_prompt() {  # $1 arm $2 name -> writes $OUT/$2.$1.json
    curl -s -m 3600 "127.0.0.1:$PORT/v1/completions" -H 'Content-Type: application/json' \
        --data-binary @"$OUT/$2.body.json" > "$OUT/$2.$1.json"
    python3 -c '
import json,sys; d=json.load(open(sys.argv[1])); c=d["choices"][0]
print(sys.argv[2], sys.argv[3], "tokens=%s finish=%s" % (d.get("usage",{}).get("completion_tokens"), c.get("finish_reason")))' "$OUT/$2.$1.json" "$1" "$2"
}

echo "== $(date +%T) prompts"
prompt_file boundary 2000 >/dev/null
prompt_file long100k 100000 >/dev/null

for arm in mtp plain; do
    if [[ $arm == mtp ]]; then export GLM_DS4_MTP=1; else export GLM_DS4_MTP=0; fi
    echo "== $(date +%T) start $arm"
    start_server "$arm"
    for name in boundary long100k; do
        echo "== $(date +%T) $arm $name"
        run_prompt "$arm" "$name"
        grep -E 'prompt done' "$OUT/server-$arm.log" | tail -1 | cut -c1-160
    done
    stop_server
    echo "== $(date +%T) stop $arm"
done

echo "== $(date +%T) compare"
python3 - "$OUT" <<'EOF'
import json, sys, re
out = sys.argv[1]
ok = True
for name in ("boundary", "long100k"):
    a = json.load(open(f"{out}/{name}.mtp.json"))["choices"][0]["text"]
    b = json.load(open(f"{out}/{name}.plain.json"))["choices"][0]["text"]
    same = a == b
    ok &= same
    print(f"{name}: {'IDENTICAL' if same else 'MISMATCH'} ({len(a)} vs {len(b)} chars)")
    if not same:
        i = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), min(len(a), len(b)))
        print("  first divergence at char", i, repr(a[i:i+40]), "vs", repr(b[i:i+40]))
log = open(f"{out}/server-mtp.log").read()
cycles = re.findall(r"glm mtp utility: width=2 pos=(\d+) .*?result=(\w+)", log)
rej = [int(p) for p, r in cycles if r.lower().startswith("reject")]
acc = [int(p) for p, r in cycles if r.upper().startswith("ACCEPT")]
print(f"mtp cycles={len(cycles)} accept={len(acc)} reject={len(rej)}")
print(f"rejections above 2051: {sum(1 for p in rej if p > 2051)}; around boundary (2040-2060): {sorted(p for p in rej if 2040 <= p <= 2060)}")
print(f"rejections above 90000: {sum(1 for p in rej if p > 90000)}")
if not cycles: print("WARNING: no mtp timing lines found; was --mtp-timing active?"); ok = False
if not rej: print("WARNING: no rejection observed; the equivalence test did not exercise rollback"); ok = False
print("RESULT:", "PASS" if ok else "FAIL")
EOF
