#!/usr/bin/env bash
# decode-ab.sh — one arm of an interleaved decode A/B on the daily Vision-Exp config.
#
#   ./tasks/decode-ab.sh <label> [ENV=VAL ...]
#   ./tasks/decode-ab.sh base
#   ./tasks/decode-ab.sh thr2048 DS4_METAL_DECODE_INDEXER_SPARSE_THRESHOLD=2048
#   ./tasks/decode-ab.sh ngram5  DS4_NGRAM_SPEC=5
#   DECODE_AB_MODEL=~/models/gguf/...-AProjQ4K.gguf ./tasks/decode-ab.sh q4k
#
# Each arm cold-starts ds4-server on :8009 with the run.md flag set, then measures:
#   short  — ~60-token prompt, 512 generated tokens          (near-empty decode)
#   mid    — ~2K-token prompt, 256 generated                 (the only depth where the
#                                                             sparse threshold can matter)
#   deep   — ~30K-token prompt, 256 generated                (daily agent depth)
# Every prompt runs twice; the second run is the number (warm KV, cached prefill).
# Requests are temperature:0 + reasoning_effort:"none" — thinking mode discards sampling
# params and speculation only runs on the greedy path. THINKING in the log = invalid arm.
#
# Toggle assertions (read from the SERVER log, never from this script's variables):
#   thr_active   — the "decode indexer sparse threshold=" line, printed only when the env
#                  parsed; DS4_METAL_DECODE_INDEXER_SPARSE_THRESHOLD set without it = bad arm
#   ngram        — DSpark stats line at SIGINT: proposed>0 means n-gram actually drafted;
#                  DS4_NGRAM_SPEC set with proposed=0 = never engaged
#
# Interleave arms (A B A B, then B A B A) on a cooled machine. Only believe a delta when
# every window of one arm beats every window of the other — this Mac drifts 20-40%.
# The prod server must be stopped first: ds4 refuses to start while another ds4 runs.
set -uo pipefail

LABEL="${1:?usage: decode-ab.sh <label> [ENV=VAL ...]}"; shift
case "$LABEL" in *=*|*" "*) echo "decode-ab.sh: label '$LABEL' looks like it swallowed an ENV=VAL; pass label and envs as separate args" >&2; exit 2;; esac
OUT="${DECODE_AB_OUT:-$HOME/.ds4/decode-ab}"
D="$(cd "$(dirname "$0")/.." && pwd)"
M="${DECODE_AB_MODEL:-$HOME/models/gguf/DeepSeek-V4-Flash-Vision-Exp-Abliterated-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8.gguf}"
V="$HOME/models/gguf/DeepSeek-V4-Flash-Vision-Encoder.gguf"
PORT="${DECODE_AB_PORT:-8009}"
GEN="${DECODE_AB_GEN:-512}"
LOG="$OUT/$LABEL.log"
RES="$OUT/results.tsv"
mkdir -p "$OUT"

for kv in "$@"; do export "$kv"; done
export DS4_DSPARK_STATS=1

if pgrep -x ds4-server >/dev/null || pgrep -x ds4 >/dev/null; then
  echo "$LABEL: another ds4 process is running; stop it first" >&2; pgrep -fl 'ds4' >&2; exit 1
fi
if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "$LABEL: port $PORT busy" >&2; exit 1
fi

KV="$OUT/kv-$LABEL"; rm -rf "$KV"; mkdir -p "$KV"   # fresh KV per arm: no cross-arm warming
"$D/ds4-server" --chdir "$D" --metal --model "$M" --vision "$V" \
  --ctx 262144 --tokens 32768 --warm-weights --power 100 \
  --host 127.0.0.1 --port "$PORT" \
  --kv-disk-dir "$KV" --kv-disk-space-mb 8192 --kv-cache-min-tokens 2048 \
  > "$LOG" 2>&1 &
SRV=$!
for _ in $(seq 1 200); do
  curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  kill -0 $SRV 2>/dev/null || { echo "$LABEL: server died, see $LOG" >&2; exit 1; }
  sleep 3
done

THR=$(grep -oE 'decode indexer sparse threshold=[0-9]+' "$LOG" | grep -oE '[0-9]+$' || true)
echo "$LABEL: env=[$*] thr_active=${THR:-default} ngram_env=${DS4_NGRAM_SPEC:-unset}"
# (threshold line prints on first decode, so the real assertion is post-run below)

# Prompts. deep = first ~120 KB of ds4_server.c (~30K tokens); mid = first ~8 KB (~2K).
PY=python3
mk() { # $1=prompt-file-or-inline $2=max_tokens
  $PY - "$1" "$2" <<'PYEOF'
import json,sys
src,n=sys.argv[1],int(sys.argv[2])
if src.startswith('@'):
    body=open(src[1:]).read()
    content=body+"\n\nList every function defined above with a one-line description each. Plain text, no preamble."
else:
    content=src
print(json.dumps({"model":"deepseek-v4-flash","temperature":0,"max_tokens":n,"reasoning_effort":"none",
  "messages":[{"role":"user","content":content}]}))
PYEOF
}
head -c 8192   "$D/ds4_server.c" > "$OUT/mid.txt"
head -c 122880 "$D/ds4_server.c" > "$OUT/deep.txt"
SHORT="Write a complete Python implementation of a red-black tree with insert, delete, and search. Include docstrings. Code only, no explanation."

run() { # $1=name $2=body-json  -> prints decode t/s of the run from /stats
  local name=$1 body=$2 t0 t1 st
  t0=$(date +%s.%N)
  curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d "$body" > "$OUT/$LABEL.$name.json"
  t1=$(date +%s.%N)
  st=$(curl -fsS "http://127.0.0.1:$PORT/stats")
  $PY - "$st" "$name" "$t0" "$t1" "$OUT/$LABEL.$name.json" <<'PYEOF'
import json,sys
st=json.loads(sys.argv[1]); name=sys.argv[2]; wall=float(sys.argv[4])-float(sys.argv[3])
r=json.load(open(sys.argv[5])); u=r.get("usage",{})
print(f"{name}\tdecode_tps={st['last_decode_tps']:.2f}\tprefill_tps={st['last_prefill_tps']:.1f}\tgen={u.get('completion_tokens')}\tprompt={u.get('prompt_tokens')}\tcached={u.get('prompt_tokens_details',{}).get('cached_tokens')}\twall={wall:.2f}s")
PYEOF
}

for pass in 1 2; do
  echo "$LABEL pass$pass $(run short "$(mk "$SHORT" "$GEN")")"
  echo "$LABEL pass$pass $(run mid   "$(mk "@$OUT/mid.txt"  256)")"
  echo "$LABEL pass$pass $(run deep  "$(mk "@$OUT/deep.txt" 256)")"
done | tee -a "$RES"

THR=$(grep -oE 'decode indexer sparse threshold=[0-9]+' "$LOG" | grep -oE '[0-9]+$' || true)
echo "$LABEL post-run: thr_active=${THR:-default}" | tee -a "$RES"
grep -q "THINKING" "$LOG" && echo "  !! THINKING seen in log — sampling params were discarded, arm invalid" >&2

kill -INT $SRV 2>/dev/null; sleep 8; kill $SRV 2>/dev/null; sleep 2   # -INT so DSpark/n-gram stats print
grep -oE "DSpark stats .*" "$LOG" | tail -1 | tr ' ' '\n' \
  | grep -E "^(cycles|proposed|accept_rate|avg_accept|no_draft|target|saved|net_saved)=" \
  | paste -sd'  ' - | sed "s/^/$LABEL spec-stats: /" | tee -a "$RES"
echo "DONE_$LABEL"
