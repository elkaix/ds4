#!/usr/bin/env bash
# dspark-ab.sh — measure whether --dspark actually helps on the daily ds4 config.
#
#   ./tasks/dspark-ab.sh <server-dir> <label> <on|off>
#   ./tasks/dspark-ab.sh ~/Projects/open-source/ds4 now_on  on
#   ./tasks/dspark-ab.sh ~/Projects/open-source/ds4 now_off off
#
# Run the arms INTERLEAVED (on, off, on, off) and only believe a result when both
# on-windows beat both off-windows — this Mac drifts up to ~40% across windows.
# Best run on a cooled machine (idle ~20 min first).
#
# Two traps this script exists to defeat:
#   1. temperature:0 alone does NOT reach DSpark — thinking mode discards sampling
#      params. The request below also sends reasoning_effort:"none".
#   2. A mis-passed toggle silently compares a config against itself, and the output
#      still looks like ordinary noise. This prints dspark_active=1|0 per run, read
#      from the SERVER's own log, not from the harness variables. Check it first.
set -uo pipefail

D="${1:?usage: dspark-ab.sh <server-dir> <label> <on|off>}"
LABEL="${2:?missing label}"
MODE="${3:?missing mode: on|off}"
OUT="${DSPARK_AB_OUT:-/tmp/dspark-ab}"
mkdir -p "$OUT"

M="$HOME/models/gguf/drowzeys-keys-DeepSeekV4-Flash-GA-0731-Abliterated-32-32-DS4-Q2.gguf"
S="$HOME/models/gguf/drowzeys-keys-DeepSeekV4-Flash-GA-0731-Abliterated-32-32-DS4-DSpark-support.gguf"
LOG="$OUT/r_$LABEL.log"

case "$MODE" in
  on)  EXTRA=(--mtp "$S" --dspark --dspark-confidence 0.9) ;;
  off) EXTRA=() ;;
  *)   echo "mode must be 'on' or 'off', got '$MODE'" >&2; exit 2 ;;
esac

if lsof -nP -iTCP:8000 -sTCP:LISTEN >/dev/null 2>&1; then
  echo "port 8000 busy:" >&2; lsof -nP -iTCP:8000 -sTCP:LISTEN >&2; exit 1
fi

DS4_DSPARK_STATS=1 "$D/ds4-server" --chdir "$D" --metal \
  --model "$M" --ctx 262144 --tokens 32768 \
  "${EXTRA[@]+"${EXTRA[@]}"}" \
  --warm-weights --power 100 --host 127.0.0.1 --port 8000 \
  --kv-disk-dir "$HOME/.ds4/server-kv/deepseek-v4-flash-ga-0731-q2" --kv-disk-space-mb 32768 \
  --kv-cache-min-tokens 2048 --kv-cache-reject-different-quant \
  > "$LOG" 2>&1 &
SRV=$!

for _ in $(seq 1 150); do
  curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1 && break
  kill -0 $SRV 2>/dev/null || { echo "$LABEL: server died, see $LOG" >&2; exit 1; }
  sleep 3
done

# Did the flag actually take? Read it from the server, not from $MODE.
ACTIVE=$(grep -c "DSpark target-hidden capture enabled" "$LOG" || true)
echo "$LABEL: requested=$MODE dspark_active=$ACTIVE"
[[ "$MODE" == "on"  && "$ACTIVE" == "0" ]] && echo "  !! asked for DSpark and did not get it" >&2
[[ "$MODE" == "off" && "$ACTIVE" != "0" ]] && echo "  !! asked for no DSpark and got it"     >&2

# reasoning_effort:none is REQUIRED — without it thinking mode eats temperature:0.
for _ in 1 2; do
  curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
    "model":"deepseek-v4-flash","temperature":0,"max_tokens":512,"reasoning_effort":"none",
    "messages":[{"role":"user","content":"Write a complete Python implementation of a red-black tree with insert, delete, and search. Include docstrings. Code only, no explanation."}]
  }' > /dev/null
done

if grep -q "THINKING" "$LOG"; then
  echo "  !! THINKING seen in log — temperature:0 was discarded, DSpark did not run" >&2
fi
grep -oE "gen=512 finish=[a-z]+ [0-9.]+s" "$LOG" | sed "s/^/$LABEL wall: /"

kill -INT $SRV 2>/dev/null; sleep 8; kill $SRV 2>/dev/null; sleep 2   # -INT so stats print
grep -oE "DSpark stats .*" "$LOG" | tail -1 | tr ' ' '\n' \
  | grep -E "^(cycles|proposed|accept_rate|avg_accept|target|saved|net_saved|spec_total)=" \
  | paste -sd'  ' - | sed "s/^/$LABEL stats: /"
echo "DONE_$LABEL"
