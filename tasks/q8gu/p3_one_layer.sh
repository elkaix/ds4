#!/usr/bin/env bash
# Convert (unless --skip-convert), launch, 8K+64K 3-rep, quality vs control refs, stop.
# Deletes the candidate GGUF unless KEEP_GGUF=1 (L24 plumbing).
set -Eeuo pipefail

LAYER=${1:?layer id}
TAG=L${LAYER}
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
QWEN=$ROOT
OUT=$ROOT/tasks/q8gu
HF=$HOME/models/hf/Qwen3.8-Flash-Next-Uncensored
GGUF=$HOME/models/gguf/Qwen3.8-Flash-Next-Uncensored-DS4-Q4KGateUp-Q8_0Down-MTP-${TAG}Q8GU.gguf
if [[ $LAYER == 24 ]]; then
  GGUF=$HOME/models/gguf/Qwen3.8-Flash-Next-Uncensored-DS4-Q4KGateUp-Q8_0Down-MTP-L24Q8GU.gguf
fi
KV=$HOME/.ds4/server-kv/qwen38-p3-${TAG}
REFS=$OUT/p3-control-quality.json
PY=/Users/panda/miniforge3/envs/rag-qa/bin/python
KEEP_GGUF=${KEEP_GGUF:-0}
SKIP_CONVERT=${SKIP_CONVERT:-0}

if [[ ! -f $REFS ]]; then
  echo "missing control quality refs: $REFS" >&2
  exit 1
fi

python3 -c 'import json,subprocess; d=json.loads(subprocess.check_output(["thermalforge","status"])); m={f["mode"] for f in d["fans"]}; assert m<={"auto","automatic","default"}, m'

if [[ $SKIP_CONVERT != 1 && ! -f $GGUF ]]; then
  echo "=== convert $TAG ==="
  "$PY" "$QWEN/gguf-tools/qwen4_exp_convert.py" \
    --src "$HF" \
    --out "$GGUF" \
    --outtype q8_0 --experts q4_k --experts-down q8_0 --no-ple \
    --q8-gate-up-layers "$LAYER" \
    --llama-cpp "$HOME/bin/llama.cpp" \
    --quants-library "$QWEN/gguf-tools/libds4quants.dylib"
fi
if [[ ! -f $GGUF ]]; then
  echo "missing $GGUF" >&2
  exit 1
fi

if lsof -nP -iTCP:8000 -sTCP:LISTEN >/dev/null 2>&1; then
  echo "port 8000 busy" >&2
  exit 1
fi

mkdir -p "$KV"
export QWEN_DS4_MODEL=$GGUF
export QWEN_DS4_KV_DIR=$KV
LOG=$OUT/p3-${TAG}-server.log
echo "=== launch $TAG $(date -Iseconds) ==="
(
  cd "$QWEN"
  exec ./run-qwen38-ds4.sh
) >"$LOG" 2>&1 &
LAUNCHER=$!
cleanup() {
  if kill -0 "$LAUNCHER" 2>/dev/null; then
    kill -INT "$LAUNCHER" 2>/dev/null || true
    for _ in $(seq 1 60); do
      kill -0 "$LAUNCHER" 2>/dev/null || break
      sleep 1
    done
  fi
}
trap cleanup EXIT

for i in $(seq 1 90); do
  if curl -fsS --max-time 2 http://127.0.0.1:8000/health >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$LAUNCHER" 2>/dev/null; then
    echo "launcher died" >&2
    tail -30 "$LOG" >&2
    exit 1
  fi
  sleep 2
done
curl -fsS http://127.0.0.1:8000/health
echo
SERVER=$(ps -axo pid,ppid,command | awk -v p="$LAUNCHER" '$2==p && $0 ~ /ds4-server / {print $1}')
SERVER=$(echo "$SERVER" | awk 'NF{print; exit}')
if [[ -z ${SERVER:-} ]]; then
  echo "could not find ds4-server child of launcher $LAUNCHER" >&2
  exit 1
fi
echo "server pid=$SERVER launcher=$LAUNCHER"

echo "=== speed 8K+64K $TAG ==="
cd "$OUT"
uv run --script p0_longctx_bench.py --sizes 8192,65536 --out "p3-${TAG}-longctx.json" --pid "$SERVER"

echo "=== quality $TAG ==="
"$PY" "$OUT/p3_quality.py" --mode score --n 40 --refs "$REFS" --out "p3-${TAG}-quality.json"

echo "=== stop $TAG ==="
kill -INT "$LAUNCHER" 2>/dev/null || true
kill -INT "$SERVER" 2>/dev/null || true
for i in $(seq 1 90); do
  if ! lsof -nP -iTCP:8000 -sTCP:LISTEN >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
if lsof -nP -iTCP:8000 -sTCP:LISTEN >/dev/null 2>&1; then
  kill -TERM "$SERVER" 2>/dev/null || true
  sleep 3
fi
trap - EXIT
thermalforge auto >/dev/null || true
python3 -c 'import json,subprocess; d=json.loads(subprocess.check_output(["thermalforge","status"])); print("fans",[f["mode"] for f in d["fans"]])'

if [[ $KEEP_GGUF != 1 && $LAYER != 24 ]]; then
  echo "=== delete $GGUF ==="
  rm -f "$GGUF"
fi
echo "DONE $TAG"
