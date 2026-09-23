#!/usr/bin/env bash
# Launch Hybrid v2 imatrix GGUF, 8K+64K 3-rep, greedy quality vs control, stop.
# Never touches the control GGUF. Never deletes the candidate.
set -Eeuo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
OUT=$ROOT/tasks/q8gu
GGUF=$HOME/models/gguf/Qwen3.8-Flash-Next-Uncensored-DS4-Q4KGateUp-Q8_0Down-MTP-imatrix.gguf
CONTROL=$HOME/models/gguf/Qwen3.8-Flash-Next-Uncensored-DS4-Q4KGateUp-Q8_0Down-MTP.gguf
KV=$HOME/.ds4/server-kv/qwen38-v2-reconvert
REFS=$OUT/p3-control-quality.json
PY=/Users/panda/miniforge3/envs/rag-qa/bin/python
TAG=v2-reconvert

if [[ ! -f $REFS ]]; then
  echo "missing control quality refs: $REFS" >&2
  exit 1
fi
if [[ ! -f $GGUF ]]; then
  echo "missing candidate $GGUF" >&2
  exit 1
fi
if [[ ! -f $OUT/p3-v2-reconvert-dequant.json ]]; then
  echo "run verify_v2_gguf.py first" >&2
  exit 1
fi

python3 -c 'import json,subprocess; d=json.loads(subprocess.check_output(["thermalforge","status"])); m={f["mode"] for f in d["fans"]}; assert m<={"auto","automatic","default"}, m'
python3 -c '
import os, sys
c=os.stat(sys.argv[1]); b=os.stat(sys.argv[2])
assert c.st_size==96114378688 and int(c.st_mtime)==1789208337, (c.st_size, int(c.st_mtime))
assert not os.path.samefile(sys.argv[1], sys.argv[2])
' "$CONTROL" "$GGUF"

if lsof -nP -iTCP:8000 -sTCP:LISTEN >/dev/null 2>&1; then
  echo "port 8000 busy" >&2
  exit 1
fi

mkdir -p "$KV"
export QWEN_DS4_MODEL=$GGUF
export QWEN_DS4_KV_DIR=$KV
LOG=$OUT/p3-${TAG}-server.log
echo "=== launch $TAG $(date -Iseconds) model=$GGUF kv=$KV ==="
(
  cd "$ROOT"
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

for i in $(seq 1 180); do
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
SERVER=$("$PY" - <<'PY'
import subprocess, sys
out = subprocess.check_output(["lsof", "-nP", "-iTCP:8000", "-sTCP:LISTEN", "-t"], text=True)
pids = [p for p in out.split() if p.strip()]
if not pids:
    sys.exit("no listener on :8000")
print(pids[0])
PY
)
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
thermalforge auto --stop-app >/dev/null || true
python3 -c 'import json,subprocess; d=json.loads(subprocess.check_output(["thermalforge","status"])); print("fans",[f["mode"] for f in d["fans"]])'
python3 -c '
import os, sys
c=os.stat(sys.argv[1])
print("control still", c.st_size, int(c.st_mtime))
assert c.st_size==96114378688 and int(c.st_mtime)==1789208337
' "$CONTROL"
echo "DONE $TAG (GGUF kept)"
