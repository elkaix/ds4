#!/bin/bash
# Scorer throughput ABBA on the live glm53-p4a binary.
# A = score_one_direct (default). B = DS4_METAL_GLM_INDEX_DECODE_ROWS=1.
# CHECK off. Metal4 off. Same binary both arms.
# Usage: tasks/p4b1_scorer_abba.sh [out-dir]
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=${1:-$PWD/tasks/data/p4b1/scorer-abba}
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
M=$HOME/models/gguf/GLM-5.3-Flash-UNCEN-d21b-L17-18-19Q4KExperts-Q2.gguf
P=$PWD/speed-bench/promessi_sposi.txt
BIN=$PWD/ds4-bench
GEN=${GEN:-256}
GAP=${GAP:-45}

busy() {
    lsof -nP -iTCP:8000 -sTCP:LISTEN >/dev/null 2>&1 && return 0
    pgrep -x ds4-server >/dev/null 2>&1 && return 0
    pgrep -x ds4-bench >/dev/null 2>&1 && return 0
    pgrep -f 'p3_one_layer.sh' >/dev/null 2>&1 && return 0
    return 1
}

if busy; then
    echo "GPU busy (:8000, ds4-server, ds4-bench, or p3_one_layer)" >&2
    exit 2
fi

{
    echo "bin $BIN"
    echo "sha $(shasum -a 256 "$BIN" | awk '{print substr($1,1,16)}')"
    echo "head $(git rev-parse HEAD)"
    echo "A score_one_direct (DECODE_ROWS unset, CHECK unset)"
    echo "B DS4_METAL_GLM_INDEX_DECODE_ROWS=1 CHECK unset"
    echo "GEN=$GEN GAP=$GAP Metal4=off"
} | tee "$OUT/arms.txt"

run() {  # $1 arm $2 ctx $3 runidx
    local tag="$2-$1-$3"
    local env=(DS4_METAL_DISABLE_METAL4=1)
    unset DS4_METAL_GLM_INDEX_CHECK DS4_METAL_GLM_INDEX_DECODE_ROWS
    if [[ $1 == B ]]; then
        env+=(DS4_METAL_GLM_INDEX_DECODE_ROWS=1)
    fi
    macmon pipe -s 1 2>/dev/null | head -1 > "$OUT/$tag.pre.json" || true
    env -u DS4_METAL_GLM_INDEX_CHECK -u DS4_METAL_GLM_INDEX_DECODE_ROWS \
        "${env[@]}" "$BIN" --metal -m "$M" --prompt-file "$P" \
        --ctx-start 1024 --step-incr $(( $2 - 1024 )) --ctx-max "$2" --gen-tokens "$GEN" \
        --csv "$OUT/$tag.csv" > "$OUT/$tag.log" 2>&1
    echo "$(date +%T) $tag rc=$? $(tail -1 "$OUT/$tag.csv" 2>/dev/null)"
    sleep "$GAP"
}

echo "== $(date +%T) start GEN=$GEN GAP=$GAP OUT=$OUT"
for ctx in 100000 200000; do
    nA=0 nB=0
    for a in A B B A; do
        if [[ $a == A ]]; then nA=$((nA+1)); run A "$ctx" $nA
        else nB=$((nB+1)); run B "$ctx" $nB
        fi
    done
done
echo "== $(date +%T) done"
python3 "$PWD/tasks/p4b1_analyze.py" "$OUT"
