#!/bin/bash
# M5 clean-engine campaign C.2+C.3: context curve x topk-fast A/B on glm53-m5-prod.
# Arm A = clean default (GLM DSA top-k fast OFF — it is opt-in upstream).
# Arm B = DS4_GLM_ENABLE_TOPK_FAST=1.
# One incremental sweep per process (ds4-bench prefills only frontier deltas),
# ABBA order, fixed fans, 45s cool-downs, macmon power snapshot per sweep.
# No Metal4 override, no scorer envs, no decode-rows, no SSD streaming, no
# logits dumps. Teacher-forced decode: identical 256 tokens across arms.
# Usage: tasks/m5_clean_curve.sh [out-dir]   (nothing else on the GPU!)
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=${1:-$PWD/tasks/data/m5-clean-curve-20260919}; mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
M=$HOME/models/gguf/GLM-5.3-Flash-UNCEN-d21b-L17-18-19Q4KExperts-Q2.gguf
P=$PWD/speed-bench/promessi_sposi.txt
GEN=256
GAP=45

echo "$0 $*" > "$OUT/cmdline.txt"
(git rev-parse HEAD; shasum -a 256 ds4-bench | cut -c1-16; sysctl -n machdep.cpu.brand_string) \
    | tr '\n' ' ' > "$OUT/arms.txt"; echo >> "$OUT/arms.txt"

fans_off() { thermalforge auto >/dev/null 2>&1 || true; }
trap fans_off EXIT
thermalforge max >/dev/null 2>&1 || true

run() {  # $1 arm(A|B) $2 rep
    local tag="$1-$2"
    macmon pipe -s 1 2>/dev/null | head -1 > "$OUT/$tag.pre.json"
    echo "== $(date +%T) $tag start" >> "$OUT/run.log"
    if [[ $1 == B ]]; then export DS4_GLM_ENABLE_TOPK_FAST=1; else unset DS4_GLM_ENABLE_TOPK_FAST; fi
    ./ds4-bench --metal -m "$M" --prompt-file "$P" \
        --ctx-start 32768 --step-incr 16384 --ctx-max 204800 --ctx-alloc 262144 \
        --gen-tokens "$GEN" --teacher-forced-decode \
        --power 100 \
        --csv "$OUT/$tag.csv" > "$OUT/$tag.log" 2>&1
    echo "== $(date +%T) $tag rc=$? rows=$(tail -n +2 "$OUT/$tag.csv" | wc -l | tr -d ' ')" >> "$OUT/run.log"
    unset DS4_GLM_ENABLE_TOPK_FAST
    sleep "$GAP"
}

run A 1
run B 1
run B 2
run A 2
echo "== $(date +%T) campaign done" >> "$OUT/run.log"