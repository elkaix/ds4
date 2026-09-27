#!/bin/bash
# GLM stage-2 decode A/B: generic per-expert down + moe_sum_experts reduction
# (arm A) vs the existing sum6 down+sum kernel admitted for n8 single-device
# via DS4_METAL_ENABLE_DOWNSUM8 (arm B). Same binary, same model, env-only
# difference, ABBA cycles, 30 s gaps, teacher-forced 256 tokens, Tensor off,
# no MTP. Route assertion: arm B log must contain 'n8 down+sum single-device'.
# Usage: [N=4] [SWEEP=short|mid] tasks/glm_downsum_ab_20260926.sh [out-dir]
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=${1:-$PWD/tasks/data/glm-downsum-ab-$(date +%Y%m%d)}; mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
M=$HOME/models/gguf/GLM-5.3-Flash-UNCEN-d21b-L17-23Q4KExperts-KDAvoQ4K-Q2.gguf
P=$PWD/speed-bench/promessi_sposi.txt
N=${N:-4}
SWEEP=${SWEEP:-short}
GEN=256
GAP=30
CYC=${CYC:-1}
export DS4_METAL_DISABLE_TENSOR_API=1

echo "$0 N=$N SWEEP=$SWEEP" > "$OUT/cmdline.txt"
(git rev-parse HEAD; git status --porcelain | wc -l | tr -d ' '; shasum -a 256 ds4-bench | cut -c1-16; sysctl -n machdep.cpu.brand_string; pmset -g | grep powermode; sysctl -n iogpu.wired_limit_mb) | tr '\n' ' ' > "$OUT/arms.txt"; echo >> "$OUT/arms.txt"
grep -n 'DS4_METAL_ENABLE_DOWNSUM8' ds4_metal.m | head -2 >> "$OUT/arms.txt" || true

fans_off() { thermalforge auto >/dev/null 2>&1 || true; }
trap fans_off EXIT
thermalforge max >/dev/null 2>&1 || true

run() {  # $1 tag, $2 1|0 arm
    local tag=$1 arm=$2
    if [[ $arm == 1 ]]; then export DS4_METAL_ENABLE_DOWNSUM8=1; else unset DS4_METAL_ENABLE_DOWNSUM8; fi
    echo "== $(date +%T) $tag start (DOWNSUM8=$arm)" >> "$OUT/run.log"
    ./ds4-bench --metal -m "$M" --prompt-file "$P" "${SWEEPARGS[@]}" \
        --gen-tokens "$GEN" --teacher-forced-decode --power 100 \
        --csv "$OUT/$tag.csv" > "$OUT/$tag.log" 2>&1
    echo "== $(date +%T) $tag rc=$? sum8=$(grep -c 'n8 down+sum' "$OUT/$tag.log")" >> "$OUT/run.log"
    unset DS4_METAL_ENABLE_DOWNSUM8
    sleep "$GAP"
}

if [[ $SWEEP == short ]]; then
    SWEEPARGS=(--ctx-start 2048 --ctx-max 2048 --ctx-alloc 4096)
else
    SWEEPARGS=(--ctx-start 32768 --ctx-max 32768 --ctx-alloc 102400)
fi

for ((c = CYC; c < CYC + N; c++)); do
    run "A-$c-1" 0
    run "B-$c-1" 1
    run "B-$c-2" 1
    run "A-$c-2" 0
done
echo "== $(date +%T) done" >> "$OUT/run.log"
