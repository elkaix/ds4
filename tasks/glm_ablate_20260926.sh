#!/bin/bash
# Stage-2 decode attribution at 2K: plain baseline vs DS4_GLM_DECODE_ABLATE=routed
# (and an optional second stage via ABL2). Alternating order, 30 s gaps,
# teacher-forced 256 tokens, Tensor API off, no MTP, powermode 2 checked.
# Ablated output is garbage — timing only (ds4.c:48416).
# Usage: [STAGE=routed] [ABL2=shared] [N=4] tasks/glm_ablate_20260926.sh [out-dir]
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=${1:-$PWD/tasks/data/glm-ablate-$(date +%Y%m%d)}; mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
M=$HOME/models/gguf/GLM-5.3-Flash-UNCEN-d21b-L17-23Q4KExperts-KDAvoQ4K-Q2.gguf
P=$PWD/speed-bench/promessi_sposi.txt
STAGE=${STAGE:-routed}
ABL2=${ABL2:-}
N=${N:-4}
GEN=256
GAP=30
export DS4_METAL_DISABLE_TENSOR_API=1

echo "$0 STAGE=$STAGE ABL2=$ABL2 N=$N" > "$OUT/cmdline.txt"
(git rev-parse HEAD; git status --porcelain | wc -l | tr -d ' '; shasum -a 256 ds4-bench | cut -c1-16; sysctl -n machdep.cpu.brand_string; pmset -g | grep powermode; sysctl -n iogpu.wired_limit_mb) | tr '\n' ' ' > "$OUT/arms.txt"; echo >> "$OUT/arms.txt"

fans_off() { thermalforge auto >/dev/null 2>&1 || true; }
trap fans_off EXIT
thermalforge max >/dev/null 2>&1 || true

run() {  # $1 tag, $2 ablation stage (empty = baseline)
    local tag=$1 stage=$2
    if [[ -n $stage ]]; then export DS4_GLM_DECODE_ABLATE=$stage; else unset DS4_GLM_DECODE_ABLATE; fi
    echo "== $(date +%T) $tag start (ABLATE=${stage:-none})" >> "$OUT/run.log"
    ./ds4-bench --metal -m "$M" --prompt-file "$P" \
        --ctx-start 2048 --ctx-max 2048 --ctx-alloc 4096 \
        --gen-tokens "$GEN" --teacher-forced-decode --power 100 \
        --csv "$OUT/$tag.csv" > "$OUT/$tag.log" 2>&1
    echo "== $(date +%T) $tag rc=$? steady=$(awk -F, 'NR>1{print $NF}' "$OUT/$tag.csv" 2>/dev/null | tail -1)" >> "$OUT/run.log"
    unset DS4_GLM_DECODE_ABLATE
    sleep "$GAP"
}

i=0
while (( i < N )); do
    ((i+=1))
    run "base-$i" ""
    run "abl-$STAGE-$i" "$STAGE"
done
if [[ -n $ABL2 ]]; then
    run "abl-$ABL2-1" "$ABL2"
    run "base-extra" ""
    run "abl-$ABL2-2" "$ABL2"
fi
echo "== $(date +%T) done" >> "$OUT/run.log"
