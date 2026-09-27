#!/bin/bash
# Continuation after wrapper timeout: abl-routed-4, shared triplet, and MoE
# stage profiler (LAYER-filtered only) on one IQ2 layer (5) and one Q4_K layer (20).
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=$PWD/tasks/data/glm-ablate-20260926
M=$HOME/models/gguf/GLM-5.3-Flash-UNCEN-d21b-L17-23Q4KExperts-KDAvoQ4K-Q2.gguf
P=$PWD/speed-bench/promessi_sposi.txt
GAP=30
export DS4_METAL_DISABLE_TENSOR_API=1
thermalforge max >/dev/null 2>&1 || true

run() {  # $1 tag, $2 ablate stage or empty, $3 extra env assignments (space-sep)
    local tag=$1 stage=$2 extra=$3
    [[ -n $stage ]] && export DS4_GLM_DECODE_ABLATE=$stage || unset DS4_GLM_DECODE_ABLATE
    if [[ -n $extra ]]; then
        local item
        for item in $extra; do export "${item%%=*}=${item#*=}"; done
    fi
    ./ds4-bench --metal -m "$M" --prompt-file "$P" \
        --ctx-start 2048 --ctx-max 2048 --ctx-alloc 4096 \
        --gen-tokens 256 --teacher-forced-decode --power 100 \
        --csv "$OUT/$tag.csv" > "$OUT/$tag.log" 2>&1
    echo "== $(date +%T) $tag rc=$?" >> "$OUT/run.log"
    unset DS4_GLM_DECODE_ABLATE DS4_METAL_MOE_ONE_STAGE_PROFILE DS4_METAL_MOE_ONE_STAGE_PROFILE_LAYER
    sleep "$GAP"
}


run abl-shared-1 shared ""
run base-extra "" ""
run abl-shared-2 shared ""
export DS4_METAL_MOE_ONE_STAGE_PROFILE_LAYER=5
run prof-l5f "" "DS4_METAL_MOE_ONE_STAGE_PROFILE=1"
export DS4_METAL_MOE_ONE_STAGE_PROFILE_LAYER=20
run prof-l20f "" "DS4_METAL_MOE_ONE_STAGE_PROFILE=1"
thermalforge auto >/dev/null 2>&1 || true
echo "== $(date +%T) cont-done" >> "$OUT/run.log"
