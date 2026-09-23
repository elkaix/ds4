#!/bin/bash
# Stage A decode A/B: O1-uncen (kda_v/kda_output Q8_0) vs the same model with
# only those 68 tensors at Q4_K. Production engine settings: Tensor API off,
# no MTP. Teacher-forced decode so both arms decode identical tokens.
# Short sweep: 2K frontier, ABBA x REPS. Long sweep: 32K and 100K frontiers,
# one ABBA cycle. Fixed fans, cool-down between processes, nothing else on GPU.
# Mid sweep: 32K frontier alone, same 100K allocation as the long sweep, ABBA x REPS.
# MA/MB override the two arms (default: O1-uncen vs stage A).
# Usage: [MA=a.gguf MB=b.gguf] [SWEEP=short,long|mid] [REPS=n] tasks/glm_stageA_ab.sh [out-dir]
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=${1:-$PWD/tasks/data/glm-stageA-ab-$(date +%Y%m%d)}; mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
MA=${MA:-$HOME/models/gguf/GLM-5.3-Flash-UNCEN-d21b-L17-18-19Q4KExperts-Q2.gguf}
MB=${MB:-$HOME/models/gguf/GLM-5.3-Flash-UNCEN-d21b-L17-18-19Q4KExperts-KDAvoQ4K-Q2.gguf}
P=$PWD/speed-bench/promessi_sposi.txt
GEN=256
GAP=30
REPS=${REPS:-5}   # ABBA cycles for the short and mid sweeps (2 runs per arm per cycle)
SWEEP=${SWEEP:-short,long}
export DS4_METAL_DISABLE_TENSOR_API=1

echo "$0 $*" > "$OUT/cmdline.txt"
(git rev-parse HEAD; shasum -a 256 ds4-bench | cut -c1-16; sysctl -n machdep.cpu.brand_string; pmset -g | grep powermode) \
    | tr '\n' ' ' > "$OUT/arms.txt"; echo >> "$OUT/arms.txt"
echo "A=$MA" >> "$OUT/arms.txt"; echo "B=$MB" >> "$OUT/arms.txt"

fans_off() { thermalforge auto >/dev/null 2>&1 || true; }
trap fans_off EXIT
thermalforge max >/dev/null 2>&1 || true

run() {  # $1 arm(A|B) $2 tag $3.. bench sweep args
    local arm=$1 tag=$2; shift 2
    local model=$MA; [[ $arm == B ]] && model=$MB
    echo "== $(date +%T) $tag start" >> "$OUT/run.log"
    ./ds4-bench --metal -m "$model" --prompt-file "$P" "$@" \
        --gen-tokens "$GEN" --teacher-forced-decode --power 100 \
        --csv "$OUT/$tag.csv" > "$OUT/$tag.log" 2>&1
    echo "== $(date +%T) $tag rc=$? rows=$(tail -n +2 "$OUT/$tag.csv" 2>/dev/null | wc -l | tr -d ' ')" >> "$OUT/run.log"
    sleep "$GAP"
}

SHORT=(--ctx-start 2048 --ctx-max 2048 --ctx-alloc 4096)
LONG=(--ctx-start 32768 --step-incr 67584 --ctx-max 100352 --ctx-alloc 102400)
MID=(--ctx-start 32768 --ctx-max 32768 --ctx-alloc 102400)

abba() {  # $1 tag prefix, $2.. sweep args
    local kind=$1 c; shift
    for ((c = 1; c <= REPS; c++)); do
        run A "$kind-A-$c-1" "$@"
        run B "$kind-B-$c-1" "$@"
        run B "$kind-B-$c-2" "$@"
        run A "$kind-A-$c-2" "$@"
    done
}

if [[ ,$SWEEP, == *,short,* ]]; then abba short "${SHORT[@]}"; fi
if [[ ,$SWEEP, == *,long,* ]]; then
    run A long-A-1 "${LONG[@]}"
    run B long-B-1 "${LONG[@]}"
    run B long-B-2 "${LONG[@]}"
    run A long-A-2 "${LONG[@]}"
fi
if [[ ,$SWEEP, == *,mid,* ]]; then abba mid "${MID[@]}"; fi
echo "== $(date +%T) done" >> "$OUT/run.log"
