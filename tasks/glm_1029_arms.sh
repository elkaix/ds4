#!/bin/bash
# Issue #1029 attribution: which half of 0e9cc2d costs GLM 5.3 prefill on M5?
# One GGUF (stage 2), three ds4-bench binaries built from the same commit with
# exactly one diff each (arms A/B/C below). B and C are NOT exact; speed only.
#   A base  - HEAD
#   B dense - pre-0e9cc2d dense-attention limit (ctx_cap instead of top_k+pool-1)
#   C valid - pre-0e9cc2d sparse prefill kernel (_valid_tensor for glm53)
# Per run: cold prefill 8K/16K/24K (separate processes) and a 2K continued
# prefill on a 14K prefix. ABC CBA x REPS, fixed fans, 30 s gaps.
# Usage: ARMS=/path/to/worktrees [REPS=n] tasks/glm_1029_arms.sh [out-dir]
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=${1:-$PWD/tasks/data/glm-1029-arms-$(date +%Y%m%d)}; mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
ARMS=${ARMS:-$HOME/Projects/open-source/ds4-1029-arms}
M=${M:-$HOME/models/gguf/GLM-5.3-Flash-UNCEN-d21b-L17-23Q4KExperts-KDAvoQ4K-Q2.gguf}
GEN=256
GAP=30
ALLOC=102400
REPS=${REPS:-3}
export DS4_METAL_DISABLE_TENSOR_API=1

dir() { case $1 in A) echo base ;; B) echo dense ;; C) echo valid ;; esac; }
echo "$0 $*" > "$OUT/cmdline.txt"
{
    echo "base_commit $(git -C "$ARMS/base" rev-parse HEAD)"
    sysctl -n machdep.cpu.brand_string; pmset -g | grep powermode
    echo "model $M"
    for a in A B C; do
        echo "$a=$(dir $a) $(shasum -a 256 "$ARMS/$(dir $a)/ds4-bench" | cut -c1-16)"
    done
    for a in B C; do git -C "$ARMS/$(dir $a)" diff; done
} > "$OUT/arms.txt"

fans_off() { thermalforge auto >/dev/null 2>&1 || true; }
trap fans_off EXIT
thermalforge max >/dev/null 2>&1 || true

bench() {  # $1 arm $2 tag $3.. sweep args
    local arm=$1 tag=$2; shift 2
    echo "== $(date +%T) $tag start" >> "$OUT/run.log"
    (cd "$ARMS/$(dir "$arm")" && ./ds4-bench --metal -m "$M" \
        --prompt-file speed-bench/promessi_sposi.txt "$@" --ctx-alloc "$ALLOC" \
        --gen-tokens "$GEN" --teacher-forced-decode --power 100 \
        --csv "$OUT/$tag.csv") > "$OUT/$tag.log" 2>&1
    echo "== $(date +%T) $tag rc=$? rows=$(tail -n +2 "$OUT/$tag.csv" 2>/dev/null | wc -l | tr -d ' ')" >> "$OUT/run.log"
    sleep "$GAP"
}

run() {  # $1 arm $2 cycle $3 slot
    local arm=$1 id="$1-$2-$3" n
    for n in 8192 16384 24576; do
        bench "$arm" "cold$((n / 1024))k-$id" --ctx-start "$n" --ctx-max "$n"
    done
    bench "$arm" "cont14k-$id" --ctx-start 14336 --step-incr 2048 --ctx-max 16384
}

for ((c = 1; c <= REPS; c++)); do
    run A "$c" 1; run B "$c" 1; run C "$c" 1
    run C "$c" 2; run B "$c" 2; run A "$c" 2
done
echo "== $(date +%T) done" >> "$OUT/run.log"
