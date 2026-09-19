#!/bin/bash
# P4b1 Protocol B: PR #964 engine-only (pr964-engine) vs its own base (pr964-base),
# frozen worktree binaries, MTP off, Metal 4 tensor route disabled on both arms,
# interleaved arm order per frontier, logits dumped at every frontier/run.
# Each run has a 1024-token warm-up frontier first (model pages re-fault after a
# process restart; the first measured frontier otherwise pays the page-in).
# Usage: tasks/p4b1_bench.sh [out-dir]  (no server on :8000, nothing else on the GPU)
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=${1:-$PWD/tasks/data/p4b1/bench}; mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
M=$HOME/models/gguf/GLM-5.3-Flash-UNCEN-d21b-L17-18-19Q4KExperts-Q2.gguf
P=$PWD/speed-bench/promessi_sposi.txt
GEN=${GEN:-256}
ARM_A=$HOME/Projects/open-source/ds4-glm53-964base; ARM_B=$HOME/Projects/open-source/ds4-glm53-964eng
armdir() { if [[ $1 == A ]]; then echo "$ARM_A"; else echo "$ARM_B"; fi; }
for a in A B; do (cd "$(armdir $a)" && echo "$a $(git rev-parse HEAD) $(shasum -a 256 ds4-bench | cut -c1-16)"); done | tee "$OUT/arms.txt"
run() {  # $1 arm $2 ctx $3 runidx
    local d; d=$(armdir "$1"); local tag="$2-$1-$3"
    mkdir -p "$OUT/logits-$tag"
    macmon pipe -s 1 2>/dev/null | head -1 > "$OUT/$tag.pre.json"
    (cd "$d" && DS4_METAL_DISABLE_METAL4=1 ./ds4-bench --metal -m "$M" --prompt-file "$P" \
        --ctx-start 1024 --step-incr $(( $2 - 1024 )) --ctx-max "$2" --gen-tokens "$GEN" \
        --csv "$OUT/$tag.csv" --dump-frontier-logits-dir "$OUT/logits-$tag" > "$OUT/$tag.log" 2>&1)
    echo "$(date +%T) $tag rc=$? $(tail -1 "$OUT/$tag.csv")"
    sleep "${GAP:-45}"
}
sched() {  # $1 ctx, rest = order
    local ctx=$1; shift; local nA=0 nB=0
    for a in "$@"; do if [[ $a == A ]]; then nA=$((nA+1)); run A "$ctx" $nA; else nB=$((nB+1)); run B "$ctx" $nB; fi; done
}
echo "== $(date +%T) start GEN=$GEN"
sched 2048   A B B A B A A B
sched 32768  A B B A B A A B
sched 102400 B A A B
sched 204800 A B B A
echo "== $(date +%T) done"
