#!/bin/bash
# P4b1-C: 256-frame F32 decode traces A vs B at 2K (Metal4-off, MTP off).
# Requires rebuilt ds4-bench with DS4_BENCH_DUMP_GEN_LOGITS_DIR support.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:-$PWD/tasks/data/p4b1/decode-trace-2k}
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
M=$HOME/models/gguf/GLM-5.3-Flash-UNCEN-d21b-L17-18-19Q4KExperts-Q2.gguf
P=$PWD/speed-bench/promessi_sposi.txt
GEN=${GEN:-256}
CTX=${CTX:-2048}
GAP=${GAP:-45}
ARM_A=$HOME/Projects/open-source/ds4-glm53-964base
ARM_B=$HOME/Projects/open-source/ds4-glm53-964eng

run_arm() {
  local arm=$1 dir=$2
  local tag="2048-$arm"
  mkdir -p "$OUT/logits-$tag"
  echo "$(date +%T) start $tag"
  (cd "$dir" && DS4_METAL_DISABLE_METAL4=1 \
    DS4_BENCH_DUMP_GEN_LOGITS_DIR="$OUT/logits-$tag" \
    ./ds4-bench --metal -m "$M" --prompt-file "$P" \
      --ctx-start 1024 --step-incr $((CTX - 1024)) --ctx-max "$CTX" \
      --gen-tokens "$GEN" --csv "$OUT/$tag.csv" \
      > "$OUT/$tag.log" 2>&1)
  local rc=$?
  echo "$(date +%T) done $tag rc=$rc steps=$(wc -l < "$OUT/logits-$tag/tokens.txt" 2>/dev/null || echo 0)"
  return $rc
}

echo "== $(date +%T) decode-trace CTX=$CTX GEN=$GEN OUT=$OUT"
for a in A B; do
  d=$ARM_A; [[ $a == B ]] && d=$ARM_B
  echo "$a $(cd "$d" && git rev-parse HEAD) $(shasum -a 256 "$d/ds4-bench" | cut -c1-16)"
done | tee "$OUT/arms.txt"

run_arm A "$ARM_A"
sleep "$GAP"
run_arm B "$ARM_B"

python3 "$(dirname "$0")/p4b1_compare_decode_traces.py" \
  "$OUT/logits-2048-A" "$OUT/logits-2048-B" | tee "$OUT/compare.txt"
echo "== $(date +%T) done"
