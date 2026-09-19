#!/bin/bash
# Targeted #1051 reachability: count GLM decode attention kernel choice.
# Uses instrumented ds4-bench (NOT ds4-bench.p4b1-frozen timing binaries).
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:-$PWD/tasks/data/p4b1/dispatch-count}
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
M=$HOME/models/gguf/GLM-5.3-Flash-UNCEN-d21b-L17-18-19Q4KExperts-Q2.gguf
P=$PWD/speed-bench/promessi_sposi.txt
GEN=${GEN:-256}
GAP=${GAP:-20}
ARM_A=$HOME/Projects/open-source/ds4-glm53-964base
ARM_B=$HOME/Projects/open-source/ds4-glm53-964eng

run() {
  local arm=$1 dir=$2 ctx=$3
  local tag="${ctx}-${arm}"
  echo "$(date +%T) start $tag"
  (cd "$dir" && DS4_METAL_DISABLE_METAL4=1 DS4_GLM_DISPATCH_COUNT=1 \
    ./ds4-bench --metal -m "$M" --prompt-file "$P" \
      --ctx-start 1024 --step-incr $((ctx - 1024)) --ctx-max "$ctx" \
      --gen-tokens "$GEN" --csv "$OUT/$tag.csv" > "$OUT/$tag.log" 2>&1)
  echo "$(date +%T) done $tag"
  grep 'ds4-dispatch:' "$OUT/$tag.log" | tee -a "$OUT/summary.txt" || {
    echo "FAIL: no ds4-dispatch line in $tag.log" | tee -a "$OUT/summary.txt"
    return 1
  }
}

echo "== $(date +%T) dispatch-count GEN=$GEN" | tee "$OUT/summary.txt"
for a in A B; do
  d=$ARM_A; [[ $a == B ]] && d=$ARM_B
  echo "$a $(cd "$d" && git rev-parse HEAD) $(shasum -a 256 "$d/ds4-bench" | cut -c1-16)"
done | tee -a "$OUT/summary.txt"

run A "$ARM_A" 2048
sleep "$GAP"
run B "$ARM_B" 2048
sleep "$GAP"
run A "$ARM_A" 32768
sleep "$GAP"
run B "$ARM_B" 32768
echo "== $(date +%T) done" | tee -a "$OUT/summary.txt"
python3 - <<'PY' "$OUT"
from pathlib import Path
import re, sys
out = Path(sys.argv[1])
pat = re.compile(r"ds4-dispatch: glm53_exact_sparse=(\d+) split_group8_checked=(\d+) split_group8_unchecked=(\d+) generic_indexed=(\d+)")
print("\n#1051 reachability")
print(f"{'tag':<12} {'exact':>8} {'sg8_chk':>8} {'sg8_un':>8} {'generic':>8} verdict")
fail=False
for tag, expect in [
    ("2048-A", "A"), ("2048-B", "B"),
    ("32768-A", "A"), ("32768-B", "B"),
]:
    log = (out/f"{tag}.log").read_text(errors="replace")
    m = pat.search(log)
    if not m:
        print(f"{tag:<12} MISSING"); fail=True; continue
    exact, chk, un, gen = map(int, m.groups())
    if expect=="A":
        ok = exact==0 and chk==0 and un==0 and gen>0
        why = "generic only" if ok else "UNEXPECTED"
    else:
        ok = exact>0 and chk==0 and un==0
        why = "exact, no sg8" if ok else "UNEXPECTED"
    if not ok: fail=True
    print(f"{tag:<12} {exact:8d} {chk:8d} {un:8d} {gen:8d} {why}")
print("DISPATCH_COUNT_PASS" if not fail else "DISPATCH_COUNT_FAIL")
sys.exit(0 if not fail else 1)
PY
