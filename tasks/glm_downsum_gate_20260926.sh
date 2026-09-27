#!/bin/bash
# Equivalence gate for DS4_METAL_ENABLE_DOWNSUM8 (GLM n8 down+sum single-device):
# 1) --perplexity-file over a fixed 16 KiB fixture, env off vs on: token-weighted
#    avg_nll must agree to <=1e-4 relative (only reduction order changes).
# 2) 48-token greedy generation from the same fixture, env off vs on: report
#    token-identical or first-divergence position (drift is expected to be rare;
#    any divergence must be explained before any speed claim is promoted).
# Usage: tasks/glm_downsum_gate_20260926.sh [out-dir]
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=${1:-$PWD/tasks/data/glm-downsum-gate-$(date +%Y%m%d)}; mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
M=$HOME/models/gguf/GLM-5.3-Flash-UNCEN-d21b-L17-23Q4KExperts-KDAvoQ4K-Q2.gguf
FIX=$OUT/fixture.txt
head -c 16384 speed-bench/promessi_sposi.txt > "$FIX"
export DS4_METAL_DISABLE_TENSOR_API=1

(git rev-parse HEAD; shasum -a 256 ds4 | cut -c1-16) | tr '\n' ' ' > "$OUT/arms.txt"; echo >> "$OUT/arms.txt"

for arm in off on; do
    if [[ $arm == on ]]; then export DS4_METAL_ENABLE_DOWNSUM8=1; else unset DS4_METAL_ENABLE_DOWNSUM8; fi
    ./ds4 --metal -m "$M" --perplexity-file "$FIX" > "$OUT/nll-$arm.log" 2>&1
    echo "ppl rc=$? arm=$arm" >> "$OUT/run.log"
    ./ds4 --metal -m "$M" --prompt-file "$FIX" -n 48 --temp 0 \
        --dump-logprobs "$OUT/lp-$arm.json" > "$OUT/greedy-$arm.log" 2>&1
    unset DS4_METAL_ENABLE_DOWNSUM8
    sleep 15
done

echo '--- NLL ---'; grep -h 'avg_nll\|perplexity' "$OUT/nll-off.log" "$OUT/nll-on.log" | tail -6
echo '--- GREEDY DIFF ---'
if diff -q <(sed -n '/^/p' "$OUT/greedy-off.log") <(sed -n '/^/p' "$OUT/greedy-on.log") >/dev/null; then
    echo "greedy outputs identical"
else
    diff "$OUT/greedy-off.log" "$OUT/greedy-on.log" | head -20
fi
