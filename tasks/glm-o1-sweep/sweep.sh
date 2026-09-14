#!/usr/bin/env bash
# GLM O1-uncen layer-sensitivity sweep. Phase 1: six 7-layer groups over MoE layers 3-44
# spliced to Q4_K, scored with score_official on the GLM fp8-100 fixture. Phase 2: every
# single layer inside the two best groups. Output: results.tsv (arm, layers, avg_nll,
# first_match, avg_lcp). GPU must be free; each arm ~100 GiB resident.
set -Eeuo pipefail
cd ~/Projects/open-source/ds4-glm53
S=tasks/glm-o1-sweep
Q2=~/models/gguf/GLM-5.3-Flash-UNCEN-d21b-Q2.gguf
D=~/models/gguf/tmp-glm53-uncen-d21b-q4-donor.gguf
MIX=~/models/gguf/tmp-glm-sweep-arm.gguf
MAN=gguf-tools/quality-testing/data/glm53-flash-openrouter-zai-fp8-100/manifest.tsv
RES=$S/results.tsv
exec > >(tee -a "$S/sweep.log") 2>&1

while pgrep -f 'score_official|ds4-server|/ds4 --metal' >/dev/null; do sleep 20; done
[[ -s $RES ]] || printf 'arm\tlayers\tavg_nll\tfirst_match\tavg_lcp\n' > "$RES"

run_arm() {  # $1 arm name, $2 layer spec
  grep -q "^$1	" "$RES" && { echo "== $1 already scored"; return; }
  echo "== $(date +%T) $1 layers=$2 splice"
  python3 $S/splice_glm.py --base "$Q2" --donor "$D" --out "$MIX" --q4-layers "$2" --force >/dev/null
  echo "== $(date +%T) $1 score"
  ./gguf-tools/quality-testing/score_official "$MIX" "$MAN" "$S/$1.tsv" 4096 > "$S/$1.log" 2>&1 || echo "!! score $1 nonzero"
  local sum; sum=$(grep -m1 '^summary' "$S/$1.log" || echo "summary avg_nll=NA first_match=NA avg_lcp=NA")
  local nll fm lcp
  nll=$(sed -n 's/.*avg_nll=\([^ ]*\).*/\1/p' <<<"$sum"); fm=$(sed -n 's/.*first_match=\([^ ]*\).*/\1/p' <<<"$sum"); lcp=$(sed -n 's/.*avg_lcp=\([^ ]*\).*/\1/p' <<<"$sum")
  printf '%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$nll" "$fm" "$lcp" >> "$RES"
  echo "== $(date +%T) $1 $sum"
  rm -f "$MIX"
}

base=$(grep -m1 '^summary' $S/base-q2.log | sed -n 's/.*avg_nll=\([^ ]*\).*/\1/p')
grep -q '^base	' "$RES" || printf 'base\t-\t%s\t%s\t%s\n' "$base" \
  "$(sed -n 's/.*first_match=\([^ ]*\).*/\1/p' <<<"$(grep -m1 ^summary $S/base-q2.log)")" \
  "$(sed -n 's/.*avg_lcp=\([^ ]*\).*/\1/p' <<<"$(grep -m1 ^summary $S/base-q2.log)")" >> "$RES"

for g in 3-9 10-16 17-23 24-30 31-37 38-44; do run_arm "g$g" "$g"; done

# phase 2: singles in the two best groups (lowest avg_nll)
for g in $(awk -F'\t' 'NR>1 && $1 ~ /^g/ && $3!="NA" {print $3"\t"$2}' "$RES" | sort -n | head -2 | cut -f2); do
  lo=${g%-*}; hi=${g#*-}
  for l in $(seq "$lo" "$hi"); do run_arm "l$l" "$l"; done
done

echo "== $(date +%T) RANKING (lower avg_nll is better)"
awk -F'\t' 'NR>1 && $1 ~ /^l/ {print $3"\t"$1}' "$RES" | sort -n | head -10
echo "== $(date) GLM_SWEEP_COMPLETE"
