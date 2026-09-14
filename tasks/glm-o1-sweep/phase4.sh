#!/usr/bin/env bash
# Phases 4-6, run after phase3.sh exits (explicit PID + sentinel, no process-name matching).
#  4: paired bootstrap analysis of the 10 triples; confirm-score every triple whose 95% CI
#     of (delta vs avg_nll winner) contains 0, not just a fixed 0.001 gap.
#  5: one-layer-swap neighborhood search around the winner over layers 10..23, repeated
#     until no swap lowers avg_nll (local optimum among 3-layer sets).
#  6: held-out check: wikitext-2 test ppl (dev fixture never saw it) for fresh Q2, local
#     optimum, best other triple, g10-16, g17-23.
set -Eeuo pipefail
cd ~/Projects/open-source/ds4-glm53
S=tasks/glm-o1-sweep
RES=$S/results.tsv
exec > >(tee -a "$S/phase4.log") 2>&1
P3PID=${PHASE3_PID:?set PHASE3_PID}
while kill -0 "$P3PID" 2>/dev/null; do sleep 30; done
grep -q GLM_PHASE3_COMPLETE "$S/phase3.log" || { echo "!! phase3 exited without sentinel"; exit 1; }
eval "$(sed -n '/^run_arm()/,/^}/p' "$S/sweep.sh")"
Q2=~/models/gguf/GLM-5.3-Flash-UNCEN-d21b-Q2.gguf
D=~/models/gguf/tmp-glm53-uncen-d21b-q4-donor.gguf
MIX=~/models/gguf/tmp-glm-sweep-arm.gguf
MAN=gguf-tools/quality-testing/data/glm53-flash-openrouter-zai-fp8-100/manifest.tsv
WIKI=~/models/ds4/assets/heldout/wikitext-2-raw/wiki.test.raw
HELD=$S/heldout.tsv

nll_of() { awk -F'\t' -v a="$1" '$1==a {print $3}' "$RES" | head -1; }
spec_of() { awk -F'\t' -v a="$1" '$1==a {print $2}' "$RES" | head -1; }
best_arm() { awk -F'\t' 'NR>1 && $1 ~ /^(t|n)[0-9]/ && $3!="NA" {print $3"\t"$1}' "$RES" | sort -n | head -1 | cut -f2; }

echo "== $(date +%T) phase 4: paired analysis"
mapfile -t triples < <(awk -F'\t' 'NR>1 && $1 ~ /^t[0-9]/ && $3!="NA" {print $3"\t"$1}' "$RES" | sort -n | cut -f2)
win=${triples[0]}
python3 $S/paired.py --json $S/paired-vs-q2.json base-q2 "${triples[@]}"
python3 $S/paired.py --json $S/paired-vs-win.json "$win" "${triples[@]:1}"
mapfile -t close < <(python3 -c "
import json; j=json.load(open('$S/paired-vs-win.json'))
print('\n'.join(a for a,s in j['arms'].items() if s['ci_lo']<=0<=s['ci_hi']))")
echo "== winner $win; CI-overlapping: ${close[*]:-none}"
for a in "${close[@]}"; do run_arm "confirm-$(spec_of "$a")" "$(spec_of "$a")"; done

echo "== $(date +%T) phase 5: one-layer-swap search from $win"
cur=$(spec_of "$win")
while :; do
  IFS=, read -r a b c <<<"$cur"; improved=0
  for x in $(seq 10 23); do
    for cand in "$x,$b,$c" "$a,$x,$c" "$a,$b,$x"; do
      sorted=$(tr , '\n' <<<"$cand" | sort -n | uniq | tr '\n' , | sed 's/,$//')
      [[ $(tr , '\n' <<<"$sorted" | wc -l) -eq 3 ]] || continue
      name="n$(tr , - <<<"$sorted")"
      grep -q "^t$(tr , - <<<"$sorted")	" "$RES" && continue   # already scored as a triple
      run_arm "$name" "$sorted"
    done
  done
  new=$(best_arm)
  if [[ "$(spec_of "$new")" != "$cur" ]]; then echo "== swap improved: $cur -> $(spec_of "$new") ($(nll_of "$new"))"; cur=$(spec_of "$new"); improved=1; fi
  [[ $improved -eq 1 ]] || break
done
opt=$(best_arm); echo "== local optimum $opt spec=$(spec_of "$opt") avg_nll=$(nll_of "$opt")"
run_arm "confirm-opt-$(spec_of "$opt")" "$(spec_of "$opt")"
python3 $S/paired.py base-q2 "$opt" $(awk -F'\t' 'NR>1 && $1 ~ /^n[0-9]/ && $3!="NA" {print $3"\t"$1}' "$RES" | sort -n | head -5 | cut -f2 | grep -v "^$opt$" || true)

echo "== $(date +%T) phase 6: held-out wikitext-2 ppl (12000 scored tokens, ctx 16384, MTP off)"
[[ -s $HELD ]] || printf 'arm\tlayers\tscored\tavg_nll\tppl\n' > "$HELD"
held() {  # $1 name, $2 spec ('-' = fresh Q2)
  grep -q "^$1	" "$HELD" && return
  local m=$Q2
  if [[ $2 != - ]]; then python3 $S/splice_glm.py --base "$Q2" --donor "$D" --out "$MIX" --q4-layers "$2" --force >/dev/null; m=$MIX; fi
  local line; line=$(./ds4 -m "$m" --metal --ctx 16384 -n 12000 --perplexity-file "$WIKI" 2>"$S/held-$1.log" | grep -m1 '^tokens=' || echo 'scored=NA avg_nll=NA ppl=NA')
  printf '%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$(sed -n 's/.*scored=\([^ ]*\).*/\1/p' <<<"$line")" \
    "$(sed -n 's/.*avg_nll=\([^ ]*\).*/\1/p' <<<"$line")" "$(sed -n 's/.*ppl=\([^ ]*\).*/\1/p' <<<"$line")" >> "$HELD"
  echo "== $(date +%T) held $1 $line"; rm -f "$MIX"
}
held base-q2 -
held "$opt" "$(spec_of "$opt")"
second=$(awk -F'\t' 'NR>1 && $1 ~ /^(t|n)[0-9]/ && $3!="NA" {print $3"\t"$1}' "$RES" | sort -n | cut -f2 | grep -v "^$opt$" | head -1)
held "$second" "$(spec_of "$second")"
held g10-16 10-16
held g17-23 17-23
cat "$HELD"
echo "== $(date) GLM_PHASE4_COMPLETE optimum=$opt"
