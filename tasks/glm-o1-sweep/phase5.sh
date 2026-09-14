#!/usr/bin/env bash
# Phase 5: multi-start one-layer-swap search over MoE layers 10..23, seeded from the three
# statistically tied phase-3 triples (best mean, robustness pick, best imp/reg count).
# Each seed climbs until no swap lowers avg_nll; all arms dedup'd through results.tsv.
# Then: paired robustness table (P95, worst-5 mean, imp/reg, max) vs fresh Q2 and vs the
# best arm, and WikiText-2 ppl as an independent sanity metric (not the promotion gate).
set -Eeuo pipefail
cd ~/Projects/open-source/ds4-glm53
S=tasks/glm-o1-sweep
RES=$S/results.tsv
exec > >(tee -a "$S/phase5.log") 2>&1
eval "$(sed -n '/^run_arm()/,/^}/p' "$S/sweep.sh")"
Q2=~/models/gguf/GLM-5.3-Flash-UNCEN-d21b-Q2.gguf
D=~/models/gguf/tmp-glm53-uncen-d21b-q4-donor.gguf
MIX=~/models/gguf/tmp-glm-sweep-arm.gguf
MAN=gguf-tools/quality-testing/data/glm53-flash-openrouter-zai-fp8-100/manifest.tsv
WIKI=~/models/ds4/assets/heldout/wikitext-2-raw/wiki.test.raw
HELD=$S/heldout.tsv
SEEDS="16,18,19 17,18,19 14,17,19"

norm() { tr , '\n' <<<"$1" | sort -n | uniq | tr '\n' , | sed 's/,$//'; }
nll_of_spec() { awk -F'\t' -v s="$1" '$2==s && $3!="NA" {print $3; exit}' "$RES"; }
name_of_spec() { awk -F'\t' -v s="$1" '$2==s && $3!="NA" {print $1; exit}' "$RES"; }
score_spec() {  # scores spec once under name n<a>-<b>-<c> unless any arm already has it
  [[ -n "$(nll_of_spec "$1")" ]] || run_arm "n$(tr , - <<<"$1")" "$1"
}

echo "== $(date +%T) phase 5: multi-start swap search, seeds: $SEEDS"
for seed in $SEEDS; do
  cur=$seed; echo "== seed $seed ($(nll_of_spec "$seed"))"
  while :; do
    IFS=, read -r a b c <<<"$cur"; best=$cur; bestn=$(nll_of_spec "$cur")
    for x in $(seq 10 23); do
      for cand in "$x,$b,$c" "$a,$x,$c" "$a,$b,$x"; do
        n=$(norm "$cand"); [[ $(tr , '\n' <<<"$n" | wc -l) -eq 3 ]] || continue
        score_spec "$n"
        v=$(nll_of_spec "$n"); [[ -n $v ]] || continue
        if python3 -c "import sys; sys.exit(0 if $v < $bestn else 1)"; then best=$n; bestn=$v; fi
      done
    done
    [[ $best == "$cur" ]] && break
    echo "== seed $seed: $cur -> $best ($bestn)"; cur=$best
  done
  echo "== seed $seed local optimum: $cur ($(nll_of_spec "$cur"))"
done

echo "== $(date +%T) ranking of all 3-layer arms"
awk -F'\t' 'NR>1 && $1 ~ /^(t|n)[0-9]/ && $3!="NA" {print $3"\t"$4"\t"$5"\t"$1"\t"$2}' "$RES" | sort -n | head -12
mapfile -t top < <(awk -F'\t' 'NR>1 && $1 ~ /^(t|n)[0-9]/ && $3!="NA" {print $3"\t"$1}' "$RES" | sort -n | cut -f2 | head -8)
best=${top[0]}
echo "== paired vs fresh Q2"; python3 $S/paired.py --json $S/paired5-vs-q2.json base-q2 "${top[@]}"
echo "== paired vs best $best"; python3 $S/paired.py --json $S/paired5-vs-best.json "$best" "${top[@]:1}"

echo "== $(date +%T) WikiText-2 ppl (independent sanity metric; 12000 scored tokens, ctx 16384, MTP off)"
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
held t16-18-19 16,18,19
held t17-18-19 17,18,19
held t14-17-19 14,17,19
[[ $best == t16-18-19 || $best == t17-18-19 || $best == t14-17-19 ]] || held "$best" "$(awk -F'\t' -v a="$best" '$1==a {print $2; exit}' "$RES")"
held g17-23 17-23
cat "$HELD"
echo "== $(date) GLM_PHASE5_COMPLETE best=$best"
