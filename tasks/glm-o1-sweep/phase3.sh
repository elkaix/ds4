#!/usr/bin/env bash
# Phase 3: top-5 single layers -> all C(5,3)=10 triples, scored vs the fresh Q2 control.
# Winner = lowest avg_nll, ties broken by first_match then avg_lcp. The winner is
# re-spliced and re-scored once (confirm) before it becomes GLM O1-uncen.
# Also prints per-group additivity: observed group gain vs sum of its singles.
set -Eeuo pipefail
cd ~/Projects/open-source/ds4-glm53
S=tasks/glm-o1-sweep
RES=$S/results.tsv
exec > >(tee -a "$S/phase3.log") 2>&1
until grep -q GLM_SWEEP_COMPLETE "$S/sweep.log" 2>/dev/null; do sleep 60; done
while pgrep -f 'score_official|ds4-server|/ds4 --metal' >/dev/null; do sleep 20; done
# reuse run_arm from sweep.sh without re-running its phases
eval "$(sed -n '/^run_arm()/,/^}/p' "$S/sweep.sh")"
Q2=~/models/gguf/GLM-5.3-Flash-UNCEN-d21b-Q2.gguf
D=~/models/gguf/tmp-glm53-uncen-d21b-q4-donor.gguf
MIX=~/models/gguf/tmp-glm-sweep-arm.gguf
MAN=gguf-tools/quality-testing/data/glm53-flash-openrouter-zai-fp8-100/manifest.tsv

echo "== $(date +%T) additivity (group gain vs sum of single gains, vs base)"
python3 - "$RES" <<'EOF'
import csv,sys
r={x['arm']:x for x in csv.DictReader(open(sys.argv[1]),delimiter='\t') if x['avg_nll']!='NA'}
b=float(r['base']['avg_nll'])
for g in [a for a in r if a.startswith('g')]:
    lo,hi=map(int,r[g]['layers'].split('-'))
    s=[b-float(r[f'l{l}']['avg_nll']) for l in range(lo,hi+1) if f'l{l}' in r]
    if s: print(f"{g}: observed={b-float(r[g]['avg_nll']):+.4f} sum_singles={sum(s):+.4f} n={len(s)}")
EOF

top5=$(awk -F'\t' 'NR>1 && $1 ~ /^l[0-9]+$/ && $3!="NA" {print $3"\t"substr($1,2)}' "$RES" | sort -n | head -5 | cut -f2 | sort -n | tr '\n' ' ')
echo "== top5 singles: $top5"
read -ra T <<<"$top5"
for ((i=0;i<5;i++)); do for ((j=i+1;j<5;j++)); do for ((k=j+1;k<5;k++)); do
  spec="${T[i]},${T[j]},${T[k]}"; run_arm "t${T[i]}-${T[j]}-${T[k]}" "$spec"
done; done; done

win=$(awk -F'\t' 'NR>1 && $1 ~ /^t/ && $3!="NA" {print $3"\t"(1000-$4)"\t"(100-$5)"\t"$2}' "$RES" | sort -n -k1,1 -k2,2 -k3,3 | head -1 | cut -f4)
echo "== winning triple: $win"
run_arm "confirm-$win" "$win"
echo "== $(date +%T) triples ranking"
awk -F'\t' 'NR>1 && $1 ~ /^(t|confirm)/ {print $3"\t"$4"\t"$5"\t"$1}' "$RES" | sort -n
echo "== $(date) GLM_PHASE3_COMPLETE winner=$win"
