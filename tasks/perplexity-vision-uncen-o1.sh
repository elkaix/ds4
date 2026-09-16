#!/usr/bin/env bash
# Teacher-forced NLL: orcarouter Vision-Uncensored Q2 (base) vs O1 (layers 37-42 Q4_K experts).
# Deterministic, so no interleaving needed. Needs the GPU free (no ds4-server running).
set -uo pipefail
cd ~/Projects/open-source/ds4
FIX=~/.ds4/ppl/ppl-readme-ds4c.txt
V=~/models/gguf/DeepSeek-V4-Flash-Vision-Encoder.gguf
BASE=~/models/gguf/DeepSeek-V4-Flash-Vision-Uncensored-orcarouter-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8.gguf
O1=~/models/gguf/DeepSeek-V4-Flash-Vision-Uncensored-orcarouter-Layers37-42Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-AProjQ8-SExpQ8-OutQ8.gguf
LOG=tasks/$(date +%F)-perplexity-vision-uncen-base-vs-o1.log
for arm in base o1; do
  case $arm in base) M=$BASE;; o1) M=$O1;; esac
  echo "=== $arm start $(date +%H:%M:%S) model=$(basename "$M")"
  ./ds4 --metal -m "$M" --vision "$V" --ctx 32768 --perplexity-file "$FIX" 2>&1 | grep -E 'memory: KV|tokens=.*avg_nll'
  echo "=== $arm end $(date +%H:%M:%S)"
done 2>&1 | tee "$LOG"
echo PPL_AB_COMPLETE
