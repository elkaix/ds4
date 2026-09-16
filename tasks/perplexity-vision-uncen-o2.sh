#!/usr/bin/env bash
# Teacher-forced NLL: O1 (layers 37-42 Q4_K experts) vs O2b (33-42), back to back on the
# current binary, same fixture as the base-vs-O1 run. The working tree changed since O1's
# 05:37 measurement (metal/cpy.metal), so O1 is re-measured here rather than read from
# tasks/2026-09-13-perplexity-vision-uncen-base-vs-o1.log. Needs the GPU free.
set -uo pipefail
cd ~/Projects/open-source/ds4
FIX=~/.ds4/ppl/ppl-readme-ds4c.txt
V=~/models/gguf/DeepSeek-V4-Flash-Vision-Encoder.gguf
O1=~/models/gguf/DeepSeek-V4-Flash-Vision-Uncensored-orcarouter-Layers37-42Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-AProjQ8-SExpQ8-OutQ8.gguf
O2=~/models/gguf/DeepSeek-V4-Flash-Vision-Uncensored-orcarouter-Layers33-42Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-AProjQ8-SExpQ8-OutQ8.gguf
LOG=tasks/$(date +%F)-perplexity-vision-uncen-o1-vs-o2.log
for arm in o1 o2; do
  case $arm in o1) M=$O1;; o2) M=$O2;; esac
  echo "=== $arm start $(date +%H:%M:%S) model=$(basename "$M")"
  ./ds4 --metal -m "$M" --vision "$V" --ctx 32768 --perplexity-file "$FIX" 2>&1 | grep -E 'memory: KV|tokens=.*avg_nll'
  echo "=== $arm end $(date +%H:%M:%S)"
done 2>&1 | tee "$LOG"
echo PPL_O2_COMPLETE
