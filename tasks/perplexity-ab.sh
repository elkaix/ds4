#!/usr/bin/env bash
set -uo pipefail
cd ~/Projects/open-source/ds4
S=/private/tmp/claude-501/-Users-panda-Projects-open-source-ds4/55a87f0a-3bb2-433c-a6b3-f3694ca920e4/scratchpad
V=~/models/gguf/DeepSeek-V4-Flash-Vision-Encoder.gguf
Q8=~/models/gguf/DeepSeek-V4-Flash-Vision-Exp-Abliterated-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8.gguf
Q4=~/models/gguf/DeepSeek-V4-Flash-Vision-Exp-Abliterated-IQ2XXS-w2Q2K-AProjQ4K-SExpQ8-OutQ4K.gguf
for arm in q8 q4k; do
  case $arm in q8) M=$Q8;; q4k) M=$Q4;; esac
  echo "=== $arm start $(date +%H:%M:%S) model=$(basename $M)"
  ./ds4 --metal -m "$M" --vision "$V" --ctx 32768 --perplexity-file "$S/ppl.txt" 2>&1 | tail -25
  echo "=== $arm end $(date +%H:%M:%S)"
done
echo PPL_AB_COMPLETE
