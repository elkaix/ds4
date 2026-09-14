#!/usr/bin/env bash
# GLM O1-uncen step 1+2: fresh Q2 control and full Q4_K donor from the pinned
# uncensored FP8 source. Sequential; CPU-only (libds4quants), no GPU needed.
# The existing UNCEN-Q2.gguf may come from a different revision, so the fresh Q2
# is the only valid control for the layer-sensitivity sweep.
set -Eeuo pipefail
cd ~/Projects/open-source/ds4-glm53
HF=~/models/hf/GLM-5.3-Flash-UNCENSORED-FP8
REV=d21b19569d30e6f471c433b11e672b3bbb80552a
TPL=~/models/gguf/GLM-5.3-Flash-UNCEN-d21b-Q2.gguf          # tokenizer metadata only
LIB=gguf-tools/libds4quants.dylib
Q2=~/models/gguf/GLM-5.3-Flash-UNCEN-d21b-Q2.gguf
Q4=~/models/gguf/tmp-glm53-uncen-d21b-q4-donor.gguf
LOG=~/models/gguf/logs/glm53-uncen-o1-sources.log
exec > >(tee -a "$LOG") 2>&1

[[ "$(cat "$HF/.pinned-revision")" == "$REV" ]] || { echo "!! source revision mismatch"; exit 1; }
for art in q2 q4; do
  case $art in q2) OUT=$Q2;; q4) OUT=$Q4;; esac
  [[ -s "$OUT" ]] && { echo "== $(date) $art exists, skipping"; continue; }
  echo "== $(date) start $art -> $OUT"
  python3 gguf-tools/glm53_quantize.py --hf "$HF" --tokenizer-template "$TPL" --artifact "$art" \
    --source-revision "$REV" --quants-library "$LIB" --threads 10 --resume --out "$OUT"
  echo "== $(date) done $art $(stat -f%z "$OUT") bytes"
  python3 gguf-tools/glm53_validate_gguf.py --hf "$HF" --gguf "$OUT" --artifact "$art" --source-revision "$REV" 2>&1 | tail -5 \
    || echo "!! validate $art returned nonzero"
done
echo "== $(date) GLM_SOURCES_COMPLETE"
