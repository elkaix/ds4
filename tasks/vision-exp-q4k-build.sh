#!/usr/bin/env bash
# vision-exp-q4k-build.sh — build the Q4_K-attention Vision-Exp Abliterated quant.
#
# Reproduces the 2026-08-27 AProjQ4K recipe (+13.2% decode measured on SuperDeepseek) on
# the daily Vision-Exp Abliterated model, from apetersson's Reference-Native-FP8 safetensors.
# Stages, each logged to $LOG:
#   1. wait for the `hf download` to exit, check all shards in model.safetensors.index.json
#   2. verify SHA256SUMS (skips files not listed)
#   3. --compare-tensor pre-flight, one tensor per dtype family, against the on-disk
#      Vision-Exp IQ2 GGUF. A non-zero exit aborts. A byte mismatch on the routed experts
#      is logged, not fatal: it would mean antirez built the reference with a different
#      imatrix, not that the dequant path is broken.
#   4. full quantize, --threads 10, ~75 min.
# The --attention-proj glob also matches blk.N.indexer.attn_q_b.weight on even layers;
# the 21 --tensor-type guards keep those at f16. Dry-run must report exactly 216 changes.
set -uo pipefail
D="$(cd "$(dirname "$0")/.." && pwd)"
HF="$HOME/models/hf/DeepSeek-V4-Flash-Vision-Exp-Abliterated/Reference-Native-FP8"
T="$HOME/models/gguf/DeepSeek-V4-Flash-Vision-Exp-Abliterated-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8.gguf"
IM="$HOME/models/ds4/assets/imatrix/DeepSeek-V4-Flash-chat-v2-routed-1p5m-plus-dense-220k-merged.dat"  # routed from 1p5m + dense/shexp from PR #621 refs/pr/22 220k (merged 2026-09-02)
OUT="$HOME/models/gguf/DeepSeek-V4-Flash-Vision-Exp-Abliterated-IQ2XXS-w2Q2K-AProjQ4K-SExpQ8-OutQ4K.gguf"
LOG="$HOME/models/gguf/logs/vision-exp-q4k-build.log"
Q="$D/gguf-tools/deepseek4-quantize"
exec > >(tee -a "$LOG") 2>&1
echo "== $(date) start"

GUARDS=(); for n in 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42; do GUARDS+=(--tensor-type "blk.$n.indexer.attn_q_b.weight=f16"); done
RECIPE=(--hf "$HF" --template "$T" --imatrix "$IM" --attention-proj q4_k --output q4_k "${GUARDS[@]}")

# 1. wait for download
while pgrep -f 'hf download apetersson' >/dev/null; do sleep 60; done
echo "== $(date) download process gone"
python3 - "$HF" <<'PY' || { echo "!! shards missing"; exit 1; }
import json,os,sys
d=sys.argv[1]; idx=json.load(open(os.path.join(d,'model.safetensors.index.json')))
files=sorted(set(idx['weight_map'].values())); missing=[f for f in files if not os.path.exists(os.path.join(d,f))]
print(f"{len(files)} shards, missing {len(missing)}"); sys.exit(1 if missing else 0)
PY

# 2. checksums
cd "$HF" && if [[ -f SHA256SUMS ]]; then
  grep -E '\.safetensors$|config.json$|index.json$' SHA256SUMS > /tmp/vision-exp-sums.txt
  shasum -a 256 -c /tmp/vision-exp-sums.txt | grep -v ': OK$' ; echo "== $(date) sha256 checked ($(wc -l < /tmp/vision-exp-sums.txt) files)"
  shasum -a 256 -c --quiet /tmp/vision-exp-sums.txt || { echo "!! checksum failure"; exit 1; }
fi
cd "$D/gguf-tools"

# 3. pre-flight
for t in blk.0.ffn_gate_exps.weight blk.0.ffn_down_exps.weight blk.0.ffn_gate_shexp.weight token_embd.weight blk.0.attn_output_b.weight; do
  echo "== $(date) compare $t"
  "$Q" "${RECIPE[@]}" --compare-gguf "$T" --compare-tensor "$t" --out /tmp/vision-exp-compare.gguf --overwrite || { echo "!! compare-tensor $t failed"; exit 1; }
done

# 4. dry-run gate, then the real thing
N=$("$Q" "${RECIPE[@]}" --dry-run --out /tmp/vision-exp-dry.gguf --overwrite 2>&1 | grep -c 'q8_0 -> q4_K')
[[ "$N" == 216 ]] || { echo "!! dry-run reported $N q8_0->q4_K changes, expected 216"; exit 1; }
echo "== $(date) quantize start"
"$Q" "${RECIPE[@]}" --out "$OUT" --threads 10 && echo "== $(date) DONE $OUT $(stat -f%z "$OUT") bytes" || echo "!! quantize failed"
