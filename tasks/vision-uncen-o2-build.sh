#!/usr/bin/env bash
# O2: orcarouter Vision-Uncensored Q2 pack with layers 33-42 routed experts at Q4_K.
# Same recipe as O1 (tasks/vision-uncen-o1-build.sh), extended four layers down:
# final GGUF = orcarouter's Q2 bytes everywhere except the 30 layer 33-42 expert
# tensors, which come from our imatrix-guided Q4_K pass. Chains the perplexity check.
set -uo pipefail
D="$(cd "$(dirname "$0")/.." && pwd)"
HF="$HOME/models/hf/DeepSeek-V4-Flash-Vision-Uncensored"
BASE="$HOME/models/gguf/DeepSeek-V4-Flash-Vision-Uncensored-orcarouter-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8.gguf"
IM="$HOME/models/ds4/assets/imatrix/DeepSeek-V4-Flash-chat-v2-routed-1p5m-plus-dense-220k-merged.dat"
CAND="$HOME/models/gguf/tmp-vision-uncen-o2-candidate.gguf"
OUT="$HOME/models/gguf/DeepSeek-V4-Flash-Vision-Uncensored-orcarouter-Layers33-42Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-AProjQ8-SExpQ8-OutQ8.gguf"
LOG="$HOME/models/gguf/logs/vision-uncen-o2-build.log"
Q="$D/gguf-tools/deepseek4-quantize"
SPLICE="$D/gguf-tools/mixed/splice_mixed_expert_layers_gguf.py"
LAYERS=(33 34 35 36 37 38 39 40 41 42)
exec > >(tee -a "$LOG") 2>&1
echo "== $(date) start"

pgrep -f 'ds4-server|/ds4 --metal' >/dev/null && { echo "!! a ds4 process is running; stop it first"; exit 1; }

python3 - "$HF" <<'PY' || { echo "!! shards missing"; exit 1; }
import json,os,sys
d=sys.argv[1]; idx=json.load(open(os.path.join(d,'model.safetensors.index.json')))
files=sorted(set(idx['weight_map'].values())); missing=[f for f in files if not os.path.exists(os.path.join(d,f))]
print(f"{len(files)} shards, missing {len(missing)}"); sys.exit(1 if missing else 0)
PY

Q4=(); for n in "${LAYERS[@]}"; do for p in gate up down; do Q4+=(--tensor-type "blk.$n.ffn_${p}_exps.weight=q4_k"); done; done
# Vision-Exp configs require the official revision string; metadata only, matches $BASE.
RECIPE=(--hf "$HF" --template "$BASE" --imatrix "$IM" --source-revision e46e16bf6035c6f317eb2ac7458eb0362926d402)

cd "$D/gguf-tools"
DRY=$("$Q" "${RECIPE[@]}" "${Q4[@]}" --dry-run --out /tmp/vision-uncen-o2-dry.gguf --overwrite 2>/dev/null)
N=$(awk '/^type_changes:/{print $2}' <<<"$DRY")
NQ4=$(grep -cE '^type_change: blk\.(3[3-9]|4[0-2])\.ffn_(gate|up|down)_exps\.weight .* -> q4_K$' <<<"$DRY")
[[ "$N" == 30 && "$NQ4" == 30 ]] || { echo "!! dry-run type_changes=$N (layer33-42 exps->q4_K=$NQ4), expected 30/30"; exit 1; }
grep -E '^approx_file_bytes:' <<<"$DRY"
echo "== $(date) dry-run OK: 30 tensors -> q4_K"

echo "== $(date) quantize start"
"$Q" "${RECIPE[@]}" "${Q4[@]}" --out "$CAND" --threads 10 || { echo "!! quantize failed"; exit 1; }
echo "== $(date) candidate $(stat -f%z "$CAND") bytes"

python3 "$SPLICE" --base "$BASE" --donor "$CAND" --q4-layers 33-42 --out "$OUT" || { echo "!! splice failed"; exit 1; }
echo "== $(date) DONE $OUT $(stat -f%z "$OUT") bytes"
rm -f "$CAND" && echo "== $(date) removed candidate"

bash "$D/tasks/perplexity-vision-uncen-o2.sh"
echo "== $(date) BUILD_AND_PPL_COMPLETE"
