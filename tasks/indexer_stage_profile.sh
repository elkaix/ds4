#!/bin/bash
# Roadmap #1 (2026-09-19 replan): indexer score/top-k stage attribution on the
# clean M5-lifted #1090 engine. No code changes needed — #1090 already ships
# metal_graph_indexer_stage_profile_boundary() behind
# DS4_METAL_INDEXER_STAGE_PROFILE=1, emitting per-layer stderr lines:
#   ds4: metal indexer stage layer=L pos=P tokens=N comp=C score|topk=X.XXX ms
# (decode path also tags decode_score/decode_topk/decode_attention.)
#
# One incremental ds4-bench sweep (frontier deltas only), teacher-forced 64
# tokens per frontier, fans max. The profiler syncs the GPU queue per boundary,
# so t/s here is NOT comparable to the clean curve — only ms attribution is.
# Frontiers: 49152, 65536, 81920, 98304, 114688, 131072, 147456, 163840
# (~48K/64K/80K/98K/115K/128K/147K/160K).
# Usage: tasks/indexer_stage_profile.sh [out-dir]   (nothing else on the GPU!)
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=${1:-$PWD/tasks/data/indexer-stage-profile-$(date +%Y%m%d)}; mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
M=$HOME/models/gguf/GLM-5.3-Flash-UNCEN-d21b-L17-18-19Q4KExperts-Q2.gguf
P=$PWD/speed-bench/promessi_sposi.txt
GEN=64
GAP=20

echo "$0 $*" > "$OUT/cmdline.txt"
{ git rev-parse HEAD; shasum -a 256 ds4-bench | cut -c1-16; sysctl -n machdep.cpu.brand_string; } \
    | tr '\n' ' ' > "$OUT/env.txt"; echo >> "$OUT/env.txt"

fans_off() { thermalforge auto >/dev/null 2>&1 || true; }
trap fans_off EXIT
thermalforge max >/dev/null 2>&1 || true

export DS4_METAL_INDEXER_STAGE_PROFILE=1
TAG=prof
echo "== $(date +%T) $TAG start (DS4_METAL_INDEXER_STAGE_PROFILE=1)" >> "$OUT/run.log"
./ds4-bench --metal -m "$M" --prompt-file "$P" \
    --ctx-start 49152 --step-incr 16384 --ctx-max 163840 --ctx-alloc 262144 \
    --gen-tokens "$GEN" --teacher-forced-decode \
    --power 100 \
    --csv "$OUT/$TAG.csv" > "$OUT/$TAG.log" 2>&1
echo "== $(date +%T) $TAG rc=$?" >> "$OUT/run.log"
unset DS4_METAL_INDEXER_STAGE_PROFILE
sleep "$GAP"

# ---- Parse: decode rows only (tokens=1). Sum ms per stage per pos, bucket to
# the frontier each pos belongs to, average per-token cost across the bucket.
# stage key: score|decode_score -> SCORE ; topk|decode_topk -> TOPK ;
# decode_attention -> ATTN (context gather, for contrast).
awk '
/ds4: metal indexer stage/ {
    layer=""; pos=""; toks=""; stage=""; ms=""
    for (i = 1; i <= NF; i++) {
        if ($i ~ /^layer=/)   { layer = substr($i, 7) }
        else if ($i ~ /^pos=/)  { pos = substr($i, 5) + 0 }
        else if ($i ~ /^tokens=/) { toks = substr($i, 8) + 0 }
        else if ($i ~ /^comp=/) { }
        else if ($i ~ /^[a-z_0-9]+=[0-9.]+$/) {
            split($i, kv, "=")
            stage = kv[1]; ms = kv[2] + 0
        }
    }
    if (toks != 1 || stage == "" || pos == 0) next
    key = stage
    if (stage == "score" || stage == "decode_score") key = "SCORE"
    else if (stage == "topk" || stage == "decode_topk") key = "TOPK"
    else if (stage == "decode_attention") key = "ATTN"
    sum[pos, key] += ms
    seen[pos, key] = 1
    if (!(key in ntok)) ntok[key] = 0
    toklist[pos] = 1
}
END {
    n = split("49152 65536 81920 98304 114688 131072 147456 163840", F, " ")
    printf "frontier,stage,per_token_ms,samples\n"
    for (p in toklist) {
        f = ""
        for (j = n; j >= 1; j--) { if (p <= F[j] + 2048) { f = F[j] } }
        if (f == "") continue
        for (k in ntok) {
            if (seen[p, k]) {
                acc[f, k] += sum[p, k]
                cnt[f, k] += 1
            }
        }
    }
    for (j = 1; j <= n; j++) {
        f = F[j]
        for (k in ntok) {
            if (cnt[f, k] > 0)
                printf "%s,%s,%.4f,%d\n", f, k, acc[f, k] / cnt[f, k], cnt[f, k]
        }
    }
}' "$OUT/$TAG.log" > "$OUT/summary.csv"

echo "== summary ==" ; cat "$OUT/summary.csv" | tee -a "$OUT/run.log"
echo "== $(date +%T) done" >> "$OUT/run.log"
