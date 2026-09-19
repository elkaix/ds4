#!/bin/bash
# Quiet P4b1 2K-only: frozen binaries, GEN=256, ABBABAAB, Metal4-off, MTP off.
# Uses ds4-bench.p4b1-frozen (same SHA as decode-trace-2k / bench-quiet-2k).
# Does not run 32K/100K/200K. Refuses if a server is on :8000 or ds4-server lives.
# WAIT=1 polls until the GPU is free, then cools down and runs.
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=${1:-$PWD/tasks/data/p4b1/bench-quiet-2k-n4}
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
M=$HOME/models/gguf/GLM-5.3-Flash-UNCEN-d21b-L17-18-19Q4KExperts-Q2.gguf
P=$PWD/speed-bench/promessi_sposi.txt
GEN=${GEN:-256}
GAP=${GAP:-45}
ARM_A=$HOME/Projects/open-source/ds4-glm53-964base
ARM_B=$HOME/Projects/open-source/ds4-glm53-964eng
BIN=ds4-bench.p4b1-frozen
WANT_A=53a60221143a2bfc
WANT_B=c5b572279c468844

armdir() { if [[ $1 == A ]]; then echo "$ARM_A"; else echo "$ARM_B"; fi; }

busy() {
    lsof -nP -iTCP:8000 -sTCP:LISTEN >/dev/null 2>&1 && return 0
    pgrep -x ds4-server >/dev/null 2>&1 && return 0
    pgrep -x ds4-bench >/dev/null 2>&1 && return 0
    pgrep -f 'p3_one_layer.sh' >/dev/null 2>&1 && return 0
    return 1
}

if [[ ${WAIT:-0} == 1 ]]; then
    while true; do
        echo "$(date +%T) waiting for GPU (port 8000 / ds4-server / p3_one_layer)"
        while busy; do sleep 30; done
        echo "$(date +%T) free; cooldown ${COOLDOWN:-120}s"
        sleep "${COOLDOWN:-120}"
        if busy; then
            echo "$(date +%T) busy again after cooldown; resume wait"
            continue
        fi
        break
    done
elif busy; then
    echo "GPU busy (:8000, ds4-server, ds4-bench, or p3_one_layer). WAIT=1 to poll." >&2
    lsof -nP -iTCP:8000 -sTCP:LISTEN 2>/dev/null | head -3 >&2
    exit 2
fi

got_a=$(shasum -a 256 "$ARM_A/$BIN" | awk '{print substr($1,1,16)}')
got_b=$(shasum -a 256 "$ARM_B/$BIN" | awk '{print substr($1,1,16)}')
if [[ $got_a != "$WANT_A" || $got_b != "$WANT_B" ]]; then
    echo "frozen SHA mismatch: A $got_a want $WANT_A; B $got_b want $WANT_B" >&2
    exit 3
fi
{
    echo "A $(git -C "$ARM_A" rev-parse HEAD) $got_a"
    echo "B $(git -C "$ARM_B" rev-parse HEAD) $got_b"
} | tee "$OUT/arms.txt"

run() {
    local d tag
    d=$(armdir "$1")
    tag="$2-$1-$3"
    mkdir -p "$OUT/logits-$tag"
    macmon pipe -s 1 2>/dev/null | head -1 > "$OUT/$tag.pre.json" || true
    (cd "$d" && DS4_METAL_DISABLE_METAL4=1 ./"$BIN" --metal -m "$M" --prompt-file "$P" \
        --ctx-start 1024 --step-incr $(( $2 - 1024 )) --ctx-max "$2" --gen-tokens "$GEN" \
        --csv "$OUT/$tag.csv" --dump-frontier-logits-dir "$OUT/logits-$tag" > "$OUT/$tag.log" 2>&1)
    echo "$(date +%T) $tag rc=$? $(tail -1 "$OUT/$tag.csv")"
    sleep "$GAP"
}

echo "== $(date +%T) start GEN=$GEN GAP=$GAP OUT=$OUT"
nA=0 nB=0
for a in A B B A B A A B; do
    if [[ $a == A ]]; then nA=$((nA+1)); run A 2048 $nA
    else nB=$((nB+1)); run B 2048 $nB
    fi
done
echo "== $(date +%T) done"
python3 "$PWD/tasks/p4b1_analyze.py" "$OUT"
