#!/bin/bash
# P4b1-A3: isolate constructed-B prefill regression at 2K.
# Arms are all on 964eng binary; treatment = env rollback switches.
# Order ABBA with A=B-default, B=rollback. Then per-family if aggregate clears it.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:-$PWD/tasks/data/p4b1/prefill-ablate-2k}
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
M=$HOME/models/gguf/GLM-5.3-Flash-UNCEN-d21b-L17-18-19Q4KExperts-Q2.gguf
P=$PWD/speed-bench/promessi_sposi.txt
GEN=${GEN:-64}   # decode still run for gen_first_ms; prefill is the signal
GAP=${GAP:-30}
DIR=$HOME/Projects/open-source/ds4-glm53-964eng
# Prefer frozen P4b1 timing binary; fall back to live instrumented bench.
BENCH=$DIR/ds4-bench.p4b1-frozen
[[ -x $BENCH ]] || BENCH=$DIR/ds4-bench
CTX=2048

run() {  # $1 label $2 runidx  remaining env already exported by caller via env string in $EXTRA_ENV
  local label=$1 idx=$2
  local tag="$label-$idx"
  echo "$(date +%T) start $tag EXTRA=${EXTRA_ENV:-} bin=$(basename "$BENCH")"
  # env KEY=val must be literal or passed through env(1); $EXTRA_ENV is not
  # re-parsed as an assignment after expansion.
  (cd "$DIR" && env DS4_METAL_DISABLE_METAL4=1 ${EXTRA_ENV:+"$EXTRA_ENV"} \
    "$BENCH" --metal -m "$M" --prompt-file "$P" \
      --ctx-start 1024 --step-incr $((CTX - 1024)) --ctx-max "$CTX" \
      --gen-tokens "$GEN" --csv "$OUT/$tag.csv" > "$OUT/$tag.log" 2>&1)
  echo "$(date +%T) done $tag $(tail -1 "$OUT/$tag.csv")"
  sleep "$GAP"
}

summarize() {
  python3 - <<'PY' "$OUT" "$@"
import csv, sys, statistics as st
from pathlib import Path
out = Path(sys.argv[1])
labels = sys.argv[2:]
print(f"{'label':<28} {'n':>2} {'pref_med':>9} {'gen_med':>8} {'first_ms':>9}")
for lab in labels:
    prefs, gens, firsts = [], [], []
    for p in sorted(out.glob(f"{lab}-*.csv")):
        rows = list(csv.DictReader(p.open()))
        if not rows: continue
        r = rows[-1]
        prefs.append(float(r["prefill_tps"]))
        gens.append(float(r["gen_steady_tps"]))
        firsts.append(float(r["gen_first_ms"]))
    if not prefs:
        print(f"{lab:<28}  0")
        continue
    print(f"{lab:<28} {len(prefs):>2} {st.median(prefs):9.2f} {st.median(gens):8.2f} {st.median(firsts):9.3f}")
PY
}

echo "== $(date +%T) prefill-ablate OUT=$OUT"
echo "B $(cd "$DIR" && git rev-parse HEAD) bin=$(basename "$BENCH") $(shasum -a 256 "$BENCH" | cut -c1-16)" | tee "$OUT/arms.txt"

# Phase 1: B default vs aggregate FLASH_TUNING off, ABBA
echo "== phase1 aggregate FLASH_TUNING"
EXTRA_ENV= run default 1
EXTRA_ENV="DS4_METAL_DISABLE_GLM53_FLASH_TUNING=1" run flash_off 1
EXTRA_ENV="DS4_METAL_DISABLE_GLM53_FLASH_TUNING=1" run flash_off 2
EXTRA_ENV= run default 2
summarize default flash_off | tee "$OUT/phase1.txt"

# Decide whether to fan out: if flash_off median prefill is meaningfully higher, isolate families.
python3 - <<'PY' "$OUT"
import csv, statistics as st, sys
from pathlib import Path
out = Path(sys.argv[1])
def med(lab):
    xs=[]
    for p in sorted(out.glob(f"{lab}-*.csv")):
        r=list(csv.DictReader(p.open()))[-1]
        xs.append(float(r["prefill_tps"]))
    return st.median(xs) if xs else float("nan")
d, f = med("default"), med("flash_off")
gain = 100*(f-d)/d if d else float("nan")
print(f"phase1 default_med={d:.2f} flash_off_med={f:.2f} gain_vs_default={gain:+.2f}%")
open(out/"phase1_decision.txt","w").write(f"default={d}\nflash_off={f}\ngain_pct={gain}\n")
# Continue families if disabling tuning recovers >=5% prefill vs default
sys.exit(0 if gain >= 5.0 else 3)
PY
rc=$?
if [[ $rc -eq 3 ]]; then
  echo "phase1: FLASH_TUNING off did not recover >=5% prefill; skip family fan-out (see phase1.txt)"
  echo "== $(date +%T) done"
  exit 0
fi

echo "== phase2 per-family prefill rollbacks (AB vs default interleaved lightly)"
families=(
  "qk_low:DS4_METAL_DISABLE_GLM53_PREFILL_QK_LOW=1"
  "indexed_attn:DS4_METAL_DISABLE_GLM53_PREFILL_INDEXED_ATTN=1"
  "moe_tail:DS4_METAL_DISABLE_GLM53_PREFILL_MOE_TAIL_CULL=1"
  "kda_prepare:DS4_METAL_DISABLE_GLM53_PREFILL_KDA_PREPARE=1"
  "kda_recurrence:DS4_METAL_DISABLE_GLM53_PREFILL_KDA_RECURRENCE=1"
)
for spec in "${families[@]}"; do
  lab=${spec%%:*}; env=${spec#*:}
  EXTRA_ENV= run default_f 1
  EXTRA_ENV="$env" run "$lab" 1
  EXTRA_ENV="$env" run "$lab" 2
  EXTRA_ENV= run default_f 2
done
summarize default_f qk_low indexed_attn moe_tail kda_prepare kda_recurrence | tee "$OUT/phase2.txt"
echo "== $(date +%T) done"
