## Benchmarking

Here we collect prefill and generation speed obtained with different hardware.

Run `ds4-bench` as:

```
./ds4-bench \
  -m ds4flash.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 \
  --ctx-max 65536 \
  --step-incr 2048 \
  --gen-tokens 128
```

Provide PR including your numbers if your hardware was not already tested.
Call the benchmark csv file something like `m3_max.csv` or alike, so that
it is clear what hardware was used for the benchmark.

To generate an SVG graph from a CSV file:

```
python3 speed-bench/plot_speed.py speed-bench/m3_max.csv --title "M3 Max t/s"
```

The script uses only the Python standard library. By default it writes a file
next to the CSV using the `_ts.svg` suffix, such as `speed-bench/m3_max_ts.svg`.

### Metal decode schedule A/B

Build the balanced, same-engine Metal decode comparison with:

```
make metal-decode-schedule-bench
./speed-bench/metal_decode_schedule_bench \
  -m ds4flash.gguf \
  --include-selection
```

The harness prefills two sessions and alternates both variant order and
variant-to-session assignment. It aborts unless every full-vocabulary logit
row is bit-identical and, with `--include-selection`, both variants select the
same non-EOS token. Use `--candidate-env NAME` to measure a rollback control,
or `--help` to compare explicit split schedules.

To compare the default pre-M5 ratio-4 compressor pack/transpose fusion with the
legacy decode path, including token selection, use:

```
./speed-bench/metal_decode_schedule_bench \
  --candidate-env DS4_METAL_DISABLE_PRE_M5_COMPRESSOR_RATIO4_DECODE_PACK_FUSION \
  --include-selection \
  --tokens 1024
```

### GLM-5.3 width-2 MTP verification A/B

Build and run the exact optimized/rollback comparison with a GLM-5.3 model:

```
make glm53-mtp-head-bench
./speed-bench/glm53_mtp_head_bench \
  /path/to/GLM-5.3-Flash-Q2.gguf
```

The harness uses one engine and two synchronized sessions. It runs three
512-token blocks, reverses arm order every 64 tokens, and uses
`DS4_GLM_MTP_DISCARDED_HEAD=1` as the rollback control. It aborts
unless token IDs, acceptance schedules, positions, and full-vocabulary logits
are bit-identical after every chunk.

Pass a second argument to compare another dynamic rollback switch. The
snapshot-aware width-2 KDA path uses:

```
./speed-bench/glm53_mtp_head_bench \
  /path/to/GLM-5.3-Flash-Q2.gguf \
  DS4_METAL_DISABLE_GLM53_KDA_VERIFY2_SNAPSHOT_FUSION
```

Two complete 2026-08-31 M4 Max runs recorded:

| Run / block | Optimized t/s | Rollback t/s | Delta |
|---|---:|---:|---:|
| 1 / 1 | 29.136657 | 28.484798 | +2.2884% |
| 1 / 2 | 28.876846 | 28.477513 | +1.4023% |
| 1 / 3 | 28.473334 | 28.054908 | +1.4915% |
| 1 aggregate | 28.826356 | 28.337641 | +1.7246% |
| 2 / 1 | 29.598591 | 28.867628 | +2.5321% |
| 2 / 2 | 29.832179 | 29.228604 | +2.0650% |
| 2 / 3 | 29.882952 | 29.192804 | +2.3641% |
| 2 aggregate | 29.770724 | 29.095435 | +2.3209% |
| Combined | **29.290900** | **28.711500** | **+2.0180%** |

See `docs/research/glm53_kda_verify2_snapshot.md` for the exact model,
runtime revision, confidence calculation, whole-stack control, and Metal trace.

### GLM-5.3 MTP cycle-budget analysis

Rebuild and verify the exact 1,157,096-byte natural corpus used by the measured
S55-200 ladder:

```
python3 speed-bench/build_s55_corpus.py /tmp/s55-corpus.md
```

The builder reads the frozen Git objects, not the mutable working tree, and
rejects any byte-count or SHA-256 mismatch.

Parse one contiguous `--mtp-timing` request and combine its acceptance with the
matched profiler-OFF decode rate:

```
python3 speed-bench/analyze_glm53_mtp.py server.log \
  --min-pos 190000 \
  --expect-committed 511 \
  --decode-tps 25.00  # replace with the matched timing-OFF rate
```

The parser reports acceptance, tokens/cycle, a Wilson interval, diagnostic
stage timing, and the measured cycle-time deficit to 55 t/s. If a log contains
multiple contiguous requests it fails closed and requires `--run N`. The timing
means themselves receive zero S55 credit because per-cycle logging perturbs
runtime; `--decode-tps` must come from the matched timing-OFF arm. After the
first parse, use `--expect-cycles N` to lock the recorded cycle count. For a
512-token generation, `--expect-committed 511` accounts for the initial token
outside the speculative-cycle log and rejects early-stop evidence.

Run its standard-library tests with:

```
PYTHONDONTWRITEBYTECODE=1 python3 speed-bench/test_analyze_glm53_mtp.py
```

Drive the matched real-200K Phase 2 arms through the existing server:

```
python3 speed-bench/run_s55_200_phase2.py manifest

python3 speed-bench/run_s55_200_phase2.py freeze \
  --server-binary ./ds4-server \
  --model-file ~/models/gguf/GLM-5.3-Flash-UNCEN-Q2.gguf \
  --output phase2-provenance.json

GLM_DS4_S55=1 GLM_DS4_MTP=1 GLM_DS4_MTP_TIMING=1 ./run-glm-ds4.sh >phase2-timing.server.log 2>&1
python3 speed-bench/run_s55_200_phase2.py run --mode timing \
  --server-log phase2-timing.server.log \
  --provenance phase2-provenance.json --output phase2-timing.json

GLM_DS4_S55=1 GLM_DS4_MTP=1 ./run-glm-ds4.sh >phase2-clean.server.log 2>&1
python3 speed-bench/run_s55_200_phase2.py run --mode clean \
  --server-log phase2-clean.server.log \
  --provenance phase2-provenance.json --output phase2-clean.json

python3 speed-bench/run_s55_200_phase2.py combine \
  phase2-timing.json phase2-clean.json --output phase2-budget.json
```

Run `freeze` only after the machine lease is idle: it reads and hashes the full
model once; each arm rehashes it before the post-hash idle settle. Stop the
server between arms. Each arm requires request counters 0 -> 1, identical
cold-or-hit KV-cache outcomes, fresh logs, and
binds each request to the current server PID/instance, frozen binary/model,
exact 512-token response, real 190K-205K prompt, active width-2 counters, and a
matched 511-token post-seed timing window. Per-cycle logging remains diagnostic;
only the clean post-seed duration enters the cycle budget. `combine` reopens and
rehashes the frozen source, binary, model, driver, launcher, and exact log ranges;
moving or changing evidence is a hard failure. The combined report
is explicitly `phase2-cycle-budget-only`: it does not certify the five-context
ladder, quality, workload diversity, or 60-minute stability. Counter overhead
and timing overhead both remain `NOT_EVALUATED` until balanced A/B measurements.

### S55 M0: width-2 MTP versus MATCHED-STATE NOMTP

Run the one-request, warm-200K diagnostic only after the machine lease is free:

```
DS4_S55_M0_ARTIFACT_DIR=$(mktemp -d /private/tmp/ds4-s55-m0.XXXXXX)
printf 'M0 artifacts: %s\n' "$DS4_S55_M0_ARTIFACT_DIR"

python3 speed-bench/run_s55_m0.py manifest
python3 -m unittest speed-bench/test_run_s55_m0.py

python3 speed-bench/run_s55_m0.py freeze \
  --server-binary ./ds4-server \
  --model-file ~/models/gguf/GLM-5.3-Flash-UNCEN-Q2.gguf \
  --output "$DS4_S55_M0_ARTIFACT_DIR/provenance.json"

GLM_DS4_S55=1 \
GLM_DS4_S55_M0_SEGMENT_TOKENS=64 \
GLM_DS4_S55_M0_REPEATS=2 \
./run-glm-ds4.sh >"$DS4_S55_M0_ARTIFACT_DIR/server.log" 2>&1
```

With that server running, set `DS4_S55_M0_ARTIFACT_DIR` to the printed path in
another terminal, then issue the single diagnostic request:

```
python3 speed-bench/run_s55_m0.py run \
  --server-log "$DS4_S55_M0_ARTIFACT_DIR/server.log" \
  --provenance "$DS4_S55_M0_ARTIFACT_DIR/provenance.json" \
  --output "$DS4_S55_M0_ARTIFACT_DIR/result.json"
```

M0 refuses provenance, logs, or results inside the source tree because creating
them would invalidate the frozen Git status.

The scored sequence is `ABBA BAAB` repeated twice (`A=MTP`,
`B=MATCHED-STATE NOMTP`). A 64-token unscored MTP wash follows every `B -> A`
transition. Switching happens only between complete speculative calls, so a
two-token commit cannot straddle arms. The fixed request ceiling is 1,422
tokens: at least 512 scored tokens per arm plus worst-case two-token boundary
overshoot; any unused ceiling is completed only by unscored MATCHED-STATE NOMTP.
M0 disables engine cycle counters so only MTP is not taxed; cycle/accept/commit
counts use the exact pre-call verifier predicate plus emitted-token count. The
server atomically accepts one inference request. The report rejects reseed
leakage, early stop, response/raw-output/state mismatch, missing request-start
telemetry, fan-topology/max-RPM drift, non-finite or zero sensors, fans below
98%, TP/config drift, dependency drift, or artifact replacement. It binds the
loaded binary, numeric model FD plus mapped model inode, and log FDs. Raw log and
response artifacts publish and rehash first; source/binary/model/state seal next;
the immutable report publishes last.

Control instability does not erase a completed iteration. A run above the 5%
control-spread or adjacent-step gate publishes as `REJECTED_UNSTABLE`, retaining
the exact response, log, state, telemetry, and raw arm measurements while still
granting zero S55 credit.

`CONFIRMED` requires both a two-sided sign-test p-value <=0.05 and a paired 95%
CI wholly above a 3% MATCHED-NOMTP cost floor. Smaller effects remain
`NEEDS_MORE_DATA`; failed stability is always `REJECTED_UNSTABLE`.

`MATCHED-STATE NOMTP` still advances nextn state with its output head skipped;
it is a conservative diagnostic control, not production MTP-off. Its result
decides MTP gating only and receives zero S55 throughput credit.

Before M0, certify that control at the canonical long-context checkpoint:

```sh
ORACLE_DIR=$(mktemp -d /private/tmp/ds4-s55-oracle.XXXXXX)
python3 speed-bench/run_s55_200_phase2.py manifest \
  --prompt-output "$ORACLE_DIR/prompt.txt"
DS4_TEST_MODEL=/path/to/GLM-5.3-Flash-UNCEN-Q2.gguf \
DS4_TEST_GLM_MTP=1 \
DS4_TEST_SNAPSHOT_CTX=262144 \
DS4_TEST_SNAPSHOT_PROMPT="$ORACLE_DIR/prompt.txt" \
  ./ds4_test --glm53-matched-nomtp
```

This gate requires byte-identical target snapshots, nextn compact-KV row, and
post-NOMTP wash. It does not equate width-2 verification with serial decode:
that existing path has identical committed tokens but non-identical floating
state, and therefore still blocks an exact-semantics `MTP_AUTO` release.

### H26 topology-neutral Metal command-buffer diagnostic

Enable command-buffer timing with selected-layer metadata:

```sh
DS4_METAL_GLM_MACRO_PROFILE=1 ./run-glm-ds4.sh
```

The default layer selector rotates across all trunk layers and reverses each
sweep. `DS4_METAL_GLM_MACRO_PROFILE_LAYER=N` fixes the metadata layer.
SSD streaming, placement/TP, other profilers, and decode ablation are rejected.
Records use command-buffer `GPUStartTime`/`GPUEndTime` after the existing normal
completion wait; no region boundary waits. Verify, target output heads, and MTP
draft/state calls share one logical cycle ID. The cycle summary reports wall time
and the sum of labelled GPU spans; unlabelled time remains explicitly
`not_computed`, not inferred as CPU time.

Region names attached to one command buffer are metadata, not independent
timings. On the current indexed verifier, selected-layer labels share the whole
forward command-buffer span. This diagnostic can validate GPU occupancy and
whole-cycle accounting, but cannot rank KDA versus routed MoE.

Certify profiler OFF/ON serialized-state exactness from one snapshot:

```sh
DS4_TEST_MODEL=/path/to/GLM-5.3-Flash-UNCEN-Q2.gguf \
DS4_TEST_GLM_MTP=1 \
DS4_TEST_GLM_MACRO_PROFILE_EXACT=1 \
DS4_TEST_SNAPSHOT_CTX=32768 \
DS4_TEST_SNAPSHOT_PROMPT=README.md \
  ./ds4_test --session-snapshot
```

Certify exactness and overhead on one real indexed width-2 operation:

```sh
DS4_TEST_MODEL=/path/to/GLM-5.3-Flash-UNCEN-Q2.gguf \
DS4_TEST_GLM_MTP=1 \
DS4_TEST_SNAPSHOT_CTX=32768 \
DS4_TEST_SNAPSHOT_PROMPT=README.md \
DS4_GLM_INDEXED_VERIFY_SCAN=8 \
DS4_GLM_MACRO_PROFILE_AB=1 \
  ./ds4_test --session-snapshot
```

The balanced same-process scanner compares profiler OFF/ON at identical n=2,
checks full hidden/logit equality, snapshots/restores all persistent DSA indexer
tails, and reports mean/median/range plus a raw adjacent-pair trace and paired
delta interval. Reject an absolute mean or median change above 3%. At actual
position 197,623, mean/median overhead passed at +0.41%/+0.62%; full replay
matched all 4,885,013,984 serialized bytes. H26 is retained only as a
whole-command-buffer diagnostic and receives zero S55 throughput credit.

### Metal prefill variant A/B

Build the balanced prefill comparison. To compare the default resident pre-M5
MXFP4 pair tail-SIMDgroup cull against the original pair kernel, make the
rollback path the candidate:

```
make metal-prefill-variant-bench
./speed-bench/metal_prefill_variant_bench \
  --candidate-env DS4_METAL_DISABLE_PRE_M5_MXFP4_MOE_MM_ID_PAIR_TAIL_SIMDGROUP_CULL
```

To isolate the default routed-down tail-SIMDgroup cull from the retained pair
default, use its down-specific rollback as the candidate:

```
./speed-bench/metal_prefill_variant_bench \
  --candidate-env DS4_METAL_DISABLE_PRE_M5_MXFP4_MOE_MM_ID_DOWN_TAIL_SIMDGROUP_CULL
```

The harness uses one Metal engine and fresh sessions for every run. It warms
both variants with at least 32 tokens, alternates control/candidate order in
ABBA and BAAB blocks, poisons host logit buffers before copying, and aborts
unless every final full-vocabulary logit row is bit-identical. Defaults are an
8192-token prefix, an automatically sized 8193-token context, and two repeats;
use `--help` to override them.
