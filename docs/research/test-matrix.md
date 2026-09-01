# S55 Local Validation Matrix

Scope: current shared `b265e22` DS4 C / Objective-C / Metal tree on this M5 Max,
including the committed discarded-draft-head H19 change.
Run only after the concurrent Claude server/benchmark lease is released. CUDA,
ROCm, physical multi-Mac TP, Q4, and vision-sidecar execution are not locally
applicable to this exact Q2/M5 target and must be reported as unvalidated—not
silently counted as passes.

## Required Before the Next Kernel Change

| Gate | Command | Status / acceptance |
|---|---|---|
| Clean Metal build | `make clean && make` | Pending; warning-free |
| CPU compile check | `make clean && make cpu` | Pending; warning-free, then rebuild Metal |
| Core suite | `make test` | Observed before/after H19: same two assertions at `tests/ds4_test.c:5446`; no H19 regression, but the suite is not green. Both are the same `short_code_completion` top-1 mismatch: once in `logprob-vectors`, then again when that case is reused by `metal-ssd-streaming-cache-pressure`. The default `ds4flash.gguf` symlink is an abliterated DeepSeek V4 checkpoint while the fixture declares official checkpoint `0731`; this is a local fixture/model mismatch, not GLM target evidence. Independent rerun remains pending. |
| GLM53 Metal primitives | `make test-glm53-kda` | Pending |
| Metal MXFP4 primitives | `make test-mxfp4-metal` | Pending; unrelated format but applicable backend regression |
| Discarded draft head A/B | `make glm53-mtp-head-bench && speed-bench/glm53_mtp_head_bench "$MODEL" DS4_GLM_MTP_DISCARDED_HEAD` | Separate-process ABBA observed +2.256% at 2K and +1.789% at 65K with identical text. Paired exact IDs, cycle shapes, positions, counters, and logits still pending; report all three blocks and aggregate |
| Quantizer unit suite | `python3 gguf-tools/tests/test_glm53_quantize.py` | Pass: 14 tests |
| Vision target compile | `make tests/test_glm53_vision_engine tests/test_glm53_vision_prompt` | Pending; execution blocked by missing vision sidecar |
| Python syntax | AST parse over every tracked Python file | Pass: 39 files; no bytecode or generated files written |
| Shell syntax | `sh -n` / `bash -n` selected by each tracked script's shebang | Pass: 7 tracked scripts |
| Whitespace | `git diff --check` | Pass |

## Exact GLM-5.3 Q2 Model Gates

Model: `/Users/panda/models/gguf/GLM-5.3-Flash-UNCEN-Q2.gguf`.

Issue #925 reports correctness failures for a similarly named community
IQ2/Q2 artifact. It is not proof against this exact file, but it makes the
official-vector and 100-case gates release-blocking rather than optional.

| Gate | Invocation | Status / acceptance |
|---|---|---|
| Metal two-row oracle | `DS4_TEST_MODEL=... DS4_TEST_SESSION_COUNT=2 DS4_TEST_LOGIT_TOLERANCE=0.001 make test-metal-session-batch` | Pending; selected tokens match, max logit delta <=0.001 |
| Metal four-row oracle | Same with `DS4_TEST_SESSION_COUNT=4` | Pending; same gate |
| Native GLM MTP verifier | `DS4_TEST_MODEL=... DS4_TEST_GLM_MTP=1 ./ds4_test --mtp-verify-depth` | Pending; zero verifier failure |
| Snapshot + native MTP | `DS4_TEST_MODEL=... DS4_TEST_GLM_MTP=1 ./ds4_test --session-snapshot` | Pending; exact committed tokens and top-eight replay |
| Continued prefill | `DS4_TEST_MODEL=... ./ds4_test --glm53-continued-prefill` | Pending; cold/resumed top-token agreement |
| Fast/quality equivalence | `DS4_TEST_MODEL=... ./ds4_test --metal-tensor-equivalence` | Observed after H19: pass, 0 top-1 mismatches. Independent rerun pending |
| 100-case quality | `gguf-tools/quality-testing/score_official ... gguf-tools/quality-testing/data/glm53-flash-openrouter-zai-fp8-100/manifest.tsv ... 4096` | Pending; preserve `summary` and `api_summary`, compare to accepted Q2 band |
| Long-context corruption smoke | `tests/glm_long_context_smoke.sh ...` | Pending; begins `>` and no corruption marker; diagnostic, not S55 credit |
| Dedicated server batching | Launch this tree on a private port, then `python3 tests/test_server_batching.py --url ...` | Pending; paired seeded outputs identical; never target Claude's live server |

The 4,096-4,100 boundary, natural long-context tail task, and S55 workload
suite require generated prompts with verified rendered-token frontiers. They
are separate model gates, not substitutes for the compiled/unit tests above.

## Explicitly Blocked or Inapplicable Locally

| Area | Reason |
|---|---|
| CUDA targets/tests | No CUDA device/toolchain on the Apple target |
| ROCm targets/tests | No ROCm device/toolchain on the Apple target |
| Physical Metal TP TCP/RDMA | A second authorized Mac is outside this local run |
| GLM53 vision execution/6-case quality | Required `GLM-5.3-Flash-Vision-Encoder.gguf` is absent; compile remains required |
| Directional-steering oracle | No declared GLM53 direction fixture is present |
| Legacy MTP / DSpark verifier | Available support GGUF is DeepSeek V4, not checkpoint-compatible with this GLM53 target |
| GLM Q4 quality | Q4 artifact is not the model under optimization |

## Post-Change Rule

Repeat every applicable gate touched by the diff, then the full required local
matrix. A kernel change additionally requires bit-identical pure-runtime output,
balanced ABBA speed evidence, real 200K measurement, four workload classes,
and the 60-minute no-decay gate. Passing tests without higher sustained decode
throughput earns zero S55 credit.
