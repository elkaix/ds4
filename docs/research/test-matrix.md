# S55 Local Validation Matrix

Scope: isolated `codex-s55-runtime` worktree at `b05ecd7` plus the current
Phase 2/M0 telemetry and correctness diff on this M5 Max. The user confirmed
the Metal/server lease is free. CUDA,
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
| S55 MTP analyzer | `PYTHONDONTWRITEBYTECODE=1 python3 speed-bench/test_analyze_glm53_mtp.py` | Pass: 4 tests; historical fixture reproduces 297 cycles, 214 accepts, 83 rejects, 511 committed |
| Phase 2 diagnostic driver | `PYTHONDONTWRITEBYTECODE=1 python3 speed-bench/test_run_s55_200_phase2.py` | Pass: 16 tests. Diagnostic-only output; full S55 remains `NOT_EVALUATED` |
| M0 matched-state driver | `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest speed-bench/test_run_s55_m0.py` | Pass: 16 tests; runtime `CONFIRMED` diagnostic at context 197,395. MTP 44.1838 ms/token versus MATCHED NOMTP 39.7307 ms/token; paired 95% CI [3.1080, 5.7085], sign p=0.0078125. Zero S55 credit |
| MATCHED-STATE NOMTP oracle | `DS4_TEST_MODEL=... DS4_TEST_GLM_MTP=1 DS4_TEST_SNAPSHOT_CTX=262144 DS4_TEST_SNAPSHOT_PROMPT=... ./ds4_test --glm53-matched-nomtp` | Pass at canonical prompt position 197,622: target snapshot, nextn compact-KV row, and NOMTP-to-MTP wash are byte-identical. This certifies the M0 control, not MTP-versus-plain numerical identity |
| Indexed verifier n=1/n=2 | `DS4_GLM_INDEXED_VERIFY_SCAN=8 DS4_GLM_INDEXED_VERIFY_SCAN_POS=...`, then natural-prefix request | Pass at positions 4,728/32,171/64,134/128,130/197,396. Excess grows 5.704/7.884/10.217/15.173/20.594 ms; linear R2=0.999917. Diagnostic only; state-neutrality pending |
| Exact GLM-5.3 pair indexer, 4K screen | `DS4_GLM_INDEXED_PAIR_AB=1 DS4_GLM_INDEXED_VERIFY_SCAN=8`, actual position 4,728 | Hidden output and full logits byte-identical; 44.223 -> 43.633 ms, -0.590 ms / +1.33%, paired CI [-1.168, -0.012] ms. Continue only to predeclared 197K gate |
| Exact GLM-5.3 pair indexer, 197K decision | Same-process ABBA/BAAB, actual position 197,402, one server/request | Hidden output and full logits byte-identical; 86.781 -> 68.769 ms, -18.013 ms / +20.76%, delta CI [-19.053, -16.972] ms. KEEP opt-in; zero S55 credit pending clean decode gate |
| Exact GLM-5.3 pair raw scores | `./ds4_test --metal-kernels`; forced scalar versus forced exact-pair dispatch | Pass: bit-identical 514/514 FP32-cache and 514/514 FP16-cache score values, including pooled visibility boundary |
| Exact GLM-5.3 pair serialized state | `DS4_TEST_GLM_MTP=1 DS4_TEST_SNAPSHOT_CTX=... DS4_TEST_SNAPSHOT_PROMPT=... ./ds4_test --session-snapshot` | Pass at position 4,728: 267,106,276 bytes exact. Pass at canonical ~197K: 4,885,013,984 bytes exact, per-cycle full logits exact, identical 5 single / 11 double / 27-token schedule |
| Exact-pair real 197K M0 attempt 1 | Frozen binary/model, one request, 1,422 tokens, ABBA/BAAB | REJECT: raw MTP 25.915 t/s vs MATCHED NOMTP 22.939 t/s, but NOMTP control spread 15.264% exceeds 5%. All token/output/trajectory hashes match prior baseline; zero S55 credit; stable rerun required |
| Exact-pair real 197K M0 attempt 2 | Frozen binary/model, one request, 1,422 tokens, ABBA/BAAB | `REJECTED_UNSTABLE`: raw MTP 23.011 t/s vs MATCHED NOMTP 21.085 t/s; control spread 14.707%, worst adjacent 14.513%. Swap rose 11.4 -> 13.6 GiB during decode. Exact token/output/trajectory hashes match baseline; zero S55 credit |
| H26 command-buffer timing primitive | `./ds4_test --metal-kernels` | Pass: three metadata labels join one unchanged command buffer, producing one valid GPU span at the existing completion wait; no encoder split or region wait |
| H26 scanner boundary isolation | `DS4_GLM_INDEXED_VERIFY_SCAN_POS=20987 DS4_GLM_INDEXED_VERIFY_SCAN=8 DS4_GLM_MACRO_PROFILE_AB=1 ... ./ds4_test --session-snapshot` | Red before fix: HC/logits and 656,274,916-byte state differed. Green after scanner-local restore of 45,056 DSA indexer-tail bytes: HC/logits/state exact; mean/median overhead -0.44%/-0.47% |
| H26 profiler full-state exactness | `DS4_TEST_GLM_MACRO_PROFILE_EXACT=1 DS4_TEST_GLM_MTP=1 ... ./ds4_test --session-snapshot` at canonical ~197K | Pass: every cycle's logits and all 4,885,013,984 serialized bytes exact; identical 5 single / 11 double / 27-token schedule |
| H26 profiler overhead | `DS4_GLM_INDEXED_VERIFY_SCAN=32 DS4_GLM_INDEXED_VERIFY_SCAN_POS=197623 DS4_GLM_MACRO_PROFILE_AB=1 ... ./ds4_test --session-snapshot` | Pass at actual 197,623: OFF/ON mean 85.992/86.345 ms, median 88.662/89.214 ms; +0.41%/+0.62%; paired delta CI [-0.202, 0.910] ms. Backend KEEP as whole-command-buffer diagnostic; H26 REJECT as macro-region target selector because labels share one whole-forward span. Zero S55 credit |
| Phase 2 prompt | `PYTHONDONTWRITEBYTECODE=1 python3 speed-bench/run_s55_200_phase2.py manifest` | Pass: 769,966 bytes; SHA-256 `9b5b32d34afe5cd67c55805a67aadcfcbc0b26b407c0dc13215d0b92f48f0be2` |
| Frozen S55 corpus | `PYTHONDONTWRITEBYTECODE=1 python3 speed-bench/build_s55_corpus.py --verify-only` | Pass: 1,157,096 bytes; SHA-256 `78493835239cb7a3b35228bf7304f72d3f293d9b99c4a7ef10e0aca206c7150b` |
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
| Metal two-row oracle | `DS4_TEST_MODEL=... DS4_TEST_SESSION_COUNT=2 DS4_TEST_LOGIT_TOLERANCE=0.001 make test-metal-session-batch` | Pass; selected tokens match, observed max absolute logit delta 0.000312805 |
| Metal four-row oracle | Same with `DS4_TEST_SESSION_COUNT=4` | Pending; same gate |
| Native GLM MTP verifier | `DS4_TEST_MODEL=... DS4_TEST_GLM_MTP=1 ./ds4_test --mtp-verify-depth` | Pass: 256 generated tokens, width-2 exercised, worst plain-replay argmax gap 0.000 |
| Final-token MTP budget | Same verifier after the current cap fix | Pending build/run; every returned chunk and retained session position must remain within caller budget |
| MATCHED-STATE NOMTP M0 | One-request warm 200K, >=8 blocks, >=512 tokens/arm, balanced ABBA/BAAB, exact output/state artifact, clean thermal gate | `CONFIRMED` diagnostic: 8 blocks/arm, 515/512 tokens, MTP 22.6327 t/s versus MATCHED NOMTP 25.1694 t/s; MTP penalty 11.2081%. Five wash blocks, max fans, exact sealed artifacts, final pos 198,817. Zero S55 credit; not authorization for `MTP_AUTO` |
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
