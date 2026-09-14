# P4a — correctness foundation (branch `glm53-p4a`, 2026-09-14)

Base: `31a7296` (frozen baseline `b69fd2c` + P0 instrumentation). Rollback point: `31a7296`.

| commit | upstream | how | notes |
|---|---|---|---|
| `2f47409` | `233eeb8` | manual port, subset | see below |
| `2c33a75` | `0e9cc2d` | cherry-pick, conflicts in .gitignore/Makefile/docs/test only | ds4.c + Metal applied clean. Dense boundary is now 2051 (`glm53_graph_indexer_selected_limit`), was ctx_cap=4096. `docs/SERVER.md` not present in this tree, dropped. Added `ds4_linux_memory.h` from origin/main for the test. |
| `6c7a1cf` | `fb2abb2` | cherry-pick clean | Metal scratch 16-byte alignment |
| `3f332d7` | `e965352` | cherry-pick, QA doc conflict only | kept the new test paragraph, not the Sept 6 SSD notes |
| `a8e8a21` | `c0a6119` | cherry-pick, test conflict only | snapshot final byte; test ported |

## 233eeb8 port — what was taken and what was not

Taken (ds4.c hunks 1,7,11-15,26-28,30-32,34,37; ds4_agent.c whole; ds4.h comment):
- `ds4_session_rewind`: no-op past checkpoint; invalidate when recurrent state cannot
  be restored; reset `glm_mtp_have` and rollback validity on invalidation.
- `checkpoint_valid` guards on argmax / argmax_excluding / argmax_ignoring_eos /
  sample / eval / eval_speculative entry points.
- Metal GLM memory accounting: `glm_session_count`, `glm_session_graph_bytes`,
  `ds4_engine_glm_graph_budget`, reservation in session_create / release in free.
- `ds4_agent.c`: `agent_worker_rewind` rebuilds the retained prefix.
- `tests/test_session_state.c` (test_rewind, test_session_memory), NO_GPU and Metal
  variants, `make test-session-state`.

Not taken:
- `mtp_kda_backup -> mtp_state_backup` whole-state snapshot including
  `layer_indexer_tail_k`. This tree keeps the PR #920 spec transaction for KDA and
  makes the pooled-indexer tail consistent in the update kernel (retain all rows,
  rebuild the pool from retained rows + replacement; `a5c52e2`,
  `tests/test_glm53_kda.c`). Same bug class, different mechanism, already tested.
- TP protocol v10 / `ds4_tp_send_glm_mtp` / `ds4_session_glm_tp_spec_cycle` /
  payload token rebuild: single host here.
- `tests/test_tp_commands.c`.

## Consequence for later phases

The 2051 dense boundary means `glm_graph_verify_rows_eligible` now refuses the row
verifier above 2051 instead of 4096. All F23/todo boundary numbers must be
re-measured (plan R5).

## Gates

Results are appended below by `p4a-gates.sh` (`tasks/data/p4a/*.log`, `paired.json`).

### Results (2026-09-14 16:33-16:50)

| gate | baseline `31a7296` | P4a `a8e8a21` | verdict |
|---|---|---|---|
| official 100-case avg_nll (ctx 4096) | 0.452509016, first_match 88, avg_lcp 7.670 | 0.452509016, first_match 88, avg_lcp 7.670 | per-case TSV byte-identical (`diff -q`), bootstrap not needed |
| `--metal-tensor-equivalence` route=auto | — | 5/5 cases, logits_fail 0, greedy_fail 0, worst_rms 0 | PASS |
| `--metal-tensor-equivalence` route=tensor-optin | — | logits_fail 1, greedy_fail 20 (known M5 matmul2d accumulate gap, opt-in only, documented in QA) | informational |
| `--session-snapshot` | — | OK | PASS |
| `--glm53-continued-prefill` | — | OK (continued/cold top token 39058/39058) | PASS |
| `--server` | — | OK | PASS |
| `make test-session-state` (NO_GPU + Metal) | — | ok / ok | PASS |
| `tests/test_glm_attention` | — | 4 attention cases + reductions PASS | PASS |

Both scorers were built from their own commit (baseline in a throwaway worktree,
deleted afterwards) with the Metal 4 tensor route off, so the comparison is
binary-vs-binary on the same model file. Logs: `base-score.log`, `p4a-score.log`,
`tensor-equiv.log`, `engine-tests.log`.

### Completion gate (2026-09-14 16:57-17:15, head `9da9683`)

| gate | result |
|---|---|
| `tests/test_glm53_kda` | PASS (link fixed: `ds4_image.o`) |
| `ds4_test --mtp-verify-depth` (GLM MTP) | OK, nspec=256, worst_argmax_gap=0.056 |
| tensor-route assertion | `/stats.tensor_route`; `tasks/replay.py` refuses != `auto`; launcher refuses `DS4_METAL_ENABLE_TENSOR` unless `GLM_DS4_ALLOW_TENSOR_ROUTE=1` |
| live equivalence, prompt 2822 tok (`tasks/mtp_equiv.sh`) | IDENTICAL; 18 rejections, all above 2051 |
| live equivalence, prompt 146K tok (same run) | IDENTICAL; 23 rejections above 90000 (prefill 802 s cold) |
| live equivalence crossing, prompt 1961 tok (`tasks/mtp_equiv_cross.sh`) | IDENTICAL; verifier rows→batch switch at 2049→2050; rejection at 2049 (rows path) |
| live equivalence crossing, prompt 2023 tok (`TARGET=2040`) | IDENTICAL; rejections at 2048 (rows), 2051 (batch), 2056 (batch); 6 at/below, 9 above 2051 |

MTP arm: `GLM_DS4_MTP=1 GLM_DS4_MTP_MAX_CTX=0 GLM_DS4_MTP_TIMING=1`; plain arm `GLM_DS4_MTP=0`;
greedy, 96 tokens. Crossing runs use a private empty KV dir per arm (both cold);
in the first chain the plain arm consumed the disk checkpoints written by the MTP arm
(`source=disk-text`) and still matched. Artifacts: `mtp-equiv/`, `mtp-equiv-cross-1961/`,
`mtp-equiv-cross/`.

0.4518 vs 0.4525 official NLL: the sweep log was scored with the Metal 4 tensor route on
(`tensor_matmul=on`); today's series is route=auto. Different series, same model file.

**P4a PASS.** Rollback tag `p4a-rollback` = `a8e8a21` (engine content); tooling on top
(`e545cd2`, `9da9683`, this commit) is test/instrumentation only.
