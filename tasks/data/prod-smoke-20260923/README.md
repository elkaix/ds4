# m5-prod smoke (2026-09-23)

m5-prod = glm53-m5-prod (d9d592f) + the Qwen branch's 4 launcher commits
(run-qwen38-ds4.sh, run-qwen38-agentic.sh, run-ds4-monitored.sh). The Qwen
branch's server/dashboard commits were already in glm53-m5-prod (same fixes,
GLM dashboard is the newer superset).

Build 99f3a34, `ds4_test --server`, `test-session-state`, `tests/test_glm53_kda`: pass.

`tasks/prod_tool_loop_smoke.py`, 4 turns, 3 tool calls, each model via its launcher:

| model | follow-up live reuse | misses | decode |
|---|---|---|---|
| GLM 5.3 stage 2 | 7/7 (memory-token, thinking-visible) | 0 | 39.7 t/s |
| Qwen3.8 native-BF16-ngrams, MTP | 6/6 (thinking-visible) | 0 | 73.5 t/s, MTP 179/223 |
| DeepSeek V4 Flash O2b | 5/6 (memory-token) | 1 token-mismatch after the first tool call | 43.1 t/s |

The DeepSeek miss reproduces identically on the old ds4 tree
(fix-1088-response-routing 41c0559, `deepseek-OLDTREE-*`): same tokens, same
answers, 41.9 t/s. It is pre-existing, not caused by this branch.

Not taken from upstream origin/main (see ledger glm.sync.prod-not-on-origin-main):
PR 765 slot routing (no effect at 1 slot), 5994f062 image-position validation
for live-prefix reuse (depends on the PR 765 reuse probe; known gap for
image requests that reuse a live prefix), V4.1 Engram, tests, docs, ROCm.
Syncing them means porting ~18 GLM cache commits into the reuse probe
(ds4_server.c, ~410 conflicting lines).
