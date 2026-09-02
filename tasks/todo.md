# Decode-speed enhancements — 2026-09-02 (user: "go all enhancement")

Backup tag: `pre-enhancements-20260902` = `e16d327`. Prod server (pid 89260) is live with a
user test session — GPU-bound steps wait for it.

Source: `tasks/2026-09-02-decode-speed-research.md` §3.

- [x] A5 cherry-pick #873 (BPE O(n log n)) → verify: build zero warnings, `./ds4_test --server`
- [x] A5 cherry-pick #942 (tokenizer storage off mmap; Makefile conflict) → verify: build + new `test_vocab_storage`
- [x] A5 cherry-pick #832 (build only; GPU suites pending) stack (7 commits, contains #830/#831) → verify: build; GPU suites deferred
- [x] A6 #846 n-gram speculation — extract the n-gram part only (skip M1 tuning/docs); conflicts in ds4.c, ds4_agent.c → verify: build, `./ds4_agent_test`
- [x] A4 find a safetensors source for Vision-Exp Abliterated (research subagent)
- [x] A7 #949/#948 status (both open, 0 comments — nothing to do)
- [x] A1 harness `tasks/decode-ab.sh` written — RUN BLOCKED on GPU
- [ ] A3 `ds4-eval` arm 2 on Q4_K — RUN BLOCKED on GPU (~3 h)
- [x] A2 hygiene: sysctl already 122880 ✓; ctx 262144 ✓
- [x] Rebuild, tests that don't need GPU, commit on `prod`, update CLAUDE.local.md

## Review (2026-09-02 13:35)
Done: 11 commits on prod (`213ddce`), zero warnings, `--server`/agent/eval-extractor suites green.
Safetensors source found; download + Q4_K build pipeline running (`tasks/vision-exp-q4k-build.sh`).
Deviations: #846 agent hunks dropped (upstream already generalized the block loop); #832 taken
GPU-unverified. Blocked on GPU: metal suites, decode-ab campaign, ds4-eval arm 2.
Next session: see CLAUDE.local.md "Decode-speed enhancements adopted" section.
