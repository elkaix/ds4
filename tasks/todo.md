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

## Update (2026-09-02 evening) — GPU freed, campaign #1 invalid, #2 running

- [x] GPU suites for #832: `test_topk_ab` 0 diffs, `long-context`, `metal-short-prefill`,
      `metal-kernels`, `metal-tensor-equivalence` (top1_mismatch=0, worst_max_abs 3.789 vs
      3.722 on the pre-enhancement binary, same Vision-Exp model) all OK.
      `local-golden-vectors` 3/5 on BOTH binaries → pre-existing (Vision-Exp vs 0731 fixtures).
- [x] Campaign #1 (8 arms) — **INVALID, discarded**: `"ngram1 DS4_NGRAM_SPEC=5"` was passed
      as one arg → label only, env never exported; all 8 arms ran default config. Caught by
      the harness assertion (`env=[] thr_active=default ngram_env=unset`). Archived at
      `~/.ds4/decode-ab/results-invalid-noenv-20260902.tsv`. Byproduct: pure-drift band,
      decode 44.3 → 31–34 t/s over ~50 min, never recovered.
- [x] Harness guard added: label containing `=` or a space now exits 2.
- [x] Campaign #2 (label + env as separate args) — complete but noise-dominated: agy/CodexBar/
      OrbStack at 80-110% CPU each, 5 GB swap. Kept only as the under-load reference.
- [x] Harness: post-run `thr_active=` re-grep (server prints the threshold line on first decode,
      not at startup); the stale early warning removed.
- [x] Campaign #3 (user stopped the other processes) — 8 arms, see review below. **No software
      knob beats base**; n-gram net_saved −11..−14 s per arm, deterministic.
- [x] Q4_K Vision-Exp build: download done (48/48, sha OK). Pipeline restarted with the
      **merged imatrix** (routed 1p5m + dense/shexp from PR #621 refs/pr/22 220k).
- [ ] ds4-eval Vision-Exp **Q8 baseline** — running (`tasks/2026-09-02-eval-vision-exp-q8.log`).
- [ ] ds4-eval Vision-Exp Q4_K-imatrix — after the build and the baseline.
- [ ] decode-ab pair base vs q4k (`DECODE_AB_MODEL=`) once the file exists.
- [ ] Restart prod on `a278f44` after GPU work.
- [ ] SuperDeepseek Q4_K eval arm 2 — deprioritized: no dense imatrix in that build, the
      Vision-Exp pair is the real gate now.

## Review — campaign #3 (2026-09-02, ~14:30-15:20)

Decode t/s pass1/pass2 (short 512 gen · mid ~2.7K prompt · deep ~34K prompt):

| arm | short | mid | deep | mean |
|---|---|---|---|---:|
| base1 | 45.3/35.9 | 41.0/37.8 | 32.0/30.2 | 37.0 |
| ngram1 | 41.3/39.6 | 35.6/35.2 | 32.2/27.9 | 35.3 |
| thr2048a | 42.5/40.3 | 39.7/40.8 | 34.8/30.5 | 38.1 |
| thr512a | 35.8/28.1 | 35.3/30.6 | 27.3/26.4 | 30.6 |
| ngram2 | 40.6/35.7 | 35.2/29.7 | 29.1/30.2 | 33.4 |
| base2 | 40.6/35.5 | 35.8/30.9 | 30.2/25.3 | 33.1 |
| thr512b | 29.9/33.3 | 31.9/29.1 | 28.5/27.1 | 30.0 |
| thr2048b | 35.9/30.2 | 37.1/28.8 | 22.8/25.4 | 30.0 |

Verdicts (every-window rule): n-gram — not separable, and its own accounting is net-negative on
every run (cycles=1748 proposed=1402 accept 20.97% avg_accept 0.168, net_saved −10.9/−11.2/−12.0/
−14.4 s). Do not set `DS4_NGRAM_SPEC`. thr2048 — flat. thr512 — leans worse, not separable.
Machine drift dominates: base 37.0 → 33.1 across the session, 5-10 t/s pass1→pass2 inside every
arm; agy returned at ~280% CPU by the last arm (thr512b/thr2048b contaminated). Matches the
within-session decay evandhoffman reported on #621 the same day (Q4 −10%, Q8 −4%, not thermal).
**The only lever with a measured win is the AProjQ4 recipe.**

PR #621 take-aways (verified on GitHub 2026-09-02): the +15.5% is the GGUF recipe we already
reproduced 08-27; the new asset is the **dense imatrix** (`routed-and-dense-ds4-220k.dat`, 473
entries) — our 1p5m file has routed entries only, which is why the 08-27 quality gate stalled.
Merged file keeps 1p5m routed bytes (35× more calibration chunks) and adds 344 dense/shexp entries.
PR code itself not merged (100+ files, rewrites ds4_server.c); wait for upstream.
