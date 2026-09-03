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

### Campaign #3 — runtime decode knobs — STATUS: CLOSED (2026-09-02)

| knob | verdict |
|---|---|
| n-gram speculation (`DS4_NGRAM_SPEC`, #846 port) | REJECT — net_saved −11..−14 s per ~50 s arm, deterministic 20.97% accept / 0.168 tok per cycle |
| DSpark (`--mtp --dspark`) | REJECT — −16% decode (08-09), no Vision-Exp support GGUF anyway |
| `DS4_METAL_DECODE_INDEXER_SPARSE_THRESHOLD=512` | REJECT — leans worse |
| `DS4_METAL_DECODE_INDEXER_SPARSE_THRESHOLD=2048` | REJECT — flat |
| default threshold 1024 | KEEP |

Reason: no candidate separates from BASE under valid clean testing. Additional finding: large
within-session M5 performance drift (base 37.0 → 33.1 t/s mean across one session, 5-10 t/s
inside every arm) contaminates late-position absolute throughput; only interleaved every-window
comparisons are admissible. Do NOT reopen #846 or threshold tuning on the strength of one
favourable isolated run.

Only proven decode lever: the AProjQ4 model recipe, +13–15% (08-27 here, PR #621 M5 retest).
Note the measured 08-27 recipe was `--attention-proj q4_k --output q4_k` — AProj **and** head
together (216 tensors). The Vision-Exp Q4_K build uses the same combined recipe, so it is a
like-for-like Campaign #4B; any OutQ4K-vs-OutQ8 attribution needs a separate causal control.

### Campaign #4B re-scoped: speed only, no quality gate (user, 2026-09-02 17:45)

User: "no need for that [eval] … i just want speed and performance optimisation". Q8 baseline
eval stopped at 33/92 (partial trace kept in `tasks/2026-09-02-eval-vision-exp-q8.*`); the
Q4_K eval will not run. Remaining path:
- [ ] quantize finishes (resumed at full speed once the eval released the machine)
- [ ] splice: reference bytes for the 1112 unchanged tensors, our 216 q4_K → candidate
- [ ] `tasks/decode-ab.sh` base ↔ q4k, ABAB, short/~2K/~30K; adopt if q4k wins every window
- [ ] adopt: `ds4flash.gguf`, `--model` in run.md + run-ds4-*.sh, fresh `--kv-disk-dir`, restart prod

#### Campaign #4B progress (2026-09-02 19:05)
- [x] quantize finished 19:03 (16:18→19:03 incl. ~1h SIGSTOP; ~1.3 min/layer at full speed). Output renamed
      `…AProjQ4K-SExpQ8-OutQ4K-fullregen.gguf` (routed experts re-quantized with our imatrix — NOT the reference bytes).
- [x] splice: `cp -c` fullregen → `…AProjQ4K-SExpQ8-OutQ4K.gguf` (candidate), `gguf_splice.py REF CAND`:
      `tensors=1328 unchanged=1112 type_changed=216 size_mismatch=0`, 81.3 GB copied in 25 s. Spot-check sha256:
      ffn_gate_exps / ffn_down_exps / ffn_gate_shexp / token_embd / blk.2.indexer.attn_q_b == reference;
      attn_output_b / output.weight == fullregen (q4_K, type 12). Candidate differs from Q8 baseline in exactly 216 tensors.
- [x] CLI load smoke 20:21: loads with the 889.64 MiB vision tower, resident 78.37 GiB, 17x24=408, 44.02 t/s cold.
- [x] decode-ab base ↔ q4k ABAB — q4k wins every window (see Campaign #4B review below).
- [x] adopt + restart prod — live on :8000, 48.76 t/s decode, vision verified.

### Review — Campaign #4B: AProjQ4K adopted as the daily model (2026-09-02 20:22-20:42)

Raw log: `tasks/2026-09-02-decode-ab-campaign4.log`. Six arms on :8009 with the run.md flag
set, fresh KV dir each: two throwaway cold-mmap warm-ups (`q4kdiscard`, `basediscard`) then
`base1 q4k1 base2 q4k2`. No `THINKING` in any arm, `thr_active=default` throughout, and each
arm records its weights file + size + inode — for a model A/B the GGUF *is* the toggle, so the
model line is the toggle assertion the harness otherwise lacks.

| window | base (4 runs, t/s) | q4k (4 runs, t/s) | strict every-window test |
|---|---|---|---|
| short, 512 gen | 39.77 37.83 30.36 34.35 | 49.02 47.89 40.51 43.51 | q4k min 40.51 > base max 39.77 **PASS** |
| mid, ~2.8K prompt | 36.22 36.62 32.43 35.10 | 49.90 48.85 45.79 47.61 | 45.79 > 36.62 **PASS** |
| deep, 34K prompt | 32.61 30.18 29.69 26.52 | 39.88 37.72 37.27 33.17 | 33.17 > 32.61 **PASS** |

Drift-fair adjacent pair (base1 immediately followed by q4k1): **+24.9% short, +35.6% mid,
+23.6% deep**. The every-window rule passes on all four arms even so, which is the strongest
result any campaign here has produced — campaign #3's knobs could not separate at all.

**Prefill does NOT regress.** q4k >= base in every adjacent pair (deep 528.0/428.2 vs
479.5/406.6; 451.6/412.5 vs 420.2/407.6). This contradicts the -11.3% prefill cost measured
on 2026-08-27 with the SuperDeepseek build. The difference between the two runs is the
imatrix: 08-27 used the routed-only 1p5m file, which calibrated none of the 216 dense tensors
being cut to 4 bit; this build used the merged routed-1p5m + dense-220k file.

Two honesty notes:
- **`base2` is contaminated and was excluded from the headline number.** A `shasum -a 256`
  of the 84 GB candidate that I had launched myself was competing for disk I/O from 20:35,
  i.e. during `base2` only. It depresses base, which biases *toward* q4k — the wrong
  direction to accept. Killed at 20:36; `q4k2` ran clean. The verdict rests on base1<->q4k1.
  Lesson: nothing else may touch the disk during an arm, including our own bookkeeping.
- **The mid prompt generates fewer tokens on q4k** (76 vs 256) — the model stops earlier on
  identical input. That is a behavioural difference from the q4_K output head, not a timing
  artifact (t/s is per-token). It is unmeasured for quality; the eval gate was waived by the
  user, so this is a known open risk, not a cleared one.

Adopted 20:35-20:43: `ds4flash.gguf` repointed, `--model` updated in `run.md` and all four
`run-ds4-*.sh`, fresh `--kv-disk-dir` (`...-ablit-q4k[-tag]`; the Q8 checkpoints stay in
`...-q2`). run.md's model + rollback sections rewritten, and a stale paragraph claiming the
three variant scripts still run the drowzeys 0731 pair was corrected. Prod restarted on
:8000 with the run.md §2 command: `/health` ok, **48.76 t/s** decode, vision caption on
`earth.jpg` correct.

**Context lowered to `--ctx 262144` (user, 20:47).** This also closes the 2026-08-23
inconsistency: the four wrappers were already `CTX=262144`, and checking the clients on disk
showed **both were already 262144 too** — run.md §5's table claiming `393216 ✓` for Pi and
Pythinker had never been true. Everything is now 256K end to end. Planned memory
**82.74 GiB** (KV 2.36 + buffers 2.00 + model 78.37), 48.36 t/s decode after the restart;
`/v1/models` reports `context_length: 262144`. Think Max would need 393216 on the server
*and* both clients raised together.

Not fixed, pre-existing: `run-ds4-{conf,strict,dspark}.sh` still hardcode
`THERMALFORGE="$HOME/.mtplx/bin/thermalforge"`, a path deleted with MTPLX on 2026-09-01 —
only `run-ds4-monitored.sh` resolves it via `command -v`.
