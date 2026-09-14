# Plan v3 — faster decode, MTP and turn latency for long-running coding sessions

Date: 2026-09-14 (v3 after second review). Branch `glm53-pr920` (HEAD 0a77fb0 + uncommitted
Metal-4/indexer edits). Model: GLM-5.3-Flash-UNCEN-d21b-L17-18-19Q4KExperts-Q2
(95.6 GiB resident), M5 Max 128 GB.

v3 changes vs v2: PR #964 split into engine-only perf A/B (P4b1) and full
integration (P4b2); Protocol B arms use cloned KV directories, never a shared one;
A0.4 replays exact recorded request payloads (server-side recorder), not log
reconstructions; CIs use paired block bootstrap over turns; P2.5 ROI gate before
P3; correctness required at 100K and 200K before adopting #964; checkpoint policy
has both a cost bound and a durability bound; explicit rollback points.

v2 changes vs v1: tree freeze first; correctness cherry-picks before any tuning;
upstream PR #964 evaluated before custom MTP work; MTP go/no-go derived from
`C/S < 1+a`, not a fixed acceptance threshold; A3.4 score/top-k reuse removed
pending proof of query-independence; A2.1 hardened; A2.2 target is a budget,
not zero; two benchmark protocols; tool-call server commits deferred.

## Evidence this plan rests on

Today's pi session (75 requests, 55K -> 106K ctx, `scratchpad/server.log`):

| metric | value |
|---|---|
| decode, MTP off (ceiling 32K) | mean 23.0 t/s, steady 21-22 t/s at 70-105K |
| prompt-done per turn | avg 3.41 s; appends <= 64 tok avg 1.08 s; a 5-tok append 2.07 s |
| continued KV checkpoint | 1.56 / 2.02 / 2.49 GB at 61K / 82K / 102K, save 0.4-0.9 s |
| first decode chunk after prefill | ONE observation of 12.6 t/s vs 25 t/s steady |
| errors / cancels / cache misses | 0 / 0 / 0 |

Ledger facts (`tasks/ledger.md`):
- F32: MTP **off**, decode decays 5% from 2K to 200K (27.3 -> 26.0 t/s).
- MTP **on**, decode decays 32% (33.0 -> 22.5 t/s). The penalty is in speculation.
- F23/todo F3: above the dense window the width-2 row verifier is refused and verify
  falls to the indexed batch path (PR #920 keeps that fallback by design, so a sparse
  row verifier is new work). The boundary is 4096 on this tree; upstream `0e9cc2d`
  moves GLM to the model-defined **2051**-token dense boundary, so every "above 4K"
  number must be re-measured after taking it.
- F17/F20: width-2 acceptance 0.721 at short ctx; acceptance at 100K+ unmeasured.

Break-even model (width 2): plain step `S`, speculative cycle `C`, acceptance `a`.
MTP wins iff `C/S < 1 + a`. Today at 200K: S = 38.5 ms, C = 76.5 ms, C/S = 1.99 ->
needs a > 0.99: hopeless, hence the ceiling. If A3.3 gets C ~ 50 ms, C/S = 1.30 ->
break-even a = 0.30; at a = 0.6 that is ~32 t/s vs ~26 plain. Gate on measured
`a` and measured `C`, never on a fixed threshold.

Upstream candidate: PR #964 "Metal: accelerate GLM 5.3 Flash decode and prefill"
(open, head `0a908192`, 70 files, +14363/-373, last pushed 2026-09-13). Reported
+17.6% median decode on an M5 Max 128 GB Q2 at 2K-16K on an earlier revision; no
data on this hybrid quant or at 100K+. Must be measured here, not assumed.

## Targets

All measured on the replay harness (A0.4) at 100K context unless stated.

| id | metric | now | gate |
|---|---|---|---|
| T1 | MTP decode at 100K | 22 t/s (MTP off) | >= 10% over post-#964 plain decode, paired CI > 0 |
| T2 | prompt-done, appends <= 64 tok | avg 1.08 s | p50 <= 0.30 s, p95 <= 0.50 s |
| T3 | first chunk after prefill | one 12.6 t/s outlier | first prove it recurs; then p95 <= 1.10x steady |
| T4 | checkpoint blocking the request path | 0.4-0.9 s sync write | p95 <= 25-50 ms, no synchronous disk write |
| T5 | 75-turn session | 28 min total, ~4 min server | server <= 3 min; total tracked separately |

## Phase -1 — freeze the tree (hours)

- A-1.1 Commit the uncommitted Metal-4 opt-out / indexer-query prune / launcher edits
  as one explicit baseline commit (or revert them). No A/B until this is done.
- A-1.2 Record: binary SHA-256, `ds4_metal.m` + `metal/*.metal` SHA, model SHA-256,
  launcher env, macOS build, benchmark inputs. Store in `tasks/data/baseline-v2.json`.
  Verify: file exists; `git status` clean.

## Phase 0 — instrument and capture the current baseline (1 day)

- A0.1 Per-request timing breakdown on the `prompt done` line: cache lookup,
  restore/rewind, template render, tokenize, prefill compute, checkpoint save.
  Verify: fields sum to the existing total within 5%.
- A0.2 Per speculative cycle (behind `--mtp-timing`): `pos plain_step_ms draft_ms
  verify_ms indexer_ms rollback_ms accepted_tokens cycle_tokens cycle_ms
  effective_tps verify_path phys_footprint swap_delta`. This computes break-even live.
  Verify: emitted at pos 100K.
- A0.3 Launcher monitor: `phys_footprint` and `ps -o %cpu` instead of rss/cpu that
  read 0.7 GiB / 0% on a busy 96 GB model. Verify: within 5% of Activity Monitor.
- A0.4 Exact replay harness. `server.log` has timing only, no bodies, so add a
  server-side recorder (`DS4_SERVER_RECORD_DIR`): per request it writes the raw
  JSON payload, api kind, rendered-prompt SHA-256, prompt/cached token counts,
  generation settings, and the ctx span. `tasks/replay.py` replays those payloads
  in order against a warm cache and reports T1-T5 with p50/p95 per turn. The
  2026-09-14 session cannot be replayed exactly; the baseline is captured on the
  next recorded pi session. Verify: two back-to-back replays agree within 3%;
  rendered-prompt hashes match the recording.
- A0.5 Capture the v2 baseline on the frozen tree. Verify: numbers in `tasks/data/`.

## Phase 4a — correctness foundation (1-2 days)

Fresh branch off the frozen baseline. Cherry-pick, in order:
- `233eeb8` restore GLM recurrent/pooled-indexer state after rejected speculation;
  Metal GLM memory accounting. Prerequisite for any ceiling change.
- `0e9cc2d` GLM attention masking/reductions, 2051 dense boundary, `test_glm_attention.c`.
- `fb2abb2`, `e965352`, `c0a6119` small correctness fixes.
Then re-run: O1 gates, `--metal-tensor-equivalence`, session-state tests, and the
A0.4 harness. **This is the new baseline; P0 numbers are superseded.**
Verify: all gates green; harness deltas recorded.

Deferred, not in this branch: the tool-call server set (`5b3cc8b d077fa6 fc6414c
759dd7c 6eac69e 930ab73`). They alter token streams/templates and would confound
the performance work. Take them afterwards on top of `2215830`, unless the pi
replay exposes a tool-call bug that one of them fixes.

## Phase 4b1 — PR #964 engine-only performance A/B (1-2 days)

#964's current head also carries server/checkpoint/recovery changes (tool-block
checkpoint replay, checkpoint scanning, recovery). Those are exactly what P4a
deferred, so the PR is not one performance arm.
- A4b1.1 Split the PR: cherry-pick only its Metal/engine commits (ds4.c, ds4_metal.m,
  metal/*.metal, ds4_gpu*) onto the P4a baseline in its own worktree. Two frozen
  binaries. Record the commit list.
- A4b1.2 Protocol B with `ds4-bench`, MTP OFF, at 2K / 32K / 100K / 200K:
  interleaved ABBABAAB, >= 4 runs per arm. Each arm run gets its own APFS clone of
  one immutable seed KV directory (`seed-kv/` -> `A-run-1/`, `B-run-1/`, ...);
  the seed's metadata is hashed before the run and never written. Report decode
  t/s, prefill t/s, first-token latency, phys_footprint, swap.
- A4b1.3 Correctness at every point, not just performance: 2K kernel/reference
  sanity; 32K coding regime; 100K primary; 200K stress. Full-logit equivalence
  where practical, otherwise greedy/state equivalence plus the official NLL
  fixture. The PR's own traces are not proof for this hybrid quant.
- Adopt iff 100K decode wins with block-bootstrap CI > 0 and no correctness
  regression. Expected: plain 100K decode 22 -> 25-27 t/s; upstream saw +17.6%
  decode / -1.9% prefill on an M5 Max Q2, +27% / +18% on M3 Ultra, so it is
  architecture-dependent.

## Phase 4b2 — PR #964 full integration (1 day)

- A4b2.1 Only if P4b1 adopted: bring in the PR's server/session commits on a branch
  off P4b1 and run the replay harness, session-state tests, disk-KV and tool-loop
  suites separately. Adopt or reject on correctness, not performance.
  Rollback: the P4b1 engine-only binary.

## Phase 1 — config tuning on the production baseline (same day)

- A1.1 MTP ceiling sweep 0 / 32768 / 65536 using A0.2 break-even data (Protocol A).
  Pick the crossover from `C/S < 1+a`; update the launcher default with the data.
- A1.2 Checkpoint policy: replace the fixed 20480 interval with two simultaneous
  bounds: cost (write duty <= 1-2% of server wall time) and durability
  (uncheckpointed progress <= N tokens OR <= T minutes, whichever first) so an idle
  interactive session still checkpoints. Snapshots grow with context (1.56 -> 2.49
  GB), so a constant token interval makes write cost grow anyway.
  Verify: duty cycle and staleness logged; cold restart restores within N tokens.

## Phase 2 — turn latency (2-3 days)

- A2.1 Measure first with A0.1: hashing of the full token-text key, template
  render, retokenize, or rewind-target walk. Fix only the dominant term.
  If it is O(L) hashing: keep an incremental/chunked fingerprint of the live
  prefix and **still verify the boundary exactly** (compare tokens at and after the
  claimed match point). Never skip the cache lookup because a request "looks like an
  append": template or tool-message formatting can change tokens before the tail.
  Verify: T2 p50/p95 on harness; cache hit count and greedy streams unchanged.
- A2.2 Asynchronous checkpoint save: single-flight worker, bounded staging buffer
  (reuse the existing save buffer; the machine is wired to 101.7 GiB planned),
  coalesce obsolete saves, immutable ownership + generation id so an old write can
  never publish over a newer checkpoint, `temp -> fsync(file) -> rename ->
  fsync(dir)`. Keep the synchronous path behind a rollback flag.
  Verify: T4 p95; kill during save leaves the previous checkpoint valid
  (extend `test_session_state.c`).
- A2.3 First-chunk penalty: only if A0.2 shows it recurring (>= 3 of 20 prefills).
  Candidates: decode graph re-plan after a differently shaped prefill, indexer pool
  re-expand, KDA state copy. May vanish with #964, which reports first-token gains.

## Phase 2.5 — ROI gate before any P3 work

Re-measure the 75-turn session on the P2 baseline. Proceed to P3 only if
projected agent-session wall-clock improvement >= 3-5% **or** projected MTP decode
gain >= 15% over the final plain baseline, using S, C', a from A0.2/A3.1. If
ordinary decode is already high-20s and server time is a small share of the
session, STOP and ship the P2 baseline.

## Hard constraint — GPU-resident only (set 2026-09-14)

Target machine is M5 Max 128 GB with GLM-5.3 Flash fully resident in unified
memory. Therefore, for this branch:

- No SSD-streamed experts, no expert offloading, no SSD decode path.
- No optimization whose benefit depends on expert `pread`/disk traffic.
- PR #1047 (Qwen3.8 Metal SSD expert streaming, +72.7% MTP decode on M1 Max
  from -75% `pread` bytes) is **not** an implementation candidate. Its only
  borrowable idea is selective-expert staging, and only if applied entirely in
  unified memory/Metal.
- PR #1049 (V4.1 gathered-KV reuse, +0.79%) below the 1% bar; ignore.

Roadmap order under this constraint:

```
#964 resident Metal path -> plain decode profiling -> routed-MoE/dispatch
optimization -> KDA/indexer optimization -> turn latency -> MTP verifier
profiling -> resident selective-expert staging ONLY if profiling proves
unnecessary in-memory copies
```

MTP verifier profiling (A3.0, before A3.1) must attribute per verify step:

```
selected experts -> resident bytes touched/copied -> staging/gather time
-> routed-MoE time -> verify time
```

If the verifier copies or gathers materially more expert data than the
selected set, optimize those RAM/Metal copies. If not, #1047 is irrelevant
and A3 proceeds to verifier arithmetic as planned. Note MTP is currently gated
off above ctx 32,768, so this branch pays nothing for the >32K pi workload
until the gate question is settled.

## Phase 3 — MTP above the dense window (1-2 weeks)

- A3.0 Expert data-movement attribution for one verify step at <=32K (see hard
  constraint above). Go/no-go for resident selective-expert staging.

- A3.1 Protocol A in-process ABBA at 100K, MTP on/off per 64-token segment via the
  mask-program stepping (`attribseg.py`). Yields S, C, a, `verify[batch]` ms.
- A3.2 Go/no-go from the model: proceed iff the projected sparse verifier cost
  `C'` (from A0.2's per-stage split) satisfies `C'/S < 1 + a_measured` with margin
  >= 10%. No fixed acceptance threshold.
- A3.3 Sparse row verifier: extend `glm_graph_verify_rows` to attend past the dense
  window via the decode-style path (`indexer_q -> scores(visible/4) -> topk ->
  pooled expand`) for the 2 rows, one shared launch per stage. Target C ~ 50 ms
  at 100K (C/S ~ 1.3). Rollback contract stays exactly the batch path's.
  Verify: `verify[rows]` at pos 100K; full-logit equality vs the batch path on the
  tensor-equivalence gate at every transition point (see gates); T1 on harness.
- A3.4 Reuse across draft and verify — **restricted**. Scores and top-k depend on
  the query row, so overlapping visible sets do not make them reusable. Only
  query-independent structures (pooled K / index tables) may be shared, and only
  after (1) a source-level proof that the reused quantity has no query dependence
  and (2) full-logit equality on the gate. Score/top-k reuse is prohibited until
  both exist.
- A3.5 Width-3 only if measured long-ctx `a >= 0.75` on the sparse verifier.

## Benchmark protocols

- **Protocol A (runtime switch, same binary):** in-process synchronized ABBA with
  interleaved controls; check tokens, logits and accept/reject schedule identity
  where the switch should not change them. Used for P1, A2.x, A3.x.
- **Protocol B (code revision):** frozen worktrees and binaries, interleaved
  multi-run ABBA/ABBABAAB, one immutable seed KV cloned per arm run, thermal gap
  logged. Used for P4a, P4b1/2, A3.3.
- **Statistics:** segments at long context are serially correlated (shared model,
  thermal, cache state). Report median paired gain, mean paired gain, and a 95%
  paired block-bootstrap CI with the turn (or a contiguous multi-segment block) as
  the resampling unit. Never an IID CI over 64-token segments.
Cross-restart single-run comparisons remain inadmissible.

## Correctness gates (A3.3 and any verifier change)

Tensor-equivalence and greedy-stream identity at: below the dense boundary, around
2051 (post-`0e9cc2d`), 4K, 32K, 100K, >= 200K. Exercise accept and reject, partial
rollback, checkpoint restore mid-speculation, cancel/interrupt, repeated tool calls.
Plus the release list: GLM MTP comparison, long-context smoke, session correctness,
disk KV, server tool-loop.

## Execution order and rollback points

```
P-1  freeze          -> gate: git clean, baseline-v2.json written
P0   instrument      -> gate: recorder + replay repeatable within 3%; baseline captured
P4a  correctness     -> gate: all gates green; harness re-baselined
                        rollback: frozen pre-P4a commit b69fd2c
P4b1 #964 engine A/B -> gate: Protocol B, block-bootstrap CI > 0 at 100K, correctness
                        at 2K/32K/100K/200K -> adopt   rollback: P4a binary
P4b2 #964 integrate  -> gate: replay/session/tool suites green   rollback: P4b1 binary
P1   config          -> gate: ceiling + checkpoint policy chosen from data
P2   latency         -> gate: T2, T4 met; T3 only if it recurs
                        rollback: sync checkpoint + old prefix path stay feature-flagged
P2.5 ROI             -> insufficient -> STOP and ship P2 baseline
P3   MTP             -> gate: C'/S < 1+a with 10% margin; T1 met
                        fallback: batch verifier remains the hard fallback
```

## Risks

- R1 A3.3 touches the verify/rollback contract next to the F19 rewind bug.
  Mitigation: `233eeb8` first; session-state tests extended before the kernel work.
- R2 Uncommitted edits in the live binary. Mitigation: P-1.
- R3 #964 is a 70-file open PR that was rebased on 2026-09-13; it may conflict with
  the local MoE/router work. Mitigation: isolated worktree, adopt only on data.
- R4 Memory: 101.7 GiB planned on 128 GB. A2.2 must not add a second staging buffer.
- R5 The 2051 boundary change invalidates today's 4K-based measurements. Mitigation:
  nothing in P1-P3 is measured before P4a.
