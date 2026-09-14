# Plan v2 — faster decode, MTP and turn latency for long-running coding sessions

Date: 2026-09-14 (v2 after review). Branch `glm53-pr920` (HEAD 0a77fb0 + uncommitted
Metal-4/indexer edits). Model: GLM-5.3-Flash-UNCEN-d21b-L17-18-19Q4KExperts-Q2
(95.6 GiB resident), M5 Max 128 GB.

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
- A0.4 Replay harness from today's `server.log`: same 75 prompts, same order, warm
  cache; reports T1-T5 with p50/p95. Verify: two back-to-back runs agree within 3%.
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

## Phase 4b — PR #964 isolated A/B on this hybrid quant (1-2 days)

- A4b.1 Build #964 head `0a908192` rebased onto the P4a baseline in its own worktree.
  Two frozen binaries.
- A4b.2 Protocol B (cross-binary): interleaved ABBABAAB, >= 4 runs per arm, same
  warm KV, MTP OFF, at 2K / 32K / 100K / 200K. Report decode t/s, prefill t/s,
  first-token latency, phys_footprint, swap.
- A4b.3 Correctness: tensor-equivalence, greedy-stream identity, O1 quality gates.
- Adopt iff decode wins with CI > 0 at 100K and no correctness regression. If adopted,
  it becomes the production baseline and the T1 comparator.
  Expected: plain 100K decode 22 -> 25-27 t/s is a bigger, lower-risk win than P3.

## Phase 1 — config tuning on the production baseline (same day)

- A1.1 MTP ceiling sweep 0 / 32768 / 65536 using A0.2 break-even data (Protocol A).
  Pick the crossover from `C/S < 1+a`; update the launcher default with the data.
- A1.2 Checkpoint policy: replace the fixed 20480 interval with an adaptive
  write-duty rule: checkpoint writes <= 1-2% of server wall time, bounded replay
  after crash (<= N tokens). Snapshots grow with context (1.56 -> 2.49 GB), so a
  constant token interval makes write cost grow anyway.
  Verify: duty cycle logged; cold restart restores within N tokens.

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

## Phase 3 — MTP above the dense window (1-2 weeks)

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
  multi-run ABBA/ABBABAAB, same warm KV, thermal gap logged. Used for P4a, P4b, A3.3.
Cross-restart single-run comparisons remain inadmissible.

## Correctness gates (A3.3 and any verifier change)

Tensor-equivalence and greedy-stream identity at: below the dense boundary, around
2051 (post-`0e9cc2d`), 4K, 32K, 100K, >= 200K. Exercise accept and reject, partial
rollback, checkpoint restore mid-speculation, cancel/interrupt, repeated tool calls.
Plus the release list: GLM MTP comparison, long-context smoke, session correctness,
disk KV, server tool-loop.

## Execution order

```
P-1 freeze        -> gate: git clean, baseline-v2.json written
P0  instrument    -> gate: harness repeatable within 3%; baseline captured
P4a correctness   -> gate: all gates green; harness re-baselined
P4b PR #964 A/B   -> gate: Protocol B, CI > 0 at 100K, no regression -> adopt
P1  config        -> gate: ceiling + checkpoint policy chosen from data
P2  latency       -> gate: T2, T4 met; T3 only if it recurs
P3  MTP           -> gate: C'/S < 1+a with 10% margin; T1 met
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
