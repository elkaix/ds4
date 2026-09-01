# GLM-5.3 long-context decode: 4K boundary work

## Established (source-verified)
- `glm_graph_dense_compact_attention_limit()` (ds4.c:41212) returns `g->ctx_cap`
  for GLM-5.3 on the compact DSA cache. Banner prints `cap=4096`.
  **Still an inference that ctx_cap == 4096** -- must be confirmed by the pos probe.
- Decode splits at ds4.c:51565: `visible <= limit` -> `indexer_fill` (dense);
  else `indexer_q/weights/scores(visible/4)/topk` -> pooled expand -> attention.
- `glm_graph_verify_rows` refuses at `pos + n > limit` (ds4.c:47166+patch).
- GGUF: indexer top_k=2048 pool_size=4; linear_attn head_count=64 head_dim=128
  conv_kernel=4. `il % 4 != 3` => 34 KDA / 11 DSA of 45 trunk layers.
- One `glm53_graph_copy_kda_state` moves 145.6 MiB + forces `ds4_gpu_end_commands()`.
  Above 4094 it ran twice per cycle; the second was pure waste.

## Done
- [x] P0 patch: `glm_graph_verify_rows_eligible()` shared predicate; gate the
      attempt in `glm53_spec_verify` so the guaranteed-fail rows pass and its
      KDA `restore_base` are skipped above the dense window.
      A/B lever: `DS4_GLM_MTP_ROWS_EAGER_ATTEMPT=1` restores old behaviour.
- [x] `pos=%u` added to the `--mtp-timing` emit (ds4.c:65153).
- [x] `-fsyntax-only` clean, 0 warnings. **Not built yet** (server is live).
- [x] Harness: `SWEEP_SET=micro` target ladder; retry once on short decode.

## Next, in order
1. Let the wide MTP-ON sweep finish (pass 1 of 3 in flight). Do not `make`.
2. `make` + restart with `--mtp-timing`. **Boundary probe first**: one request at
   ~4050 prompt tokens, 256 decode. Confirm `verify[rows]` below pos 4094 and
   `verify[batch]` at/above. If it never flips, ctx_cap != 4096 and the P0 patch
   is a no-op -- stop and re-derive.
3. Acceptance table at ~500 / 8K / 28K / 34K (timing on).
4. Restart WITHOUT `--mtp-timing` (the emit fprintf+2 malloc/free per cycle
   contaminates throughput). Run `SWEEP_SET=micro`, MTP-ON then MTP-OFF
   (`DS4_MTP_SPEC_DISABLE=1`), alternating short blocks to average thermal drift.
5. P0-patch A/B via `DS4_GLM_MTP_ROWS_EAGER_ATTEMPT`, alternating blocks, at a
   context above 4K. Cross-check against `verify[batch]=` ms from the timing arm.
6. Stage profile on a **DSA layer** (il % 4 == 3) at ~2K/3.8K/4.2K/16K/64K.

## Open bug, deliberately not bundled
`ds4_session_glm_spec_cycle_impl` (ds4.c:64805) takes neither `ignore_eos` nor
`think_mode`; n1/n2 come from plain `glm_session_logits_argmax` (64847/64961/65029).
So GLM MTP can commit a stop token that `ds4_session_argmax_ignoring_eos` would
have excluded. The server catches it at ds4_server.c:12504 and ends the response,
so nothing leaks to the client -- but `ignore_eos: true` cannot hold a fixed
generation length under MTP. Fixing it here would confound the perf A/B.

## 2026-08-31 boundary probe results (unpatched binary, --mtp-timing)

Prompt 4050 tok, 256 gen, temperature 0. Position reconstructed as
live_len(4050) + cumsum(committed). Sanity: sum(committed)=256=requested; both
arms produced 147 cycles with an IDENTICAL position + accept/reject schedule.

F1. BOUNDARY CONFIRMED EXACTLY. last verify[rows] pos=4094, first
    verify[batch] pos=4095. Guard is `pos + n > dense_limit` with n=2, so
    dense_limit = 4096. The ctx_cap=4096 inference was correct; the P0 patch
    does target the live path.

F2. THE WASTED RESTORE IS SMALL. `restore_base` runs inside glm53_spec_verify
    (ds4.c:64769) and is therefore inside the timed verify[] window. One KDA
    copy of 145.6 MiB is directly priced by two other fields that each perform
    exactly one: setup=0.80 ms (transaction_begin save) and rollback=0.80 ms.
    => the patch removes ~0.8 ms from a 52.75 ms cycle = ~1.5%. Real and free,
    but ~100x smaller than the earlier framing implied. ~380 GB/s effective,
    i.e. the copy is already bandwidth-bound and near peak.

F3. CROSSING 4K IS A STEP DOWN IN COST, NOT A CLIFF UP. Within a single
    process (drift-immune, adjacent in time), per-cycle total drops at 4095:
      fast-verify ON : rows 58.60 -> batch 52.75 ms  (-5.85)
      fast-verify OFF: batch 66.10 -> batch 60.60 ms (-5.50)
    The OFF arm is batch on BOTH sides, so its -5.50 is purely the dense ->
    sparse attention regime change. The two drops agree, so the boundary step
    is the attention regime, NOT the verifier. Sparse attention above 4096
    (top_k=2048 < 4096 dense rows) is cheaper than the dense window below it.
    => the 507->34K decay is not caused by a 4K cliff; 4K is a step DOWN.
    Micro-sweep prediction: ms/tok dips just above 4096, then rises with the
    O(L/4) indexer_scores term.

F4. THE ROWS FAST PATH'S VALUE AT 4K IS UNRESOLVED, NOT ZERO. Cross-restart
    drift is not scalar (setup 1.25x, draft 1.20x, ver 1.14x), so a ratio-of-
    ratios cannot extract a +-2% effect through a drift model that is itself
    off by +-10%. The honest statement: the width-2 row verifier's value at
    ~4K is BELOW this machine's cross-restart resolution. Deciding it requires
    interleaved arms. Do not cite it for or against the P1 sparse-verify item.
    NB: DS4_GLM_DISABLE_MTP_FAST_VERIFY=1 equals the P0 patch only ABOVE 4094;
    below 4094 they differ (patch keeps rows, the env var kills it).

F5. SEQUENTIAL A/B ACROSS RESTARTS IS INVALID ON THIS MACHINE. The second
    process was uniformly ~14-25% slower on EVERY stage including ones the
    flag cannot touch (setup 0.80->1.00, draft 4.40->5.30), i.e. broad
    multiplicative drift, not a code-path effect. This is larger than every
    effect being chased. Any future A/B must interleave arms, or be justified
    by direct measurement of the removed work (as F2 does) rather than by an
    end-to-end delta.

F6. THE 4096 STEP IS SHARP, AND THERE IS A ONE-TIME CROSSING SPIKE.
    Arm B (batch on both sides, so only attention changes) per-cycle verify ms:
      pos 4076..4094  58.2 59.9 58.1 58.0 57.6 59.2 57.8 58.0 59.0 58.4   (flat,
                      mild ramp: first-half median 57.70 -> second-half 58.40)
      pos 4095        81.2   <-- one-time +23 ms crossing cycle
      pos 4097..4111  51.7 50.8 50.7 51.5 50.8 50.5 52.8 51.8 53.3 53.2   (plateau)
    Arm A shows the same shape (below ramps 51.00 -> 54.20, above flat 46.1).
    So: dense attention ramps with pos, pays a one-time cost to build the first
    sparse selection, then drops onto a plateau. Not a smooth decline => F3 holds.

F7. ABOVE 4096 THE ATTENTION COST IS FLAT, SO indexer_scores IS THE ONLY
    CONTEXT-GROWING TERM. Arm B verify medians: 53.50 (pos 4095-4126),
    54.00 (4128-4190), 52.20 (4228-4304); arm A 46.30 / 46.10 / 46.05.
    Consistent with top_k pinned at 2048 => attention work constant. The
    long-context decay must therefore come from indexer_scores over visible/4
    rows: 1024 rows at 4K (invisible here) vs 16384 at 64K.
    => THE DSA-LAYER STAGE PROFILE IS NOW THE HIGHEST-VALUE REMAINING ITEM,
    ahead of both verifier work items.

## Revised plan
- P0  build the P0 patch; justify on removed work (F2, 0.8 ms) not on an A/B.
- P0  DSA-layer stage profile at ~2K/3.8K/4.2K/16K/64K (layer with il%4==3).
- P1  micro-sweep 1K..8K; PREDICTION: ms/tok dips just above 4096.
      FIX: shuffle target order per pass -- ascending order confounds context
      with time-in-pass, and replication does not remove that.
- P2  rows-vs-batch value at 4K, only with interleaved arms.
- --   GLM MTP ignore_eos/think_mode correctness bug (separate, unscheduled).

## DSA-layer stage profile (layer 43, il%4==3), n=2 per cell
Caveat: each stage boundary flushes the GPU queue, so absolutes are inflated;
only growth against context is meaningful. Single layer, n=2 -- directional.

F8. F7 IS FALSIFIED. indexer_scores does NOT explain the long-context decay.
    Over 7176 -> 84925 tokens (11.8x context, so 11.8x rows at visible/4):
      indexer_scores  0.201 -> 0.283 ms   (1.4x, +0.08 ms)
      sum of ALL profiled stages  4.52 -> 4.66 ms  (+3%)
    The DSA layer's per-decode cost is essentially context-independent. At
    these sizes indexer_scores is launch/overhead-bound, not compute-bound
    (21231 rows x 32 heads is small for an M5 Max). => optimising
    indexer_scores/indexer_topk cannot recover the decay. Deprioritise.

F9. THE DECAY IS NOT IN DSA-LAYER COMPUTE AT ALL. Same run, end-to-end:
      pos  3968 -> 32.19 t/s
      pos  7742 -> 35.25 t/s
      pos 24020 -> 29.21 t/s
      pos 84925 -> 26.75 t/s      (-24% vs 7742)
    while the profiled per-layer stage sum moved +3%. The decay is therefore
    outside per-layer compute. KDA layers are recurrent (O(1) in context) and
    attention is pinned at top_k=2048, so the remaining candidate is the
    memory system: KV residency / page-ins. Consistent with the sweep's
    residual correlation (pageins +0.372) and with pageins at the 84925 point
    (306589) and the 24020 point (286425).
    => the next target is the MEMORY SYSTEM, not kernel work. CAUTION: this
    is correlation, not cause, and it is NOT the same problem as hot
    multi-session residency (that one fixes the 17-20 s client-switch TTFT).
    Four rival explanations remain untested:
      A. the live KV itself is paged
      B. growing KV displaces mapped model weights
      C. macOS compressor/swap pressure stalls execution
      D. page-ins merely accompany the slowdown without causing it
    The fixes differ sharply: under B, adding permanently hot KV sessions
    would make long-context decode WORSE. So the architectural target is
    "memory-budgeted hot session residency that protects the model's hot
    working set", not "keep every session KV resident".
    DECISIVE CONTROL (not yet run): repeat one identical high-context decode
    twice on a quiet machine. If run 2 is warmer, has fewer page-ins and is
    faster, residency is causal. If page-ins differ greatly but t/s barely
    moves, page-ins are an accompanying signal (D).

F10. THE 4096 STEP IS CONFIRMED AT THE STAGE LEVEL, INDEPENDENTLY.
     attention stage, DSA layer 43:
       pos 3968 (dense window)  1.259 ms
       pos 7176+ (sparse)       0.694 / 0.695 / 0.703 / 0.720 ms
     A drop of ~0.56 ms per DSA layer x 11 DSA layers = ~6.2 ms per decode,
     against the 5.5-6.5 ms step measured end-to-end in F3/F6 from a
     completely different instrument. Two independent methods agree.
     indexer_fill appears ONLY at pos 3968 and the indexer_q/weights/scores/
     topk chain ONLY above 4096, confirming the regime switch directly.

## P0 verifier patch: PRICED (in-process chop, 2026-08-31)

Whole-process A/B could not resolve this (8 runs ABBA: 90% CI [-2.49,+0.81] ms,
P(effect>0)=0.34) because between-run spread of the above-boundary median is
~3 ms against a ~0.8 ms effect. Fixed by moving the toggle INSIDE the process:
DS4_GLM_MTP_ROWS_EAGER_CHOP=1 alternates the treatment every speculative cycle,
so treated and untreated cycles are adjacent in time and drift is common-mode.

F11. DIRECT COST OF THE REMOVED OPERATION. A timer around exactly
     glm53_spec_transaction_restore_base() in the attempted_rows fallback,
     accumulated in memory and summarised once per 256 occurrences (per-cycle
     fprintf would perturb more than the effect):
       n=256  mean=0.807 ms  min=0.732  max=1.042
     Matches the 0.80 ms implied independently by the setup= and rollback=
     fields, and the 145.6 MiB / ~380 GB/s bandwidth estimate. Three methods
     agree.

F12. HOW MUCH SURVIVES END-TO-END. 720 above-boundary cycles, alternation
     verified at exactly 50.0%, arms labelled verify[batch+rows] vs
     verify[batch] so the treatment is asserted and not assumed. Statistics on
     NON-OVERLAPPING balanced 4-cycle blocks (n=180), so no pseudo-replication:
       verify_ms  +0.568 ms  95% CI [+0.303, +0.797]   = +1.25%
       total_ms   +0.486 ms  95% CI [+0.189, +0.748]   = +0.93%
     So ~60% of the 0.807 ms restore survives into whole-cycle latency; the
     rest is absorbed by overlap with other GPU work.
     => P0 patch is worth ~0.5 ms/cycle, ~0.9% of a speculative cycle above
     4094. Correct and free, so keep it -- but it is a cleanup, not the prize.

## Revised priorities (three independent problems)
P0-A  multi-client TTFT: hot session states vs one serial MTP compute lane.
      Addresses the proven KV disk save/load/re-prefill thrash.
P0-B  long-context decode residency: determine WHAT is paged as context grows
      (A/B/C/D above). Run the decisive control first.
P1    this verifier cleanup. Confirmed ~0.9%; keep, stop benchmarking it.
P2    kernel work. Only revisit if a repeated stage profile contradicts F8/F9.

## REVIEW — 2026-09-01 session

### Delivered
1. **P0 verifier cleanup committed, split in two** (`ac9f6ec` perf, `bd2ca17`
   instrumentation), index-only so the running binary stayed the measured
   artifact. Commit 1 syntax-checked in isolation to prove it does not depend on
   commit 2's A/B lever. `run-glm-ds4.sh` left untouched per standing constraint;
   `tasks/`, `roadmap.md`, `docs/architecture/` deliberately not committed.
2. **Warm-repeat control run** (roadmap steps 3-4), redesigned from "run it
   twice" to N identical back-to-back decodes in one process — two runs is n=1
   per arm against 14-25% drift.
3. **`tasks/ledger.md` written** — durable evidence ledger, F1-F16, with the
   evidence hierarchy and the method rules.
4. **`roadmap.md` updated** — status banner plus a dated RESULTS section
   recording which of its premises failed and the revised priorities.
5. **Disk KV cache cleared** (77 GiB, 113 files) and `clear_kv.sh` written with
   a path guard; the rule is now in the ledger, the roadmap and memory.

### The three findings that matter
- **F14 splits F9 in two.** A same-condition 2K/60K/2K bracket gives context
  **-13.8%** (drift-corrected) and pressure drift **-6.9% per 60K block**
  (-21.6% over a longer series, at *fixed* context). Ascending sweep order made
  them inseparable, so F9 charged all 24% to context. Note: an intermediate
  reading of "context costs only 1.5%" was itself confounded — it compared
  blocks sitting at different points on the drift trajectory. Bracket, don't
  just repeat.
- **F15 refutes hypothesis B.** The model is *wired* (102.33 GiB of 128), so KV
  cannot displace it; the kernel swaps the rest of the workstation instead. The
  global page-in counter was mostly other processes (hypothesis D).
- **F16 is the largest single item.** 95% of a *repeated* 60K request is
  re-prefill, on a prompt reported as a 59,904-token cache hit, because
  `live_prefix_rewind_target` promises a rewind `ds4_session_glm_mtp_rewind`
  can only honour within +-2 positions of the MTP frontier. Scope tested, not
  assumed: appending is **578x cheaper** than re-sending (0.3 s vs 173.4 s), so
  normal chat is safe and regeneration/editing/benchmarks are not.

### Deviations from the plan I was given
- Did not run the roadmap's two-run control as specified; it cannot resolve the
  effect size the existing data implied. Ran repeats-in-one-process instead.
- Did not run the section 8 memory-relief control: it requires closing the
  user's applications. The 2K series supplies the same evidence incidentally —
  throughput recovered as free memory recovered, without touching anything.
- Did not pursue steps 5-12 (residency manager). Two of their premises are now
  refuted; they need redesign before implementation, not execution.

### Gotchas for next session
- The wrapper prints `binary: STALE` because the commits are newer than the
  build timestamp. The binary content **is** HEAD (`git diff --quiet HEAD --
  ds4.c` passes); only the timestamp moved. Rebuild once to silence it.
- `warm_repeat.py` originally called `main()` at module scope, so importing it
  launched a full 65K sweep. Now guarded by `__main__`; keep it that way.
- Raw data and harnesses are copied into `tasks/data/` — the session scratchpad
  does not survive.

### THE TARGET (set 2026-09-01)
**45-50 tok/s sustained, no decay.** Today: 34.24 best (2K, fresh), 28.51 at
60K, 23.63 after drift. Needs ~1.4x. Reachable in principle — only 4.7% of
309.5 B params are active per token (14.6 B = 4.56 GB at 2.49 bits), so 34 t/s
is ~156 GB/s, about a quarter of the 546 GB/s peak. Decode is not
bandwidth-bound. F14 already locates about half the gap (drift + context term);
the rest must come from MTP acceptance/width and per-step overhead.
Gate PEAK and SUSTAINED separately — see roadmap "THE TARGET".

### Next session starts here
0. **Build a non-serializing profiler** (Metal timestamp counters, or bisect by
   stage groups). The current one drains the queue per boundary: a real layer is
   ~0.79 ms against 4.659 ms profiled, so ~84% of every number is its own
   overhead, spread near-uniformly (F18). Nothing inside the forward pass can be
   attributed until this exists.
0b. **Measure verify at n = 1, 2, 3, 4.** Decides width-3, which is worth
   between +1% and +18% (F17). Small harness; `glm_graph_verify_rows` already
   takes `n`. MTP *acceptance* is closed (p = 0.721, near-saturated); width is
   not.
1. Fix the rewind contract (F16). Either refuse rewinds the backend cannot serve
   — so the server stops reporting a cache hit it did not deliver — or add
   periodic KDA checkpoints (145.6 MiB each, ~2.1 GiB per 60K at every 4096).
2. Decide the wired-memory budget. 96 GiB planned on a 128 GiB machine is what
   actually degrades throughput over a session (F14/F15).
3. Find the residual ~14% context term. F8 put the profiled DSA stages at +3%
   across 11.8x context, so it is somewhere the stage profiler never looked:
   sampling, KV compress/decompress, command submission, the 34 KDA layers.
4. Re-measure multi-client TTFT afterwards; some of the 17-20 s may be F16
   rather than session switching.
5. Still open, still separate: GLM MTP `ignore_eos` / `think_mode`.


---

## REVIEW — 2026-09-01, second pass (target raised to S55)

### What landed

```text
4ce350c  server: ask the session whether a GLM prefix rewind is restorable
c2175e0  glm53: measurement-only verify-row scan for MTP width decisions
```

Docs: `roadmap.md` opening statement, four-problem table, 12-step execution
order, and the whole TARGET section replaced with **S55**. `tasks/ledger.md`
gained the target change, F19 (rewind contract) and F20 (width-3 closed).

### Delivered

1. **Target replaced.** 45-50 sustained -> **S55 = min(sustained t/s at 2K,
   32K, 60K, 85K) after >=30 min runtime, >= 55, with <=2% context decay and
   <=2% session drift.** Gate table, milestones M1/M2/M3, and the balanced
   bracket protocol are in `roadmap.md`.
2. **P0-A fixed and verified.** `ds4_session_can_rewind()` shares its predicate
   with the backend guard; the server asks it under `inference_mu`. Live log
   confirms the fake `memory-rewind` hit is gone and the refusal path fires.
   Turn 3 now takes a 20,480-token disk-KV hit it previously skipped.
3. **Width-3 closed with a direct measurement**, replacing an extrapolation
   from a code comment that compared two different kernels. 28.6 t/s of
   marginal rate vs 33.9 current; the marginal *draft step* is what kills it.

### What did NOT change

**Decode throughput. 33.9 t/s is exactly where it was.** This pass bought
correctness (P0-A) and knowledge (F20 closes a branch that would otherwise have
been implemented), not speed. S55 is still 1.61x away and the path to it beyond
the recoverable drift/context terms is still unidentified.

### Note for the user

The goal message contradicts itself: the headline commits to 55/S55, a tail
paragraph still says "keep 45-50 as a stretch target". The headline was taken.

### Next session starts here

0. **Build the non-serializing profiler.** Now the explicit blocker, with a
   concrete reason: F20 measured the verify *slope* (13.57 ms/row) but its
   *intercept* is only bounded, and every remaining decode lever -- MoE expert
   gather, launch overhead, the residual ~13.8% context term -- sits behind
   attributing that intercept. Option A: Metal GPU timestamp counters, one sync
   at normal completion. Option B: stage-group bisection (0-10, 11-20, 21-30,
   31-44, then subdivide). Gate: instrumented decode within a few percent of
   normal decode, not 5-6x slower.
1. **Phase 2 — bounded KDA rewind checkpoints.** P0-A made the contract
   truthful; it did not make a 257-token rewind fast. Prefer turn-frontier
   snapshots over fixed 4096-token ones, under a byte budget, not a count.
2. **Decide the wired-memory envelope.** 96 GiB planned on a 128 GiB box; run
   the 2K/60K/2K bracket at CTX 64K / 128K / 262K and record wired, free,
   compressor, swap alongside the three medians.
3. **Attribute the residual ~13.8% context term.** Needs (0).
4. **Re-measure multi-client TTFT** now that P0-A is in.
5. **Still open, still separate:** GLM MTP `ignore_eos` / `think_mode` bug.
6. **Costing exercise, not yet a task:** tree / multi-candidate speculation.
   Verification amortizes rows at 13.57 ms each; the open question is whether
   any drafter can produce candidates for less than that.
