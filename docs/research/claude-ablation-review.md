# Claude Live Ablation Review

Date: 2026-09-01

Scope: read-only review of the uncommitted `ds4.c`, `ds4.h`, and
`ds4_server.c` diff in `ds4-glm53`, frozen at SHA-256
`88ce011e1d7f79873913ef0951aac5d9d78d8caf6a5177693bbd03efccd5091a`,
plus its active `attrib200.py` harness. No server, model, build, or running test
was changed.

## Verdict

**REJECT as S55 evidence and as production code.** Path-visit counters are a
useful diagnostic concept. The current implementation and experiment cannot
rank components, authorize an optimization, or count toward sustained decode.

## Findings

- **C1 — Forbidden cross-arm state.** `attrib200.py` explicitly appends each
  invalid-output treatment response and `Continue.` to the next arm. Prompts,
  positions, KDA state, routed experts, and DSA selections differ. Flat control
  time cannot remove treatment carryover. The result is not same-state ABBA.
- **C2 — CPU link break.** The three new diagnostic definitions live inside
  `DS4_NO_GPU` exclusions, while `ds4_server.c` calls them unconditionally.
  `make cpu` therefore has unresolved symbols.
- **C3 — Process-global data races.** Dynamic mask, visit/skip counters, and
  cycle totals are unsynchronized globals shared by supported slot workers.
  One request can change or drain another request's diagnostic state.
- **C4 — Disabled-path overhead.** Every normal gate scans and updates global
  counters; every MTP cycle takes two clocks and updates totals. Only printing
  is environment-gated. No <=3% OFF/ON gate exists.
- **C5 — Counter boundary mismatch.** Counters begin before prompt sync but the
  decode clock begins later; progress logging resets them. A report can include
  untimed prompt work or earlier intervals, so it does not prove the timed path.
- **C6 — Exact-sampling acceptance is not measured.** Both acceptance and a
  rejection followed by replacement commit two tokens. `commit/cycle` cannot
  distinguish them; explicit accepted/rejected cycle counts are required.
- **C7 — Ablation is only a deletion ceiling.** Garbage hidden state changes
  downstream routing, locality, and acceptance. Whole-token deltas are not
  realizable per-stage costs, even after the harness is repaired.

## Minimum Correction

1. Use non-invasive Metal timestamps/trace on a valid baseline to rank macro
   regions; do not substitute invalid-output ablation for the mandatory profiler.
2. If path counters remain, make them disabled-fast-path and request/session
   local, or refuse multi-slot diagnostics; add CPU no-op definitions and reset
   exactly at the decode timer boundary.
3. Restore one frozen 200K checkpoint before every ablation arm with identical
   prompt, tokens, seed, and state; never append treatment output. Label all
   resulting deltas deletion ceilings only.
4. Require CPU link, concurrent isolation, prompt-exclusion, exact-sampling,
   and <=3% diagnostics-OFF/ON gates before retaining instrumentation.

The running MTP-off 200K attribution cannot rescue this design. S55 remains
the measured 22.48 t/s baseline until a valid-output, MTP-on controlled result
changes it.

## Replacement Snapshot Review

Claude replaced the reviewed diff with an in-generation mask program. A second
max-effort Sol review froze the replacement at SHA-256
`2987a32031b3380a4282a69807a6766dbc80d3a67ef16477c88b33d79afbded8`.
Verdict remains **REJECT**:

- **C8 — Still not same-state ABBA.** Advancing changes only the mask. It does
  not restore tokens, RNG, KV/dense caches, KDA recurrent/conv state, MTP draft
  state, logits, or checkpoint. Later controls inherit invalid arm state.
- **C9 — CPU and concurrency blockers remain.** Four public functions are
  defined only in GPU code but called unconditionally by the CPU server.
  Program, mask, counters, and cycle timers remain unsynchronized process
  globals shared by concurrent slot workers.
- **C10 — Arm boundaries can split one MTP cycle.** A cycle may commit two
  tokens, while mask advancement occurs in the per-token emission loop. The
  second token can be labeled under a mask that did not compute it; interval
  one can advance twice and skip an arm without model execution.
- **C11 — Timing does not isolate decode.** The first mask applies before prompt
  sync and its counters include untimed resume/prefill work. Progress logging
  captures time before formatting and mask advancement, then charges that
  asymmetric overhead to the following arm.
- **C12 — Metrics remain invalid and always-on.** Every ordinary gate mutates
  counters and every MTP cycle takes clocks even when diagnostics are disabled.
  `commit/cycle` is not acceptance: an accepted draft and an exact-sampling
  rejection plus replacement can both commit two tokens.

The partial MTP-off 200K run completed only control and routed arms before the
subsequent server restart failed with Metal out-of-memory. Those arms use
different state, cannot score S55, and do not authorize kernel work.

Minimum correction is unchanged, with two additions: switch masks only between
complete MTP cycles, and record true accepted/rejected successful cycles. The
production S55 measurement must run separately with diagnostics off, MTP on,
valid output, real 200K context, and the required quality and decay gates.
