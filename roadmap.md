# DS4 GLM-5.3 Performance Investigation Guide

> **STATUS 2026-09-01 — this banner and the TARGET/RESULTS sections are
> canonical. Everything below them is the original plan, kept for the record.**

**Opening statement of this guide, as of 2026-09-01:**

> Stop treating long-context decode as the only problem. The largest real-world
> loss found so far is a **failed recurrent-state rewind that recomputes an
> entire 60K prompt**, while the sustained decode problem itself separates into
> **~14% genuine context cost plus significant system-memory-pressure drift**.

The narrow question this document originally opens with ("why does decode lose
~24% between 7.7K and 85K") is answered and was wrong as posed: **~13.8% is a
genuine context effect and a comparable amount is memory-pressure drift that
ascending test order made inseparable from it.** Hypothesis B is refuted
outright (the model is *wired*, not evictable), C is supported as a co-driver,
D was right about the page-in counter. Sections 9 and 11 rest on the refuted
premise. The 32K knee, the indexer-score bottleneck and the 4K verifier problem
are **no longer main targets**; the verifier cleanup is committed and passed its
gates — stop spending benchmark time there.

## The four problems — keep them isolated

| Priority | Problem | Observed impact | Status |
| --- | --- | ---: | --- |
| **P0-A** | Failed GLM rewind contract | ~95% of a repeated 60K request spent re-prefilling | Largest item |
| **P0-B** | Excessive wired-memory pressure | ~7% per 60K block, up to ~21% over a long series | Confirmed co-driver |
| **P1-A** | Genuine context decode penalty | ~13.8% from 2K to 60K | Cause unresolved |
| **P1-B** | Multi-client TTFT | previously 17-20 s | Remeasure after P0-A |
| **P1-C** | GLM MTP `ignore_eos` bug | correctness | Open |
| **P2** | Raw kernel / MoE optimization | unknown remaining opportunity | Needs a real profiler |

## Execution order

1. **Fix/define the rewind contract.** (P0-A)
2. **Design bounded KDA rewind checkpoints.**
3. **Determine a sustainable wired-memory envelope on the 128 GB machine.**
4. **Establish PEAK + SUSTAINED benchmark gates.**
5. **Build the non-serializing Metal profiler.**
6. **Attribute the remaining ~13.8% context penalty.**
7. ~~Measure width-3 verification directly.~~ **Done — closed at ~0.95x.**
8. **Optimize MoE / launch behavior only if the profiler supports it.**
9. **Re-measure multi-client TTFT after rewind repair.**
10. **Design one budgeted state manager for hot sessions + rewind checkpoints.**
11. **Fix `ignore_eos` separately.**
12. **Only then chase additional kernel micro-optimizations.**

---

## THE TARGET — S55 (set 2026-09-01, supersedes the 45-50 target)

> **55 tok/s sustained generation, with no meaningful decay.**
>
> This is not a fresh-context peak target. DS4 must hold **>=55 tok/s at 2K,
> 32K, 60K and ~85K context after prolonged server operation**, with no more
> than **2% context-dependent or session-time decay**. 55 is not allowed to be
> a short benchmark burst.

Acceptance is the *lowest* sustained rate across those conditions, never the
highest observed one — otherwise the easiest context gets optimized:

```text
S55 = min(
  sustained_tps_2K,
  sustained_tps_32K,
  sustained_tps_60K,
  sustained_tps_85K
)

PASS iff:
  S55 >= 55 tok/s
  context_decay <= 2%
  session_drift <= 2%
```

### Hard acceptance gates

| Gate | Requirement |
| --- | ---: |
| Fresh peak (fresh process, 2K) | >= **55 tok/s** |
| 60K sustained | >= **55 tok/s** median |
| 85K sustained | >= **55 tok/s** median |
| 30-minute sustained | >= **55 tok/s** |
| 60-minute sustained | >= **55 tok/s** |
| Context decay | <= **2%** |
| Session-time drift | <= **2%** |
| MTP | enabled |
| Output correctness | unchanged |
| Full-context rewind | no accidental re-prefill |
| Thermal throttling | none |
| Memory-pressure regression | none |
| Multi-client mode | must not destroy single-stream 55 t/s |

### How far away we actually are

```text
                        t/s     eff. bandwidth   % of 546 GB/s peak   x to 55
best,  2K, fresh       34.24        156 GB/s          28.6%            1.61x
       60K             28.51        130 GB/s          23.8%            1.93x
worst, after drift     23.63        108 GB/s          19.7%            2.33x
TARGET S55             55.00        251 GB/s          46.0%              --
```

The bandwidth column is **accounting, not proof of a bandwidth limit** — the
evidence below says the engine is not bandwidth-bound today.

This splits into two independent problems:

```text
A. Recover stability      23.6-28.5  ->  ~34 sustained
B. Raise engine throughput      ~34  ->  55         = +61% beyond today's
                                                      best clean 2K rate
```

Part B is substantially more aggressive than the retired 45-50 goal, and it is
**not yet technically established**: see "The gap arithmetic" below — roughly
half of it is located and half is not. Treat 55 as the committed target and the
unidentified remainder as the next validation gate, not as a promise.

### Milestones — do not wait until the end to know if the direction is right

```text
M1  Stable 35   >=35 t/s, 2K-85K, 30+ min, <=3% decay
                needs: rewind fix, memory-pressure control, no hidden
                re-prefill, stable benchmark environment
M2  Stable 45   >=45 t/s, 2K-85K, 30+ min, <=2.5% decay
                needs: real forward-pass optimization, possibly wider MTP
M3  S55         >=55 t/s, 2K-85K, 30-60 min, <=2% decay      <- finish line
```

### The benchmark protocol every candidate optimization gets

Do not walk upward once — ascending order made context and elapsed-session
pressure rise together, which is exactly how the original "24% context decay"
came out wrong. Use balanced blocks, repeated:

```text
2K  -> 60K -> 2K
32K -> 85K -> 32K
```

and for A/B, `A B B A` or an equivalent balanced block. Keep: same process,
fixed generated-token count, byte-identical output assertion, server decode-line
timing, KV-store accounting, thermal and swap/compressor telemetry, raw data
saved. The dashboard is operational telemetry, not a sub-percent instrument.

### Why some headroom exists in principle

The model is a 288-expert MoE using 8 + 1 shared per token
(`expert_feed_forward_length=2048`, 42 MoE layers of 45), so only **4.7% of
309.5 B parameters are active per token**:

```text
active params/token   14.6 B
bytes/param (Q2)      0.312  (2.49 bits)
active bytes/token    4.56 GB
```

At 34 t/s that is ~156 GB/s, about **a quarter of the machine's bandwidth** —
and less than that in reality, because MTP width-2 reads the weights once to
verify two rows. **Decode is not bandwidth-bound**, which is consistent with F8
finding the profiled stages launch-bound rather than compute-bound. 55 t/s
implies ~251 GB/s of active-byte traffic, 46% of peak — demanding, not absurd.

### What is settled about MTP — measured 2026-09-01

**Acceptance is 0.721** (297 cycles at each of 2K and 32K), so tokens/cycle is
1.721 against a width-2 maximum of 2.0 and a perfect drafter is worth at most
x1.16. **Drafter quality cannot supply a 1.61x — closed.**

**Verify is 86-89% of every MTP cycle** (43.7 ms of 50.7 at 2K; draft is 9%,
setup 2%, rollback 0%). Whatever is slow is the model forward, not the
speculation machinery around it.

**Width-3 is CLOSED — measured 2026-09-01, not extrapolated.** New tool
`glm53_verify_scan()` (`DS4_GLM_VERIFY_SCAN`) times `glm_graph_verify_rows()`
at n = 1..4 in one process at a real position, each rep bracketed by the MTP
cycle's own KDA snapshot/restore. Warm, pos = 2651, 40 reps each:

```text
n   mean ms   marginal
1   44.843       --
2   57.456    +12.61
3   72.184    +14.73
4   85.538    +13.35        marginal row cost = 13.57 ms/row
```

The slope is a difference, so it cancels any fixed overhead in the harness.
Width-3 pays iff the extra tokens arrive faster than the current rate:

```text
delta tokens = p^2 = 0.520          (p = 0.721 measured; see note)
delta time   = 13.57 verify + 4.60 draft = 18.17 ms
delta rate   = 28.6 t/s   <   current 33.9 t/s
```

Full cycle: 69.3 ms for 2.2405 tokens = **32.3 t/s vs 33.9 today, ~0.95x.**
Break-even needs the second draft step under ~1.7 ms against a measured 4.60.
**It is the drafter's per-step cost that closes width-3, not the verifier.**
`p^2` assumes independence and **biases in width-3's favour** (the measured p is
for a draft conditioned on the *true* previous token, not on another draft), so
the closure holds a fortiori.
The earlier +1%-to-+18% range is superseded; delete it.

What the scan found instead: the verify pass is dominated by a
**row-count-independent** component — 4 rows cost 1.91x one row, and against
the real in-cycle `verify(2) = 43.70 ms` the fixed part is ~16.6 ms (~38%).
That intercept is **bounded, not measured** (the scan's own n=2 is 57.46 vs the
in-cycle 43.70, confounded by HEAD_LAST vs HEAD_FIRST, position, and the
harness's own drain), so do not convert it into a bandwidth claim until the
non-serializing profiler exists. It does open one direction worth costing out:
verification amortizes extra rows very well, so **tree / multi-candidate
speculation** (several candidates verified in one pass at ~13.6 ms each) is the
shape that fits this cost curve — though it still pays a drafter per candidate,
which is exactly what closed width-3.

### Where the gap IS — and the instrument problem blocking it

```text
active bytes/token 4.56 GB / verify 43.70 ms = 104 GB/s = 19% of peak
```

**The forward pass runs at under a fifth of memory bandwidth.** That is the
whole gap and it is not a hardware limit. 65% of those bytes are the MoE expert
gather (9.5 B of 14.6 B active params, 9 of 288 experts across 42 layers).

But **the attribution inside the forward pass is not currently measurable.** The
stage profiler drains the GPU queue at every boundary: a real decode layer is
~0.79 ms against 4.659 ms profiled, so **~84% of every profiled number is the
profiler's own drain** (~0.195 ms x 20 boundaries), applied near-uniformly and
therefore flattening exactly the ratios needed. `routed_moe` *is* emitted and
*was* captured — the numbers simply cannot be trusted at this resolution.

**Build a non-serializing profiler first** — Metal GPU timestamp counters
(one synchronization at normal completion, no host wait per stage), or
bisection by stage groups (0-10, 11-20, 21-30, 31-44, then subdivide). The
acceptance requirement is that instrumented decode lands within a few percent
of normal decode, not 5-6x slower. Until then, expert-gather kernel work would
be aimed by a number that is 84% artifact.

Once it works, the MoE targets to investigate are specific, not "optimize
Metal": expert gather locality, router->gather fusion, MTP row weight reuse,
dispatch-count reduction, expert scheduling/order, shared+routed expert fusion,
Metal command-buffer batching. The metric is `useful tokens / expert-stage
time` under two-row MTP verify — not theoretical GB/s.

### The gap arithmetic

```text
23.63  worst sustained today
  x    remove pressure drift        +7% per 60K block, up to +21% over a series
  x    remove the context term      +13.8% from 2K to 60K
  ~34  sustained at today's peak
  x    ??? +61% more                <- this part is not yet identified
55.00  S55
```

So roughly **a third of the gap to 55 is already located** (F14: drift and
context) and the rest is not. Candidate sources for the remainder, cheapest
first:

1. ~~MTP acceptance rate~~ — **measured at 0.721, near-saturated. Closed.**
2. **A non-serializing profiler.** Prerequisite for everything below it: the
   current one is 84% its own drain overhead.
3. ~~MTP width-3~~ — **measured 2026-09-01 at ~0.95x. Closed.** The marginal
   verify row is cheap (13.57 ms); the marginal *draft step* (4.60 ms) is what
   makes it a loss.
4. **The MoE expert gather.** 65% of active bytes; forward pass at 19% of peak.
   The likeliest home for the remaining gap, but do not start until (2).
5. **Launch/overhead reduction.** F8 showed the DSA stages are launch-bound at
   these sizes, and F14's residual ~14% context term is invisible to the stage
   profiler — both point at per-step overhead rather than math.
6. **Wired-memory reduction.** 96 GiB planned on a 128 GiB box is what causes
   the drift. Anything that shrinks the resident footprint buys back the
   sustained half of the target directly.

### Reporting rule

Report PEAK and SUSTAINED separately; one never substitutes for the other.

```text
PEAK       t/s in the first minute of a fresh process at 2K
SUSTAINED  t/s in the last block of a >=30 min session at >=32K,
           measured with the 2K/60K/2K bracket so drift is visible
```

A change that raises PEAK while leaving SUSTAINED behind has not met this goal.
`34 -> 39 peak` with `29 -> 24 sustained` is a failed server optimization.

**Note on scope:** the rewind fix (P0-A) moves *prefill* wall time. S55 is a
decode-rate metric, so P0-A contributes ~0% toward 55 t/s — it is P0 for
real-world latency and for benchmark validity, and those are separate ledgers.


---

# RESULTS — 2026-09-01

Written after executing steps 1-4 of the roadmap below. **Two of this document's
central premises did not survive the experiments.** Full evidence and method are
in `tasks/ledger.md` (F14, F15, F16); this section records what changed and what
the plan should be now.

## What was completed

**Step 2 — the verifier cleanup is committed, separately, as prescribed.**

```text
ac9f6ec  glm53: skip ineligible MTP row verification above dense window
bd2ca17  glm53: measurement-only MTP verifier instrumentation
```

Split index-only so the running binary stayed the artifact that was measured.
Commit 1 was syntax-checked in isolation (`-Wall -Wextra`, zero warnings) to
prove it does not depend on commit 2's A/B lever. Gates from section 2 all met:
default-off levers, clean build, token identity and accept/reject schedule
preserved, no change below 4K, direct-timer elimination confirmed above it.
The EOS bug and the memory work were not bundled. **Stop benchmarking this.**

**Steps 3-4 — the warm-repeat control was run, redesigned.**

Section 5's "run the exact same decode twice" is n=1 per arm, which against
14-25% cross-run drift can only detect a Case-1 collapse. Replaced with: one
process, one fixed prompt, N identical back-to-back decodes, byte-identical
output asserted on every repeat, decode isolated from prefill by parsing the
server's `decoding chunk=... avg=` line rather than the dashboard. Six repeats
at 59,905 tokens, then eight at 2,033 tokens in the same process.

## Premise 1, revised: the ~24% is context AND drift, in comparable parts

Section 4 and the "one narrow question" framing rest on decode losing ~24%
between 7.7K and 85K, attributed to context. That attribution was too generous,
but only by about half.

Six identical decodes of one fixed 59,905-token prompt (byte-identical output
asserted each time, decode isolated from prefill via the server's own
`decoding chunk=... avg=` line) fell **30.13 -> 23.63 t/s, -21.6%, at fixed
context**, GPU temperature falling 74.3 -> 72.2 C. Eight identical decodes at
2,033 tokens minutes later, same process, *rose* 30.58 -> 32.94 t/s as free
memory recovered 2.5 -> 6.5 GiB. So a large context-independent drift is real.

But comparing first-repeat to first-repeat across those two blocks is invalid —
they sit at opposite ends of that same pressure trajectory. A same-condition
bracket settles it: 2K, then 60K, then 2K again, three repeats each, one sitting.

```text
block      decode t/s              median
2k-pre     33.99  34.24  34.29     34.24
60k        30.00  28.51  27.83     28.51
2k-post    31.88  31.40  32.43     31.88

drift across the 60K block                      -6.9%
drift-corrected 2K reference (34.24+31.88)/2 =  33.06
60K against that reference                     -13.8%
```

**Context genuinely costs ~13.8% from 2K to 60K. Pressure drift costs a further
~6.9% per 60K block, and -21.6% over a longer series.** Every earlier sweep
walked context upward over time, so the two rose together by construction and
F9 charged all of it to context.

The residual context term is still unexplained by per-layer compute: F8 put the
profiled DSA stage sum at +3% across an 11.8x context range. Whatever the ~14%
is, the stage profiler does not see it — so look at the parts of the decode step
the profiler cannot resolve — and note it resolves almost nothing, since ~84% of
every profiled number is its own per-boundary queue drain (see "the instrument
problem" below). Candidates once a usable profiler exists: sampling, KV
compression/decompression (the server plans `KV 2.92 GiB raw 0.00 + compressed
2.92`), command submission, the 34 KDA layers, and per-step overhead scaling
with live length.

## Premise 2 that failed: KV cannot displace the model, because the model is wired

```text
Pages wired down   102.33 GiB      Free        2.49 GiB
File-backed          5.00 GiB      Compressor  9.96 GiB (26.3 GiB stored)
Anonymous            6.79 GiB      Swap used   4.99 / 6.14 GB
```

matching the server's own plan line: `KV 2.92 GiB + buffers 3.16 GiB +
resident model 89.87 GiB = 95.96 GiB planned`.

The 97 GB GGUF is **wired**, not evictable file cache. Hypothesis B — "growing
KV displaces mapped model weights" — is mechanically impossible, and sections 9
and 11 are built on it. DS4 wires 102 GiB of 128 GiB and the kernel swaps out
the rest of the workstation instead; the global `Pageins` counter that looked
like a residency signal is mostly *other processes* faulting back in.

The section 11 conclusion still holds, but for a different reason. The budget
that matters is not "KV vs model working set" — it is **DS4's wired total vs the
rest of the machine**. At 96 GiB planned on a 128 GiB box there is no headroom,
and hot-session KV is spent directly against system stability.

## The real finding: 95% of a repeated long-context request is re-prefill

Not on this roadmap at all, and larger than everything that was.

```text
60K request:  decode 8.5 s      total 164 s      95% re-prefill
```

on a prompt the server reports as a 59,904-token cache hit.
`live_prefix_rewind_target()` (`ds4_server.c:10725`) returns `prompt_len - 1`
whenever the prompt is a prefix of the live context and credits it as
`cache_source = "memory-rewind"`. But `ds4_session_glm_mtp_rewind()` can only
honour a rewind of **one or two positions** from the last MTP rollback point.
The actual rewind here is 257 positions; it fails, `checkpoint_valid` is
cleared, and the whole context is re-prefilled while still being accounted as
cached — which is why the progress line reads `chunk 0/1 (0.0%)` while
sustaining 351 t/s for 152 s.

This is architectural: **34 of 45 trunk layers are KDA and recurrent**, so their
state cannot be truncated. DS4 keeps exactly one snapshot, at the MTP frontier.

Scope was measured, not assumed. Three requests against one live 60K context:

```text
turn                            prompt_tok  prefilled   non-decode  rewind
2  APPEND  [P, assistant, new Q]    60173      12 tok      0.3 s    none
3  RE-SEND [P]                      59905       1 tok*   173.4 s    60237 -> 59904
                                            * reported; ~60K actually run
```

**Appending is 578x cheaper than re-sending the same content.** So ordinary
multi-turn chat does not hit this; it fires on **regeneration, message editing,
branching, and any benchmark that re-sends a prompt** — so the sweep harness's
premise that each point prefills only its delta was false at every point. Decode
figures read from the decoding lines are unaffected; wall-clock and TTFT are not.

Fix: periodic KDA checkpoints. One snapshot is 145.6 MiB, so every 4096 tokens
costs ~2.1 GiB at 60K. That is a memory-budget decision, so it belongs in the
same design as hot-session residency, not in a separate workstream.

## Operating rule added

**Clear the disk KV cache between long runs, with the server stopped.** It is
configured at `--kv-disk-space-mb 131072` and reached **77 GiB across 113 `.kv`
files** during this work. Beyond disk usage, a stale entry makes a run reported
as cold actually warm, and cold-vs-warm is the axis these experiments measure.
Use `clear_kv.sh`, which refuses any path outside `~/.ds4/server-kv/` and
deletes only `*.kv`.

## Revised priorities

| | Work | Why |
|---|---|---|
| **P0** | Honour the rewind contract: make `live_prefix_rewind_target` return only what the GLM backend can serve, or add periodic KDA checkpoints so it can serve more | 95% of a repeated 60K request; also stops the server reporting a cache hit it did not deliver |
| **P0** | Decide DS4's wired-memory budget. 96 GiB planned on a 128 GiB machine leaves no headroom and drives the drift half of the decay | Worth ~7-21% over a session; gates every residency design |
| **P1** | Find the residual ~14% context term. The profiled DSA stages grow only +3%, but that profile is ~84% its own drain overhead, so it has not actually been excluded — needs the non-serializing profiler first | The genuine context cost, now separated from drift |
| **P1** | Multi-client TTFT (old P0-A), re-measured now that re-prefill is understood — some of the 17-20 s may be this same bug, not session switching | Cannot be scoped until P0 lands |
| **P1** | GLM MTP `ignore_eos` / `think_mode` correctness bug, own patch | Unchanged, still separate |
| **P2** | Kernel work | Unchanged. F8/F14 both say compute is not the problem |
| **Closed** | 4K boundary, indexer kernels, 32K knee, verifier cleanup | Settled or shipped; stop measuring |

## What to carry forward about method

The confound that produced the 24% figure was ordering, not instrumentation:
every sweep walked context upward, so anything that drifts over a session was
indistinguishable from a context effect. Section 17's rules were right and this
adds one — **bracket the independent variable, do not merely
repeat it.** Running 2K/60K/2K in one sitting made the drift estimable (-6.9%)
and separable from the context effect (-13.8%). Repeats alone produced two
confounded answers in a row — first 24% charged to context, then 1.5% — because
each compared blocks sitting at different points on the drift trajectory.

Also: prefer the server's own `decoding chunk=... avg=` line to wall-clock or
the dashboard. Wall clock at 60K is 95% prefill and would have hidden all of
this.

---

# 1. Freeze what is already settled

Do not keep reopening these questions unless new evidence contradicts them.

### 4K boundary

Settled:

```text
pos <= 4094
    width-2 pair fits
    verify[rows]

pos >= 4095
    pair crosses dense_limit=4096
    verify[batch]
```

The transition is real and exact.

But importantly:

```text
dense DSA <= 4096
        ↓
cross 4096
        ↓
sparse DSA > 4096
```

makes steady-state computation **faster**, not slower.

Independent measurements agree:

```text
DSA attention stage
1.259 ms dense
~0.70 ms sparse

delta ≈ -0.56 ms/layer

× 11 DSA layers
≈ -6.2 ms/decode
```

versus the independent MTP-cycle observation:

```text
~ -5.5 to -6.5 ms
```

That is excellent agreement.

There is a separate one-time cost of about:

```text
+20 ms
```

when the first sparse selection is constructed.

Do not confuse that one-time crossing event with steady-state performance.

---

# 2. Close the verifier cleanup

Your P0 verifier patch is now understood.

It removes the guaranteed failed fast-row attempt above 4094 and therefore removes one unnecessary KDA restore.

Measured eliminated operation:

```text
~0.807 ms
```

Direct timer:

```text
n = 256
~0.807 ms
```

Independent estimates agree with:

```text
145.6 MiB KDA state
÷ effective ~380 GB/s
≈ same order
```

But only about:

```text
~0.49 ms
```

survives at whole speculative-cycle level.

So this is approximately:

```text
~0.9% improvement above 4094
```

Treat that result correctly:

> **Real optimization, correct cleanup, worth keeping, but not the explanation for long-context decay.**

Do not spend another day trying to turn this into a 10% optimization.

Before committing it, require:

```text
default-off diagnostic levers
build clean
zero warnings
token identity preserved
accept/reject schedule preserved
no performance regression below 4K
expected direct-timer elimination above 4K
```

Then commit it independently from every other change.

Suggested commit scope:

```text
glm53: skip ineligible MTP row verification above dense window
```

Do not bundle the EOS bug or memory experiments with it.

---

# 3. Retire the wrong performance targets

These are currently off the main path.

### 32K knee

There is no demonstrated 32K execution transition.

The original apparent slope was heavily influenced by single-pass noise.

Replicated slope:

```text
~0.16–0.19 ms/token per 1K
R² ~0.52–0.67
```

instead of the original:

```text
0.317 ms/token per 1K
R² ~0.84
```

So stop densely sampling around 32768.

### Indexer kernels

The original hypothesis was:

```text
context grows
→ indexer_scores rows grow
→ indexer_scores dominates decay
```

The profiler falsified it.

Measured:

```text
context
7,176 → 84,925
= 11.8×
```

while:

```text
indexer_scores
0.201 → 0.283 ms
= 1.4×
```

and the entire profiled DSA stage set:

```text
4.52 → 4.66 ms
≈ +3%
```

Yet end-to-end decode changed approximately:

```text
35.25 → 26.75 tok/s
≈ -24%
```

Do not optimize `indexer_scores` or `indexer_topk` now.

They may eventually be worth micro-optimizing, but they cannot plausibly recover the observed loss according to the current measurements.

---

# 4. State the live hypothesis correctly

The remaining problem is not:

> “long context makes the DSA math expensive.”

The evidence currently supports:

> **Longer context changes memory-system behavior while the actual model compute remains nearly constant.**

Four explanations remain.

| Hypothesis | Meaning                                            | If confirmed                                 |
| ---------- | -------------------------------------------------- | -------------------------------------------- |
| **A**      | Live KV pages themselves are being paged/faulted   | Bound hot KV residency                       |
| **B**      | Growing KV evicts mapped model-weight pages        | Extra resident KV can make performance worse |
| **C**      | Compressor/swap pressure stalls execution globally | Fix memory pressure/system policy            |
| **D**      | Page-ins correlate but do not cause slowdown       | Look outside VM residency                    |

Do not choose among them yet.

The next experiment exists specifically to separate them.

---

# 5. First next experiment: warm-repeat high-context control

This should be the highest priority experiment.

Use a **quiet machine**.

No Pi agent.

No competing DS4 clients.

No compiler.

No browser or process intentionally exercising the model.

No benchmark sweep running in parallel.

Use one fixed high-context prompt, ideally around:

```text
64K–85K
```

because that is where the slowdown is already obvious.

Run the exact same decode twice.

Conceptually:

```text
High-context state
      │
      ├── Run 1: cold-ish
      │
      └── Run 2: immediately repeated/warm
```

Keep all generation properties identical:

```text
same binary
same GGUF
same context
same prompt tokens
same generation length
same temperature
same MTP state
same power/fans
same server process if possible
```

Collect for each run:

```text
context tokens
generated tokens
decode elapsed
corrected ms/token
tok/s
pageins delta
swapins delta
swapouts delta
compressor delta
RSS delta
GPU power
GPU temperature
KV-store count/time
prefill time
```

Do not use dashboard `last_decode_tps` as the authoritative measurement.

Use your harness timing and exclude or explicitly subtract KV-store time.

---

# 6. Interpret the warm-repeat control

This experiment has a very useful decision tree.

### Case 1: second run is much faster and page-ins collapse

Example:

```text
Run 1
27.0 tok/s
300K pageins

Run 2
33.5 tok/s
40K pageins
```

This is strong evidence that memory residency is causal.

Then proceed immediately to **memory attribution**.

### Case 2: page-ins collapse but throughput barely changes

Example:

```text
Run 1
27.0 tok/s
300K pageins

Run 2
27.4 tok/s
40K pageins
```

Then ordinary page-in count is mostly an accompanying signal.

Do not design a residency architecture around it.

Investigate another system-level cost.

### Case 3: page-ins remain large on both runs and both remain slow

This suggests the live working set exceeds comfortable residency even after warming.

Then A/B/C remain plausible.

You need to determine **what pages are faulting**.

### Case 4: second run becomes faster despite similar page-in counts

Then gross `pageins` is too coarse.

The identity of the pages matters more than the count.

Move directly to page/residency attribution.

---

# 7. If memory causality survives: determine what is being displaced

This is the decisive follow-up.

You want to distinguish:

```text
KV pages faulting
```

from:

```text
model/mmap pages faulting
```

from:

```text
compressor/swap pressure
```

because the resulting DS4 architecture is different.

Track three conceptual memory classes:

```text
MODEL
GGUF / mmap / Metal-resident hot weights

SESSION
KV
KDA recurrent state
MTP buffers
other context-dependent state

SYSTEM
compressed memory
swap
other applications
filesystem cache
```

At approximately:

```text
8K
32K
64K
85K
```

measure the same fixed decode.

Do not densely sweep.

Four good memory states are more valuable than fifty noisy contexts.

---

# 8. Perform a memory-relief control

Once you have one reproducible slow high-context state, test whether reducing unrelated memory pressure restores decode performance.

Important: do this as a **separate experiment**, not halfway through your existing sweep.

For example:

```text
control
normal workstation state

experimental
close memory-heavy unrelated processes
reduce compressor/swap pressure
leave DS4/model/config identical
```

Then repeat the high-context decode.

If:

```text
same context
same DS4
same model

27 → 33 tok/s
```

just from making system memory less pressured, hypothesis C becomes much stronger.

If nothing changes, C weakens.

---

# 9. Test whether KV growth displaces model working-set pages

This is the most important distinction before implementing hot multi-session residency.

The dangerous case is:

```text
more resident KV
      ↓
less room for model working set
      ↓
model pages fault
      ↓
decode slows
```

If that is what is happening, blindly keeping multiple sessions hot will improve client switching but damage steady decode.

So before implementing:

```text
resident_sessions = many
```

test:

```text
one session
small context

one session
large context
```

under otherwise identical conditions.

Then compare memory behavior.

If larger single-session KV alone causes model residency to deteriorate, multi-session hot KV must be aggressively budgeted.

---

# 10. Design P0-A separately: multi-client TTFT

Do not mix this with P0-B.

P0-A is already well established:

```text
client A active
       ↓
client B arrives
       ↓
live KV A serialized
       ↓
KV B loaded
       ↓
partial prefix mismatch possible
       ↓
re-prefill
       ↓
17–20 s TTFT
```

That is a **session-switching problem**.

The eventual architecture should still look like:

```text
                 MEMORY BUDGET
                      │
        ┌─────────────┴─────────────┐
        │                           │
   protected model             session pool
   working set                     │
                           ┌───────┼───────┐
                           A       B       C
                         hot KV  hot KV  hot KV

                                   │
                                   ▼
                          ONE execution lane
                                   │
                          MTP width-2 enabled
```

The key word is:

> **budgeted**

not:

> “keep every session resident.”

---

# 11. Define the residency manager around bytes, not session count

Avoid designing:

```text
--resident-sessions 4
```

as the primary resource control.

A 4K session and an 80K session are radically different.

Prefer something conceptually like:

```text
--hot-session-budget-mb N
```

or automatic budgeting.

Then residency is:

```text
model safety reserve
      +
execution/MTP reserve
      +
system headroom
      +
hot-session budget
      <= safe unified-memory envelope
```

Sessions should be evicted according to actual byte cost.

A practical policy might ultimately be:

```text
protect model working set
protect current session
retain recently used sessions while budget permits
evict coldest/largest sessions when budget would be exceeded
disk KV remains cold tier
```

Do not implement this until P0-B tells you how much memory the model itself needs protected.

---

# 12. Separate residency from compute batching

This architectural distinction remains critical.

Do not equate:

```text
multiple resident sessions
```

with:

```text
batched inference
```

You want:

```text
N resident states
1 active compute lane
MTP ON
```

not:

```text
N batched decode lanes
MTP OFF
```

So internally DS4 should eventually separate:

```text
residency_mode
```

from:

```text
execution_mode
```

Something like:

```text
execution:
  serial
  batched

residency:
  single
  pooled
```

MTP compatibility should depend on serial execution, not on there being only one resident session.

---

# 13. Add metrics before implementing the residency manager

The dashboard currently hides the difference between a good cache and a good working set.

You previously had:

```text
95–99% token-cache efficiency
```

while still suffering large client-switch pauses.

So expose separate telemetry.

I would eventually want the dashboard to show:

```text
Prefix reuse
99.0%

Live sessions
3

Hot resident
2

Cold/disk
1

KV resident
1.3 GiB

KV evictions
4

KV restores
3

Recovery prefill
6.2 s

Session switch overhead
8.1 s

Memory pressure
...

Swap delta
...

Pageins
...
```

This turns “cache works” and “working set fits” into separate concepts.

---

# 14. Treat P0-B and P0-A as independent acceptance tests

For **P0-B**, the success metric is:

```text
high-context steady decode t/s
```

For **P0-A**, it is:

```text
session-switch TTFT
```

Do not report one combined “server got faster” number.

A residency change could theoretically produce:

```text
TTFT
18 s → 1 s       excellent

decode
30 → 26 tok/s    bad
```

That is not automatically a win.

The feature should have explicit gates for both.

---

# 15. Keep the EOS bug completely separate

This is now a correctness issue, not a benchmark annoyance.

Current behavior:

```text
ignore_eos=true
        │
        ▼
ordinary path filters stop token

GLM MTP
        │
        ├ drops ignore_eos / think_mode
        │
        ▼
plain argmax
        │
        ▼
can commit stop token
```

The server later catches the stop condition, so it does not leak malformed output, but fixed-generation-length semantics are violated.

You already reproduced it deterministically around the same ~31.5K point across all three passes.

That deserves its own patch.

Do not merely suppress the emitted token afterward.

The stop-token filtering semantics must be applied wherever MTP establishes the target truth:

```text
row-0 argmax
row-1 argmax
draft comparison
accepted token
next draft parent
```

Correctness test:

```text
MTP OFF + ignore_eos
vs
MTP ON + ignore_eos
```

Require:

```text
same deterministic continuation
same requested fixed length
no premature stop
```

Keep this out of performance commits.

---

# 16. Clean up the current uncommitted work

You currently have approximately:

```text
ds4.c
+84 / -8
```

with three measurement-only levers default-off.

Before moving into the memory architecture, split that work.

I would aim for three commits:

```text
1. perf:
   skip impossible GLM MTP row verify above dense window

2. instrumentation:
   add measurement-only GLM MTP timing / A-B controls
   if you actually want those upstream

3. correctness:
   GLM MTP ignore_eos / think stop handling
```

Do not make the memory-residency implementation the fourth section of the same giant diff.

Start it from a clean baseline.

---

# 17. Preserve measurement discipline

This investigation already demonstrated that benchmark methodology can create false conclusions.

Cross-restart variation:

```text
14–25%
```

is vastly larger than many of the optimizations you're measuring.

So from now on use these rules:

```text
same-process comparisons whenever possible
interleaved treatments
ABBA rather than one-direction ABAB when drift is possible
block/run as statistical unit
not individual speculative cycles
fail closed if treatment flag didn't activate
exclude known one-time initialization separately
record KV stores
record early stops
record page/swap behavior
record thermal state
preserve raw data
```

For effects below about 2%, prefer direct instrumentation of the operation being removed over server restarts.

Your 0.807 ms restore measurement is a perfect example.

---

# 18. Do not overfit profiler absolute times

Your stage profiler deliberately drains the GPU queue at every stage boundary.

Therefore:

```text
absolute stage ms
```

is inflated and should not be compared directly against ordinary decode timings.

Use it for:

```text
relative growth
presence/absence of stages
regime changes
```

The strongest profiler result is therefore not:

> “DSA costs exactly 4.66 ms.”

It is:

> “Context increased 11.8× while profiled DSA stage cost increased only ~3%.”

That's the result to carry forward.

---

# 19. Evidence hierarchy from now on

When evidence conflicts, use roughly this ordering:

**Strongest**

```text
direct timer around exact operation
```

then:

```text
same-process controlled A/B
```

then:

```text
independent instruments agreeing
```

then:

```text
replicated clean throughput measurements
```

then:

```text
stage profiler relative trends
```

then:

```text
dashboard rolling metrics
```

The dashboard remains useful operationally, but it should not decide sub-percent optimization claims.

---

# 20. Final experimental roadmap

This is the order I would follow now:

1. **Let all current measurements finish untouched.**
2. **Commit/finalize the ~0.9% verifier cleanup separately.**
3. **Run the same high-context decode twice on a completely quiet machine.**
4. Compare corrected t/s against page-ins, swap and compressor behavior.
5. If warming changes throughput, attribute which pages/working sets are involved.
6. Run a separate memory-relief control.
7. Determine whether large KV is displacing model pages.
8. Establish a safe memory budget for hot session residency.
9. Design multiple resident sessions with **one serial MTP lane**.
10. Benchmark P0-A using session-switch TTFT and P0-B using high-context decode separately.
11. Fix the GLM MTP `ignore_eos` correctness issue in its own patch.
12. Only return to kernels if repeated profiling produces evidence that compute is again material.

---

# The target architecture

If the memory experiments support it, the end state should look approximately like:

```text
                     DS4 SERVER
                         │
                Request scheduler
                         │
          ┌──────────────┴───────────────┐
          │                              │
   Session residency               Compute policy
          │                              │
   memory-budgeted pool            serial lane = 1
          │                        GLM MTP width-2
     ┌────┼────┐                        │
     A    B    C                        ▼
    hot  hot  cold                  PR #920 path
     │    │    │
     │    │    └──────────── disk KV
     │    │
     └────┴──── protected in memory
          │
          ▼
  model-working-set guard
          │
          ▼
   unified-memory governor
```

The governor is the important new concept.

It should decide:

> **How much memory may session KV consume without evicting the model's performance-critical working set?**

That is the question your next experiments need to answer before you implement multi-hot-session residency.

## Current status in one sentence

You started with a suspected GLM kernel/context-scaling problem and have narrowed it to something substantially more interesting:

> **The model's actual per-layer work is almost flat at long context, while end-to-end decode loses ~24%, so the next investigation belongs at the unified-memory/residency layer, with multi-client session residency treated as a separate latency problem rather than the explanation for long-context decode decay.**

That is now a much cleaner engineering target than the one you started with.
