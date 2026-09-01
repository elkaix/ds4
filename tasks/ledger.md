# GLM-5.3-Flash / DS4 performance investigation — evidence ledger

Durable record of what was measured, how, and how much to trust it. One row per
finding. Working notes and the live plan stay in `todo.md`; this file is the
part that should survive the session.

Machine: MacBook Pro M5 Max, 128 GiB unified memory, macOS 26.6.2.
Binary: worktree `ds4-glm53`, branch `glm53-pr920`.
Model: `~/models/gguf/GLM-5.3-Flash-UNCEN-Q2.gguf` — **97 GB file, `glm5-next` arch.**
Server: `--metal --mtp --ctx 262144 --power 100`, fans forced to maximum, one slot.

## Evidence hierarchy used throughout

Strongest first. When two rows conflict, the higher tier wins.

1. Direct timer around the exact operation being added or removed
2. Same-process controlled A/B with interleaved arms
3. Two independent instruments agreeing
4. Replicated clean end-to-end throughput
5. Stage-profiler *relative* trends (absolutes are inflated by queue drains)
6. Dashboard rolling metrics

## Architecture facts (source-verified, not measured)

| Fact | Where |
|---|---|
| `glm_graph_dense_compact_attention_limit()` returns `ctx_cap`; DSA dense window = **4096** | `ds4.c:41212`, confirmed by F1 |
| Decode splits at `visible <= limit` → `indexer_fill`, else `indexer_q/weights/scores/topk` → pooled expand → attention | `ds4.c:51565` |
| Layer split `il % 4 != 3` → **34 KDA / 11 DSA** of 45 trunk layers | `ds4_glm53_layer_is_kda` |
| One `glm53_graph_copy_kda_state()` moves **145.6 MiB** (68 tensor copies) + forces `ds4_gpu_end_commands()` | — |
| GGUF: indexer `top_k=2048`, `pool_size=4`; `block_count=46`, `trunk=45`, `leading_dense=3` | GGUF metadata |

## Findings

| # | Finding | Tier | Status |
|---|---|---|---|
| F1 | Dense/sparse boundary is exact: last `verify[rows]` at `pos=4094`, first `verify[batch]` at `pos=4095`; with `n=2` and guard `pos+n > limit`, `dense_limit = 4096` | 3 | **Settled** |
| F2 | The wasted KDA restore is small: ~0.8 ms, priced independently by `setup=` and `rollback=`, which each perform exactly one 145.6 MiB copy (~380 GB/s, bandwidth-bound and near peak) | 3 | Superseded in precision by F11 |
| F3 | Crossing 4096 is a step **down** in cost, not a cliff up: −5.85 ms (fast-verify ON) / −5.50 ms (OFF). The OFF arm is `batch` on both sides, so its −5.50 is purely the dense→sparse attention regime change | 2 | **Settled** |
| F4 | The rows fast path's value at 4K is **unresolved, not zero**. Cross-restart drift is not scalar (setup 1.25×, draft 1.20×, ver 1.14×), so a ratio-of-ratios cannot extract ±2% through a drift model itself off by ±10% | — | Open, low priority |
| F5 | **Sequential A/B across restarts is invalid on this machine.** The second process was 14–25% slower on every stage, including stages the flag cannot touch | 2 | **Settled — methodological** |
| F6 | The 4096 step is sharp and has a **one-time crossing spike**: pos 4095 costs 81.2 ms vs a 50.5–53.3 ms plateau above, i.e. ~+20 ms once per request to build the first sparse selection | 2 | **Settled** |
| F7 | Above 4096 attention is flat, so `indexer_scores` is the only context-growing term | 4 | **FALSIFIED by F8** |
| F8 | `indexer_scores` does not explain the decay. Context 7176→84925 (11.8×): `indexer_scores` 0.201→0.283 ms (1.4×); **all profiled stages 4.52→4.66 ms (+3%)**. Launch-bound, not compute-bound | 5 | **Settled** |
| F9 | The decay is not in per-layer compute at all: 35.25 t/s @7742 → 26.75 t/s @84925 (−24%) while the stage sum moved +3%. Points at the memory system | 4+5 | **Settled that it is not compute** |
| F10 | The 4096 step confirmed independently at stage level: attention 1.259 ms dense → ~0.70 ms sparse, ×11 DSA layers ≈ −6.2 ms/decode, vs −5.5…−6.5 ms end-to-end | 3 | **Settled** |
| F11 | Direct cost of the operation P0 removes: **0.807 ms** (n=256, min 0.732, max 1.042). Agrees with F2's 0.80 ms and with 145.6 MiB ÷ ~380 GB/s | **1** | **Settled** |
| F12 | How much survives end-to-end: in-process alternating A/B, 720 above-boundary cycles, alternation asserted at 50.0%, balanced non-overlapping 4-cycle blocks (n=180): `verify` **+0.568 ms** [+0.303, +0.797]; `total` **+0.486 ms** [+0.189, +0.748] ≈ **+0.93%** | **1+2** | **Settled** |
| F13 | **Superseded by F15** — read the `SM=SHM` mapping as evidence the model was evictable. The model is a **97 GB file mapped `SM=SHM` at 89.9 G virtual with only 8.7 MB resident in the process page table**; `phys_footprint` is just 6.7 GB (6.6 GB IOAccelerator). Machine-wide **File-backed pages = 90.9 GiB** of 128 GiB. So the model's working set is file-backed and evictable, and KV growth competes with it directly | 3 | **New — see below** |

## F13 — why it reframes the residency question

`vmmap` on the live server:

```text
mapped file  7000000000-8678320000  [ 89.9G  8720K  0K  0K ]  r--/r-x SM=SHM
             /Users/panda/models/gguf/GLM-5.3-Flash-UNCEN-Q2.gguf
phys_footprint: 6851 MB      IOAccelerator (graphics): 6775 MB dirty
```

and machine-wide (`vm_stat`, 16 KiB pages):

```text
File-backed pages   5,953,161  =  90.9 GiB
Anonymous pages       633,397  =   9.7 GiB
Free                  685,299  =  10.5 GiB
Compressor            598,648  =   9.1 GiB
```

The model is not "loaded" into the process in the ordinary sense — it is a
file-backed mapping the VM may evict at any time, and it is 76% of physical
memory. Only ~6.6 GB is pinned GPU-side. That makes **hypothesis B** (growing
KV displaces mapped model weights) mechanically plausible in a way it was not
before this was measured, and it sharpens the architectural constraint:

> Any hot-session KV residency budget is spent directly against the model's
> file cache. On this machine there is roughly 10 GiB of headroom before hot KV
> starts evicting weights that the next token will need.

This is the strongest argument yet for *budgeted* residency and against
"keep every session KV in RAM".

## Operating rule: clear the disk KV cache between long runs

**Do this before every long benchmark run, without being asked.**

```bash
bash <scratchpad>/clear_kv.sh          # ~/.ds4/server-kv/glm-5.3-flash-uncen-q2
```

The server is configured with `--kv-disk-space-mb 131072`, so the on-disk KV
store grows to **128 GiB** unchecked. It reached **77 GiB across 113 `.kv`
files** during this investigation. Two reasons this is not merely housekeeping:

- The volume has ~650 GiB free out of 3.6 TiB. A 128 GiB cache plus a 97 GiB
  model is a meaningful share of what is left.
- Stale entries change what a "cold" run means. A run that silently hits a
  90 GiB cache from an earlier session is not the cold run it is reported as,
  and cold-vs-warm is exactly the axis the residency experiments measure.

Clear it with the server **stopped**, so nothing is deleted from under an open
handle. `clear_kv.sh` refuses any path that is not under `~/.ds4/server-kv/`
and only ever removes `*.kv`, never the directory.


## F14 — the decay is BOTH context and memory pressure, in comparable parts (2026-09-01)

F9 attributed ~24% (7.7K -> 85K) entirely to context. That was too generous to
context; a first attempt at this finding then swung too far the other way and
claimed context costs only ~1.5%. Both were confounded. The corrected answer
needed a same-condition bracket.

### The confounded attempts, and why each failed

**Attempt 1 — six identical decodes at fixed 60K, then eight at fixed 2K.**
Byte-identical output verified on every repeat; decode isolated from prefill via
the server's `decoding chunk=... avg=` line.

```text
60K, 6 identical repeats            2K, 8 identical repeats (same process, after)
rep  decode t/s  pageins  free MB   rep  decode t/s  pageins  free MB
 0     30.13     142112       89     0     30.58      21410     2535
 1     29.26     186800      421     1     30.81      12284     2210
 2     29.35     425802     1876     2     29.91      27165     2182
 3     28.11     414624      105     3     28.65      27551     3105
 4     24.18     467416      123     4     29.16      27455     4672
 5     23.63     533718       65     5     31.57      10641     6775
                                      6     32.65      21227     6492
     -21.6% at FIXED context          7     32.94       8667     6459
                                            +7.7% at FIXED context
```

This establishes one thing solidly: **a -21.6% decline occurs at fixed context**,
with GPU temperature falling (74.3 -> 72.2 C), so it is neither context nor
thermal. It tracks free memory and page-ins.

But comparing "first repeat vs first repeat" across the two blocks (30.13 vs
30.58, hence the discarded 1.5%) is invalid: 60K rep0 ran on a fresh
post-restart machine at free=89 MB while 2K rep0 ran afterwards with free
recovering. Those are different points on the pressure trajectory.

### The bracket that resolves it

2K, then 60K, then 2K again — three repeats each, one process, one sitting. The
flanking 2K blocks make the drift *estimable* rather than assumed away.

```text
block      decode t/s              median
2k-pre     33.99  34.24  34.29     34.24
60k        30.00  28.51  27.83     28.51
2k-post    31.88  31.40  32.43     31.88

drift across the 60K block (2k-post vs 2k-pre)     -6.9%
drift-corrected 2K reference  (34.24+31.88)/2   =  33.06
60K against that reference                         -13.8%
```

### Conclusion

**Both effects are real and comparable in size.**

- **Context costs ~13.8%** from 2K to 60K, drift-corrected. Real, and larger
  than the 1.5% the confounded first attempt suggested.
- **Memory-pressure drift costs ~6.9% per 60K block** and reached -21.6% over
  the longer six-repeat series. Independent of context, since it appears at
  fixed 2K and fixed 60K alike.

So F9's -24% over 7.7K -> 85K was a **mixture**, and every earlier sweep walked
context upward over time, making the two inseparable by construction. Of the
four rivals, **C is supported as a co-driver, A and B are refuted (see F15), and
D was right about the global page-in counter** — but none of them replaces
context as an explanation, they add to it.

The residual ~14% context term is still not explained by per-layer compute:
F8's stage profile put the profiled DSA stage sum at +3% across an 11.8x context
range. Whatever the ~14% is, the stage profiler does not see it, which makes the
un-profiled parts of the decode step — sampling, KV compression/decompression
(the server plans `KV 2.92 GiB raw 0.00 + compressed 2.92`), command submission,
the KDA layers, and per-step overhead that scales with live length — the place
to look next.

## F15 — the model is wired, so KV cannot displace it

```text
Pages wired down     102.33 GiB      Free              2.49 GiB
File-backed           5.00 GiB       Compressor        9.96 GiB (26.3 GiB stored)
Anonymous             6.79 GiB       Swap used         4.99 / 6.14 GB
```

matching the server's own plan line:

```text
ds4: memory: KV 2.92 GiB (raw 0.00 + compressed 2.92) + buffers 3.16 GiB
     + resident model 89.87 GiB = 95.96 GiB planned
```

The 97 GB GGUF is **wired**, not merely file-backed. So hypothesis B is
mechanically impossible: growing KV cannot evict model weights, because the
kernel will swap out every other process first. That is exactly what it does —
DS4 wires 102 GiB of 128 GiB and the rest of the workstation lives in the
compressor. The global `Pageins` counter that looked like a residency signal is
therefore dominated by *other* processes faulting back in, which is hypothesis D.

This also corrects F13, which read the 89.9 G `SM=SHM` mapping with 8.7 MB
resident in the process page table as evidence the model was evictable. It is
not: the pages are wired, and simply are not attributed to the mapping in the
process's own page-table accounting.

## F16 — every repeated long-context request pays a full re-prefill

Decode is **8.5 s of a 164 s request at 60K — 95% of the wall clock is
re-prefill**, on a prompt the server reports as a 59,904-token cache hit.

```text
ds4-server: rewound GLM live prefix from 60161 to 59904; final prompt token
            will be reevaluated
ds4-server: chat ctx=59904..59905:1 prefill chunk 0/1 (0.0%) chunk=351.02 t/s
```

`chunk 0/1 (0.0%)` while sustaining 351 t/s for 152 s: the progress denominator
is `prompt_len - cached`, and `cached` was already credited as 59,904, so ~60K
tokens of real prefill are reported as one token.

The cause is a contract mismatch between two functions:

- `live_prefix_rewind_target()` (`ds4_server.c:10725`) returns `prompt_len - 1`
  whenever the prompt is a prefix of the live context, and the caller records
  `cache_source = "memory-rewind"`, `cached = rewind_to`.
- `ds4_session_glm_mtp_rewind()` (`ds4.c`) can only honour a rewind of **one or
  two positions** from the last MTP rollback point:

  ```c
  s->checkpoint.len != (int)s->glm_mtp_rollback_pos + 2 ||
  (pos != (int)s->glm_mtp_rollback_pos &&
   pos != (int)s->glm_mtp_rollback_pos + 1)   ->  return false;
  ```

Here the rewind is 60161 -> 59904, i.e. 257 positions. It fails,
`ds4_session_rewind()` sets `checkpoint_valid = false`, and the whole context is
re-prefilled while still being accounted as cached.

This is architectural, not merely a bug: **34 of 45 trunk layers are KDA and
recurrent**, so their state cannot be truncated to an earlier position. Serving
any prefix of the live context requires either replaying from a snapshot or
replaying from zero, and DS4 keeps exactly one snapshot, at the MTP frontier.

### Scope — tested, not assumed

The scope claim decides whether F16 is a P0 or a harness footnote, so it was
measured rather than read off the source. Three requests, one process, one live
60K context:

```text
turn                            prompt_tok  prefilled   non-decode  rewind line
1  establish 60K context            59905   39425 tok    104.9 s    none
2  APPEND  [P, assistant, new Q]    60173      12 tok      0.3 s    none
3  RE-SEND [P]                      59905       1 tok*   173.4 s    60237 -> 59904
                                            * reported; ~60K actually run
```

**Appending is 578x cheaper than re-sending the same content.** Turn 2 extends
the frontier, matches the live prefix, prefills 12 tokens in 0.195 s and logs no
rewind. Turn 3 truncates, triggers the rewind, and pays 173.4 s while its
progress line reads `chunk 1/1` at 204 t/s.

So ordinary multi-turn chat does **not** hit this. It fires on **regeneration,
editing an earlier message, branching, and any benchmark that repeats or
re-sends a prompt** — which means the sweep harness's premise, that each point
prefills only its delta, was false at every point. Decode numbers read from the
decoding lines are unaffected; wall-clock and TTFT numbers are not.

A fix is periodic KDA checkpoints: one snapshot is 145.6 MiB, so a checkpoint
every 4096 tokens costs ~2.1 GiB at 60K context. That is a memory budget
question, which puts it in the same design as hot-session residency rather than
in a separate workstream.


## F17 — MTP acceptance is near-saturated; width is UNRESOLVED (2026-09-01)

Measured with `--mtp-timing` at 2K and 32K, 297 cycles each. The token stream is
deterministic so the two runs are the same work at different context — treat
them as one measurement at two contexts, not as independent replication.

```text
                        pos <= 4094 (2K)     pos > 30000 (32K)
median cycle total          50.70 ms             57.90 ms
  setup  (KDA save)          1.20   2.4%          0.80   1.4%
  verify                    43.70  86.2%         51.30  88.6%
  rollback                   0.00   0.0%          0.00   0.0%
  draft                      4.60   9.1%          4.50   7.8%
  other                      1.60   3.2%          1.50   2.6%
verify path                 rows (297)           batch (297)
acceptance p                0.721                0.721
tokens/cycle                1.721                1.721
```

**Settled: acceptance is 0.721**, so tokens/cycle is 1.721 against a width-2
maximum of 2.0 and a perfect drafter is worth at most **x1.16**. Drafter quality
is not where a 1.4x lives.

**Settled: verify is 86-89% of the cycle**; draft is 9%. Whatever is slow is the
model forward, not the speculation machinery around it.

**NOT settled: whether width-3 helps.** It depends entirely on `verify(3)`,
which has not been measured. A first attempt here extrapolated
`verify(n) = 43.7 x (1 + 0.23(n-2))` from the code comment "~1.4 ms/layer at
n=2 ... decode does the same math in 0.79" and concluded width-3 buys 1%. That
extrapolation does not hold: the comment compares **two different kernels**
(indexed-batch vs decode-style), not one kernel at two row counts, and a verify
pass is weight-dominated — the weight traffic is fixed and only activations grow
with rows, so the marginal cost of row 3 is plausibly far below that of row 2.

The answer is entirely determined by one unmeasured number:

```text
verify(3)          tok/cycle   cycle ms     t/s   vs now
53.8  (+23%/row)       2.241       65.8   34.08   1.01x
47.0  (+7.5%)          2.241       59.0   37.98   1.13x
44.5  (+2%)            2.241       56.5   39.66   1.18x
```

**Measure `verify` at n = 1, 2, 3, 4 directly before ruling width in or out.**
`glm_graph_verify_rows` already takes `n` and `--mtp-timing` already prints
`verify[...]=` per cycle, so this is a small harness, not a redesign.

## F18 — the forward pass is at 19% of peak, and no current instrument says where

The solid part:

```text
active bytes/token  4.56 GB   (14.6 B params of 309.5 B, 4.7%, at 2.49 bits)
verify              43.70 ms
effective bandwidth  104 GB/s  =  19% of the M5 Max's 546 GB/s
```

**The forward pass runs at under a fifth of memory bandwidth.** That is the
whole 1.4x gap, and it is not a hardware limit. Where the bytes are:

```text
expert FFN (8 + 1 shared of 288, 42 layers)   9.5 B params   65% of active bytes
attention + dense FFN + embeddings            5.1 B params   35%
```

**Retracted:** a first version of this finding claimed the expert path "has
never been profiled". That is false. The profiler emits `routed_moe`,
`routed_moe_folded`, `router`, `shared_gate_up`, `shared_down` among 32 decode
stage labels, and `routed_moe` appears in the F8/F10 capture with 10 samples.

The real problem is that **the profiler cannot resolve cost at all**, because it
drains the GPU queue at every stage boundary:

```text
stage (layer 43, pos 84925)   profiled    sync-corrected
attention                       0.720 ms       ~0.53
routed_moe                      0.365          ~0.17
attn_output                     0.326          ~0.13
router                          0.267          ~0.07
...20 stages...
TOTAL                           4.659          ~0.76
```

A real decode layer is ~0.79 ms (45 layers, ~35.6 ms forward), against 4.659 ms
profiled. So **~84% of every profiled number is the profiler's own drain**,
roughly 0.195 ms per boundary, applied 20 times. That overhead is near-constant
per stage, so it compresses every stage toward a common floor and destroys
exactly the ratios needed here. The sync-corrected column above is a *model*
(subtract a uniform floor), not a measurement, and layer 43 is a DSA layer — 1
of only 11, against 34 KDA layers — so it is not representative either.

**Conclusion: the 19%-of-peak figure is solid; the attribution inside it is not
available from any instrument currently in the tree.** Getting it needs a
profiler that does not serialize — Metal GPU timestamp counters, or bisection
by enabling stage groups rather than every boundary at once. Build that before
committing to expert-gather kernel work, or the work will be aimed by a number
that is 84% artifact.

## Method rules earned the hard way

- Never A/B across restarts here (F5). Interleave arms inside one process.
- For sub-2% effects, prefer a direct timer on the removed operation (F11) to
  an end-to-end delta.
- Assert the treatment took effect from the log (the `verify[batch+rows]` vs
  `verify[batch]` label), never assume the flag was live.
- Use the block, not the individual speculative cycle, as the statistical unit,
  and make the blocks non-overlapping — adjacent-neighbour estimators
  pseudo-replicate and produce CIs that are too narrow.
- The stage profiler drains the GPU queue at every boundary. Its absolutes are
  inflated; only relative growth and the presence/absence of stages are usable.
- Guard every measurement on `clients == 1`. A stray second client already
  destroyed one full sweep pass.
- Subtract `kv cache stored` time before calling anything a decode rate.

## Commits

```text
ac9f6ec  glm53: skip ineligible MTP row verification above dense window   (perf, ~0.9%)
bd2ca17  glm53: measurement-only MTP verifier instrumentation             (all default-off)
```

Split index-only so the running binary stayed the artifact that was measured.
Commit 1 was syntax-checked in isolation (`-Wall -Wextra`, 0 warnings) to
confirm it does not depend on commit 2's A/B lever.

## Known open bug — deliberately unbundled

`ds4_session_glm_spec_cycle_impl` takes neither `ignore_eos` nor `think_mode`;
`n1`/`n2` come from plain `glm_session_logits_argmax`. GLM MTP can therefore
commit a stop token that `ds4_session_argmax_ignoring_eos` would have excluded.
The server catches it and ends the response, so nothing malformed reaches the
client — but `ignore_eos: true` cannot hold a fixed generation length under MTP.
Reproduced deterministically at ~31.5K in all three sweep passes. Correctness
issue, owns its own patch, must not ride in a performance commit.

## Raw data

`tasks/data/` holds what the numbers above were computed from, because the
session scratchpad does not survive:

```text
warm-64k.jsonl   6 identical decodes at 59,905 tokens
warm-2k.jsonl    8 identical decodes at 2,033 tokens
bracket.json     the 2K / 60K / 2K same-condition bracket (F14)
warm_repeat.py   repeats harness      bracket.py    bracket harness
append_test.py   F16 scope test       clear_kv.sh   KV cleaner, path-guarded
```

---

# 2026-09-01 (later) — target raised to S55, P0-A patched

## Target change

The 45-50 sustained target is retired. The committed target is **S55**:

```text
S55 = min(sustained_tps at 2K, 32K, 60K, 85K)  after >= 30 min server runtime
PASS iff S55 >= 55 t/s AND context_decay <= 2% AND session_drift <= 2%
```

Distance from measured reality: **1.61x** the best clean fresh 2K rate (34.24),
**1.93x** the 60K rate (28.51), **2.33x** the worst drifted state (23.63).
Milestones M1 = stable 35, M2 = stable 45, M3 = S55. Full definition, gate
table and protocol in `roadmap.md` -> THE TARGET — S55.

One inconsistency in the instruction is noted for the record: the headline sets
55/S55 while a tail paragraph still says "keep 45-50 as a stretch target". The
headline was taken. The tail's *caution* was kept in a different form — the
roadmap now states plainly that the +61% remainder is not yet technically
established and is the next validation gate.

## F19 — the rewind contract lied because it asked the wrong question

`live_prefix_rewind_target()` (`ds4_server.c:10725`) gated only on
`ds4_engine_is_glm_dsa(engine)` — a *model-family* capability. The backend's
actual guard (`ds4_session_glm_mtp_rewind`, `ds4.c`) is far narrower: GLM-5.3
carries recurrent KDA state across 34 of 45 trunk layers, so the only restorable
positions are the two snapshotted by the **last two-token MTP cycle**:

```text
restorable iff  glm_mtp_rollback_valid
            AND checkpoint.len == rollback_pos + 2
            AND (pos == rollback_pos OR pos == rollback_pos + 1)
```

A 257-position rewind therefore always failed, after the server had already
recorded `cache_source=memory-rewind` and counted the request as a cache hit.

**Patch (P0-A):** new public query `ds4_session_can_rewind(session, pos)` in
`ds4.c` / `ds4.h`, mirroring the backend guard via a shared predicate
`ds4_session_glm_mtp_rewind_available()` so the two cannot drift apart. The
server asks it **inside `inference_mu`, adjacent to the rewind**, so the answer
cannot go stale between check and use. When the answer is no, the request no
longer claims a hit and logs the refusal instead.

### What this patch does and does not do — measured claims only

**Correctness is not at risk from the skipped rewind.** Verified by reading
`ds4_session_sync_internal`: the GLM path reconciles independently of the
server's `cached` value —

```c
if (checkpoint_valid && prompt->len >= checkpoint.len &&
    starts_with(prompt, checkpoint))  -> resume at checkpoint.len
else                                  -> reset KDA state, prefill from 0
```

In the re-send case `prompt->len (59905) < checkpoint.len (60237)`, so sync
resets and full-prefills either way. The old failed rewind reached the same
place by a different route (`checkpoint_valid = false`). **Wall time on that
request is expected to be unchanged; what changes is that it is now accounted
and logged as the prefill it is.**

**The disk-KV path is newly reachable, but that is a hypothesis, not a result.**
Today `cached = 59904 > 0` skips `kv_cache_try_load()` entirely. With the patch
`cached = 0`, so the disk cache gets a chance. Unverified: whether the kvstore
serializes the 34 layers of KDA recurrent state at all. If it does not, the
patch is honesty-only and perf-neutral, plus one wasted `store_current` write.
**Do not claim a prefill win until a populated-cache run shows one.**

**Scope, so the two ledgers do not merge:** this moves *prefill* wall time. S55
is a decode-rate metric. P0-A contributes **~0% toward 55 t/s**. It is P0 for
real-world latency and for benchmark validity, nothing else.

### Test note

`./ds4_test` reports 2 failures, both `short_code_completion step 0 selected
token mismatch` (`logprob-vectors`, `metal-ssd-streaming-cache-pressure`).
Golden-vector mismatches in the decode path; the patch touches only rewind
eligibility and cannot reach them. Confirmed pre-existing by rebuilding and
rerunning at HEAD (see `scratchpad/test-baseline.txt`). `server:` OK, which
covers `test_live_prefix_rewind_target()`.

## P0-A verified against the live server

Rebuilt, restarted, reran `append_test.py`. The fake hit is gone and the new
refusal path fires exactly where predicted:

```text
0901 01:30:20 GLM live prefix rewind from 60237 to 59904 is not restorable;
              falling back to the prefill path
0901 01:30:20 live kv cache miss live=60237 prompt=59905 common=59905
              reason=token-mismatch
```

Turn 3 wall 165.4 s vs 173.4 s before, and the request now takes a **20,480-token
disk-KV hit it previously skipped** (`ctx=20480..59905:39425` instead of
single-token chunks). That is one unpaired comparison across a server restart
and a differently-populated cache — **not a controlled measurement**; treat the
-4.6% as suggestive only. The disk path *is* now reachable, which was the
open question, and the accounting is now truthful, which was the point.
Turn 2 (append) unchanged at 0.3 s non-decode.

## F20 — width-3 MTP is closed, by direct measurement

New measurement-only tool `glm53_verify_scan()` (`DS4_GLM_VERIFY_SCAN=<reps>`,
`DS4_GLM_VERIFY_SCAN_POS=<pos>`): times `glm_graph_verify_rows()` at n = 1..4 in
one process at a real session position, each rep bracketed by the same KDA
snapshot/restore transaction the MTP cycle uses, so model state is left as
found. One unmeasured warm-up rep per n covers workspace sizing. This replaces
the extrapolation from a code comment that compared two different kernels.

**Warm, pos = 2651, 40 reps each** (a cold run at pos=16 was discarded — that is
what `DS4_GLM_VERIFY_SCAN_POS` exists for):

```text
n   mean ms   min      max      marginal
1   44.843    43.442   49.986      --
2   57.456    52.908   59.051   +12.61
3   72.184    69.997   74.726   +14.73
4   85.538    82.283   87.154   +13.35
```

**Marginal row cost = 13.57 ms/row** (least-squares slope over n=1..4). This is
a *difference*, so it cancels any fixed overhead in the scan harness and is the
robust number here.

### The width-3 decision

Width-3 pays iff the extra tokens it buys arrive faster than the current rate:

```text
delta tokens  = p^2                     = 0.520   (p = 0.721, measured, 594 cycles)
                                                  -- see the note below
delta time    = verify row 3  + draft 2  = 13.57 + 4.60 = 18.17 ms
delta rate    = 0.520 / 18.17 ms        = 28.6 t/s
current rate  = 1.721 / 50.70 ms        = 33.9 t/s
```

`p^2` is an **independence assumption and it biases in width-3's favour**: the
measured p = 0.721 is for a draft conditioned on the *true* previous token,
while width-3's second draft is conditioned on the first *draft*, which is
itself only 72% likely to be right. If the MTP head degrades when fed its own
output the real third-token acceptance is below p^2. The closure therefore
holds a fortiori -- the true number can only be worse than the one used here.

**28.6 < 33.9 -> width-3 is a net loss (~0.95x).** Full cycle check: verify(3)
= 43.70 + 13.57 = 57.27 ms, cycle = 1.2 + 57.27 + 9.2 + 1.6 = 69.3 ms,
2.2405 / 0.0693 = **32.3 t/s vs 33.9 today**. The break-even needs the second
draft step under ~1.7 ms against a measured 4.60 ms. **Width-3 is closed. It is
the drafter's per-step cost that kills it, not the verifier.**

### What the scan found instead — and the caveat on it

The verifier is dominated by a **row-count-independent** component: 4 rows cost
1.91x one row. Against the real in-cycle `verify(2) = 43.70 ms` and the measured
slope, the row-independent part is ~16.6 ms, ~38% of the pass.

**Do not yet convert that into a bandwidth claim.** The scan's own n=2 is
57.46 ms against the in-cycle 43.70 ms, and that 13.8 ms gap is confounded three
ways: `GLM_VERIFY_HEAD_LAST` vs the in-cycle `HEAD_FIRST`, position 2651 vs
~2033, and whatever fixed drain the scan harness itself adds. The intercept is
therefore bounded, not measured. **Separating it needs the non-serializing
profiler** — the slope did not, which is why this finding lands and the
attribution one still does not.

Direction this does open (a direction, not a claim): verification amortizes
extra rows very well, so **tree / multi-candidate speculation** — several
candidate continuations verified in one pass at ~13.6 ms each — is worth
costing out. Any such scheme still has to pay a drafter per candidate, which is
exactly what closed width-3.


---

# 2026-09-01 (third pass) — F20 RETRACTED, and the first real forward-pass attribution

## Speed test on the current build — the numbers the user saw

Bracket, one process, server up 3.8 h (idle), fans on **Apple auto, not forced
to max** — the earlier 34.24 baseline was taken with fans forced:

```text
2K before 30.85 | 60K 26.76 | 2K after 29.25
drift across the 60K block  -5.2%
60K vs drift-corrected 2K reference (30.05)  -10.9%
S55 today = 26.76  ->  2.06x short of 55
```

Machine state at the time: fans 3456/5349 and 3733/5777 RPM (auto), **swap
8.57 GiB of 9.66 GiB used**. This run is a *drifted* machine, not a regression
from the two commits — neither touches decode. Re-run with fans forced before
comparing against 34.24. `thermalforge max` needs sudo, so it needs the user.

## F20 is RETRACTED — width-3 is not closed

The first scan measured a marginal row cost of 13.57 ms. Two things were wrong
with it:

1. **It passed the same token in every row.** Identical tokens route to the
   same 8-of-288 experts, so extra rows reuse row 1's expert weights. The scan
   now takes distinct tokens from the tail of the live checkpoint.
2. **It was a single run on a drifted machine.** Its n=1 was 44.84 ms against
   38.7 ms for the same measurement later.

Corrected scan — distinct tokens, 40 reps, **the whole sweep run twice in one
process** so run-to-run spread is visible rather than assumed:

```text
       n=1      n=2      n=3      n=4
pass0  38.844   45.073   54.720   63.487
pass1  38.597   45.022   54.670   62.562     spread between passes < 0.6%
```

Least-squares slope over n=1..4: **8.26 ms per marginal row**, not 13.57.
Redoing the decision with it:

```text
delta tokens = p^2 = 0.520
delta time   = 8.26 verify + 4.60 draft = 12.86 ms
delta rate   = 40.4 t/s   >   current 33.9 t/s
full cycle   = 1.2 + 51.96 + 9.2 + 1.6 = 63.96 ms -> 35.0 t/s  (+3.3%)
```

So width-3 is **roughly break-even to +3%**, not the ~0.95x loss recorded in
F20, and not the +18% the old extrapolation allowed either. It is still not an
S55 lever, and it is still sensitive to the `p^2` independence assumption
(width-4 lands at 34.1 t/s by the same arithmetic). **Verdict: not worth
implementing for S55, but the "closed at 0.95x" claim was wrong and is
withdrawn.** The lesson is the method one: one run of a new instrument is not a
finding, and a synthetic input that collapses the routing collapses the answer.

## F21 — the verify pass is 99% transformer layers, uniformly, at ~23% of peak bandwidth

The scan gained a layer sweep (`DS4_GLM_VERIFY_SCAN_LAYERS=<step>`): cap the
layer loop at k and time the whole pass, so cost(k) - cost(k-step) attributes
time to a layer range with **one host sync per measurement** instead of the
stage profiler's ~20 queue drains per layer. The truncated pass computes
nonsense — the head reads a hidden state that never finished — which is why it
only ever runs inside the scan's snapshot/restore transaction.

```text
n=1                          n=2
layers=1    2.321            layers=1    2.386
layers=5    5.731  0.853/l   layers=5    6.172  0.947/l
layers=10   9.646  0.783/l   layers=10  10.792  0.924/l
layers=15  13.572  0.785/l   layers=15  15.397  0.921/l
layers=20  18.287  0.943/l   layers=20  20.874  1.095/l
layers=25  22.218  0.786/l   layers=25  25.494  0.924/l
layers=30  25.960  0.749/l   layers=30  30.115  0.924/l
layers=35  29.917  0.791/l   layers=35  34.608  0.899/l
layers=40  34.700  0.957/l   layers=40  40.241  1.126/l
layers=45  38.544  0.769/l   layers=45  44.746  0.901/l
```

Three results, all new:

**1. There is essentially no fixed overhead.** Full 45-layer pass 38.544 ms
against a whole-pass n=1 of 38.844 ms: the embed prologue, the output head over
a 154,880-token vocab, and every host-side cost together are **~0.35 ms, under
1%**. The "intercept" that F20 could only bound does not live outside the
layers. The earlier hypothesis that a large row-independent overhead sits
around the pass is **refuted**.

**2. Cost is flat across depth.** 0.75-0.96 ms/layer at n=1, with no hot
region — the mild bumps at the 16-20 and 36-40 buckets are ~20%, not a
bottleneck. **There is no single slow stage to fix.** Reaching S55 needs the
*typical* layer to get cheaper, not an outlier removed.

**3. Extra rows are nearly free per layer: 0.138 ms** ((44.746 - 38.544)/45),
against 0.82 ms for the first row — a **6:1 ratio**. Per-layer time is
dominated by something row-count-independent, i.e. reading weights, not by
per-row arithmetic.

### What that implies, and the fork it leaves

Per layer the model touches ~101 MB of active weights (14.6 B active params x
0.312 bytes / 45 layers). At 0.82 ms that is **~123 GB/s, ~23% of the 546 GB/s
peak.** Two hypotheses fit the 6:1 ratio equally well and they have different
fixes:

```text
H1  memory-bound but inefficient gather
    9 experts x 3 matrices, strided reads, poor coalescing
    fix: gather layout / locality

H2  launch-bound
    9 experts x 3 matrices = 27 skinny dispatches per layer
    x 45 layers = ~1215 dispatches per token; at 20-30 us each
    that is 24-36 ms, which is the entire budget
    fix: grouped/fused expert GEMM, fewer command encodings
```

**H2 is the one that would also explain why extra rows are nearly free** — a
dispatch costs the same for 1 row as for 4 — and it agrees with F8's finding
that the DSA stages are launch-bound at these sizes. The decisive experiment is
a step=1 layer sweep separating the **3 leading dense layers** (one FFN,
few dispatches) from the **42 MoE layers** (27 dispatches). If dense layers are
much cheaper per byte moved, it is H2. That sweep is running.

If the per-layer cost can be brought from 0.82 ms to ~0.30 ms — i.e. the gather
reaching ~60% of peak, which is ordinary for a well-shaped kernel — verify(2)
falls to ~15 ms, the cycle to ~27 ms, and 1.721/0.027 = **64 t/s. S55 is
reachable through this path and, on today's evidence, only through it.**


## F22 — DSA attention layers cost 2.5x a KDA layer and are ~32% of the forward pass

**F21's "cost is flat across depth" was an artifact of 5-layer buckets**, which
averaged the two layer types together. Re-run at step = 1 and the period-4
structure is unmistakable. Delta at k is the cost of layer index k-1; the layer
split is `il % 4 != 3` -> KDA, `il % 4 == 3` -> DSA.

```text
n=1, layers 1-20 (the low-noise region)
  DSA (il%4==3), 5 layers   mean 2.110 ms   [1.82 2.00 2.03 2.06 2.63]
  KDA,          15 layers   mean 0.826 ms   [0.49 ... 0.90] + one 2.34 outlier
  premium 1.284 ms/layer  x 11 DSA layers = 14.1 ms

n=2, same region
  DSA   mean 2.384 ms      KDA mean 0.832 ms
  premium 1.552 ms/layer  x 11 = 17.1 ms
```

The five DSA layers are the five largest values in the window — **perfect rank
separation, no overlap with the 15 KDA layers.**

**~14 ms of a 44.6 ms pass, about 32%, is DSA attention on 24% of the layers.**
That is the single largest identified block of time in decode, and F21 missed it.

### Two corollaries that redirect the MoE hypothesis

**The MoE FFN is not the anomaly.** Layers 0-2 are the leading *dense* blocks
(`leading_dense_block_count=3`, one FFN of 12288 -> ~151 M params) and layer 1
costs 0.804 ms — indistinguishable from the ~0.72-0.83 ms of a *routed* KDA MoE
layer (9 experts x 2048 x 4096 x 3 -> ~226 M params). **A routed MoE layer costs
about the same as, or less than, a dense layer holding fewer parameters.** H1
and H3 predicted the opposite. The 9-of-288 gather is not obviously pathological,
and the missing `group6/group8/group24` kernels for IQ2_XXS/Q2_K may matter much
less than the raw kernel inventory suggested.

**H3's premise is nevertheless confirmed at the file level** (worth keeping, but
demoted): the expert tensors in `GLM-5.3-Flash-UNCEN-Q2.gguf` really are
**86 IQ2_XXS + 43 Q2_K**, and `ds4_metal.m` really does provide
`kernel_mul_mv_group6/group8/group24_q4_K_*` and mxfp4 `fixed_route*` variants
with **no IQ2_XXS or Q2_K equivalent**. So this model does run a less-optimised
expert path than a Q4_K model would. F22 says that gap is not where the 45 ms
goes.

### Caveat that bounds this finding

The scan requires `pos + n` inside the 4096 dense window, so **this was measured
at pos = 2651, where DSA is in its dense mode.** At 60K the DSA layers take the
indexer + top-k 2048 sparse path instead, which is a different kernel mix. The
32% figure is therefore a short-context number, and the residual ~13.8% context
penalty (P1-A) plausibly lives in the same 11 layers. Measuring DSA above 4096
needs the scan's eligibility gate lifted or a different instrument.

### Where this points

```text
pass 44.6 ms  =  ~14.1 DSA attention  +  ~28 KDA+FFN  +  ~2.5 layer 0  +  0.35 head
```

Removing the DSA premium entirely — not achievable, but as an upper bound —
takes the pass to ~30.5 ms, the cycle to ~37 ms, and 1.721/0.037 = **46 t/s**.
That alone does not reach 55, but it is the largest single lever measured so
far and it is 3x anything else on the list.

---

## Contract change — 2026-09-01: S55 becomes **S55-200**

The target is no longer bracketed at 85K. The committed gate is now:

```text
S55-200 = min(tps_2K, tps_32K, tps_64K, tps_128K, tps_200K)
PASS iff S55-200 >= 55 t/s, context_decay <= 2%, session_drift <= 2%,
         runtime >= 60 min, MTP on, quality unchanged, no early-stop.
```

The **minimum** is the score. The metric is sustained autoregressive decode
throughput and nothing else. Explicitly *not* counted as progress: prefill,
TTFT, cache-hit rate, rewind, disk KV, free RAM, wired GB, theoretical
bandwidth, GPU utilisation, a single peak, or a short-context-only result.

Three permanent ledgers, and only the first scores:

| Ledger | Counts toward S55-200 |
|---|---|
| Decode t/s at real context | yes |
| Prefill / rewind / TTFT / KV | no — real latency work, zero decode credit |
| Memory / stability | only when it demonstrably stops decode *decay* |

**This retroactively demotes P0-A.** The rewind-contract fix (commit `4ce350c`)
is correct and stays, but it is a latency-ledger item worth **0 t/s**, exactly
as the previous entry already said.

**It also invalidates every long-context number in this file as a baseline.**
The best measurements here are 2K and 60K. There is no trustworthy 200K result
in this repository, and 60K must not be extrapolated to 200K. Phase 1 of the new
contract is to produce `baseline-s55-200.md` with real measurements at 2K, 32K,
64K, 128K and 200K before any further optimisation is attempted.

---

## F23 — the fast row verifier is switched off above 4096 tokens

**This is the first finding that is specific to long context, and it may be the
whole context penalty.**

`glm_graph_verify_rows_eligible` (ds4.c:47170) gates the MTP row verifier on the
*dense* window, not on the cache capacity:

```c
const uint32_t dense = glm_graph_dense_compact_attention_limit(g);   /* 4096 */
return pos <= g->compact_cache_cap && n <= g->compact_cache_cap - pos &&
       pos <= dense && n <= dense - pos;
```

So for any position past 4096 the row pass is refused and `glm53_spec_verify`
falls through to `glm_graph_forward_indexed_tokens` — the indexed batch path.
The row verifier's own header comment prices that difference:

```text
indexed batch fn  ~1.40 ms/layer at n=2   (gpu-wait 1.22 ms/layer)
decode-style rows  0.79 ms/layer
```

Measured directly: a scan at **pos = 5611** logs `verify_rows refused
(eligible=0)` for every n in 1..4, where the same scan at pos = 2651 ran all
four. Confirmed in `scratchpad/server-pos3700.out`.

**Every decode measurement in this file taken at 2K used the fast path; every
measurement at 32K/60K and above used the slow one.** The two arms of the
"context penalty" were never the same code.

Consequences to test, in order:

1. Price the two verify paths against each other at the same position, using
   `DS4_GLM_MTP_ROWS_EAGER` (the eager-attempt hook already exists at
   ds4.c:64776) versus the natural refusal, same process, ABBA.
2. If the delta is large, the engineering question is whether the row verifier
   can attend through the sparse indexer path instead of the dense compact
   cache. That is the only change identified so far that is *inherently* a
   200K-context change rather than a 2K one.
3. F22's DSA premium was measured at pos = 2651, inside the dense window, so it
   is a short-context number and may itself be a symptom of the same boundary.

Caveat: the arithmetic does not close on its own. At 2K the cycle is
1.72/30.85 = 55.8 ms and at 60K it is 1.72/26.76 = 64.3 ms, a gap of 8.5 ms,
not the ~27 ms that 0.6 ms/layer x 45 would predict. Either the header's
1.40 ms/layer is stale, or the indexed path is cheaper than advertised, or
acceptance differs between the two contexts. Measure before believing either.

---

## H3 is dead — DS4 already ships the fused 2-bit expert path

The earlier note in this file ("this model runs a less-optimised expert path
because `group6/group8/group24` exist only for q4_K") was drawn from an
incomplete grep and is **withdrawn**. The IQ2_XXS and Q2_K expert paths are not
the unoptimised ones; they are separately and more aggressively fused:

```text
ds4_metal.m pipelines that exist for this checkpoint's expert tensors

IQ2_XXS (gate/up, 86 tensors)
  kernel_mul_mv_id_iq2_xxs_f32
  kernel_mul_mv_id_iq2_xxs_pair_f32                     gate+up, one pass over x
  kernel_mul_mv_id_iq2_xxs_pair_swiglu_f32              + SiLU and the product
  kernel_mul_mv_id_iq2_xxs_pair_swiglu_pack2_overlap    + two experts per dispatch
  kernel_mul_mv_id_iq2_xxs_sum6_f32                     6 experts, one dispatch
  kernel_mul_mv_slots6_iq2_xxs_pair_swiglu_f32
  kernel_mul_mv_addr_iq2_xxs_pair_swiglu[_masked]_f32

Q2_K (down, 43 tensors)
  kernel_mul_mv_id_q2_k_f32
  kernel_mul_mv_id_q2_k_sum6_f32                        6 experts, one dispatch
  kernel_mul_mv_slots6_q2_k_sum6_f32
  kernel_mul_mv_addr_q2_k_sum6[_masked]_f32
```

Gate and up are already paired against one read of the activation, SwiGLU is
already fused into that kernel, two experts are already packed per dispatch, and
the down projection already sums six experts in a single dispatch. Every fusion
that a from-scratch design would propose for this FFN is already present.

**This closes the last MoE-side hypothesis and it agrees with F22**, which found
by direct measurement that a dense 151 M-parameter layer and a routed 226 M-
parameter layer both cost ~0.80 ms. The FFN is not where the forward pass goes.

Verdict: `CLOSED`. Do not reopen without a measurement showing expert time above
what the byte count predicts.

Remaining live levers, in measured order of size:

| # | Lever | Evidence | Size |
|---|---|---|---|
| 1 | DSA attention premium (F22) | 2.11 ms/layer vs 0.83 KDA, 11 layers | ~14 ms of a 44.6 ms pass |
| 2 | Row verifier disabled above 4096 (F23) | `verify_rows refused` at pos 5611 | unpriced, long-context only |
| 3 | Width-3 MTP (F20, retracted closure) | 8.26 ms/marginal row | +3.3%, not an S55 lever |

---

## F24 — the GPU is 94% busy during decode; host scheduling is not the lever

Codex's ranked change 1 ("overlap host encoding with GPU execution") is
**rejected on measurement**. Its premise — that the whole pass is encoded into
one command buffer committed only at the end, so encoding runs before execution
instead of overlapping it — is architecturally true but empirically worth almost
nothing.

Method: `DS4_METAL_GPU_BUSY_PROFILE=1` accumulates `cb.GPUEndTime -
cb.GPUStartTime` after a wait that already happens, so it adds no drains. The
timed request is decode-only: turn 1 establishes the context, turn 2 *appends*
to it and extends the live frontier, which prefills only the delta. Re-sending
the same prompt would have re-prefilled the context and put prefill busy-time
into a decode-only denominator — an artifact that would have read as ">100%
busy" and killed the lever for the wrong reason.

```text
ctx    gen   decode      wall    gpu busy        cbs   ms/cb  cbs/tok  busy/decode
2033   512   15.47 s   15.6 s   14,588 ms      1472   9.91     2.88      94.3%
32750  512   15.91 s   16.1 s   14,923 ms      1984   7.52     3.88      93.8%
```

Host plus idle is **5.7–6.2% of decode time**, and that residue includes submit
latency that no amount of chunked committing removes. There is no whole-pass
scheduling win here.

Two corollaries:

1. **~3–4 command buffers per token**, each with its own `waitUntilCompleted`,
   and the GPU is still 94% busy. The sync points are real but cheap.
2. **The work itself is the problem.** At 2K the GPU runs 28.5 ms per generated
   token, i.e. ~49 ms of GPU time per 1.721-token cycle, against a 4.56 GB /
   546 GB/s = 8.35 ms roofline for the bytes a cycle must read. That is
   **~17% of peak bandwidth** — 93 GB/s effective.

So S55-200 has to come from doing less work or doing it at a higher fraction of
the bus, not from scheduling it better. That directs the next step to contract
§8: read the same byte volume sequentially, in the real expert access pattern,
and through the real kernel, and find out which of the three is the wall.

Verdict on the hypothesis ledger:

| ID | Hypothesis | Verdict |
|---|---|---|
| H2b | fixed host dispatch/encoding cost dominates | **REJECTED** (F24) |
| H3 | missing grouped 2-bit expert kernels | **CLOSED** (already shipped) |
| H1 | poor expert-gather coalescing | open — but now the leading candidate, since 17% of peak is exactly what bad coalescing looks like |

---

## F25 — skip-ablation attributes the pass: routed MoE is 33.6%, and 60% is unattributed

`DS4_GLM_DECODE_ABLATE` already exists (ds4.c:44144) and is the non-serializing
profiler the contract asks for in Phase 3: it skips one stage's dispatches,
leaves every other dispatch and gate in place, and costs nothing in
instrumentation because the measurement is a whole-token time delta. Output is
garbage, so it is timing-only.

Nine arms, baseline first and last, one server process per arm, decode-only via
the append path, 512 generated tokens each, at ctx = 2033:

```text
     stage   ms/token    saved  % of pass
  baseline     30.248
    routed     20.178   10.223      33.6%
 attn_core     29.412    0.989       3.3%
    shared     29.719    0.681       2.2%
     qpath     30.293    0.108       0.4%
   indexer     30.328    0.073       0.2%
  attn_out     30.680   -0.279      -0.9%
     qklow     31.482   -1.081      -3.6%
  baseline     30.553

baseline drift across the whole sweep: +1.0%
```

Conditions were unusually clean: the two baselines agree to 1.0%, so every arm
above ~1.1% is resolved.

### Reading

**Routed MoE is 10.22 ms of a 30.40 ms token — a third of the pass.** Its
roofline: 8 experts x 3 matrices x 2048 x 4096 weights x ~0.28 bytes = 56 MB per
layer, 2.37 GB over 42 MoE layers, which at 546 GB/s is **4.3 ms**. So the
routed path runs at **42% of peak bandwidth**, and perfecting it caps out at
10.22 - 4.3 = 5.9 ms, taking the token to 24.5 ms and 2K throughput to ~41 t/s.
Real, the largest single attributed item, and still not S55 on its own.

**This does not contradict F22.** F22 compared per-layer costs and found DSA
layers dearer than KDA layers; the routed FFN is present in all 42 MoE layers,
so 10.22 ms spread over them is ~0.24 ms/layer — small per layer, large in
total. Both are true.

**`qklow` and `attn_out` came out negative**, i.e. skipping them made the token
slower. That is not noise at this drift level; it means those masks keep a gate
or change a branch such that the ablated path is more expensive. They are not
usable cost estimates and are recorded as such.

### The real headline: 60% of the pass is unattributed

```text
routed 33.6 + attn_core 3.3 + shared 2.2 + qpath 0.4 + indexer 0.2  =  39.7%
unattributed                                                        =  60.3%
```

No existing ablation mask covers the KDA recurrence (34 of 45 layers), the mHC
residual mixing, the norms, the 154,880-row output head, or the MTP draft step.
F21 already priced the head plus prologue at under 1% of a pass, so the
remaining ~18 ms per token is dominated by **the KDA layers' own state update**
and the MTP draft.

That is now the single largest unknown in this investigation, and it is bigger
than every attributed item combined. Next instrument: an ablation mask for the
KDA recurrence, same method.

### Consequences for the ranking

| # | Lever | Measured size at 2K | Ceiling if perfected |
|---|---|---|---|
| 1 | unattributed (KDA recurrence + MTP draft) | ~18.3 ms, 60% | unknown — measure first |
| 2 | routed MoE bandwidth (42% of peak) | 10.22 ms, 34% | -5.9 ms -> ~41 t/s |
| 3 | attn_core | 0.99 ms, 3.3% | negligible |
| 4 | shared expert | 0.68 ms, 2.2% | negligible |
| 5 | indexer / qpath | <0.2 ms | dead |

Do not start on the routed MoE kernel until the 60% is named. Spending on a
34% item while a 60% item is unmeasured is exactly the failure the contract's
§6 cost-budget rule exists to prevent.

---

## F26 — the ablation floor is real GPU work, and it is 66% of the token

Added a `kda` mask to `DS4_GLM_DECODE_ABLATE` (ds4.c, commit pending) because no
existing mask covered the KDA recurrence, which is 34 of 45 trunk layers. Same
method as F25: baselines first and last, one process per arm, decode-only append
path, 512 generated tokens, ctx = 2033.

```text
     stage   ms/token    saved  % of pass
  baseline     30.25
       kda     27.69     2.66     8.8%
    routed     20.12    10.23    33.7%
kda,routed     20.30    10.06    33.1%     <-- NOT ADDITIVE
  baseline     30.45              drift +0.7%
```

**Ablating both saves no more than ablating routed alone.** Two readings, and
`DS4_METAL_GPU_BUSY_PROFILE=1` separates them:

```text
        arm     t/s   ms/token   gpu ms/token   busy%   cbs/token
   baseline   32.92     30.379         28.555   94.0%       2.88
 kda,routed   49.24     20.309         20.160   99.3%       2.75
```

The ablated arm is **99.3% GPU busy**. The floor is not exposed host time — it
is 20.16 ms/token of real GPU work in stages that no mask touches. Hypothesis A
(the floor is host/sync cost hiding behind GPU work) is **rejected**, and F24's
rejection of the scheduling lever is strengthened rather than qualified.

### What the floor has to be

Active bytes per token, from the GGUF:

```text
total active                                4.56 GB
  routed experts  8 x 3 x 2048 x 4096 x 42  2.37 GB   52%
  everything else                           2.19 GB   48%
```

"Everything else" is the Q8_0 (450 tensors) and BF16 (151 tensors) non-expert
weights: the KDA and DSA projections, the shared expert, the norms, the mHC
mixing and the 154,880-row output head. At 546 GB/s those 2.19 GB are **4.0 ms**.
The floor measures **34.7 ms per cycle** for them.

```text
routed path      10.22 ms measured vs  4.3 ms roofline   ->  42% of peak
non-expert path  34.7  ms measured vs  4.0 ms roofline   ->  12% of peak
```

**The non-expert path is three and a half times further from the bus than the
routed path is, and it is the larger of the two in wall time.** Every hypothesis
in this investigation so far — H1, H2, H3, width-3, the DSA premium, the fused
FFN — has been aimed at the 42%-of-peak half. The 12%-of-peak half was never
examined because no ablation mask exposed it.

That reverses the ranking:

| # | Target | Wall time | Fraction of peak | Ceiling if brought to 42% of peak |
|---|---|---|---|---|
| 1 | non-expert Q8_0/BF16 path | 34.7 ms/cycle | 12% | -24.8 ms/cycle |
| 2 | routed MoE | 17.6 ms/cycle | 42% | -6.9 ms/cycle |

Item 1 alone is 24.8 ms against the 19.41 ms that S55 needs at short context.
It is the first lever measured in this investigation that is large enough.

### Caveats that must be closed before acting

1. The masks are not proven to be clean skips. `qklow` and `attn_out` both came
   out *negative* in F25 — skipping them cost time — so at least two masks
   change a branch rather than removing work. The floor's composition is
   inferred from what the masks do not cover, not from a direct measurement.
2. Whether `DS4_GLM_DECODE_ABLATE` reaches the MTP draft step is unverified. If
   the draft runs an unablated stack, part of the floor is the draft and the
   "routed = 33.7%" figure is really "33.7% of verify". Delegated to review.
3. All of this is at ctx = 2033. Only long-context numbers score under S55-200.

---

## F27 — the active-bytes figure was wrong by 2.1x, and it settles the target

Every roofline in this file, including mine in F24 and F26, used **4.56 GB of
active weights per token**. That number comes from the model card's 14.6 B
active parameters times the whole-file average of 0.312 bytes/parameter. It is
wrong, because the average is dominated by the 87 GB of 2-bit expert tensors
while the tensors that are actually read *every* token are not 2-bit at all.

Measured from the GGUF header, per token, 45 trunk layers, counting 8 of 288
experts as active:

```text
  KDA projections      3.845 GB    39.9%     Q8_0 (8.5 bpw) and Q4_K (4.5 bpw)
  routed experts       2.378 GB    24.7%     IQ2_XXS / Q2_K (2.06 / 2.63 bpw)
  DSA attention        1.373 GB    14.2%
  shared expert        1.123 GB    11.7%     Q8_0
  dense FFN            0.482 GB     5.0%
  norms / mHC          0.236 GB     2.4%
  router               0.198 GB     2.1%
  ------------------------------------
  trunk total          9.635 GB
  + output head etc    1.348 GB
```

Per MoE layer the split is **146 MB non-expert against 56 MB of experts**. A
single `kda_v` or `kda_output` tensor is 35.65 MB at Q8_0 — each one on its own
is *two thirds* of that layer's entire routed-expert traffic.

### What this does to every conclusion so far

```text
                        old (4.56 GB)      corrected (9.635 GB)
roofline per token          8.35 ms             17.65 ms
bus ceiling                  120 t/s              56.7 t/s   (53.0 with the head)
measured 28.555 ms GPU     17% of peak          61.8% of peak
```

**The engine is not running at 17% of the bus. It is running at 62%.** DS4 is
far better than this investigation has been assuming, and the "3.5x headroom in
the kernels" premise behind F24, F25 and F26's rankings is gone. The routed-MoE
"42% of peak" figure in F25/F26 is likewise wrong in the same direction.

### The decisive consequence

```text
S55 cycle budget      31.29 ms for 1.721 tokens  =  18.18 ms/token
bytes that must move                                 9.635 GB
required bandwidth                                     530 GB/s
                                                   = 97% of the 546 GB/s peak
```

**S55-200 is not reachable with this checkpoint on this hardware.** A perfect
engine at 100% of theoretical bus bandwidth tops out at 56.7 t/s at short
context, 53.0 with the output head, and long context adds KV and indexer traffic
on top of that. 55 t/s sustained at 200K sits above the ceiling, not below it.

This is not a statement that the work is impossible. It is a statement that **no
scheduling change and no kernel rewrite can get there, because the target is
above the roofline.** The remaining 38% of peak that DS4 leaves on the table is
worth at most 28.555 -> 17.65 ms/token, i.e. 2K throughput from 33 t/s to ~53,
and that is the absolute ceiling of all kernel work combined.

### THE PICK — reduce bytes per token, starting with the KDA projections

The only lever that moves the ceiling itself:

| Change | Bytes saved | New total | New ceiling | Risk |
|---|---|---|---|---|
| `kda_v` + `kda_output` Q8_0 -> Q4_K | 1.51 GB | 8.13 GB | 67.2 t/s | quality |
| + `*_shexp` Q8_0 -> Q4_K | 0.59 GB | 7.54 GB | 72.4 t/s | quality |
| + `kda_q`/`kda_k` Q4_K -> Q2_K | 0.55 GB | 6.99 GB | 78.1 t/s | high |

The first row alone takes the ceiling from 56.7 to 67.2 t/s, which is the first
time in this investigation that 55 has been on the reachable side of a bound.
At the *current* 61.8% bus efficiency it gives 8.13 GB / (546 x 0.618) = 24.1 ms
per token = **41.4 t/s at 2K**, and it composes with every kernel improvement
rather than competing with them.

Per contract §13 this is only admissible if size falls, decode rises, **and** a
predeclared quality gate passes: these are attention projections, not experts,
and 8.5 -> 4.5 bits on the value and output projections of a linear-attention
layer is exactly where recurrent state error would compound. The quality suite
must run before the throughput claim, not after.

### Retractions forced by this entry

- F24's "17% of peak" -> 62% of peak. The conclusion that the GPU is busy and
  the schedule is not the lever **stands** (94% busy is measured, not derived).
- F25/F26's "routed MoE at 42% of peak" and "non-expert path at 12% of peak"
  are both recomputed against the wrong denominator and are withdrawn. The
  non-expert path is larger than the expert path in bytes as well as in time,
  which was the correct half of that finding.
- The F26 ranking table is void. Bytes, not kernels, is the top item.
