# ivanfioravanti/ds4 — is it worth pulling? (2026-08-09)

Fetched as remote `ivan`. Three branches: `main` (== upstream `b030961`, nothing new),
`mxfp4-m3`, `batch-inference`.

## Verdict

- **`ivan/mxfp4-m3` — YES, high value.** 24 commits, strictly ahead of `origin/main`
  (0 behind). Merges into `prod` with **zero textual conflicts** (`git merge-tree` clean)
  — which is *not* the same as semantically safe. Overlap set is `ds4_metal.m`,
  `ds4_server.c`, `ds4.c`, `ds4.h`, and `prod` rewrote 3627 lines of `ds4_server.c`.
  That is the exact shape that produced two silent defects on 2026-07-27 (the vanished
  `pthread_mutex_init` block, the duplicated `enqueue` that dropped upstream dispatch) —
  both compiled, both survived shallow tests. Treat a clean merge as the start of review,
  not the end.
- **`ivan/batch-inference` — NO.** 13 ahead / **137 behind** upstream. Stale fork of the
  GLM 5.2 + batch-inference work that upstream already landed independently
  (`a185c36`, `36cd0ca` native Metal session batching, `005afed` GLM 5.2). Nothing for a
  DeepSeek-only Metal setup.

Upstream PR: **antirez/ds4#755**, OPEN, updated 2026-08-09 13:31 — same author whose
previous Metal PR #555 landed as upstream `427e281`.

## Why mxfp4-m3 matters here

Ivan benchmarked on **the same hardware and quant class we run**: M5 Max 128 GiB,
ds4f-q2 (IQ2XXS experts). His baseline decode **39.39 t/s** matches our measured
**39.32 t/s** (2026-08-05 llama-benchy), so the deltas should transfer.

| stage | decode | note |
|---|---:|---|
| baseline (== our prod) | 39.39 t/s | |
| round 1: 7 exact fusions admitted on M5 | 40.54 | +2.9% |
| round 4 | ~42.8 | +8.6% |
| round 5: concurrent FFN + HC continuation | **45.25 t/s** (steady 45.47) | **+14.9%** |

Claimed bit-exact throughout: every replica reproduced the canonical frontier SHA-256
`eb794f49…f21d`; rejected variants include ones that were faster but changed logits.
Prefill unchanged (~791 t/s on his 2048-ctx contract). Every fusion is rollback-gated by
a `DS4_METAL_DISABLE_M5_*` env var.

## Correctness finding — affects our running binary today

`023614e "metal: remove broken stream512 top-k path"` removes the streaming top-512
indexer selector, calling it **non-reproducible for wide resumed-prefill rows**.

That path is **upstream code, not his**: added by `222b2cb "metal: accelerate DeepSeek
indexed prefill"` (`origin/main:ds4_metal.m:16653`), which we merged into `prod` on
2026-08-05. So we are shipping it today.

**TESTED 2026-08-09 — the max_abs=2.90 link is REFUTED.** `./ds4_test
--metal-tensor-equivalence` run 3× on `prod` (ffc1cc3) produced bit-identical results
every time:

```
run 1/2/3: worst_max_abs=2.90212  worst_rms=0.593814  min_overlap=18/20  top1_mismatch=0
           long_memory_archive max_abs=2.90212 | long_code_audit max_abs=2.71609
           all three short cases max_abs=0
```

The 2.90 is **deterministic** fast-vs-quality route drift introduced by
`532ec8b`/`222b2cb`, not run-to-run nondeterminism. My earlier correlation between the
0.106 → 2.90 jump and the stream512 selector was wrong.

Caveat on scope: this test compares the fast and quality routes inside one process. Ivan's
claim is specifically about **wide resumed-prefill rows** (continuing from a KV
checkpoint), which this case does not exercise. So `023614e` is neither confirmed nor
disproven here — it is taken on his authority as the author of that kernel work, not on
local evidence. There is no longer a reason to cherry-pick it urgently on its own.

**But note it ships anyway:** the whole branch was merged, so the stream512 selector is
**removed** on `ivan-metal-20260809` and indexed prefill always takes the deterministic
argsort/merge path. That is a real behavior change to the prefill path riding along with
this adoption. It is not a regression by any local measure —
`metal-tensor-equivalence`, `local-golden-vectors`, `metal-short-prefill`, and
`streaming-decode-prefill-correctness` all pass without it, and prefill throughput showed
no drift-fair change.

`023614e` is a self-contained 3-file revert (ds4.c, ds4_metal.m, metal/argsort.metal) and
is cherry-pickable on its own if the perf work is not wanted.

The other two fixes in `0658dac` (stale `attn_inv_rope` flag; `mem_device_and_threadgroup`
barrier in `kernel_dsv4_kv_rope_fp8_store_f32`) are internal to his own new code —
that kernel and that fuse gate are absent from `origin/main`.

## Server-side commits in the branch (review before taking)

- `0ead8a8 server: recover truncated DSML tool calls` — overlaps the area where we
  deliberately took upstream's `51a1c14`/`9735ad8` over our forced-`</think>` restart
  during the 2026-08-05 merge. Textually clean, semantically needs a read.
- `3196149 server: canonicalize OpenAI tool schema JSON spelling` — +274 lines, standalone.

## MEASURED 2026-08-09 — branch `ivan-metal-20260809`, worktree `~/Projects/open-source/ds4-ivan`

Merged `ivan/mxfp4-m3` into `prod`. Git merged clean, **but the build failed**: Ivan's
`0ead8a8` calls upstream's `trace_event(s, trace_id, …)`, which our transaction rewrite
replaced with `production_observe_trace_event(p, …)` — `trace_id` does not exist in our
version of that function. Exactly the hazard flagged above, except it failed loudly
instead of silently. One-line adaptation, committed as `0db5826`. Rebuild: **zero
warnings**. Symbol-count check on `ds4_server.c` (`ctx_span`, `req_flags`,
`production_observe_trace_event`, `production_txn`, `server_txn_run`, `trace_cache`)
showed prod == merged on every one, so nothing else was dropped.

### Full `make test` on the merged branch — exit 0, all green

```
long-context OK | tool-call-quality OK | think-tool-recovery OK | logprob-vectors OK
metal-ssd-streaming-cache-pressure OK | local-golden-vectors OK | metal-short-prefill OK
metal-kernels OK | metal-tensor-equivalence OK | streaming-decode-prefill-correctness OK
mtp-verify-depth OK | dspark-verify-depth OK | server OK | ds4_agent_test ok
test_layer_pack 97/97 | test_engine_mgpu_placement 101/101 | test_gpu_args_cli 44/44
```

Three of these answer open questions directly:

- **`server` + `think-tool-recovery` + `tool-call-quality` all OK** — the two `ds4_server.c`
  commits and my `0db5826` rethread are sound. No need to drop them.
- **`dspark-verify-depth` and `mtp-verify-depth` OK** — speculative *correctness* under
  MTP/DSpark holds on the merged binary (the verifier commits autoregressive-identical
  tokens at draft depth > 2). It does **not** establish that Ivan's M5 concurrent-FFN gate
  engages under DSpark: that gate is documented as a one-token shape, and both "engaged"
  and "silently declined" pass this suite. Unresolved, and it matters — see the DSpark
  caveat under the benchmark below.
- **`metal-tensor-equivalence` worst_max_abs = 2.90212, worst_rms = 0.593814** — *identical
  to prod, to six digits*. His fusions did not perturb the numerics on our abliterated
  0731 Q2 quant either, which is the local confirmation of his exactness claim (his
  canonical hash is for a different weights file and does not transfer).

### llama-benchy, interleaved ABAB (drift control)

Same flags both binaries (2026-08-05 protocol: `--ctx 65536 --tokens 32768 --mtp …
--dspark --dspark-confidence 0.9`; benchy `--pp 2048 --tg 128 --depth 0 8192 32768
--runs 2 --extra-body '{"temperature":0}'`). Only the binary differs.

**The machine drifts hard** — prod's own pp2048 fell 572 → 452 between its two windows,
and tg128 38.68 → 35.23. Single-window A/B would have been worthless here; this is the
~10% drift Ivan warns about. Chronological order was prod → ivan → prod2 → ivan2.

| test | prod (A1) | ivan (B1) | prod2 (A2) | ivan2 (B2) | mean A → mean B |
|---|---:|---:|---:|---:|---:|
| pp2048 | 572.35 | 513.74 | 451.69 | 451.02 | drift-dominated |
| **tg128** | 38.68 | **43.51** | 35.23 | **40.15** | 36.96 → 41.83 = **+13.2%** |
| pp2048 @ d8192 | 582.72 | 531.60 | 506.99 | 519.89 | drift-dominated |
| **tg128 @ d8192** | 34.92 | **36.87** | 33.06 | **35.50** | 33.99 → 36.19 = **+6.5%** |
| pp2048 @ d32768 | 452.52 | 443.36 | 448.37 | 405.28 | drift-dominated |
| tg128 @ d32768 | 28.47 | 29.69 | 29.93 | 26.68 | inconclusive |

Read:

- **Decode near-empty: +13.2%, solid.** Both ivan windows beat both prod windows
  (43.51 and 40.15 vs 38.68 and 35.23), so the result survives the drift. Directionally
  consistent with Ivan's claimed +14.9%.
- **Decode @ 8192: +6.5%, solid.** Same both-beat-both property.
- **Decode @ 32768: inconclusive.** B1 > A1 but B2 < A2, error bars ±1.1, and by then the
  machine was deep in thermal decay.
- **Prefill: no regression detected.** The raw means look ~5% down, but pp2048 decays
  monotonically 572 → 514 → 452 → 451 regardless of binary. The only drift-fair
  comparison — the adjacent pair at the plateau, A2 vs B2 — is 451.69 vs 451.02, a dead
  heat. Deep-context prefill is too contaminated to call either way.

**DSpark caveat — the +13.2% is a non-DSpark number.** Both servers logged every benched
request as `THINKING` (`chat ctx=… gen=128 THINKING finish=length`), and thinking mode
discards the request's sampling params, so the `temperature: 0` DSpark requires never took
effect and no DSpark decode line appears in either log. Same trap as 2026-08-05. The
comparison stays apples-to-apples — both binaries ran the identical greedy path — but our
**daily config runs `--mtp --dspark`, and whether the M5 gate engages there is not
established**. The daily-use gain could be smaller than +13.2%. Measuring it needs a
request that reaches the sampler at temperature 0, i.e. not in thinking mode.

Thermal caveat: the machine was saturated by the end. Ivan's numbers come from *cooled*
replicas via `ds4-bench`; a cooled `ds4-bench` run would tighten prefill and the
deep-context decode figures. The decode headline does not need it.

## Decision A (independent, do first): is stream512 actually nondeterministic?

The claim in `023614e` is *non-reproducible*, not *wrong* — and that is testable on the
binary already built, before merging anything. Upstream's condition at
`ds4_metal.m:16653` has no env guard (the guard only appears in `0658dac`), so
repeat-run variance is the available discriminator:

Run `make test` on current `prod` **twice**, compare `max_abs` on the two long
metal-tensor-equivalence cases.

- Varies between runs → nondeterminism confirmed; cherry-pick `023614e` today, on its
  own, regardless of what happens with the perf work.
- Stably 2.90 → diagnosis wrong; 2.90 is deterministic numeric drift from
  `532ec8b`/`222b2cb`, and the correctness section above needs rewriting.

## ADOPTED 2026-08-09 — `prod` = `0db5826`

Done: merge, fixup, build, full `make test`, ABAB benchmark, adoption.

`prod` fast-forwarded from `ffc1cc3` to `0db5826` (a genuine fast-forward this time, unlike
the 07-27 / 08-05 adoptions — the old tip is still an ancestor). Backup tag
**`pre-ivan-metal-20260809`**; revert with `git reset --hard pre-ivan-metal-20260809`.
Clean rebuild in the main checkout: exit 0, zero warnings, all five binaries. CLI smoke:
coherent output at **44.14 t/s** generation near-empty. Worktree removed; single checkout.

Remaining, both optional:

1. **Measure the daily `--mtp --dspark` config outside thinking mode** — the one real open
   question. +13.2% is the non-DSpark number and may not survive DSpark's multi-token
   verify shape.
2. A cooled `ds4-bench` run on Ivan's own contract, to firm up the prefill and
   deep-context decode rows that thermal drift left inconclusive.

## Decision B (perf): merge path — SUPERSEDED by the measured section above, kept for the rationale

1. `git checkout -b ivan-metal-20260809 prod`, then merge **only the Metal commits** —
   drop `0ead8a8` and `3196149`. They carry all the `ds4_server.c` merge risk and none
   of the +14.9%.
2. If they are taken anyway: after merging, `git diff prod HEAD -- ds4_server.c` and
   confirm every `p->ctx_span`, `p->req_flags`, and `production_observe_trace_event`
   reference in prod's tool-recovery path survived. Ivan's `0ead8a8` is based on plain
   upstream and has none of that transaction plumbing; git will merge cleanly while
   dropping it.
3. `make clean && make -j10` — expect zero warnings.
4. Full `make test` (GPU is free — no server running as of 2026-08-09).
5. **A/B with `ds4-bench`, not llama-benchy.** The 2026-08-05 llama-benchy runs went
   through THINKING mode, which discards sampling params. Ivan publishes his contract,
   so use it — same tool, same shape, directly comparable to his table:
   ```sh
   ./ds4-bench -m MODEL --prompt-file speed-bench/promessi_sposi.txt \
     --ctx-start 2048 --ctx-max 2048 --ctx-alloc 2081 --step-incr 2048 \
     --gen-tokens 32 --prefill-chunk 4096 --power 100 --warm-weights
   ```
   Keep llama-benchy as a secondary for continuity with the 2026-08-05 numbers.
6. **Verify the DSpark/MTP path** — it is the production config and his validation list
   does not cover it. Round 5's concurrent-FFN gate is documented as "the proven
   single-device IQ2XXS/Q2_K, top-6, **one-token shape**", while `--mtp --dspark`
   verifies multiple tokens per decode step. Likely the fast path just declines and
   DSpark sees no speedup — confirm rather than assume: run with `--mtp` + `--dspark` +
   `temperature: 0` and check the log emits a DSpark decode line with correct output.
7. Adopt onto `prod` only if the A/B holds; `pre-upstream-merge-20260805` and the current
   `prod` tip stay as rollback.

Rollback of individual fusions without rebuilding: `DS4_METAL_DISABLE_M5_*=1`.

**Exactness caveat:** his canonical frontier SHA `eb794f49…` is for *his* ds4f-q2 file.
We run the drowzeys abliterated 0731 Q2 — different weights, different hash. Exactness
transfers as a property of the shape-gated kernels; verify it by comparing our own
before/after logits, never against his number.
