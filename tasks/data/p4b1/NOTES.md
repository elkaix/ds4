# P4b1 — isolated selected #964 Metal/runtime treatment on historical PR base

Date: 2026-09-14. Status: **HOLD**. 2K Protocol B is a solid inconclusive
result, not a closed or correctness-qualified adoption. Canonical record.

## Pins (reproducible after another #964 force-push)

```text
source PR snapshot/head:  0a908192450a9d0ec67d79e864c2af7e40505f1d
A base:                   bd66c402070042bf0a79ad6ece8242de4c93680c
A tree:                   d1572d260ebaabd2c2e40f97b3f734c02e30e610
constructed B commit:     b3f48f9eabf30940bacaa5723869498e8d830b56
constructed B tree:       65304d349d5b570cce8cdc4753f59f7b4b0d9c17
binary sha (arms.txt):    A 57a1591add42283b / B 5b97c53694132782
```

Saved with the artifacts: `bench/pins.txt`, `bench/commits.txt`,
`bench/diff-name-status.txt`, `bench/diff-stat.txt`.

## Arms

B is a **constructed treatment**, not current PR #964 and not “the 9 engine
commits” as a complete tree:

```text
B = bd66c40
  + selected #964 Metal/runtime changes
  - server/checkpoint integration
  - rax integration
  - requantizer
  - benchmark/docs-only changes
```

| arm | worktree | branch | head | content |
|---|---|---|---|---|
| A | `../ds4-glm53-964base` | `pr964-base` | `bd66c40` | historical PR merge-base on upstream main |
| B | `../ds4-glm53-964eng` | `pr964-engine` | `b3f48f9` | constructed Metal/runtime subset |

Cherry-picks (engine files only where noted): `5723502` profiling controls,
`b80bb52` fused BF16 HC + KDA decode, `bd72d25` sparse decode + top-8 routing,
`643593c` tiled prefill, `1d7be03` M3 Ultra router/DSA/KDA experiments,
`ad87423` prefill safety/cap sizing, `bdb743a` (Metal/test files; `ds4_server.c`
dropped), `09b6aaf` (Metal/argsort/test; server/`rax.*`/bench dropped),
`1f0c54d` HC producer repeat.

Excluded: `655ed96`, `f92ec68` (server checkpoint maps), `566def1` (requant
tool), docs/bench commits. PR head has since been force-pushed to `0a90819`
(2026-09-13) with extra integration/recovery/docs commits.

## Why the arms are on the PR's base, not on `p4a-pass`

The plan said "engine-only commits onto P4a". Tried: `b80bb52` alone conflicts in
8 hunks across `ds4.c`, `ds4_metal.m`, `metal/dsv4_hc.metal`, `metal/glm53_kda.metal`
because this branch (PR #920 lineage) restructured the same HC/KDA kernels that
#964 templatizes, and lacks main's `b6af0ad..bd66c40`. Hand-merging ~6.6K lines of
kernel code before measuring would put merge noise inside the A/B. Measuring the
constructed treatment against its own historical base isolates the engine delta
at zero merge cost. The live `glm53-p4a` merge is **not** a performance
adoption (HOLD). Absolute A/B rates are not comparable to the P4a series
(different lineage); only the paired A→B delta is the P4b1 evidence.

## Protocol (B)

**Estimand:** treatment effect of the constructed #964 Metal/runtime subset
under the **Metal4-disabled** route matching intended P4a execution semantics.
This is **not** a benchmark of #964 under upstream M5 defaults (main enables
the Metal 4 tensor route on M5). `DS4_METAL_DISABLE_METAL4=1` on both arms is
experimentally clean and changes the estimand.

`tasks/p4b1_bench.sh` → `tasks/data/p4b1/bench/`. Frozen binaries (sha in
`arms.txt`), `ds4-bench --metal`, MTP off, prompt
`speed-bench/promessi_sposi.txt`, 256 greedy tokens per frontier, a 1024-token
warm-up frontier per run (first attempt without it, `bench-attempt1-nowarmup/`,
showed 129–294 t/s prefill scatter from page re-faulting after process
restart), 45 s gap, macmon sample before each run. Orders: 2K/32K `ABBABAAB`,
100K `BAAB`, 200K `ABBA`.

`ds4-bench --dump-frontier-logits-dir` dumps **pre-generation frontier
vectors**, not per-decode-step F32 traces. `p4b1_analyze.py` reports paired
gain and an exploratory paired-bootstrap interval (unit = run pair).

Smoke (2K, 32 tokens): frontier logits byte-identical A vs B (max_abs 0,
argmax 277). That is boundary equivalence only.

## 2K Protocol B result (n=4, `bench/`) — HOLD, not a win

Primary evidence is the four paired deltas. The bootstrap column is
**exploratory** (n=4 has almost no empirical distribution; a four-pair
sign-flip test cannot reject a zero decode effect; even 4/4 same-sign
prefill cannot produce a two-sided exact sign-test p below 0.125).

ABBABAAB forms four adjacent alternating-order pairs: `A B | B A | B A | A B`.

| pair | order | decode Δ% | prefill Δ% | first-step Δ% |
|---:|---|---:|---:|---:|
| 1 | AB | **+10.02** | −12.95 | −6.20 |
| 2 | BA | **−7.62** | −16.52 | +1.39 |
| 3 | BA | −1.06 | −21.56 | −9.69 |
| 4 | AB | **+13.57** | −4.68 | −11.77 |

| metric | A med | B med | ratio of medians | mean of paired % | exploratory 95% interval |
|---|---:|---:|---:|---:|---|
| gen_steady_tps | 28.99 | 30.32 | **+4.6%** | **+3.72** | [−4.34, +11.79] |
| prefill_tps | 235.24 | 199.58 | −15.2% | **−13.93** | [−19.41, −7.64] |
| gen_first_ms | 35.17 | 32.36 | −8.0% | −6.57 | [−10.73, −1.38] |

`gen_first_ms` is the **first generation step after prefill**, not request
TTFT. Exploratory paired mean −6.57%. Suggests some optimized decode path is
running; run variability still dominates steady t/s.

Ratio of arm medians (+4.6%) and mean of paired deltas (+3.72%) are different
estimands. CI includes 0 on decode either way.

**Frontier logits A↔B are byte-identical at the measured frontiers. This
establishes pre-generation boundary equivalence, not full decode-step bit
equivalence. Full decode F32 traces remain required before the port
correctness gate closes.** #964 already showed HC fusion could match
generated text and frontier logits while decode logits differed. Required
gate, not yet run:

```text
A/B:
2K:   256 decode frames
32K:  256 decode frames
200K: 256 decode frames

require:
token ids identical
every F32 logit frame byte-identical, or documented expected tolerance
no recurrent/KDA-state divergence
```

32K/100K/200K Protocol B not run. Adoption gate, if run cost permits: 8 pairs
at 2K and 32K, 6–8 at 100K and 200K. Four pairs cannot resolve a ~4% effect
on a machine that already showed large transient collapses (pair 2).

## Conclusion

**2K steady decode: unresolved.** Selected #964 engine changes produce a
descriptive +4.6% ratio-of-medians / +3.72% mean paired improvement, but four
pairs are unstable and include one large negative excursion; the experiment
does not establish a steady-state decode gain.

**Prefill: credible regression signal requiring isolation.** Observed −13.93%
paired mean is large, directionally consistent in all four pairs, and
materially worse than prior M5 Max official-Q2 (−1.9%). Use #964 prefill
rollback switches before any long sweep or P4a port:

```text
DS4_METAL_DISABLE_GLM53_FLASH_TUNING=1          aggregate
DS4_METAL_DISABLE_GLM53_PREFILL_QK_LOW
DS4_METAL_DISABLE_GLM53_PREFILL_INDEXED_ATTN
DS4_METAL_DISABLE_GLM53_PREFILL_MOE_TAIL_CULL
DS4_METAL_DISABLE_GLM53_PREFILL_KDA_PREPARE
DS4_METAL_DISABLE_GLM53_PREFILL_KDA_RECURRENCE
```

A ~14% effect should resolve with 2K ABBA. Isolate first; do not learn “B is
slower again at 32K” the expensive way.

**Correctness: frontier-equivalent; 2K decode exactness CLOSED (2026-09-15).**
`tasks/data/p4b1/decode-trace-2k/`: 256 greedy frames at ctx 2048, Metal4-off,
MTP off. Token streams identical; every F32 logit frame byte-identical
(`vocab=154880`, `n_diff_elements=0`). See `compare.txt`. 32K/200K decode
traces still required before a full port gate.

**#1051:** closed for P4b1. Dispatch counters at 2K and 32K (256 gen each,
Metal4-off, instrumented binary; `ds4-bench.p4b1-frozen` untouched):

```text
2048-A   exact=0     sg8_*=0  generic=5632
2048-B   exact=5632  sg8_*=0  generic=0
32768-A  exact=0     sg8_*=0  generic=5632
32768-B  exact=5632  sg8_*=0  generic=0
```

`DISPATCH_COUNT_PASS`. Unchecked split-group8 never runs. #1051 has zero
bearing on P4b1 A/B validity. Track for rebases only. Artifacts:
`tasks/data/p4b1/dispatch-count/`.

**Prefill “−13.9%”:** likely false alarm from the noisy P4b1 day.
Quiet-machine ABBA n=2 GEN=64 (`prefill-ab-confirm/`): paired prefill
**+0.15%**, paired gen **+16.3 / +17.1%**. FLASH_TUNING-off does not move
prefill (+0.34%) and costs decode. Escalate to clean n=4 GEN=256 Protocol B
at 2K, then 32K, before adoption.

**Adoption status: provisional KEEP leaning.** Decode-exactness PASS; #1051
unreachable; quiet-machine decode gain matches remote M5 Max magnitude.
Still need n≥4 at 2K/32K/100K/200K. Rollback: `archive/glm53-p4a-pre964`.
