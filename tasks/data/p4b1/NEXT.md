# GLM long-context scorer experiment

Date: 2026-09-16. Quiet P4b1 2K n=4 is **parked** (`bench-quiet-2k-n4/ABORTED.md`). Do not salvage.

CHECK stashes the **full `n_rows` score vector** (25K–65K floats ≈ 100–260 KB). `DS4_METAL_GLM_INDEX_CHECK_N` (default 16) is a **call** limit, not a row limit.

Synthetic randomized F32/F16 score disagreement is ~1e-6, not bit-exact. Production stays on `score_one_direct`. Decode-rows stays opt-in (`DS4_METAL_GLM_INDEX_DECODE_ROWS=1`). **Do not adopt.**

## Sequence 1–5 — executed 2026-09-16

1. full-row CHECK (`g_glm_index_check_ref_scores` is `n_rows * 4` bytes)
2. Smoke top-k instrumentation in `tests/test_glm53_kda` (`GLM_INDEX_CHECK topk ... set_eq order_eq margin`)
3. Real-decode top-k set/order/boundary at 8K / 100K / 200K — `boundary_any=0` (CHECK `min_margin` log metric is still wrong: uses `opt[k-1]`, not min over selected set)
4. 100K / 200K greedy first-divergent-token — identical streams, `first_divergent=-1` (CHECK off, decode-rows on vs off)
5. Throughput ABBA with CHECK off — **done**

### Throughput ABBA result (step 5)

Artifacts: `tasks/data/p4b1/scorer-abba/`  
Script: `tasks/p4b1_scorer_abba.sh`  
Binary: live `ds4-bench` @ `43cac00` / sha `24ab3d0e265659dd`  
A = `score_one_direct` (default). B = `DS4_METAL_GLM_INDEX_DECODE_ROWS=1`. CHECK unset. Metal4 off. GEN=256. Order per ctx: `A B B A`.

| ctx | A med gen_steady_tps | B med | gain | 95% CI (n=2) |
|---|---:|---:|---:|---|
| 100K | 28.73 | 28.74 | **+0.12%** | [−2.90, +3.13] |
| 200K | 26.41 | 27.45 | **+3.92%** | [+3.66, +4.18] |

**Headline: 100K null; 200K +3.9% steady; no-adopt.** Isolated ~30% scorer microbench does not show up end-to-end. Leave production on `score_one_direct`.

## Enable (debug only)

```
DS4_METAL_GLM_INDEX_DECODE_ROWS=1
DS4_METAL_GLM_INDEX_CHECK=1
# optional: DS4_METAL_GLM_INDEX_CHECK_N=16
```

Do not leave CHECK on for a throughput run.

## Later

- Profile default decode at 100K/200K (CHECK off) and name the residual bottleneck. `#1060` only if argsort+merge still dominates.
- Optional: fix CHECK `min_margin` (`last_sel = min` over selected set). Instrumentation only; does not reopen adoption.
- `glm53_fast` is M3 Ultra-only (`tuning_available`). Do not bypass on M5.
- #1051: Flash `DS4_N_ROT==0`, `split_group8` needs 64. Exact DSA path already runs.

## Do not

- Resume `p4b1_quiet_2k.sh` / salvage `bench-quiet-2k-n4/`.
- Chase bit-exact `decode_rows` vs `score_one_direct`.
- Adopt decode-rows as default off this ABBA.
- Re-run the locked 1–5 sequence; it is closed.
