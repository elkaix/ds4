# GPT-5.6 Sol Max Review

Date: 2026-09-01

Scope: read-only adversarial review of the DS4 GLM-5.3 width-2 verifier at
source revision `535d6bd230a679dd801cef4a09dd772860097d53`. Shared head later
moved through `468891a57fb9a61f0f74f2fbb42425c18efab833` and analysis-only
`1e9d87b29307e2b2ba7c4695352c742ec5584cae`. No source files were changed by
the review.

## Verdict

Measure H6 first. Do not write H9 or H10 kernels. Ranking: **H6 > H9 >>> H10**.
None currently supports removing the full S55 deficit.

## H6 — Concurrent Routed and Shared FFN

- Live width-2 FFN executes routed before shared.
- Existing concurrency is partial and restricted to one row, six selected
  experts, and 256 total experts. Guard widening is incorrect.
- Required dependency graph:

```text
routed gate/up  ─┐
                 ├─ barrier ─ routed down ─┐
shared gate/up  ─┘                         ├─ barrier ─ sum8 ─ final add
                     shared down ──────────┘
```

- Maximum honest saving:
  `sum_42(min(RG,SG) + min(RD,SD))`.
- Shared weights remain 1,123,024,896 bytes/cycle; H6 removes no traffic.
- Decisive gate: same-command-buffer RG/SG/RD/SD timestamps, one completion
  synchronization, profiler within 3% of OFF. Do not prototype unless the
  predicted whole-cycle gain is at least 5%. Kill if ABBA is at most 2%.

## H9 — Cross-Row Routed Expert Reuse

- Current kernels launch 16 independent row/expert-slot groups for width two.
- Maximum logical traffic saving is
  `7,077,888 * sum(layer matched_experts)`, bounded by 2,378,170,368 bytes only
  for perfect eight-of-eight overlap in every routed layer.
- Existing caches may already reuse matching addresses, so logical bytes are
  not a DRAM or time claim.
- The existing batch expert profiler records adjacent overlap but serializes
  every layer. Use it for route data only.
- Compare identical-expert and disjoint-expert access with unchanged kernels.
  Kill if elapsed time and measured memory traffic differ by at most 2%.
- Even perfect removal of the measured 6.33 ms marginal second-row cost yields
  only about 38.8 t/s at the short-context reference. H9 cannot solve S55 alone.

## H10 — Shared Q8 Pairing

Rejected. The live batch-FFN path already selects Q8 `r1_2` gate/up and down
kernels. Gate/up dequantizes a weight block once and evaluates both rows.
Incremental ceiling: **0 bytes, 0 ms**.

## Budget Correction

The prior 19.409 ms deficit applies only to the short-context reference cycle.
It does not transfer to the 197K run. Using 0.721 acceptance only as a
sensitivity, 22.75 t/s implies about 75.65 ms/cycle and a 44.36 ms gap. Actual
200K acceptance and cycle timing remain mandatory before any code change.

## Post-Review Audit

- The 2K KDA ablation result is invalid: `glm_graph_verify_rows` calls
  `glm53_graph_kda_attention_rows` without consulting `DS4_GLM_ABLATE_KDA`.
  The mask reaches one-row decode and indexed-batch paths only.
- The 9.635 GB static trunk inventory at `1e9d87b` reproduces, but its roofline
  does not model width-2 dense reuse, routed expert union, or draft/head call
  counts and uses forbidden theoretical bandwidth as proof.
- Exact GGUF math gives 1.141 GB saved by converting the 34 KDA layers'
  `kda_v` and `kda_output` tensors from Q8_0 to Q4_K, not 1.51 GB.
- Candidate order remains measurement-first: actual `a200`, non-serializing
  macro timing, then M1. No requantization or kernel edit is authorized yet.

## Second Max-Effort Audit at `1e9d87b`

The independent reviewer rejected the prior additive stage interpretation. At
steady 2K, `glm53_spec_verify` selects `glm_graph_verify_rows`; that path checks
only the routed mask inside its batch FFN helper. Non-routed masks affect the
seed one-row call, changing subsequent tokens, acceptance, and routes without
removing their named component from the steady verifier.

| Mask | One-row decoder | Fast two-row verifier | Indexed two-row verifier | Draft |
|---|---|---|---|---|
| `kda` | Entire KDA attention | Nothing | Entire KDA attention | Nothing |
| `routed` | Routed MoE only | Routed MoE only | Routed MoE only | Nothing |
| `shared` | Resident shared branch only | Nothing | Nothing | Nothing |
| `qpath` | DSA q path | Nothing | Partial q path | Nothing |
| `indexer` | Indexer path | Nothing | Nothing | Nothing |
| `attn_core` | Indexed qk/attention branch | Nothing | Batch qk/attention | Nothing |
| `qklow` | qk-low only | Nothing | Nothing | Nothing |
| `attn_out` | DSA output projection | Nothing | DSA output projection | Nothing |

Consequences:

- Keep only the routed arm as a steady 2K deletion ceiling. Relabel every
  non-routed F25/F26 arm invalid stage attribution.
- Draft is one unique layer-45 block, not 45 trunk layers. It has no designed
  trunk-weight sharing and ends each call with the full output head, logits
  readback, and CPU argmax.
- Accepted cycles call the first draft with `&dummy`; its head result is unused.
  A no-head state-advance variant is a valid exact-runtime candidate, but even
  deleting that whole ~2.3 ms call only bounds the gain near 3% at `a=.721`.
- The single-process width-two routed path has fused IQ2 gate/up/SwiGLU, generic
  Q2 down, and a separate expert reduction. Existing pack2/direct-sum guards do
  not cover ordinary 8-of-288 on one M5 GPU.

The reviewer proposed multi-branch/tree block speculation as the same-checkpoint
structural research path. Using the scored 22.48 t/s and old `.721` acceptance
only as sensitivity gives a 76.56 ms cycle; keeping that cycle time would need
4.21 committed tokens for 55 t/s. Under constant independent `.721` acceptance,
an infinitely wide linear chain reaches only `1/(1-.721)=3.58`. Branching could
change that ceiling, but the current engine has no branch-aware KDA state or DSA
causal verifier, and verification cost cannot be assumed constant. The idea is
therefore H18 research—not a selected implementation—until actual `a200`, row
scaling, route union, M1, exactness, workload, and 60-minute gates exist.

Shared HEAD `d0a6151` incorporated the coverage correction and retracted the
“impossible” headline. Its replacement 79.5 t/s sensitivity is still not a
measured roofline: it assumes one trunk sweep for a two-row cycle, omits the
Q4_K per-row path and route-union uncertainty, and undercounts target output
heads (`1+a` per cycle). It does not alter the measurement-first verdict.
