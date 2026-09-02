# S55-200 Stack-Specific Research

Date: 2026-09-01

North star: the minimum sustained generated-token decode rate at real 2K,
32K, 64K, 128K, and 200K context. External results below create hypotheses;
none count as speedup until a controlled DS4/M5 Max/GLM-5.3 A/B passes.

## Source-Grounded Findings

1. The target M5 Max exposes the public `timestamp/GPUTimestamp` counter set
   but reports dispatch-boundary sampling unsupported. The topology-neutral H26
   fallback therefore labels existing command buffers and reads
   `GPUStartTime`/`GPUEndTime` at their existing completion wait. At position
   197,623, profiler OFF/ON hidden state and logits were byte-identical; OFF/ON
   means were 85.992/86.345 ms and medians 88.662/89.214 ms, or +0.41%/+0.62%
   overhead with paired delta CI [-0.202, 0.910] ms. The 16-cycle replay also
   matched all 4,885,013,984 serialized state bytes. This accepts the backend as
   a whole-command-buffer diagnostic. It rejects H26 as the required macro-region
   target selector: all selected-layer labels on an indexed verify share the same
   whole-forward span, so they cannot rank KDA against routed MoE. Profiling earns
   zero S55 throughput credit.
2. Apple documents ordered `enqueue()` plus parallel command-buffer encoding.
   DS4 should test coarse command-buffer splits only if wall time materially
   exceeds GPU span; otherwise host/GPU overlap has no useful ceiling.
3. Cohere measured temporal expert-route correlation across adjacent speculative
   tokens. Its reported step-1 overlap was about 0.38 for an unrelated 128-expert,
   top-8 model. This supports capturing GLM route overlap, not assuming it.
4. *Utility-Driven Speculative Decoding for Mixture-of-Experts* reports the
   opposing failure mode: speculative rows activate more unique experts and can
   increase weight traffic enough to erase acceptance gains. Local unique-expert
   count and routed bytes are therefore decisive.
5. vLLM's fused-MoE documentation groups tokens by expert before quantized expert
   work; its low-latency backend discussions emphasize small, memory-bound expert
   GEMMs and fusing routing/activation/reduction. DS4 already has one dispatch for
   each routed gate/up, down, and reduction stage, so the useful porting idea is
   cross-row weight reuse—not recreating a grouped runtime.
6. llama.cpp evidence shows compact IQ formats can be slower on Metal than a
   larger quant. Model size or nominal bits/weight cannot substitute for measured
   decode speed. The existing IQ2_XXS/Q2_K checkpoint remains fixed.
7. MLX's current documentation covers packed affine 2-bit kernels, but its own
   tracker still lacks native IQ2_XXS/Q2_K GGUF execution. It supplies no exact
   kernel result transferable to this checkpoint.
8. DS4's CUDA backend has exact-format aligned-SoA prototypes: IQ2_XXS improved
   about 12%, and its Q2_K prototype reports roughly 154 to 214 GB/s on a
   different accelerator. This is strong layout inspiration inside the same
   codebase, but remains zero-credit on Metal until a local A/B passes.
9. A balanced 2K skip-ablation on the normal Metal command-buffer topology
   measured routed MoE at 10.229 ms/generated-token, 33.7% of the 30.352 ms
   baseline. KDA measured 2.659 ms/token. The intentionally invalid-output
   ablation attributes time only; it does not distinguish bytes, addressing,
   dequant arithmetic, or row reuse.
10. Whole-command-buffer GPU span resolves the non-additive ablation floor.
    Baseline was 30.379 ms/token wall and 28.555 ms/token GPU (94.0% busy);
    with KDA+routed removed it was 20.309 wall and 20.160 GPU (99.3% busy).
    The remaining floor is real GPU work, not exposed CPU submission time.
11. The same source trace invalidates the reported 0.681 ms/token `shared`
    ablation as a price for H6. `DS4_GLM_ABLATE_SHARED` guards only the one-row
    path; `DS4_GLM_ENCODE_FFN_BATCH_SHARED` still executes in live width-2
    verification. H6 therefore remains unpriced.
12. None of the decode-ablation bits is read by `glm_graph_mtp_step`. A reject
    runs one draft call and an accept runs two, averaging `1+a` calls/cycle.
    Each call executes one dedicated nextn attention/FFN layer plus the shared
    output head—not the 45-layer target stack. Existing 2K timing is about
    2.3 ms/call, so draft cannot account for the 20.1 ms/token residual.
13. The exact routed formats are 66 bytes/256 weights for IQ2_XXS and 84/256
    for Q2_K. Two IQ2 matrices plus one equal-sized Q2 matrix average 0.28125
    bytes/weight: 7,077,888 bytes/expert, 56,623,104 bytes/layer, and
    2,378,170,368 logical bytes for one row across 42 layers. The prior 42.6%
    calculation mixed that one-row byte count with a per-generated-token
    width-2 ablation delta. At `a=0.721`, counting both verifier rows gives
    49.5% of theoretical peak instead. Neither number is measured bandwidth;
    only M1 can classify the bottleneck.
14. Current issue #925 reports loops, tool errors, and a crash for a similarly
    named GLM-5.3 IQ2/Q2 community checkpoint. It does not prove this exact
    `UNCEN-Q2` artifact is wrong, but it makes the official-vector and 100-case
    quality gates release-blocking. Speed measurements alone cannot certify it.
15. The existing MTP timing log contains the same 214 accepts and 83 rejects on
    its 2K row-verifier and 32K indexed-batch arms. Median verifier time rises
    from 43.7 to 51.4 ms/cycle and reported decode falls from about 31 to 29
    t/s. Context and implementation path changed together, so 8.1 ms is not a
    removable-cost claim. It is enough to require a matched forced-batch versus
    rows experiment at 2K before designing a long-context tiny-row verifier.
16. The GGUF header reproduces a 9,635,430,648-byte active trunk inventory for
    one target pass: KDA 3,845,383,680; routed 2,378,170,368 for eight active
    experts per MoE layer; DSA 1,537,800,192; shared 1,123,024,896; dense FFN
    481,296,384; router 198,229,248; norms/mHC 71,525,880. This is static
    tensor accounting, not measured DRAM traffic or bytes/generated-token.
17. The width-2 path breaks the one-pass roofline assumption. Q8/F16/BF16
    small-batch kernels evaluate two rows from one weight load; Q4_K KDA q/k
    currently choose the classic grid-Y path; routed bytes depend on the union
    of two independently selected expert sets; the draft executes one 295 MB
    active nextn layer plus the shared output head once on rejects and twice on
    accepts. M1 must measure the resulting traffic and time.
18. The proposed `kda_v` plus `kda_output` Q8_0-to-Q4_K change saves exactly
    1,140,850,688 bytes across 34 KDA layers under GGUF block sizes 34/32 and
    144/256. The published 1.51 GB estimate is not reproducible. More
    importantly, recurrent attention quantization is a weight/quality change
    and receives zero S55 credit until decode, 200K, and quality gates pass.
19. A second Sol max audit traced the complete ablation matrix. At steady 2K,
    fast `glm_graph_verify_rows` applies only `routed`; every non-routed arm
    changes the seed token stream without deleting the named steady verifier
    component. Retain the routed ceiling and reject all other F25/F26 stage
    attributions.
20. Accepted cycles execute two nextn calls, but the first call's drafted token
    is stored in `dummy`. Its layer-45 state advance is needed; its full output
    head, logits readback, and CPU argmax are not. Even deleting the whole first
    ~2.3 ms call on `.721` of cycles bounds this below the current 5% priority
    gate, so a no-head variant is parked until the large gap is identified.
21. Multi-branch/tree block speculation is a distinct research candidate from
    the closed linear width-3 experiment. At the scored 200K rate and old
    `.721` acceptance sensitivity, unchanged 76.56 ms cycles would require
    4.21 committed tokens, while an infinite constant-acceptance linear chain
    reaches 3.58. No branch-aware KDA/DSA verifier exists, so the candidate
    cannot outrank measured `a200`, macro attribution, and M1.
22. Shared HEAD `d0a6151` correctly retracts “S55 is impossible,” but its new
    79.5 t/s roofline is still not evidence. The width-2 verifier does not read
    every trunk tensor exactly once: paired Q8/F16/BF16 kernels share loads,
    Q4_K KDA q/k currently dispatch per row, and routed traffic depends on two
    selected-expert sets. The target output head runs once for row 0 and again
    on accepted row 1, while draft layer/head calls also average `1+a`.
23. Exact static nextn inventory is 295,005,312 active bytes per layer-45 call
    before its 674,037,760-byte output matrix. At old `a=.721`, those calls
    average 1.721/cycle. These counts are useful for constructing M1; cache
    behavior and achievable bandwidth still require measurement.

## Same-Backend PR Audit

| PR | Relevance to this target | Verdict |
|---|---|---|
| antirez/ds4#799 | M5 multi-session queue overlap; its implementation explicitly leaves GLM on the old path and measures aggregate session throughput | Research clue for idle GPU capacity only; not S55 evidence |
| #794 | DeepSeek/M3-Ultra long-context indexer and reducer work; short-context decode was unchanged | Inspect only if GLM 200K profiling identifies the same kernel family |
| #778 | M5 FFN concurrent-encoder island and HC scheduling | Structural reference for H6; reported occupancy-only changes are too small to prioritize alone |
| #832 | Exact streaming top-k; reported generation deltas were noise while prefill improved | Zero current decode credit; reconsider only after GLM 200K top-k timing |
| #755 | Merged Q2/M5 decode kernels | Already in the measured source lineage; not an unclaimed candidate |
| #828 | Stage timing by committing at each boundary without CPU waits | Still changes command-buffer topology; reject unless ON/OFF is within 2% |

## Independent Sol Max Review

GPT-5.6 Sol reviewed source revision
`535d6bd230a679dd801cef4a09dd772860097d53` read-only. Shared head later moved
through `468891a57fb9a61f0f74f2fbb42425c18efab833` and then analysis-only
`1e9d87b29307e2b2ba7c4695352c742ec5584cae`; shared head later advanced to
`b265e226a454d9e6c29c19c22650560039cc9171` with the small H19 change. It ranked the candidate
set **H6 > H9 >>> H10** and found no candidate with evidence sufficient to
remove the full S55 deficit.

1. Width-2 routed execution is exactly IQ2 gate/up+SwiGLU, Q2_K down, then
   sum8. It is not a 27-launch-per-layer path.
2. H6 must schedule routed/shared gate-up concurrently, synchronize both
   intermediate results, schedule both down projections concurrently,
   synchronize before sum8, then perform sum8 and the final residual add.
   Extending the one-row helper by widening its guard would compute only row 0.
3. H9 has a real missing reuse opportunity, but even impossibly deleting the
   entire measured 6.33 ms marginal verifier-row cost would yield only about
   38.8 t/s at the short-context reference point.
4. H10's proposed traffic saving is already present: the live batch-FFN path
   uses the Q8 `r1_2` gate/up and down kernels, which load a block once and
   evaluate both rows. Its incremental ceiling is zero.
5. Holding the old 0.721 acceptance only as sensitivity, 22.75 t/s at 197K
   implies about 75.65 ms/cycle and a 44.36 ms gap to 55. Actual 200K
   acceptance must replace this sensitivity before it becomes a budget.

## Ranked Local Experiments

| Rank | Existing DS4 lever | Maximum defensible ceiling before measurement | Gate |
|---|---|---:|---|
| M0 | Existing whole-command-buffer GPU-busy probe | Diagnostic only; measured 93.8-94.3% busy at 2K/32K | Formal profiler OFF/ON gate still required; coarse scheduling is parked |
| M1 | Equal-byte sequential / production-pattern / real routed-MoE / width-2 routed-MoE benchmark | Decides whether bytes, access pattern, dequant kernel, or width-2 scaling owns the gap | One queue, balanced arm order, one final completion wait, real full-cycle byte volume |
| R1 | Existing concurrent routed/shared FFN mechanism, generalized to indexed two-row 8/288 | `sum_42(min(RG,SG) + min(RD,SD))`; no byte reduction | Timestamp RG/SG/RD/SD in one command buffer; do not prototype below 5% predicted whole-cycle gain |
| R2 | Aligned-SoA IQ2_XXS/Q2_K expert layout, preserving every quant bit and accumulation tree | Unpriced on Metal; same-backend CUDA evidence only | Prototype one real selected-expert span after M1; reject unless exact and >=5% whole-cycle projection |
| R3 | Reuse IQ2/Q2 selected-expert weights when the two verifier rows choose the same expert | 2,378,170,368 logical bytes/cycle times measured route overlap; not proven DRAM traffic | Existing intrusive profiler for overlap only; unchanged-kernel identical-ID/disjoint-ID microbench before code |
| R4 | Fuse DSA indexer score with first block-top-k stage | Only the measured 200K score/top-k time; zero 2K gain | Needed to keep context decay within 2% |
| R5 | Replace the generic indexed MTP batch fallback with an exact two-row sparse verifier | Unpriced; existing confounded log shows an 8.1 ms/cycle row-vs-batch gap | First force both paths at the same short context, then profile at 200K; preserve KDA snapshots and full DSA history |
| R6 | Multi-branch/tree block speculation with adaptive fallback | Only structural same-checkpoint class identified as potentially large enough; no local timing ceiling | Research only after `a200`, n-row scaling, route-union, and M1; exact branch KDA/DSA semantics and four-workload 200K gate required |
| R7 | Skip the unused first draft output head on accepted cycles | 2K ABBA +2.256% (30.448 -> 29.761 ms/token); 65K +1.789% (34.013 -> 33.404); identical 512-token text in every arm; no 200K or sustained credit | Committed as `b265e22`; paired observable-state benchmark still required. Small retained cleanup, not the next large S55 target |
| R8 | Add an exact-order Q2 direct-down reduction for 8 selected experts | Removes one sum8 dispatch per routed layer and 22.0 MB/cycle of scratch write/read; selected-expert weight reads remain | Do not merely widen the existing host guard: that kernel combines experts before one SIMD reduction, unlike production's per-expert SIMD reductions followed by sum8. After M1, retain only a bit-identical variant with >=5% whole-cycle projection |
| R9 | Specialized width-2 Q4_K KDA q/k matvec preserving each row's reduction order | One full q/k sweep across 34 layers is 1,283,457,024 logical bytes; a two-row kernel can at most amortize one sweep, not proven traffic or time | Measure production q/k GPU time and 2-row/1-row ratio at 200K. Reject if the priced whole-cycle ceiling is below 5% or the ratio is already <=1.3; require bit-identical rows |

The 2K diagnostic makes the local floor concrete: baseline 30.352 ms/token,
S55 budget 18.182 ms/token, deficit 12.170 ms/token. Impossible deletion of all
routed work leaves 20.123 ms/token (49.69 t/s), still a failure. No routed
kernel candidate can win alone. S55 requires another measured large term plus
removal of the 200K context decay.

The combined-ablation control further rejects host scheduling as that second
large term: 20.160 of 20.309 ms/token remained on the GPU. The next measurement
must split this GPU floor into shared FFN, mHC/residual/norms, output/MTP draft,
and attention without per-stage queue drains.

M1 stays inside the production stack. Existing graph-dump hooks capture real
200K width-2 FFN norm/router inputs, the repository GGUF parser supplies exact
expert offsets, and the production routed-MoE APIs implement C/D. Only A/B need
new runtime code: one diagnostic checksum reader over equal 4,756,340,736-byte
logical volumes. A reads contiguous spans; B reads the captured selected spans.
Report unique bytes separately because row expert sets can overlap. Run all four
arms in balanced order with cache-defeating layer rotation and one completion
per arm; no M1 bandwidth result itself counts toward S55.

H6 cannot be implemented by merely calling the existing parallel-FFN entry
with `n_tokens=2`. That entry stores one-row Q8 gate/up and down pipeline state
and validates only one activation row. Production GLM width-2 shared gate/up
uses the `mul_mv_ext` two-row kernel, and shared down likewise follows the
two-row Q8 path. A valid H6 prototype must preserve those exact two-row PSOs,
argument layouts, and outputs while changing only the encoder dependency graph.
Price RG/SG/RD/SD first; reject immediately if the maximum measured overlap is
below 5% of the 200K cycle or if concurrent dispatch only contends for memory.

R9 is similarly measurement-gated. Metal currently routes Q4_K batches of up
to eight rows through the classic matvec with `n_tok` as grid Y, while the
alternative `mul_mv_ext` family has a two-row form but is documented in this
tree as slower for the GLM dense shapes. Do not flip that selector. Timestamp
the real q/k projections and compare one versus two rows first; only a new
exact-order pair kernel with a >=5% measured whole-cycle ceiling is eligible.

H16 is likewise not a predicate-removal patch. `glm_graph_verify_rows` is
eligible only while the requested positions fit its dense compact-attention
window. Forcing it above that bound would silently replace real 200K DSA
history with the compact window and violate S55-200 semantics. Any retained
high-context verifier must execute the existing indexed DSA selection and full
physical history while specializing only the two-row scheduling/layout; it must
also keep the KDA base/prefix transaction exact. Compare forced rows versus
indexed only at a context where both are semantically eligible, then price the
indexed DSA subregions at 200K before designing that specialization.

H19 source review found no control-flow defect. Two comments need correction:
rollback is `DS4_GLM_MTP_DISCARDED_HEAD=1`, not `0`, and the claimed 4% byte
share lacks an M1-measured traffic denominator. The already-present paired
benchmark must receive that environment name as argv[2]; it checks exact token
IDs, cycle sequence, position, counters, and full float logits after every
alternating 64-token chunk, but not raw internal nextn-cache bytes.

Skip-ablation cannot supply that additive split. A skipped stage leaves stale
buffers, changing downstream hidden values and expert selections; the negative
and non-additive arms demonstrate the interaction. It remains useful only as a
large-component deletion ceiling.

## Rejected or Parked

- New C++/`metal-cpp`, MLX, llama.cpp, or greenfield inference harness: wrong
  execution stack.
- Prefill, TTFT, KV restore, cache hit rate, free RAM, or model-size-only work:
  zero S55 decode credit.
- Native width-3 speculation: closed by local measurements unless the underlying
  cost model changes.
- Paired shared-expert Q8 loading: already implemented by the live width-2
  `r1_2` kernels; incremental ceiling is 0 bytes and 0 ms.
- Pack2/direct-sum as the primary gap: current 8/288 verifier already fuses
  IQ2 gate+up+SwiGLU. Its Q2 down still writes 16 expert rows and launches
  sum8, but the removable scratch traffic is only 22.0 MB/cycle across all 42
  routed layers and expert-weight reads do not change. Price dispatch cost after
  M1 before writing an exact-order fused reduction. Merely widening the
  reusable direct kernel's host guard changes floating-point reduction order.
- Existing `DS4_EXPERT_PROFILE` as performance evidence: it ends and restarts
  command batches for every layer. Reuse it only to measure adjacent route
  overlap offline.
- Coarse early command-buffer commits as the primary lever: measured GPU busy
  leaves only 5.7-6.2% wall residue, including unavoidable submission cost.
  Revisit only after larger kernel/access-pattern work.
- Claims based on theoretical UMA bandwidth or another accelerator's speedup:
  diagnostic only.
- The `1e9d87b` conclusion that S55 is above a 546 GB/s roofline: forbidden
  theoretical evidence and incorrect width-2 execution accounting. Preserve
  its static GGUF inventory; reject its candidate ranking until M1.
- The replacement `d0a6151` 79.5 t/s roofline: it fixes the direction of the
  conclusion but still models a heterogeneous two-row speculative cycle as one
  trunk sweep. Keep it as sensitivity only, not S55 progress or a bound.

## Sources

- Apple, *Explore Live GPU Profiling with Metal Counters*:
  https://developer.apple.com/videos/play/tech-talks/10001
- Apple, `MTLCommandBuffer.enqueue()`:
  https://developer.apple.com/documentation/metal/mtlcommandbuffer/enqueue()
- Cohere, *Why MoE Models Get More From Speculative Decoding*:
  https://cohere.com/blog/mixture-of-experts-models-get-more-from-speculative-decoding
- *Utility-Driven Speculative Decoding for Mixture-of-Experts*:
  https://arxiv.org/html/2506.20675v1
- vLLM, fused-MoE kernel features and batched Marlin MoE:
  https://docs.vllm.ai/en/stable/design/moe_kernel_features.html
  https://docs.vllm.ai/en/stable/api/vllm/model_executor/layers/fused_moe/experts/marlin_moe.html
- llama.cpp PR #5747, Metal quantized-kernel performance evidence:
  https://github.com/ggerganov/llama.cpp/pull/5747
- MLX issue #1388, exact sub-3-bpw GGUF formats still unsupported:
  https://github.com/ml-explore/mlx-lm/issues/1388
- DS4 exact-format aligned-layout prototypes:
  `cuda/mmq/test/proto_iq2_aligned.cu` and
  `cuda/mmq/test/proto_m2_q2k.cu`
- DS4 issue #925, current community checkpoint correctness report:
  https://github.com/antirez/ds4/issues/925
