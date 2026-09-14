# S55-200 Competition Plan

Success: prove `min(TPS_2K, TPS_32K, TPS_64K, TPS_128K, TPS_200K) >= 55.0`
for real autoregressive decode with MTP, unchanged semantics/quality, and no more
than 2% context or 60-minute runtime decay.

Stack lock: optimize the existing DS4 C / Objective-C / Metal path in place.
No greenfield harness, C++, `metal-cpp`, MLX, or format-substitute kernels.

## Execution

- [ ] PHASE 0 — Freeze binary, model, source, OS, hardware, and runtime config provenance.
- [ ] PHASE 1 — Measure trustworthy 2K/32K/64K/128K/200K decode baseline.
- [ ] PHASE 2 — Measure `a200`; calculate target cycle time and removable-ms deficit.
- [ ] PHASE 3 — Build macro GPU profiler; prove profiler ON is within 3% of OFF.
- [ ] PHASE 4 — Benchmark equal bytes: sequential, selected-expert pattern, routed MoE, width-2 routed MoE.
- [ ] PHASE 5 — Rank components by measured 200K cycle time.
- [ ] PHASE 6 — Optimize only the largest verified component with plausible >=5% end-to-end gain.
- [ ] PHASE 7 — Run same-process or balanced ABBA speed and correctness gate.
- [ ] PHASE 8 — Validate the retained change at real 200K context.
- [ ] PHASE 9 — Validate prose, code, tool JSON, and reasoning workloads at 200K.
- [ ] PHASE 10 — Validate 60 minutes with no cooldown/restart and <=2% decay.
- [ ] PHASE 11 — Recalculate the remaining cycle-ms deficit.
- [ ] PHASE 12 — Repeat phases 5-11 until S55-200 passes.

## Evidence Rules

- Only generated tokens divided by decode time counts toward S55-200.
- No truncation, sliding history, filler prompt, quality loss, early stop, or changed sampling.
- Prefill, TTFT, cache, RAM, utilization, and theoretical bandwidth stay in separate ledgers.
- Every candidate declares component cost, removable ceiling, expected whole-cycle gain, measurement, and kill condition before code.
- Sub-2% results require direct operation timing plus same-process A/B; otherwise reject as unresolved.

## Phase 0 Provenance

- Baseline source: `c2175e01e7c9cd8b1eedc4e7ca5898ac71d62876`
- Current shared source: `b265e226a454d9e6c29c19c22650560039cc9171`; it commits the discarded-draft-head change and ABBA evidence
- Isolated Codex source: `535d6bd230a679dd801cef4a09dd772860097d53`; shared head additionally carries the timing-only ablation work, analysis ledgers, and H19
- Baseline runtime binary SHA-256: `33214637a2c2876eebdc801c2208a65619a12e6e51f364c76a4af0bfe3e52b13`
- Current live `ds4-server` SHA-256: `3fc1971e15b13e2a92743e982f17e9f8e4d8bb3c96893f4c01c15dd72879fc25` (inode 276953551; built 2026-09-01 08:02:15 -0400; contains both H19 and the older deferred-row1 switch)
- Model: `/Users/panda/models/gguf/GLM-5.3-Flash-UNCEN-Q2.gguf` (96,505,818,432 bytes; inode 273408410; mtime 2026-08-31 18:40:44 -0400; SHA-256 pending idle I/O window)
- Hardware: M5 Max, 40-core GPU, 128 GB; macOS 26.6.2 (25G83); Metal 4
- Runtime: Metal, MTP, context ceiling 262,144, one slot, power 100
- Frozen prompt corpus: 1,157,096 bytes, SHA-256
  `78493835239cb7a3b35228bf7304f72d3f293d9b99c4a7ef10e0aca206c7150b`;
  the explicit exclusion recipe in `baseline-s55-200.md` reproduces it
  byte-for-byte

## Hypothesis Ledger

| ID | Hypothesis | Source / path | Expected max gain | Experiment | Observed | 200K | Correctness | Verdict | Commit |
|---|---|---|---:|---|---|---|---|---|---|
| H1 | Native Metal tracing or stage-boundary timestamps can attribute macro GPU time with <=3% overhead without extra completion waits. | `xctrace` ships the Metal System Trace template; M5 Max reports dispatch-boundary sampling unsupported and stage-boundary sampling supported; historical `11689e1` and PR #828 split command buffers and are rejected | Diagnostic only | Test native trace export and OFF/ON ABBA first; only if insufficient, delimit macro regions with compute-encoder boundaries inside the existing command buffer | Hardware/tool capability confirmed; trace schema and runtime overhead pending | Pending | No serving-default change | SUPPORTED, NOT YET ACCEPTED | — |
| H2 | GLM53 width-2 routed IQ2/Q2 MoE rereads selected expert weights inefficiently; generalizing existing pack2/direct-sum paths to the actual 8/288 shape may reduce its cost. | `ds4_gpu_routed_moe_batch_tensor`: n2 uses fused IQ2 gate+up+SwiGLU, Q2_K down, then sum8; existing direct sum is guarded to 6 experts | At 2K, deleting routed work saved 10.229 ms/token (33.7%); even impossible full deletion reached only 49.69 t/s | Equal-byte sequential/pattern/kernel/verify benchmark; 1-row vs 2-row; routed-stage timing before any edit | Balanced baseline/routed/baseline ablation: 30.352 -> 20.123 ms/token, output intentionally invalid; exact bottleneck type remains unknown | Pending | Bit-identical retained change | CONFIRMED LARGE, UNDIAGNOSED | — |
| H3 | Long-context KDA/DSA, not MoE, becomes the largest 200K cycle component. | Short-row scan: DSA ~44% at ~2.6K; batch verifier path differs | Unknown; the claimed 2K KDA deletion ceiling is invalid | Non-serializing 200K macro profile; 1-row vs 2-row | The KDA mask is read by one-row decode and indexed batch, but not by `glm_graph_verify_rows`; the 2K fast-row KDA arm therefore did not remove KDA | Pending | Bit-identical | OPEN, UNPRICED | — |
| H4 | PR #864 improves S55 decode. | PR #864 IQ2/Q2 MPP prefill kernels | 0% measured | Upstream same-machine A/B | Decode unchanged | Not needed | Tests reported upstream | REJECTED | — |
| H5 | Native MTP width >2 reaches S55. | PR #892 / current investigation | Optimistic width-3 cost model only ~+7%, still far below S55 | Width 2/3/4/6 matched A/B | Wider native MTP measured uncompetitive; corrected marginal-draft model does not close the target | Not reopened | Goldens matched | CLOSED | — |
| H6 | Run GLM53 shared-expert Q8 gate/up/down concurrently with the independent 8/288 routed-MoE branch during width-2 verification. | GLM batch FFN is serial; DS4 already has a one-row 6/256 concurrent mechanism, but guard widening would compute only row 0 | Cannot exceed the 10.229 ms/token 2K routed-ablation ceiling; actual ceiling is `sum_42(min(RG,SG) + min(RD,SD))` | Same-command-buffer RG/SG/RD/SD timing first; only then implement the two-barrier dependency graph | Sol ranked it first; routed work is now proven large, but branch-overlap savings remain unpriced | Pending | Same outputs; no reduction-order change | SUPPORTED, NEXT TO PRICE | — |
| H7 | Pair the two rows of GLM53 BF16 matvecs so each weight is loaded once while preserving the existing per-row reduction tree. | `ds4_gpu_glm53_matmul_bf16` dispatches grid-Y independently for `n_rows <= 8`; exact GGUF directory inventory finds 235,154,432 BF16 weight bytes in the 45-layer target verifier path per row | Perfect reuse removes at most 235 MB/cycle before kernel-efficiency effects; likely below the 5% priority gate | Measure BF16 macro time and 1-row/2-row ratio; reject unless the measured removable cost is >=2.5 ms/cycle | Source and exact model inventory confirmed; time pending | Pending | Bit-identical per row | OPEN, PARKED | — |
| H8 | Commit the long-context indexed verifier after early layers so GPU execution overlaps host encoding of the remaining layers. | Existing whole-CB timestamps measured GPU busy against decode wall | Absolute upper bound is only 5.7-6.2% at 2K/32K and includes unavoidable submit latency | Formal profiler OFF/ON gate if revisited | 94.3% busy at 2K; 93.8% at 32K; current cross-run throughput agrees within ~1%, not formal ABBA | Pending | Serial queue keeps dispatch order and outputs | REJECTED AS PRIMARY | — |
| H9 | Reuse routed-expert weights across the two verifier rows when adjacent tokens select the same expert. | Current IQ2/Q2 tiny batch kernels dispatch 16 independent expert-token pairs; existing batch profiler records adjacent overlap but synchronizes per layer | 2,378,170,368 logical bytes times measured overlap; even perfect deletion of the 6.33 ms marginal row reaches only ~38.8 t/s at the short-context reference | Intrusive profiler for overlap only; unchanged-kernel identical/disjoint pattern microbench before code | Real opportunity but provably insufficient alone; cost and overlap pending | Pending | Bit-identical logits/tokens required | OPEN, DEFERRED BEHIND H6 | — |
| H10 | Reuse each shared-expert Q8 gate/up weight load across both verifier rows. | Live batch FFN already uses Q8 `r1_2` gate/up and down kernels that reuse weights across both rows | 0 bytes; 0 ms | Reopen only if a live PSO capture contradicts the source path | Sol review falsified the premise | Not needed | Existing path | CLOSED | — |
| H11 | Repack IQ2_XXS and Q2_K expert blocks into aligned SoA sections without changing quant values or float reduction order. | Current Metal kernels traverse raw 66-byte IQ2 and 84-byte Q2 blocks; DS4 CUDA exact-format prototypes measured layout wins on another accelerator | Unpriced on Metal; no current S55 credit | M1 first; then one-layer real-span raw-vs-SoA kernel A/B with cache-defeating rotation | Source structure confirmed; Metal cost unknown | Pending | Raw-bit round-trip and bit-identical outputs | OPEN, CONDITIONAL ON M1 | — |
| H12 | Host submission/synchronization is the large residual after routed/KDA removal. | Whole-command-buffer GPU span on baseline and combined ablation | At most 0.149 ms/token in the combined-ablation arm | GPU span divided by decode wall, one normal completion | Baseline 94.0% GPU busy; combined ablation 99.3%, 20.160 of 20.309 ms/token on GPU | Pending | No runtime change | REJECTED AS LARGE TARGET | — |
| H13 | The measured 0.681 ms/token `shared` ablation prices H6. | `DS4_GLM_ABLATE_SHARED` call sites | No valid ceiling | Trace every guard into the live width-2 batch path | Guard exists only in the one-row FFN path; `DS4_GLM_ENCODE_FFN_BATCH_SHARED` still runs during verification | Not applicable | Measurement fix only | REJECTED EVIDENCE | — |
| H14 | The decode-ablation floor is mainly an unablated full-model MTP draft pass. | `glm_graph_mtp_step` and `ds4_session_glm_spec_cycle_impl` | One nextn layer plus one output head per call, not 45 trunk layers | Source dispatch inventory plus existing MTP timing log; macro timestamps still required | No ablation bit reaches MTP; rejects call it once, accepts twice, average `1+a`; each call is one dedicated nextn layer. Existing 2K log is about 2.3 ms/call | Pending | No runtime change | REJECTED AS FLOOR EXPLANATION | — |
| H15 | The routed ablation proves 42% of measured peak bandwidth. | Exact IQ2_XXS/Q2_K block sizes and width-2 cycle accounting | Diagnostic only | M1 measured sequential/pattern/kernel bandwidth | Exact mixed storage is 0.28125 B/weight and 2.378 GB/row; 42.6% divides a one-row byte count by a per-generated-token delta. Two verifier rows and `a=0.721` instead imply 49.5% of theoretical peak, but neither is measured DRAM bandwidth | Pending | No runtime change | REJECTED EVIDENCE | — |
| H16 | The generic indexed two-row verifier is a removable long-context tax; a tiny-row sparse verifier can preserve 200K semantics with less GPU work. | `glm_graph_verify_rows_eligible` stops above the dense window; `glm53_spec_verify` falls back to `glm_graph_forward_indexed_tokens` | Same timing log shows 51.8 ms batch verify versus 43.7 ms rows, but context and path are confounded; 200K ceiling pending | At 2K force rows versus indexed batch with matched tokens, then macro-profile both at 200K; prototype only if the removable delta is >=5% | At 2K/32K the same continuation had identical 214/83 acceptance counts; timing-log medians were 43.7/51.4 ms verify and reported decode fell about 31 to 29 t/s | Pending | Exact logits, KDA transaction, DSA history | SUPPORTED, UNPRICED | — |
| H17 | The static 9.635 GB trunk inventory proves a decode roofline and makes KDA requantization the top candidate. | GGUF header plus theoretical 546 GB/s | No defensible speed ceiling before M1 and real width-2 execution counting | Reproduce tensor inventory, trace verifier/draft/head multiplicities, then run M1 measured bandwidth and macro timing | Static trunk total reproduces exactly. `d0a6151` retracted “impossible” but its replacement 79.5 t/s ceiling still assumes one trunk sweep: Q4 KDA q/k use grid-Y rows, routed traffic depends on two expert sets, target heads run `1+a` times, and drafts run `1+a` layer/head calls. Theoretical bandwidth remains forbidden evidence. Exact KDA V+output Q8→Q4 saving is 1.141 GB, not 1.51 GB | Pending | Requantization needs predeclared recurrent-quality suite | REJECTED AS S55 PROOF; QUANTIZATION PARKED | — |
| H18 | Multi-branch/tree block speculation can amortize a target trunk sweep across enough exact committed tokens to close the same-checkpoint gap. | Second Sol max audit; current verifier already batches linear rows but has no branch-aware KDA/DSA state graph | Unpriced. At the scored 200K rate and assumed `a=.721`, unchanged 76.56 ms cycles require 4.21 committed tokens; an infinite linear chain with constant independent `.721` acceptance has sensitivity ceiling 3.58 | First measure actual `a200`, n-row verifier scaling, route union, and M1. Only then prototype exact branch semantics with adaptive fallback; gate on all four workloads at 200K | Structural hypothesis only; existing linear width-3 remains closed and no branch verifier exists | Pending | Exact target output, real history, unchanged quality | RESEARCH ONLY, BEHIND M1 | — |
| H19 | On accepted cycles, the first of two nextn calls can advance layer-45 state without computing its unused `dummy` output head. | `ds4_session_glm_spec_cycle_impl` formerly called `glm_graph_mtp_step(...,&dummy)` before the second draft; the committed path passes NULL and skips the unused head/readback/argmax | At `a=.721`, even deleting that entire first call is <=1.66 ms/cycle from the observed ~2.3 ms/call, roughly 3%; skipping only its head is smaller | Balanced ABBA plus paired same-process session/logit/cycle-state benchmark; then 200K and sustained gates | 2K: 30.448 -> 29.761 ms/token, +2.256%. 65K: 34.013 -> 33.404, +1.789%. Text identical across four arms at both contexts. Paired observable-state benchmark and 200K remain pending | Pending | Text passed; exact logits, cycle sizes/counts, positions, and counters pending in the paired benchmark | SHIPPED SMALL; ZERO S55 CREDIT | `b265e22` |
| H20 | Fuse GLM's 8-expert Q2 down projection with reduction instead of writing 16 expert rows and launching `sum8`. | Existing `kernel_mul_mv_id_q2_K_sum6_f32` loops over `args.nei0` and its encoder validates up to 8, while `direct_down_sum` is host-guarded to 6. GLM uses 2 tokens, 8 experts, 4096 outputs | Only one reduction dispatch per routed layer plus 22,020,096 bytes/cycle of scratch write/read across 42 layers; expert-weight traffic is unchanged, so no >=5% ceiling is established | After M1, direct operation timing; implement only an exact-order variant and kill unless whole-cycle projection is >=5% | One-line host widening is not bit-identical: direct code combines experts per lane before one `simd_sum`, while production performs one `simd_sum` per expert then scalar sum8 | Pending | Must preserve per-expert SIMD reductions and expert-order scalar addition | OPEN, PARKED BEHIND M1 | — |

## Validation Queue

Do not run while Claude owns the live Metal server.

- Full categorized matrix: `docs/research/test-matrix.md`.
- [ ] `make test`
- [ ] `make test-glm53-kda`
- [ ] `make test-mxfp4-metal`
- [ ] `DS4_TEST_MODEL=/Users/panda/models/gguf/GLM-5.3-Flash-UNCEN-Q2.gguf make test-metal-session-batch`
- [ ] `python3 gguf-tools/tests/test_glm53_quantize.py`
- [ ] Exact GLM MTP, snapshot, continued-prefill, tensor-equivalence, 100-case quality, long-context, and dedicated-server gates from the matrix

The prior core-suite pair is now attributed without rerunning it: both
assertions are the same `short_code_completion` selected-token mismatch, first
in `logprob-vectors` and then in the SSD cache-pressure test that deliberately
reuses that case. The suite default resolves `ds4flash.gguf` to a local
abliterated DeepSeek V4 artifact, while the fixture declares official checkpoint
`0731`. Preserve the failure as a fixture/model mismatch; it neither validates
nor falsifies the GLM-5.3 target and must not be hidden or disabled.

## Independent GPT-5.6 Sol Review

Max-effort read-only review completed against the live source family. It
confirmed three routed dispatches per MoE layer for both one and two rows,
rejected the 27-launch premise, and ranked measurement first: whole-command-
buffer GPU span, then routed/shared branch spans. Its provisional first code
candidate is reuse of the existing concurrent routed/shared FFN mechanism,
not a greenfield verifier or runtime. A second adversarial pass ranked
H6 > H9 >>> H10, closed H10 because two-row shared Q8 reuse is already live,
and proved H9 cannot close even the short-context gap alone. No speedup is
credited before A/B data. Full disposition: `docs/research/sol-max-review.md`.

The later 2K ablation does not price H6: source tracing shows its `shared`
mask skips only the one-row path, while the width-2 batch macro always executes
shared gate/up/down. A corrected batch-path measurement remains mandatory.

A second max-effort audit at shared HEAD `1e9d87b` independently proved the
fast-row coverage mismatch: at steady 2K only `routed` reaches the width-2
verifier; every other reported mask arm changes seed/workload without deleting
the named steady component. It also identified the accepted-cycle unused draft
head (H19) and proposed branch/tree block speculation (H18) as the only
same-checkpoint structural class plausibly large enough. Both remain zero-
credit hypotheses until `a200`, macro timing, M1, exactness, and 200K workload
gates. Full disposition: `docs/research/sol-max-review.md`.

## Live Baseline — Primary Decode Ledger

The palindromic run completed. Contract scoring discards the cold first 2K arm
and takes the lower measured rate at every other rung: 2K 33.01, 32K 26.64,
64K 24.15, 128K 23.67, 200K 22.48 t/s. Observed S55-200 is 22.48 t/s with
31.90% context decay. Acceptance/cycle and system telemetry remain pending.

## Non-Serializing Skip-Ablation — 2K Diagnostic

Claude's balanced `baseline / kda / routed / kda+routed / baseline` run kept the
normal command-buffer topology and measured generated-token time. Baselines
were 30.254 and 30.451 ms/token (+0.7% drift; midpoint 30.352). Source tracing
later invalidated the KDA arm: the 2K fast-row verifier never reads the KDA
mask. Routed-MoE removal saved 10.229 ms/token (33.7%)
and reached 49.69 t/s. Removing both reached 49.27 t/s, so their apparent costs
are not additive. The GPU-span control measured the combined arm at 20.309
ms/token wall and 20.160 ms/token GPU (99.3% busy), so the residual is GPU
work—not exposed host time. The ablation emits invalid text and is attribution
only. It proves routed work is
large, but does not identify bandwidth, access pattern, dequant arithmetic, or
width-2 reuse as the cause. It also proves that eliminating all routed work
would still miss 55 t/s at 2K.

Source tracing also bounds the unablated draft: no decode-ablation mask is read
inside `glm_graph_mtp_step`. A reject invokes one nextn layer; an accept invokes
two; the mean is `1+a` calls/cycle. The existing short-context timing log is
about 2.3 ms per call, so draft work cannot explain a 20.1 ms/token floor.
Because stale ablation outputs change later router selections, deletion deltas
are not additive component timings. The KDA and combined-arm labels must not be
used as KDA cost evidence.

## Review

In progress. `baseline-s55-200.md` contains both real 197,395-token passes and
rejects the script's mean-based score. H10 is closed and H9 is deferred. The
next gate is 200K acceptance/cycle plus whole-command-buffer GPU span and
the M1 equal-byte benchmark before any kernel edit. H6 branch timing follows
only if M1 does not reveal a larger access-pattern or kernel-efficiency ceiling.

The local M5 counter capability query returned `dispatch=false, stage=true`;
the only public counter set is `timestamp/GPUTimestamp`.
Shared head `b265e22` now contains H19. Its balanced 2K/65K results are small
and receive no S55-200 credit before the paired observable-state and 200K gates.
Static review found two documentation defects, not a runtime defect: the kill-
switch comment says `0` although rollback is `=1`, and the source/ledger call
the head about 4% of cycle traffic without a measured M1 denominator. Remove
both claims after the shared-tree lease; do not count that cleanup as speed.

The existing paired benchmark must be invoked with the committed switch name:
`speed-bench/glm53_mtp_head_bench MODEL DS4_GLM_MTP_DISCARDED_HEAD`.
Its compiled-in default names an older experiment. It compares exact generated
IDs, cycle sizes/counts, accept/reject counts, positions, and full float logits
after alternating 64-token chunks; it does not expose internal nextn cache
bytes directly, so call it paired observable-state, not raw internal state.

H6 source audit: the reusable concurrent-encoder state is concretely one-row.
Its gate/up PSO is `kernel_dsv4_shared_gate_up_swiglu_q8_0`, its down PSO is
the one-row Q8 matvec, and buffer guards cover one activation row. GLM width-2
instead uses `mul_mv_ext` two-row gate/up and Q8 down paths. Do not widen a
guard; a valid prototype must retain those two-row PSOs/arguments and alter only
the dependency graph after measured RG/SG/RD/SD overlap clears 5%.

The native `xctrace` Metal System Trace is the first profiler rung because it
requires no DS4 source change. If its export cannot close the macro ledger,
H1 falls back to start/end timestamps on successive compute encoders in the
same command buffer, with one ordinary completion wait. Either path must pass
the OFF/ON throughput gate.

## M1 Native-Stack Design

No second model parser, runtime, or inference harness is needed. Capture the
real 200K width-2 inputs through the existing graph dump using names containing
`glm_ffn_norm`, `glm_ffn_router_selected`, and `glm_ffn_router_weights`.
Use the repository GGUF parser for tensor offsets after adding its missing
BF16 type-table entry `(1, 2, "BF16")`. The exact checkpoint contains 42 routed
layers (3 through 44); one selected row reads 2,378,170,368 logical expert
bytes, so every M1 arm must process 4,756,340,736 logical bytes.

- A: one diagnostic checksum kernel reads 16 contiguous expert spans.
- B: the same kernel reads the 16 captured selected-expert spans.
- C: two production one-row routed-MoE calls using captured norm, IDs, weights.
- D: one production width-2 routed-MoE call using the same inputs.

Run ABBA/BAAB with cache-defeating layer rotation, one command-buffer completion
per arm, and report GPU elapsed time, logical bytes, unique bytes, GB/s, median,
mean, range, and C/D output equivalence. This is a diagnostic source change only
after Phase 2 and Phase 3; implement no kernel optimization unless the measured
removable whole-cycle ceiling is at least 5%.
