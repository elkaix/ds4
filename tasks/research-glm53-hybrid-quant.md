# GLM-5.3 Flash hybrid quantization for ds4 on M5 Max 128 GB (2026-09-22)

Question: which mixed-precision layout gives the best accuracy per unit of decode
speed on this machine, using only formats ds4 can load and run.

## Answer

Move bytes from the always-read dense path into the rarely-read routed experts.

- **Dense tensors** are read in full on every token.
- **Routed experts** are read at 8/288 of their size per token.
- **One more Q4_K expert layer** costs 56.6 MB/token and 1.90 GiB of memory.
- **The largest measured quality gain** on this model comes from upgrading experts:
  - 3 layers: avg_nll 0.4741 → 0.4518.
  - 7 layers: 0.4368.
  - Full Q4_K (ablit, old scorer): 0.2944.
- **What to fund it with:** Q8_0 → Q4_K on the dense KDA/MLA projections. This frees
  1.14 GB/token (stage A) up to ~2.9 GB/token (stages A–D, F), plus the matching
  resident memory.
- **Keep the shared expert at Q8_0.** Upstream's own Q4-dense test found it the most
  sensitive dense tensor [ds4-618]. On this machine it also keeps two Q8_0-only fused
  kernels.

This is a hypothesis for this checkpoint until our 100-case fixture measures it.
Upstream's AProjQ4 quality result did not transfer to our DeepSeek checkpoint
(`deepseek.quality.aprojq4k-nll`: +2.41% avg_nll).

## Evidence

### External

| Source | Model | Layout | Measured |
|---|---|---|---|
| [ds4-618] | DeepSeek V4 Flash, M1 Pro | all-Q4 dense, imatrix | avg_nll 0.3870 vs Q8 0.3732 (+1.38% ppl) |
| [ds4-618] | same | AProjQ4 + shared expert Q8 + output Q8 | avg_nll 0.3831 (+0.99% ppl), top-1 unchanged, 2.8–3.2× decode (SSD-streamed) |
| [aj9o9] | GLM-5.3-Flash, llama.cpp | experts IQ2_XXS; attention, dense FFN, shared expert, embeddings/head at Q6_K; KDA gates and indexer Q8_0 | KLD 0.7072 vs BF16 |
| [aj9o9] | same | experts IQ2_S/IQ3_XXS, rest unchanged (+25 GB) | KLD 0.3557: expert bits roughly halve the divergence |
| [rmq] | GLM-5.3 Flash, MLX, M3 Ultra | experts Q4; all always-active paths Q8 | 0.1B fixture only: top-1 94.6% vs 76.0% for uniform Q4 |
| [tq4] | small dense model, Metal | Q4_K vs Q8_0 ffn_down | ppl +0.5 points, decode 187 vs 176 t/s (native Q4_K kernel is faster than Q8_0) |

[aj9o9] also measures that GLM-5.3-Flash is 94.9% routed experts: 304.4B of 320.6B
parameters. Everything else totals 8.9B parameters.

### This machine (ledger and today's session)

- **Decode is limited by memory bandwidth.**
  - 10.479 GB/token at 614 GB/s gives a ceiling of 58.6 t/s. Measured: about 56% of the bus.
  - Q8_0 dense tensors are 59% of the bytes per token (`glm.perf.decode-bytes-o1-614`).
- **Measured O1 stage costs** (`glm.perf.stage-ablation-m5-2k`):
  - routed experts 9.62 ms;
  - KDA q/k/v 5.47 ms;
  - attn_out 3.19 ms;
  - KDA output 2.99 ms;
  - shared expert 2.54 ms.
- **ds4 GLM loader:** accepts bf16/q8_0/q4_K/q4_0 for every dense role (`ds4.c:4904`).
  - No Q5_K or Q6_K for dense roles. Routed experts are different: the loader and the GLM routed
    kernels accept Q5_K gate/up and Q6_K down (`ds4.c:4990-4998`, `ds4_metal.m:38843-38857`), but
    gguf-tools cannot emit either (`gguf-tools/quants.c:51-52`). The Q6_K layout in [aj9o9] still
    cannot be built here.
  - The KDA fused paths need BF16 or all-Q8_0 q/k/v. O1 already mixes Q4_K q/k with
    Q8_0 v, so stage A loses no fusion.
- **Fresh-scorer control (f95923c8):** O1 = 0.4525 token-weighted avg_nll, 88/100
  first_match. Tensor API on and off give identical scores.

### Byte arithmetic (verified by the quantizer's dry run)

- Q8_0 = 34 B per 32 weights; Q4_K = 144 B per 256 weights.
- One 4096×8192 KDA projection saves 16,777,216 B.
- Stage A (kda_v + kda_output, 34 layers = 68 tensors) saves 1,140,850,688 B per token
  and 1.06 GiB resident. Dry-run delta: exactly 1,140,850,688 B.
- One Q4_K expert layer, per token:
  - gate/up: 2 × (8 × 4096 × 2048) × (4.5 − 2.0625) bits;
  - down: 8 × 4096 × 2048 × (4.5 − 2.625) bits;
  - total 56.6 MB/token and 1.898 GiB resident.
- Stage A's bandwidth saving would fund about 20 expert layers, but its memory saving
  funds only about 0.56. **Memory, not bandwidth, limits expert upgrades.**

## Recommended ladder (each step gated on the 100-case fixture plus the paired worst case)

1. **A: kda_v and kda_output to Q4_K** — adopted 2026-09-22.
   - Predicted decode ×1.077–1.086; measured ×1.080 at 2K, ×1.110 at 32K, ×1.107 at 100K;
     prefill unchanged.
   - avg_nll 0.4525 → 0.4557 (paired CI includes 0).
   - Data: `tasks/data/glm-stageA-ab-20260922` and `-mid` (32K rerun).
2. **Stage 2: experts L20-23 to Q4_K (g17-23)** — adopted 2026-09-22.
   - avg_nll 0.4557 → 0.4399 (−0.0158, paired CI95 [−0.0208, −0.0110]); first_match 87 → 89.
     Sweep + stage A predicted 0.4400.
   - Decode vs stage A (steady, paired): ×0.992 at 2K, ×0.994 at 32K, ×0.944 at 100K (n=2);
     predicted ×0.978. Prefill unchanged.
   - Memory: +7.59 GiB; 110.25 GiB allocated at the first 262K start (derivation said 108.18) vs 115.23 GiB.
   - Data: `tasks/data/glm-stage2-ab-20260922` and `-mid`.
3. **B/C: MLA attn_output and the other MLA projections**, one at a time. These are the
   highest-risk dense tensors [ds4-618].
4. **D: dense FFN (layers 0–2)**, then **F: output head**. Keep the embedding, since it
   is not read in full on each token.
5. **Do not convert the shared expert** (E). Upstream reversed the same change, and it
   breaks two Q8_0-only fused kernels here.

## Not available or rejected

- **Q5_K, Q6_K experts:** loader and kernels exist; the gguf-tools emitter does not.
- **IQ2_S, IQ3_XXS:** no ds4 loader or kernel path on GLM. [aj9o9] uses them.
- **Q2_K gate/up everywhere:** IQ2_XXS 2.0625 → Q2_K 2.625 bpw costs 0.316 GiB per layer
  (4.832e9 gate/up weights × 0.5625/8), about 12.7 GiB across 40 layers, with no measured
  gain. It would move those layers off the generic IQ2_XXS routed kernel onto the GLM
  specialized one, which is the kernel lever L1 in `~/models/docs/ds4-m5max-systems-blueprint.md`.
- **imatrix for Q2:** no gain on this line (`glm.quality.imatrix-q2-no-gain`).

[ds4-618]: https://github.com/antirez/ds4/issues/618
[aj9o9]: https://huggingface.co/aj9o9/GLM-5.3-Flash-GGUF
[rmq]: https://github.com/raullenchai/rapid-mlx/blob/781fecb3e2db85d0319ef358cc9be45492e87c35/docs/engineering/performance/2026-09-12-glm53-rmq-mvp.md
[tq4]: https://github.com/thetom/turboquant_plus/blob/ba52ad107d1fdd02bc9be8fd85308226b75c905b/docs/papers/weight-compression-tq4.md
