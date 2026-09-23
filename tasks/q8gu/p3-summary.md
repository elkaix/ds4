# P3 empirical — reconstruction-ranking kill-test

Quality is **greedy agreement** vs frozen control continuations (40 `prompts.jsonl` cases, temp 0, 32 tokens, thinking off). HTTP has no logprobs; this is not official NLL.

Speed: 8K 3-rep decode (check the earlier L24 −9.3% scare) and 64K 3-rep prefill/decode (honest long-ctx anchors). 262K deferred. L0 and L13 GGUFs deleted after test. L24 plumbing file kept.

Control anchors (P0): 8K decode **56.85 t/s**, 64K **616.8 / 49.10 t/s**.

## Results

| layer | P2 rank | type | exact/40 | first/40 | mean LCP | 8K decode | 64K prefill | 64K decode | 8K MTP |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| control | — | Q4_K | 40/40 | 40/40 | 1.000 | **56.85** | **616.8** | **49.10** | 63.2 |
| 24 | 31 | GDN | **36/40** | 39/40 | **0.928** | 57.04 | 449.8 | 45.56 | 72.2 |
| 0 | 1 | GDN | 33/40 | 39/40 | 0.900 | 55.10 | 530.3 | 46.81 | 70.3 |
| 13 | 48 | GDN | 33/40 | **40/40** | 0.920 | 52.47 | 554.7 | 49.41 | 63.2 |

L0 vs L13 mismatch sets share 6/8 cases (Jaccard 0.75). Exact-match counts are **identical**.

## Verdict

**Reconstruction ranking does not predict greedy quality.** Top (L0) and bottom (L13) of the flat P2 curve are indistinguishable on the only quality metric this HTTP surface can score. Mid-pack L24 is *closer* to control than either extreme.

Do **not** convert L2, L3, L47, L38, L8. Do not build an S6–S12 hybrid from this ranking.

The earlier L24 8K decode −9.3% (51.57 vs 56.85) **did not reproduce** (57.04 this session, MTP 72%). Treat that as thermal/session noise, not a systematic Q8 gate/up tax. 64K prefill in these P3 sessions is slower and noisier than the original P0 616.8; L13 still posted one 618 t/s rep. Do not attribute a prefill tax to Q8 without a same-protocol control re-run.

## Implication

Q8 gate/up remains **technically viable** (F9). Selective Q8 as a *quality* project is unsupported by P2+P3: the remaining precision lever exists, but this ranking cannot tell you which layers to spend 0.78 GiB on. Next GiBs are better spent on something other than reconstruction-ranked routed gate/up — unless a real activation/NLL fixture (score_official vs local BF16 continuations) is collected and the experiment is redesigned around that.

Evidence: `p3-control-quality.json`, `p3-L{0,13,24}-quality.json`, `p3-L{0,13,24}-longctx.json`.
