# Task: Quality Hybrid v2 — **CLOSED / STOP**

**Stop.** No more work on this track. Housekeeping and BF16/NLL are **not** next — open them explicitly if ever.

**Frozen 2026-09-15:** **A = production/control (ships). B = retained experimental artifact only. C = closed.**

Do **not** promote B. Do **not** convert C (BF16 PLE). Do **not** overwrite control. Do **not** treat the zero-imatrix run as a speed result (empty completions).

Production remains:
`~/models/gguf/Qwen3.8-Flash-Next-Uncensored-DS4-Q4KGateUp-Q8_0Down-MTP.gguf`
+ PLE `~/models/gguf/Qwen3.8-Flash-Next-Uncensored-DS4-PLE-Q4_1.gguf`

Launcher default (`run-qwen38-ds4.sh`) still points at A. Control mtime **1789208337**, size 96114378688.

## Why

B’s speed delta is in-band and small (~**+1.6%** 8K decode, ~**+5.2%** 64K decode). Quality regression is material: greedy exact **20/40** vs layer-Q8 **33–36/40**. 37/40 first-char and 0.71 mean LCP mean coherent paraphrases, not collapse — that does **not** restore exact behavioral fidelity.

## A/B (reconvert2, 2026-09-15 20:58)

| metric | A (RTN control) | B (imatrix reconvert2) | invalid B (zeros) |
| --- | --- | --- | --- |
| greedy exact / 40 | refs | **20** | 0 (all empty) — **exclude** |
| first-char / 40 | — | 37 | 0 |
| mean LCP norm | — | 0.710 | 0 |
| 8K decode t/s | 56.85 | 57.74 | 66.43 (invalid) |
| 64K prefill / decode | 616.8 / 49.10 | 634.4 / 51.64 | 631.7 / 55.09 (invalid) |
| empty completions | 0 | 0 | 40 |

B GGUF kept: `~/models/gguf/Qwen3.8-Flash-Next-Uncensored-DS4-Q4KGateUp-Q8_0Down-MTP-imatrix.gguf` (96.11G). Not production.

Evidence: `tasks/q8gu/p3-v2-reconvert-{quality,longctx,dequant}.json`. Invalid-B evidence kept as `p3-v2-imatrix-*` (do not reuse that KV).

Also closed earlier: reconstruction-ranked selective Q8 (P2–P6). Keep `L24Q8GU` as hybrid-load proof only.
