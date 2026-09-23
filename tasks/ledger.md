# Qwen3.8 selective Q8 gate/up — evidence ledger

Machine: MacBook Pro **M5 Max**, 40 GPU cores, **128 GiB**, Metal, macOS 27.0 (26A428).
Policy: resident inference, **SSD streaming OFF**, one ds4-server on :8000.
Worktree: `ds4-qwen38-ivan` @ `9b48358` (`fix(dashboard): style section groups, health strip, and icons`).
Binary: `ds4-server` mtime 2026-09-14T20:43:11-0400, 3,128,392 bytes. Code tree clean at freeze.

## Control (immutable — never overwrite)

| | path | bytes | mtime |
|---|---|---:|---|
| main | `~/models/gguf/Qwen3.8-Flash-Next-Uncensored-DS4-Q4KGateUp-Q8_0Down-MTP.gguf` | 96,114,378,688 | 2026-09-12T06:18:57-0400 |
| PLE | `~/models/gguf/Qwen3.8-Flash-Next-Uncensored-DS4-PLE-Q4_1.gguf` | 32,000,154,848 | 2026-09-12T06:02:30-0400 |

SHA256:
- main `33f3ea38de52b2918842cbe73a0f084abe33c3ed6fb83c7d652a6cdd5d5bb7be`
- PLE  `4b3d72a087cb8e4587a3807b3934bb413d4f04b0a51e06c515431ea6aec9b89d`

Launcher defaults (`run-qwen38-ds4.sh`): `--metal --mtp --mtp-draft 1 --ctx 262144 --prefill-chunk 1024 --power 100 --host 127.0.0.1 --port 8000`, no `--ssd-streaming`. KV `~/.ds4/server-kv/qwen38-flash-next-uncen-q4k-q8down`. Wired limit 118000 MiB.

## Architecture facts (source-verified)

| Fact | Where |
|---|---|
| 48 trunk layers, 512 routed experts, hidden 2560, expert intermediate 640, top-10 routing, ctx 262144 | HF `config.json` `text_config` |
| Layer pattern 3×GDN + 1×QSA: QSA ids **3,7,11,15,19,23,27,31,35,39,43,47** (12 QSA / 36 GDN) | `layer_types` |
| MTP is `blk.48` with its own Q4_K gate/up + Q8_0 down + Q8_0 shared | GGUF inventory |
| **Main GGUF contains 49 gate + 49 up tensors: 48 trunk + 1 MTP.** All are Q4_K; all 49 corresponding down tensors are Q8_0. Selective overrides reject `>=48` (F2). | `tasks/q8gu/main-tensors.json` |
| Shared experts already Q8_0; routers F32; PLE sidecar Q4_1 (1 tensor + 3 I64) | same |
| Q4_K→Q8_0 gate+up delta = 2×512×2560×640×0.5 B = **838,860,800 B ≈ 0.78125 GiB/layer** | geometry |
| S6/S8/S10/S12 ≈ +4.69 / +6.25 / +7.81 / +9.38 GiB | 6/8/10/12 × 0.78125 |
| Mixed per-layer routed-expert quantization is proven viable under **full residency**. DS4 historically had SSD-streaming failures for mixed expert precision (`#388`); SSD streaming is **excluded from this experiment regardless of current upstream mixed-quant SSD regression coverage**. | `#388`; upstream `QA_BEFORE_RELEASES.md` |
| **This GGUF was built by `gguf-tools/qwen4_exp_convert.py`**, not `qwen4_pack.py`. Provenance: `~/models/docs/qwen38-uncensored-ds4-quant.md` (2026-09-12). Recipe B: `--experts` Q4_K gate/up, `--experts-down q8_0`, `--no-ple`, native `libds4quants.dylib` K-quants, PLE from `qwen4_ple_sidecar.py`. Source HF rev `8336e613ea508b13c2159bd0f68965d97a606b95`. `qwen4_pack.py` is a different artifact (Q4_K down padded to 768). | quant.md |
| Selective overrides belong on `qwen4_exp_convert.py --q8-gate-up-layers` (parser landed). | convert.py |
| Existing DeepSeek mixed splicer copies donor tensors; does not requantize. | `gguf-tools/mixed/` |
| BF16 source present: `~/models/hf/Qwen3.8-Flash-Next-Uncensored` 335G, 131 shards — same uncensored checkpoint. | disk |

## Findings

| # | Finding | Status |
|---|---|---|
| F1 | Control GGUF matches the stated recipe: trunk+MTP gate/up Q4_K, down Q8_0, PLE external Q4_1. | **Settled** |
| F2 | MTP lives at `blk.48`. `--q8-gate-up-layers` must reject ≥48. Layer 24 plumbing is GDN (`linear_attention`), not QSA. | **Settled** |
| F3 | Cost model 0.78125 GiB/layer matches tensor dims `[2560,640,512]` gate and up. | **Settled** |
| F4 | Converter path found. Script default `~/repo/llama.cpp` does **not** exist; the 2026-09-12 tree is `~/bin/llama.cpp` @ `95ef7fc16` (`conversion/qwen4exp.py` present). PLE sidecar already defaults there. Python: conda env `rag-qa` (`~/miniforge3/envs/rag-qa`, torch 2.11.0). Import check succeeded. `--q8-gate-up-layers` already patches `tensor_force_quant` for `FFN_GATE_EXP`/`FFN_UP_EXP`. Tests live at `gguf-tools/tests/test_qwen4_exp_convert.py` (not repo-root `tests/`). Do not invent a new llama.cpp checkout. Control is now **stopped**; layer-24 convert is the next write (new path, never overwrite the two control GGUFs). | **Settled — convert env exists** |
| F5 | Memory labels (do not conflate): Metal allocated **102241.3 MiB = 99.84 GiB** on both pids; DS4 planner **99.20 GiB** (KV 8.33 + buffers 1.36 + resident model 89.50). Main GGUF file **89.51 GiB**; PLE file **29.80 GiB**. Swap **0** throughout load + 8K/32K/64K + 262K 3-rep. ~2 MiB swap appeared only on teardown unmap — ignore. | **Settled** |
| F6 | This pack used **no imatrix** (RTN Q4_K). Documented in quant.md § Not done. Phase 2 weighted Δerror must use BF16 source energy or a newly collected imatrix, not an Unsloth official-Qwen matrix. | **Settled for P2 method** |
| F7 | 8K and 32K `/stats.prefill_ns` cluster at **~0.26 s** (implied 30.7k / 124.9k t/s) with `cached=0` `source=none`. 64K scales: **~102–110 s / 616.8 t/s**. Treat 8K/32K prefill t/s as **instrumentation/last-chunk, not comparable**. Decode and MTP at those sizes remain usable. | **Settled — use 64K+ for prefill control** |
| F8 | Native 262K **3/3 done** on restarted pid **35489** (13:58:30–14:23:27). Median prefill **527.8 t/s**, decode **50.46 t/s**, MTP acc **85.3%**, TTFT **0.1075 s**, peak RSS **9.56 GiB**, Metal **102241.3 MiB**, swap 0, thermal fair, `finish=length`. First attempt (pid 62368) completed 1 rep at **614.9 / 47.05** then SIGKILL 137; preserved in `p0-longctx-262k-attempt1-sigkill.json`. Do not mix the two sessions into one median. Official P0 262K = the 3-rep rerun. | **Settled** |
| F9 | Layer-24 plumbing GGUF **loads and runs** on resident Metal, SSD off, existing PLE sidecar, fresh KV. File Δ **exactly 0.78125 GiB**. First 8K decode 51.57 vs control 56.85 **did not reproduce** on the P3 rerun (57.04). No evidence of an inherent one-layer Q8 decode tax. | **Settled — mixed Q8 gate/up is viable** |
| F10 | P2 is a **weight-reconstruction proxy**, not activation sensitivity. 48/48 layers, 96/96 tensors, no NaN/Inf. L24 Q8 bytes match plumbing GGUF **891,289,600/891,289,600**. Native Q4_K blk.0 matches control **471,859,200/471,859,200**. Source rev `8336e613ea508b13c2159bd0f68965d97a606b95`. Combined ΔNMSE **0.010285–0.011239** (max/min 1.093). Layer 0 mild outlier; no QSA/GDN split (medians 0.01037 vs 0.01034). **Do not auto-convert top-16.** | **Settled** |
| F11 | P3 kill-test: L0 (P2 rank 1) and L13 (rank 48) both **33/40** greedy-exact vs control; L24 (rank 31) **36/40**. Mismatch Jaccard 0.75. **Reconstruction ranking does not predict quality.** Stopped remaining P3 converts (L2/L3/L47/L38/L8). First L24 8K decode −9.3% did **not** reproduce (57.04 vs control 56.85). | **Settled — ranking approach terminated** |
| F12 | Hybrid v2 (own imatrix Q4_K gate/up on rev `8336e613`) **does not beat RTN A**. Valid B: greedy exact **20/40** vs layer-Q8 33–36/40; first-char 37/40; mean LCP 0.71 (coherent paraphrases, not collapse). Speed in band and small: 8K decode 57.74 vs 56.85 (~+1.6%); 64K 634.4 / 51.64 vs 616.8 / 49.10 (~+5.2% decode). Zero-tensor first convert (invalid B) 0/40 empty — **exclude entirely**; its speed numbers are invalid. **A = production. B = artifact only. C (BF16 PLE) = closed.** | **Settled — do not promote** |

## P0 control medians (this SHA, this GGUF)

`ds4_sha=9b483585f65e402d232ed4cf84d19d25cb08b506` · main `33f3ea38…5bb7be` · PLE `4b3d72a0…c9b89d`  
`--mtp --mtp-draft 1 --ctx 262144` · SSD off · swap 0 during every timed run  
Evidence: `tasks/q8gu/p0-longctx.json` (8K/32K/64K, pid 62368), `p0-longctx-262k.json` (262K 3-rep, pid 35489), `p0-longctx-262k-attempt1-sigkill.json` (discarded 1-rep). Control **stopped**.

| Prompt (actual) | n | prefill t/s | decode t/s | MTP t/s | MTP acc % | TTFT s | peak RSS GiB | thermal |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 8250 | 3 | **30704*** | **56.85** | 56.85 | **63.2** | **0.0446** | 4.87 | nominal |
| 32826 | 3 | **124882*** | **57.50** | 57.50 | **72.2** | **0.0462** | 12.10 | fair |
| 65594 | 3 | **616.8** | **49.10** | 49.10 | **77.1** | **0.1540** | 12.71 | fair |
| 261130 | **3** | **527.8** | **50.46** | 50.46 | **85.3** | **0.1075** | 9.56 | fair |

\*F7: 8K/32K prefill t/s not comparable. Official long-ctx prefill controls: **64K 616.8 t/s** and **262K 527.8 t/s** (3-rep). Attempt1 single 262K rep was 614.9 t/s after the 8K–64K session — not pooled. Decode 56.9 → 57.5 → 49.1 → 50.5 t/s.

## Phase status

| Phase | Status | Evidence |
|---|---|---|
| P0 freeze (identity) | **done** | `tasks/q8gu/freeze.json` + `main-tensors.json` |
| P0 baseline 8K/32K/64K | **done (3-rep medians)** | `p0-longctx.json` |
| P0 baseline 262K | **done (3-rep medians)** | `p0-longctx-262k.json`; F8 |
| P1 converter selector | **landed (parser)** | `--q8-gate-up-layers`; `gguf-tools/tests/test_qwen4_exp_convert.py` |
| P1 plumbing GGUF (layer 24) | **done** | File + type inventory `p1-l24-tensors.json`; load/smoke `p1-l24-8k.json`. Server stopped. |
| P2 sensitivity | **done** | `p2-reconstruction.json`, `p2-ranking.csv`, `p2-summary.md`. F10. |
| P3 empirical | **done — ranking killed** | L24 + L0 + L13 only. `p3-summary.md`. F11. L2/L3/L47/L38/L8 **not converted**. |
| P4 greedy / P5 S6–S12 / P6 | **cancelled** | Ranking has no utility. Do not build reconstruction-ranked hybrids. |
| Quality Hybrid v2 | **done — do not promote** | Valid B A/B’d. Exact 20/40, speed in band. Production stays RTN A. C closed. F12. |

## Rollback

```text
QWEN_DS4_MODEL=$HOME/models/gguf/Qwen3.8-Flash-Next-Uncensored-DS4-Q4KGateUp-Q8_0Down-MTP.gguf
QWEN_DS4_PLE=$HOME/models/gguf/Qwen3.8-Flash-Next-Uncensored-DS4-PLE-Q4_1.gguf
```

Never replace those two files. Candidates are extra paths + `QWEN_DS4_MODEL=` override. Finalist A/B uses a **fresh KV dir**.

## Closed: reconstruction-ranked selective Q8

Negative for ranking utility, positive for plumbing. Production stays **RTN Q4_K gate/up + Q8_0 down**. Keep `...-L24Q8GU.gguf` as the hybrid-load proof. Do not convert more P2-order layers. Do not build S6–S12 from that ranking.

## Closed: Quality Hybrid v2 (2026-09-15)

**A = production/control. B = retained experimental artifact only. C = closed.**

Do not promote B. Do not convert BF16 PLE. Do not overwrite the two control GGUFs. Launcher default remains RTN A.

Valid B (`...-MTP-imatrix.gguf`, reconvert2, VERIFY_V2_OK): greedy exact **20/40**, first-char 37/40, mean LCP 0.71. Speed in band (8K decode 57.74 vs 56.85; 64K 634.4 / 51.64 vs 616.8 / 49.10). Quality loss vs layer-Q8 33–36/40 is too large for ~+1.6% / ~+5.2% decode.

Invalid B (all-zero Q4_K, 0/40 empty): **exclude entirely**. Do not cite its 8K 66.43 / 64K 55.09 as speed.

Evidence: `tasks/q8gu/p3-v2-reconvert-{quality,longctx,dequant}.json`; invalid-B `p3-v2-imatrix-*`. Control mtime still 1789208337.

**Stop (2026-09-15).** RTN A ships. No C, no further imatrix, no ranked Q8, no more GGUF writes. Housekeeping and future BF16/NLL are separate tracks only if explicitly opened.
