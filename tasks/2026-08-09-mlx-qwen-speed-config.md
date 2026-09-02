# Fastest MLX config for Qwen3.6 27B / 35B-A3B on this M5 Max

Researched 2026-08-09. Companion to [`2026-08-09-mlx-omlx-alternative-research.md`](2026-08-09-mlx-omlx-alternative-research.md).

## Verdict

**Every real speedup needs a different checkpoint. None of them is a flag you flip on the
weights already in `~/models/mlx/`.**

Two separate recommendations, because the two models sit in different regimes:

| model | biggest lever | expected | cost |
|---|---|---|---|
| **Qwen3.6-27B** (dense, memory-bound) | MTPLX + its 4-bit MTP checkpoint | **~17.6 → 40–50 t/s** | 19.9 GB download |
| **Qwen3.6-35B-A3B** (MoE, 3B active) | 8-bit → 4-bit; speculation adds little | ~2× from the quant, **+9%** from MTP | 21 GB download |

Speculative decoding is worth 2–3× on the **dense** 27B and roughly **+8.8%** on the MoE
[[5]]. A 3B-active MoE is already cheap per token, so there is far less for speculation to
recover. Do not buy one story for both.

---

## 1. True on this machine right now — verified, no download

```
mlx            0.31.2  (PyPI latest 0.32.0)      ← upgrade available
mlx-lm         0.31.3  (PyPI latest 0.31.3)      ← already current
omlx           0.5.5   (tap stable 0.5.7)        ← upgrade available
iogpu.wired_limit_mb = 118000                    ← already raised, nothing to do
~/models/mlx/Qwen3.6-27B-8bit        28 GB
~/models/mlx/Qwen3.6-35B-A3B-8bit    35 GB
```

Baseline to anchor everything against: **17.6 t/s** for `Qwen3.6-27B-8bit`, measured on
this machine 2026-07-27 (`~/.claude/skills/local-llm-serving`). That number matches the
independent M5 Max "without MTPLX → 17 tok/s" figure [[5]] almost exactly, which is why the
third-party deltas below transfer here rather than being someone else's hardware.

**Do not `pip install -U mlx` in the `mlx` conda env.** It holds the
`kernelpool/mlx-lm@add-hy3-preview` fork that Hy3-oQ2 depends on; a reinstall breaks it
(standing prohibition in `~/CLAUDE.md`). MTPLX ships its own Python engine via Homebrew, so
it sidesteps this entirely. Upgrade `mlx` only inside `mlx-native`, or not at all.

**Port map on this box:** ds4-server 8000, `start-mlx.sh` 8080, oMLX 8010, laguna 8091.
MTPLX and dflash both default to **8000** — pass an explicit port or you will collide with
ds4-server.

---

## 2. The cheapest certain win: stop running 8-bit

Decode on a dense 27B is bandwidth-bound — every token reads the whole weight set. 8-bit
reads roughly twice the bytes of 4-bit per token. This needs no new runtime, no new engine,
and no acceptance-rate gamble:

| | on disk | per-token read |
|---|---:|---|
| `Qwen3.6-27B-8bit` (current) | 28 GB | 1.0× |
| `mlx-community/Qwen3.6-27B-4bit` | 16.1 GB | ~0.57× |

Say this before any engine swap: it is the only lever here whose mechanism is arithmetic
rather than a heuristic that can fail on your workload.

Quality caveat worth knowing: **fewer bits is not automatically faster**, and group size
matters as much as bit width. Measured on this machine 2026-07-27 on Laguna — oQ4e
(group_size 128) did 62.4 t/s vs NVFP4 (group_size 16) at 44.2 t/s on the same engine. The
MTPLX 27B build is 4-bit/group 32, the 35B build 4-bit/group 64.

---

## 3. MTPLX — the 27B answer

MTPLX uses the **model's own built-in MTP heads** as the drafter: no second model, no extra
RAM, exact rejection sampling (Leviathan/Chen with residual correction), so `temperature=0.6,
top_p=0.95` behaves identically to normal decoding [[1]].

It **refuses** to bolt an MTP sidecar onto an arbitrary trunk — quoting the README: *"Use a
complete model that already includes its matching MTP weights."* Your `Qwen3.6-27B-8bit` has
no MTP heads, so this is a download, not a flag.

```bash
brew install youssofal/mtplx/mtplx
mtplx pull Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed-V2
mtplx tune --model Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed-V2 --retune
mtplx serve --model Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed-V2 --port 8020
```

`mtplx tune` is the part that matters: it measures AR baseline vs depth 1/2/3 **on this Mac**
with fans pinned, and **saves nothing if no depth beats the baseline** [[1]]. That is exactly
the honesty the DSpark result (§6) says to demand.

Catalog and sizes (HF API, 2026-08-09):

| repo | size | quant | notes |
|---|---:|---|---|
| `Qwen3.6-27B-MTPLX-Optimized-Speed-V2` | **19.9 GB** | 4-bit, g32 | README's coding pick; dynamic hybrid, sensitive tensors up to 16-bit |
| `Qwen3.6-27B-MTPLX-Optimized-Speed` | 16.4 GB | — | original, smaller, lower quality |
| `Qwen3.6-27B-MTPLX-Optimized-Quality` | 30.0 GB | — | quality build |
| `Qwen3.6-35B-A3B-MTPLX-Optimized-Speed` | **21.0 GB** | 4-bit, g64 | |
| `Qwen3.6-35B-A3B-MTPLX-Optimized-Balance` | 29.6 GB | 6-bit, g64 | |

Modes [[1]]: **Turbo** (NAX verify kernels; auto-selected for the quantized 27B) ·
**Sustained** (chunked prefill, 16K–200K prompts — the agent-work mode) · **Sustained Max**
(fans pinned) · **Burst** (short benchmarks only).

Serves OpenAI **and** Anthropic (`/v1/messages`) APIs, plus `/metrics` and a default-on SSD
session cache. Qwen's own guidance, echoed by MTPLX: keep presence/frequency penalties at
**0** for coding — nonzero penalties break MTP exactness [[1]][[8]].

---

## 4. dflash-mlx — the alternative, external drafter

Block-diffusion drafter proposes 16 tokens in one pass; target verifies in one pass; lossless
[[2]]. Needs a **separate draft model**, but they are small: 3.5 GB for the 27B, 0.8 GB for
the 35B MoE.

Its registry only maps `mlx-community/Qwen3.6-27B-4bit` and `-35B-A3B-4bit` — your 8-bit dirs
are unregistered and rejected unless you pass `--draft` explicitly, and the draft was trained
against a specific target, so overriding is not free.

```bash
pip install dflash-mlx
dflash serve --model mlx-community/Qwen3.6-27B-4bit \
             --draft z-lab/Qwen3.6-27B-DFlash --port 8020
```

Useful flags: `--chat-template-args '{"enable_thinking":false}'` (thinking is on by default
and, as on ds4, it discards your sampling params), `--fastpath-max-tokens 64` (skip
speculation on short replies), `--prefix-cache-l2` + `--prefix-cache-l2-dir` for SSD spill,
and `curl :8020/metrics` for live acceptance / `tokens_per_cycle`.

Its verify-specialized int4 qmm auto-selects the **Metal 4 NAX M=16 path on M5-class GPUs**
(`applegpu_g17*`) — so this machine gets the fast path, not the older simdgroup fallback [[2]].

---

## 5. Third-party numbers, with protocols — read the disagreement

These do **not** belong in one table. Three sources differ by ~3× on the same model because
their protocols differ.

**dflash-mlx README** — M5 Max 64 GB, MLX 0.31.1, stock `mlx_lm.stream_generate` baseline,
3 repeats, median, 60 s cooldown, single math prompt [[2]]:

| model | ctx | baseline | DFlash | speedup | acceptance |
|---|---:|---:|---:|---:|---:|
| Qwen3.6-27B-4bit | 1024 | 33.26 | 98.05 | 2.95× | 84.7% |
| Qwen3.6-27B-4bit | 8192 | 26.03 | 79.12 | 3.04× | 83.5% |
| Qwen3.6-27B-4bit | 16384 | 21.50 | 60.77 | 2.78× | 84.4% |
| Qwen3.6-35B-A3B-4bit | 1024 | 138.26 | 300.33 | 2.20× | 91.0% |
| Qwen3.6-35B-A3B-4bit | 8192 | 133.20 | 177.45 | 1.33× | 87.0% |

> Its 138 t/s MoE **baseline** is ~3× above every other source below. Internally consistent
> with its own Qwen3.5 rows, so it is a protocol artifact, not a typo — treat the *ratios*
> as transferable and the *absolutes* as not.

**largitdata / ywchiu mlx_benchmark_lab** — M5 Max 64 GB, `Qwen3.6-35B-A3B-4bit`, all engines
as local servers, **prefix caching explicitly disabled**, 5 runs, median [[3]]: oMLX leads
from 4K onward and holds **82.1 t/s at 32K**; dflash-mlx peaks at **167.3 t/s** short-context
but collapses to **12.6 t/s at 32K**; rapid-mlx is a middle ground; mlx-vlm is the multimodal
one. That collapse contradicts dflash's own 16K row — different build, different flags,
unresolved. **If your prompts are long, this is the risk.**

**Real agent workload, M5 Max, 35B-A3B** [[5]]: 89–90 t/s without MTP → 97.7 t/s with,
**+8.8%**, with draft acceptance decaying **82% → 69%** as the Pi coding agent sent varied
requests. DevoxxGenie independently reports 35B-A3B at 89 t/s under MTPLX [[6]].

**27B dense, M5 Max** [[5]]: 17 t/s → 50 t/s, 2.24×, TTFT 0.59 s → 0.25 s vs LM Studio.

Excluded: LLMCheck's "~52 t/s on M5 Max" — its own page labels it an *estimate*, not a
measurement.

**Adjacent, not yet usable:** Asher Feldman's k-quant support for MLX reports 1.37× prefill /
1.24× decode for `Qwen3.6-35B-A3B Q4_K_XL` and 1.18× / 1.29× for `Qwen3.6-27B Q4_K`, vs
llama.cpp on bit-exact weights, M5 Max 128 GB — with >2× better quality per bpw than MLX
affine quantization [[4]]. But: *"Pre-built kquant MLX checkpoints aren't published yet
(pending upstream acceptance)."* Watch it; cannot act on it today.

---

## 6. The prior you already own — do not skip the A/B

You proved **today, on this machine**, that a speculative-decoding feature marketed as a
speedup cost **16% of decode**: DSpark on ds4 showed 88.20% acceptance yet `net_saved` −21%,
because it drafted on only 19% of cycles (0.164 accepted tokens/cycle) while
propose+verify+replay ran on every cycle. Twelve interleaved samples, zero distribution
overlap.

dflash's own roadmap admits the same failure mode — *"tool-call regime auto-fallback: switch
to target-only AR when speculative surplus goes negative on structured outputs"* [[2]] — and
the field measurement above shows acceptance decaying 82% → 69% on exactly an agent workload.

So: the headline 2.24× is a clean short-prompt benchmark. **Your** workload is long-context
tool-calling, which is the regime where speculation degrades. Adopt only after an interleaved
A/B — `tasks/dspark-ab.sh` is the harness, and it already prints `feature_active` per run so
the toggle cannot silently no-op (that trap cost two sessions).

Also carry forward: **assert thinking is off** before trusting any temp-0 speculative
measurement. Both dflash and ds4 default to thinking, and thinking mode discards request
sampling params.

---

## 7. Do this, in order

1. `brew upgrade omlx` (0.5.5 → 0.5.7) — free, already-installed engine, no model change.
2. `hf download Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed-V2 --local-dir ~/models/mlx/Qwen3.6-27B-MTPLX-Speed-V2` (19.9 GB, ~10 min).
3. `brew install youssofal/mtplx/mtplx && mtplx tune --retune` — it reports honestly if no depth wins.
4. Interleaved A/B, **port 8020**, against the current 8-bit baseline, with thinking off and
   `feature_active` printed. Only then swap the daily model.
5. Only if MTPLX disappoints on long context: pull `mlx-community/Qwen3.6-27B-4bit` +
   `z-lab/Qwen3.6-27B-DFlash` (19.6 GB total) and repeat step 4.

Skip the 35B MoE speculative path unless step 4 goes well — +8.8% does not justify a second
runtime. For the MoE, take the 4-bit quant win and leave it on oMLX, which is the engine that
holds up at 32K [[3]].

---

## References

[1]: https://github.com/youssofal/MTPLX "MTPLX — native MTP speculative decoding for Apple Silicon"
[2]: https://github.com/bstnxbt/dflash-mlx "dflash-mlx — lossless DFlash speculative decoding for MLX"
[3]: https://www.largitdata.com/blog/mlx-inference-benchmark-apple-m5-max "rapid-mlx vs oMLX: MLX Inference Benchmark on Apple M5 Max"
[4]: https://www.linkedin.com/pulse/better-inference-quality-performance-mlx-apple-silicon-asher-feldman-ztm0e "Asher Feldman — Better inference quality and performance for MLX on Apple Silicon"
[5]: https://www.youtube.com/watch?v=Bd0q3cOWY90 "Execute Automation — Run Qwen3.6 27B 2x Faster on M5 Max (and the 35B MTP follow-up, youtube.com/watch?v=DhGUi4CtJRg)"
[6]: https://genie.devoxx.com/blog/mtplx-local-llm-apple-silicon "DevoxxGenie — MTPLX: 2x Faster Local LLMs on Apple Silicon"
[7]: https://news.ycombinator.com/item?id=48224337 "HN — oMLX Qwen3.6 throughput reports on M5 Pro"
[8]: https://huggingface.co/Qwen/Qwen3.6-35B-A3B "Qwen3.6-35B-A3B model card — recommended sampling parameters"
[9]: https://huggingface.co/api/models "Hugging Face model API — repo sizes and quantization configs, queried 2026-08-09"

- [1]: MTPLX README — 2.24× M5 Max, exact rejection sampling, no sidecar attach, modes, `mtplx tune`
- [2]: dflash-mlx README — benchmark table, registry, M5 NAX kernel path, roadmap admission
- [3]: largitdata — five-engine M5 Max comparison, prefix caching disabled
- [4]: Feldman — k-quant MLX, checkpoints not yet published
- [5]: Execute Automation — 17→50 t/s on 27B; 89→97.7 t/s (+8.8%) on 35B-A3B with acceptance decay
- [6]: DevoxxGenie — 35B-A3B at 89 t/s under MTPLX
- [8]: Qwen — penalties at 0 for coding
- [9]: HF API — all sizes and quant configs in §3
