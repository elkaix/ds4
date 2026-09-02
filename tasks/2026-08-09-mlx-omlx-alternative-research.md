# Is there an MLX / oMLX version of DeepSeek V4 Flash that runs better than ds4?

Researched 2026-08-09. Short answer: **MLX versions exist and they are real, but they are
slower than what you run today. Stay on ds4.**

## The headline comparison

The best MLX build was benchmarked by its own author on **the same machine as yours** —
MacBook Pro M5 Max 128 GB, 40-core GPU — so this is a like-for-like read, not an
extrapolation. [1]

| depth | oMLX 2.4bit-mixed gen | **ds4 (ours, today)** | winner |
|---|---:|---:|---|
| short / 1k | 36.1 tok/s | **44.58 tok/s** | **ds4 +23%** |
| 4k | 33.8 tok/s | — | |
| 8k | 33.1 tok/s | **36.19 tok/s** | **ds4 +9%** |
| 32k | 31.5 tok/s | ~30 tok/s | oMLX, marginally |

ds4 wins where you spend most of your time and loses slightly at deep context, where MLA
compression and the sparse indexer let MLX decay more gracefully. Note our 44.58 already
includes Ivan's fusions and the DSpark removal from today — the gap against a stock ds4
would be narrower.

Prefill is not close. oMLX reports **498 tok/s at 1k falling to 337 at 32k**, with a
**97-second** time-to-first-token at 32k. [1] Our ds4 measured 451–520 tok/s and, more
importantly, has the disk KV cache — so a repeated prefix costs nothing on a second turn
and oMLX pays that 97 s again.

## The blocker you would hit first

**Stock `mlx-lm` cannot run this model at all.** Verified two ways:

- Upstream `ml-explore/mlx-lm` `main` ships `deepseek.py`, `deepseek_v2.py`,
  `deepseek_v3.py`, `deepseek_v32.py` — **no `deepseek_v4.py`**. [2]
- Your local envs (`mlx`, `mlx-native`) are both on **0.31.3, which is the current PyPI
  latest** [3], and neither has the model file.

The model card says so outright: *"mlx-lm does not support the `deepseek_v4` architecture.
There are half a dozen open PRs ([mlx-lm#1189] among them) and I haven't checked myself
whether any of them work."* [1]

So the runtime is **oMLX**, not mlx-lm — which is the `omlx` you already run on :8010,
but it needs **oMLX 0.5.7+**. oMLX is not currently installed in any of your conda envs
(checked `mlx`, `mlx-native`, `local-llm`) and nothing is listening on 8010 right now.

## What is actually available

15 MLX-tagged DeepSeek-V4-Flash repos exist on HF. The ones that fit 128 GB:

| repo | size | notes |
|---|---:|---|
| `mlx-community/DeepSeek-V4-Flash-0731-OptiQ-2bit` | 92.5 GB | oQ2 |
| **`mlx-community/DeepSeek-V4-Flash-0731-2.4bit-mixed`** | **92.8 GB** | **best documented; the benchmarks above** |
| `Jundot/DeepSeek-V4-Flash-0731-oQ2e` | 94.3 GB | oQ2 enhanced |
| `Jundot/DeepSeek-V4-Flash-0731-oQ2e-mtp` | 105.2 GB | with MTP draft stack |
| `mlx-community/DeepSeek-V4-Flash-2bit-M-DQ` | 106.0 GB | pre-0731 |
| `ddalcu/…-MLX-Serve-mixed-2-3-8bit` | 115.3 GB | tight |
| `Vontra/DeepSeek-V4-Flash-0731-MXFP4-MLX` | 167.1 GB | **does not fit** |

All are **larger than our ds4 Q2 (86.7 GB)**. The 2.4bit-mixed peaks at 79.8–80.8 GB
resident vs our 85.13 GB planned, so memory is roughly a wash despite the bigger file.

## Quality

The card publishes an honest quality number, which most quant repos do not:
**mmlu_pro 0.573 at 2.44 bpw vs 0.647 for the bf16 hosted API** (n=600, thinking off,
greedy, same prompts both rows, ±2 points standard error). [1] That is a real ~7-point
drop from quantization — expected at 2-bit, and no better or worse than what our Q2 pays.
Nobody has published a comparable mmlu_pro figure for the ds4 Q2 recipe, so the two
cannot be ranked on quality from public data.

## Where MLX genuinely wins

**Concurrency.** oMLX continuous batching scales well: 36.1 → 43.9 → 64.1 → 83.2 tok/s at
batch 1/2/4/8, a 2.30× throughput multiplier at batch 8. [1] ds4's batched session support
exists but is not what antirez tunes for, and our own transaction work targets the
single-session path.

If you ever serve several clients at once, that is the argument for MLX. For one
interactive session — your actual use — it is irrelevant.

## Verdict

Not worth switching. You would download 93 GB, install a new runtime, lose the disk KV
cache, lose ~23% decode at short context and a large factor on repeated-prefix TTFT — to
gain graceful decay past 32k and batching you do not use.

Two things that would change this:
1. **`deepseek_v4` landing in stock mlx-lm** (watch [mlx-lm#1189]) — removes the
   extra-runtime cost and opens the mlx-lm speculative-decoding path.
2. **Needing concurrent serving**, where 2.3× at batch 8 beats anything ds4 offers today.

Worth knowing for context: antirez's own published figure for M5 Max is
"~500 t/s prefill and ~35-40 t/s decoding" on ds4. [4] We measure 44.58 t/s, above his
range, because of Ivan's Metal fusions plus dropping DSpark.

---

[1]: https://huggingface.co/mlx-community/DeepSeek-V4-Flash-0731-2.4bit-mixed/raw/main/README.md
[2]: https://api.github.com/repos/ml-explore/mlx-lm/contents/mlx_lm/models
[3]: https://pypi.org/pypi/mlx-lm/json
[4]: https://antirez.com/news/167
[mlx-lm#1189]: https://github.com/ml-explore/mlx-lm/pull/1189
