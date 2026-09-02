# M5 Max DS4 inference review — SuperDeepseek V4 Flash

- Date: 2026-08-15
- Machine: MacBook Pro, Apple M5 Max (40-core GPU), 128 GiB unified memory
- Runtime: `antirez/ds4`, local branch `prod` at `0db5826`
- Model: `SuperDeepseek-V4-Flash-abliterated-MQ-DS4-Q2.gguf`

## Decision

Keep the current single-user server command and current binary. It already uses the
right maximum-throughput path: full Metal residency, Metal 4 Tensor kernels,
`--power 100`, warmed weights, and no SSD model streaming.

Do **not** merge or cherry-pick PR #799 for this workload. Its final quiet-machine
measurement is 32.0 aggregate tok/s for eight overlapping requests versus 37.5 tok/s
when requests run FIFO. It is useful for concurrent-client fairness because it improves
the current Metal batch path from 6.1 to 32.0 aggregate tok/s, not because it makes a
single user's inference faster.[^pr799-final]

Do **not** land PR #758 yet. A minimal, correctness-preserving port was tested on this
exact model. It looked positive at 8K, but failed the drift-controlled acceptance gate
and became inconsistent at 32K. No production source changes were retained.

The one promising machine-level experiment left is macOS **High Power Mode**. The Mac
is on AC but currently reports `powermode 0` (Automatic), while the machine advertises
High Power capability. Apple says High Power Mode permits higher fan speeds to maximize
performance during intensive workloads.[^apple-high-power] Enable it in System Settings
→ Battery → Energy Mode → High Power for the power adapter, then repeat the same ABAB
benchmark. This is a system-wide thermal/noise choice, so it was not changed silently.

## What was fetched

`git fetch` completed against `origin`. At the time of this report:

- Local `HEAD`: `0db5826`
- `origin/main`: `84cc882`
- Divergence: 35 local-only commits, 8 upstream-only commits
- Common base: `5b73804`

The upstream-only commits are:

| Commit | Subject | Relevance here |
|---|---|---|
| [`84cc882`][commit-84cc882] | ROCm DSpark enablement | Not a Metal speedup |
| [`5186e2c`][commit-5186e2c] | Keep Metal decode changes portable | Build portability |
| [`42033ee`][commit-42033ee] | Pipeline DFlash verification | Verification path |
| [`0ad494e`][commit-0ad494e] | Make exact fast paths automatic; trim experiments | Cleanup around the already-adopted M5 work |
| [`fd2d5e0`][commit-fd2d5e0] | Merge Ivan Fioravanti's exact Metal speedups | Performance bundle already carried by this local branch through its own integration |
| [`0e89a0e`][commit-0e89a0e] | Commit accepted DSpark verifier state directly | DSpark only |
| [`8a703b6`][commit-8a703b6] | Test real client disconnects | Server correctness |
| [`e9ded97`][commit-e9ded97] | Cancel work when clients disconnect | Server correctness |

This is not a fast-forward update. Merging `origin/main` only to chase performance would
mix eight upstream commits with 35 local server/model changes. The already-merged upstream
performance PR reports roughly 39.4 → 45.3 tok/s on M5 Q2; the local branch already
contains and logs that exact-path bundle.[^pr755]

## Current machine and runtime fit

Apple lists the 40-core M5 Max at 614 GB/s unified-memory bandwidth and supports 128 GB
of unified memory.[^apple-spec] This matters because Apple describes first-token/prefill
work as compute-bound, while subsequent-token generation is memory-bandwidth-bound.[^apple-mlx]
Apple's Metal guidance likewise ties faster prefill to large matrix multiplication and
decode to bandwidth/cache improvements.[^apple-metal]

The current server confirms the intended DS4 path:

```text
Metal device Apple M5 Max, 128.00 GiB RAM
Metal 4 tensor API enabled for Tensor kernels
tensor_matmul=on
resident model 80.76 GiB
KV 3.37 GiB + buffers 3.00 GiB + model 80.76 GiB = 87.13 GiB planned
```

The source checkpoint is a 304B-class, 43-backbone-layer, 256-routed-expert,
top-6 DeepSeek V4 Flash model with a configured 1,048,576-token context.[^model-card]
The local GGUF is 86,720,111,488 bytes and maps 80.76 GiB. It fits fully in unified
memory, so `--ssd-streaming` would add I/O to the model hot path and is not appropriate.

## Measured baseline with macmon

Artifacts: `/Users/panda/bench-results/ds4-superdeepseek-20260815/baseline-live/`

Contract:

```text
pp4096, tg512, depth 0 and 8192, concurrency 1, 3 runs
temperature=0, reasoning_effort=none, server disk cache bypassed
macmon interval=100 ms, package field=cpu+gpu+ane+ram
```

| Workload | Prompt processing | Generation | TTFR |
|---|---:|---:|---:|
| pp4096 / tg512 | 611.91 ± 24.47 tok/s | 39.61 ± 0.54 tok/s | 6,974 ms |
| pp4096 / tg512 at depth 8192 | 476.85 ± 49.07 tok/s | 33.45 ± 2.35 tok/s | 26,920 ms |

| Power window | Mean package W | Efficiency | Net efficiency | Prefill energy |
|---|---:|---:|---:|---:|
| depth 0 decode | 72.09 W | 0.550 tok/J | 0.564 tok/J | 114.02 J / 1K prompt tokens |
| depth 8192 decode | 50.64 W | 0.658 tok/J | 0.682 tok/J | 120.80 J / 1K prompt tokens |

Whole-run mean was 63.10 W, peak 121.20 W, idle 1.79 W, and measured energy
13.84 kJ over 219.4 seconds. macOS reported no thermal or performance warning.

The practical result is that **actual active context**, not the configured context ceiling,
is the main latency lever: decode dropped from 39.61 to 33.45 tok/s after an 8K prefix.
Lowering `--ctx` alone only lowers capacity and allocated buffers; shorten or cache the
prompt when latency matters.

## Local PR #758 experiment

PR #758 stages 16 indexed-attention K/V rows per threadgroup and also proposes a fused
top-512 kernel. Its author reports +2.94% mean 8K prefill, exact logits, and large thermal
variation from 503.9 to 722.6 tok/s.[^pr758]

Only the independent rb16 attention change was ported into a temporary worktree. The
top-512 portion was excluded because the current branch already contains the upstream
removal of the broken `stream512` path; resurrecting that conflict is not a minimal or
safe inference tweak.[^remove-stream512]

Both arms used the same binary. A set
`DS4_METAL_DISABLE_INDEXED_ATTN_DUAL_RB16=1`; B used the M5 rb16 kernel. The runtime
printed the selected state in every window.

### 8K prefill ABAB

Artifacts: `/Users/panda/bench-results/ds4-superdeepseek-20260815/rb16-abab/`

| Window | rb16 | Prefill tok/s | Decode tok/s |
|---|---:|---:|---:|
| A1 | off | 704.90 | 40.53 |
| B1 | on | 715.85 | 40.50 |
| A2 | off | 665.45 | 40.12 |
| B2 | on | 703.46 | 40.38 |

Mean prefill was 685.18 tok/s for A and 709.66 tok/s for B (+3.57%). Both paired
comparisons favored B, but B2 was 0.20% below the fastest A window. Therefore B did not
beat both A windows and failed the strict thermal-drift gate. All four frontier-logit
files had the same SHA-256:

```text
421aba809e8cd98f64358299ce4fd956e1a0e8741e2a466b3e0367cff23c8764
```

### 32K prefill ABAB

Artifacts: `/Users/panda/bench-results/ds4-superdeepseek-20260815/rb16-abab-long/`

| Window | rb16 | Prefill tok/s |
|---|---:|---:|
| A1 | off | 605.87 |
| B1 | on | 482.52 |
| A2 | off | 500.10 |
| B2 | on | 518.07 |

Mean A was 552.99 tok/s and mean B was 500.30 tok/s (-9.53%). The pairs disagreed
(-20.36%, then +3.59%), proving that sustained-run drift is larger than this candidate's
likely benefit on the current Automatic power profile. All four full-logit hashes were
again identical:

```text
b73a66d44ff1a79df9bd70efae97337ebf4e3b445d1fcdfb00fa930845de8382
```

Conclusion: correctness passed; performance did not. The patch remains out.

## Metal/GPU PR triage

| PR | What the evidence says | Ruling for this server |
|---|---|---|
| [#799][pr799] | Draft, per-session Metal queues. Final quiet run: 32.0 aggregate tok/s versus FIFO 37.5, but versus native batch 6.1. Requires untracked shared model buffers for most overlap gain and carries scratch-buffer race/maintenance constraints.[^pr799-limits] | Skip for single-user speed. Evaluate only for 2–8 clients that must stream concurrently. |
| [#758][pr758] | M5 indexed-prefill rb16 plus fused top-512; upstream author reports +2.94% at 8K with large thermal spread. Conflicts with current main. | Local minimal port was exact but failed the performance gate. Do not land. |
| [#794][pr794] | Broad M3 Ultra code-layer campaign. Its own scope says the M5/NAX prefill path is not the measured target. | Do not import a large unmerged patch for this M5. |
| [#782][pr782] | LLT indexer scorer measured +6–7% decode on M3 Ultra and explicitly targets pre-M5/non-NAX paths. | Not an M5 candidate. |
| [#778][pr778] | Large DSpark folding/adaptive-gate change plus small Metal decode changes. | Defer. This exact fine-tune has no generated DSpark support GGUF, and the previous local DSpark run was 16% slower. |
| [#777][pr777] | Corrects tensor extent declarations that are identical for the currently shipped 32×32 tile. | Correctness hardening, no present speed change. |
| [#621][pr621] | Adds Q4 attention-projection correctness/support; 41K additions and no M5 speed result for this Q8-projection GGUF. | Not applicable to the loaded model. |
| [#604][pr604] | Fused single-GPU session batching, but the implementation and measurements are CUDA-only. | Not applicable to Metal. |
| [#276][pr276] | Model-free suffix speculation on repetitive prompts: author measured +3% on one workload and -1% on another after tuning; branch conflicts and is large. | Too workload-specific and unmerged for the daily server. |

Older M5 Q2/Q4 benchmark PRs suggest a mixed Q2/Q4 model can improve prefill while losing
about 4% decode beyond 32K.[^pr255] That is a different checkpoint, quant recipe, and old
runtime. The current SuperDeepseek GGUF is already mixed-precision. Do not infer that a
new Q4 conversion will be faster without an exact-model A/B.

## Recommended operating profile

Keep this command for maximum single-request throughput:

```sh
cd ~/Projects/open-source/ds4
./ds4-server --chdir "$PWD" --metal \
  --model ~/models/gguf/SuperDeepseek-V4-Flash-abliterated-MQ-DS4-Q2.gguf \
  --ctx 393216 --tokens 32768 \
  --warm-weights --power 100 \
  --host 127.0.0.1 --port 8000 \
  --kv-disk-dir ~/.ds4/server-kv/superdeepseek-mq-q2 \
  --kv-disk-space-mb 131072 \
  --kv-cache-min-tokens 2048 \
  --kv-cache-reject-different-quant
```

For real requests:

1. Keep stable instructions/tool schemas at the start of the prompt so the existing disk
   KV cache can reuse long prefixes. Do not use benchmark-only `--no-cache` in normal use.
2. Use `reasoning_effort: "none"` when reasoning output is unnecessary; it reduces work
   generated, though it does not change the model's physical tok/s.
3. Keep prompts as short as the task allows. The measured 8K prefix cost 15.6% generation
   throughput and about 4× TTFR versus the shallow run.
4. Leave `--power 100`, `--warm-weights`, residency, and Metal 4 enabled for speed.
5. Enable macOS High Power Mode on AC and rerun the saved ABAB before reconsidering #758.
6. If the goal changes from one user's latency to concurrent fairness, test #799 in a
   separate binary and power table. Do not globally set `DS4_METAL_MODEL_UNTRACKED=1` on
   the current server.

## Restored state

The original server was restarted after testing and verified healthy:

```text
PID 19476
TCP 127.0.0.1:8000 LISTEN
GET /health -> {"status":"ok","model":"DeepSeek V4 Flash"}
```

No production source files were changed. Benchmark artifacts live outside the repository;
this report is the only new repository file from the investigation.

[^apple-spec]: Apple, [MacBook Pro (16-inch, M5 Pro or M5 Max) — Tech Specs](https://support.apple.com/en-us/126319).
[^apple-mlx]: Apple Machine Learning Research, [Exploring LLMs with MLX and the Neural Accelerators in the M5 GPU](https://machinelearning.apple.com/research/exploring-llms-mlx-m5).
[^apple-metal]: Apple Developer, [Accelerate your machine learning workloads with the M5 and A19 GPUs](https://developer.apple.com/videos/play/tech-talks/111432/).
[^apple-high-power]: Apple Support, [Charge the MacBook Pro battery — High Power Mode](https://support.apple.com/en-in/guide/macbook-pro/apdbc13fd966/mac).
[^model-card]: Jiunsong, [SuperDeepseek-V4-Flash-abliterated-MQ-2xDGX model card](https://huggingface.co/Jiunsong/SuperDeepseek-V4-Flash-abliterated-MQ-2xDGX).
[^pr755]: antirez/ds4 [PR #755 — Decode optimizations pre-M5 and M5 kernels Q2 and MXFP4](https://github.com/antirez/ds4/pull/755).
[^pr758]: antirez/ds4 [PR #758 — accelerate M5 Max indexed prefill](https://github.com/antirez/ds4/pull/758).
[^remove-stream512]: antirez/ds4 [`023614e` — remove broken stream512 top-k path](https://github.com/antirez/ds4/commit/023614e).
[^pr799-final]: antirez/ds4 [PR #799 final quiet-machine synthesis](https://github.com/antirez/ds4/pull/799#issuecomment-5289023556).
[^pr799-limits]: antirez/ds4 [PR #799 limitations and design trade-offs](https://github.com/antirez/ds4/pull/799#issuecomment-5288316309).
[^pr255]: antirez/ds4 [PR #255 — M5 Max Q2/Q4-imatrix curve](https://github.com/antirez/ds4/pull/255).

[pr799]: https://github.com/antirez/ds4/pull/799
[pr794]: https://github.com/antirez/ds4/pull/794
[pr782]: https://github.com/antirez/ds4/pull/782
[pr778]: https://github.com/antirez/ds4/pull/778
[pr777]: https://github.com/antirez/ds4/pull/777
[pr621]: https://github.com/antirez/ds4/pull/621
[pr604]: https://github.com/antirez/ds4/pull/604
[pr276]: https://github.com/antirez/ds4/pull/276
[commit-84cc882]: https://github.com/antirez/ds4/commit/84cc882352757baf628a1776badf7cc54d584e28
[commit-5186e2c]: https://github.com/antirez/ds4/commit/5186e2cdc31d570704b3fdbe42d4929d5b7f6df9
[commit-42033ee]: https://github.com/antirez/ds4/commit/42033ee3f45678b71c82c6ac11e9cd9ffe3d4f59
[commit-0ad494e]: https://github.com/antirez/ds4/commit/0ad494e514b5949c715857b7df8d897177957f99
[commit-fd2d5e0]: https://github.com/antirez/ds4/commit/fd2d5e05a122340de1d37f135561664e687bd8b7
[commit-0e89a0e]: https://github.com/antirez/ds4/commit/0e89a0eeffc90701fbea1f44a96492bcbb936122
[commit-8a703b6]: https://github.com/antirez/ds4/commit/8a703b670747f5cc89281a47a85345084e8fa879
[commit-e9ded97]: https://github.com/antirez/ds4/commit/e9ded97afea4ea196d84dc9fdff42464b250b5ce
