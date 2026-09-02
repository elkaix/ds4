# Running `unsloth/DeepSeek-V4-Flash-GGUF` UD-Q2_K_XL on M5 Max 128 GB (llama.cpp)

Research date: 2026-07-31. Primary sources only (GitHub source/PRs/issues, HF model cards, official Unsloth docs); Reddit used only for the speed item, labeled anecdotal.

## 1. Architecture support

- The GGUF architecture string is **`deepseek4`** — a brand-new arch, NOT `deepseek2`/`deepseek3` (V3/V3.2). HF metadata for the repo shows "Architecture: deepseek4". Source: https://huggingface.co/unsloth/DeepSeek-V4-Flash-GGUF
- Official llama.cpp support was added by PR **#24162** ("DeepSeek V4", merged 2026-06-29T08:58Z, merge commit `8c146a83`). It implements the new CSA/HCA compressed-attention caches and the lightning indexer. Source: https://github.com/ggml-org/llama.cpp/pull/24162
- **Minimum version: release `b9840`** (2026-06-29) — the tag points exactly at the #24162 merge commit and its notes lead with "DeepSeek V4 (#24162)". Verified via GitHub compare API (b9838/b9839 = "behind" the merge commit; b9840 = "identical"). Source: https://github.com/ggml-org/llama.cpp/releases/tag/b9840
- **Practical minimum: use the latest build, not b9840.** The first weeks were buggy: multi-turn context-amnesia bugs (issue #25259, referenced in PR #24162 thread), f16-mask/FA fixes (#25370, merged 2026-07-10), quantized-KV garbage (#25382, see §4). Sources: https://github.com/ggml-org/llama.cpp/pull/24162 , https://github.com/ggml-org/llama.cpp/pull/25370
- **Unsloth's own guidance**: "To run DeepSeek-V4 correctly, ensure you use the **latest version of llama.cpp** or Unsloth Studio." Their guide gives generic build instructions (`cmake -B build`; "For Apple Mac / Metal devices, set `-DGGML_CUDA=OFF` … Metal support is on by default") — no Unsloth-specific fork required. Sources: https://huggingface.co/unsloth/DeepSeek-V4-Flash-GGUF , https://unsloth.ai/docs/models/deepseek-v4

## 2. MXFP4 on Metal

- Metal has full MXFP4 support in current master: `dequantize_mxfp4`, `kernel_mul_mv_mxfp4_f32` (decode), `kernel_mul_mm_mxfp4_f16/f32` (batch), and MoE-expert paths `kernel_mul_mm_id_mxfp4_*` / `kernel_mul_mv_id_mxfp4_f32` all exist in the Metal shader library. Source: https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/ggml-metal.metal
- MXFP4 was added to llama.cpp (incl. Metal) for gpt-oss in Aug 2025; ggerganov: "all major backends, including CUDA, Vulkan, Metal and CPU". Sources: https://x.com/ggerganov/status/1952779751736627627 , https://github.com/ggml-org/llama.cpp/discussions/15396
- So the 2 MXFP4 tensors in UD-Q2_K_XL run on the GPU natively; no CPU fallback, no meaningful cost.
- Caveat unrelated to MXFP4: the **DSV4-specific ops** (lightning indexer, hyperconnection ops) have NO Metal kernels in master yet (`ggml-metal*` sources contain zero `lightning`/`dsv4` references — verified 2026-07-31), so those small ops execute on CPU even with `-ngl 99`. It works, just slower than it could be. tarruda's Metal kernels for these ops (~6 → ~20 t/s on M1 Ultra in his fork) are being upstreamed; the Metal lightning-indexer PR **#25893 is still open**. Sources: https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/ggml-metal-device.cpp , https://github.com/fairydreaming/llama.cpp/pull/3 , https://github.com/ggml-org/llama.cpp/pull/25893

## 3. Split GGUF loading

- llama.cpp loads sharded GGUFs natively: multi-split support landed in `llama_model_loader` via PR #6187; you pass **only the first shard** (`-m ...-00001-of-00003.gguf`) and the rest are found automatically. Source: https://github.com/ggml-org/llama.cpp/pull/6187 (demo/discussion: https://github.com/ggml-org/llama.cpp/discussions/6404)
- Unsloth's official command does exactly this — points `llama-cli --model` at the `...-00001-of-00005.gguf` first shard. Source: https://unsloth.ai/docs/models/deepseek-v4
- **The 97 GB merged file is unnecessary** (but harmless; both load).

## 4. Memory fit on 128 GB

- UD-Q2_K_XL = **96.8 GB** per the HF file-size table. Unsloth's hardware-requirements table gives **2-bit → 102 GB total memory** (weights + KV + context overhead). 128 GB unified memory → fits with ~25 GB headroom. Sources: https://huggingface.co/unsloth/DeepSeek-V4-Flash-GGUF , https://unsloth.ai/docs/models/deepseek-v4
- DSV4's KV cache is tiny by design: hybrid CSA/HCA compressed attention, "~10% of the KV cache of DeepSeek-V3.2" at 1M ctx. Context is cheap here; weights dominate. Source: https://huggingface.co/unsloth/DeepSeek-V4-Flash-GGUF
- **GPU wired limit (required)**: Metal caps GPU allocations at ~75% of RAM (~96 GB on 128 GB machines) by default; a ~97 GB model will not fit under the default cap. Raise it, e.g. `sudo sysctl iogpu.wired_limit_mb=114688` (112 GB; leaves ~16 GB for macOS). Runtime-tunable, resets on reboot. Sources: https://www.reddit.com/r/LocalLLaMA/comments/186phti/ , https://github.com/ivanopcode/devnote-override-macos-metal-vram-cap , https://github.com/ggml-org/llama.cpp/discussions/15372
- Recommended flags: `-ngl 99` (everything on GPU), `-fa on` (flash attention; note #25370 made KQ masks f16 when FA is on), default mmap on (weights page in from disk; don't add `--no-mmap` on a 128 GB machine running a 97 GB model). Sources: https://github.com/ggml-org/llama.cpp/pull/25370 , https://unsloth.ai/docs/models/deepseek-v4
- **KV cache quant**: `--cache-type-k q8_0` produced confident garbage on ALL backends for deepseek4 (Hadamard-rotation diverted layers off the sparse attention path) — issue #25382, closed as completed 2026-07-07; current master handles rotation for DEEPSEEK4 explicitly (`llama-kv-cache.cpp`, `LLAMA_ATTN_ROT_DISABLE` escape hatch exists). Safe on current builds, but **f16 K cache is the zero-risk choice** and costs little because the cache is compressed anyway. Sources: https://github.com/ggml-org/llama.cpp/issues/25382 , https://github.com/ggml-org/llama.cpp/blob/master/src/llama-kv-cache.cpp
- **Max context realistically**: model supports 1,048,576 tokens; Unsloth says Think Max wants ≥384K. On 128 GB with ~25 GB headroom, start at `-c 32768`–`-c 131072`; 256K+ is plausibly reachable given the compressed cache, but no primary-source measurement exists for exact per-token KV size on Metal — increase stepwise and watch the `llama_kv_cache` size log lines at startup. Sources: https://huggingface.co/unsloth/DeepSeek-V4-Flash-GGUF , https://unsloth.ai/docs/models/deepseek-v4

## 5. Expected speed (anecdotal — this item only)

- Mac Studio **M3 Ultra 512 GB**, UD-Q4_K_XL, `llama-server -ngl all -fa on -c 10000`: **~8 t/s TG** (2026-07-08, ~1 week after merge; commenters note llama.cpp DSV4 is unoptimized). Source: https://www.reddit.com/r/LocalLLaMA/comments/1uqnbk5/
- M1 Ultra: ~6 t/s upstream vs **~20 t/s** with tarruda's not-yet-upstreamed Metal DSV4 kernels (IQ3_XXS quant). Source: https://github.com/fairydreaming/llama.cpp/pull/3
- ik_llama.cpp author's 2×RTX3090 + `--cpu-moe` rig: ~11.4 → 8.3 t/s TG over 0–30K ctx. Source: https://github.com/ikawrakow/ik_llama.cpp/pull/2110
- Estimate for M5 Max 128 GB (not a measurement): roughly **8–15 t/s TG**, bandwidth-bound; PP will feel slow. Expect improvements as Metal DSV4 ops land (#25893).

## 6. ik_llama.cpp

- DS4 exists there: community port PR #2110 was **closed unmerged** (2026-07-22) after review found crashes/PPL divergence; ikawrakow's own follow-up **#2165 "DS4: slowly approaching a meaningful performance" is merged**, so main-branch ik_llama.cpp has DS4 support. Sources: https://github.com/ikawrakow/ik_llama.cpp/pull/2110 , https://github.com/ikawrakow/ik_llama.cpp/pull/2165
- Not meaningfully better here: ik's strengths are CPU/CUDA quants (IQK); Metal is not its focus, DS4 support is days old and self-described as "approaching meaningful performance". Its MLA/fused-MoE advantages target CUDA+CPU-offload rigs. On a Mac, mainline llama.cpp (Metal) is the right runtime. Sources: https://github.com/ikawrakow/ik_llama.cpp/pull/2110 , https://github.com/ikawrakow/ik_llama.cpp/pull/2165

## 7. Recommended command (official guidance synthesis)

Unsloth's official command template (their guide): `llama-cli --model <first-shard>.gguf --temp 1.0 --top-p 1.0 --min-p 0.0`; sampling per DeepSeek: temp 1.0, top-p 1.0 (0.95 for agentic on the -0731 revision); Think High on by default; toggle via `--chat-template-kwargs '{"enable_thinking":false}'` or `--reasoning on|off`. Sources: https://unsloth.ai/docs/models/deepseek-v4 , https://huggingface.co/unsloth/DeepSeek-V4-Flash-GGUF

## Recommended command

```bash
# one-time per boot: let Metal use ~112 GB of the 128 GB unified memory
sudo sysctl iogpu.wired_limit_mb=114688

# latest llama.cpp build (>= b9840 absolute minimum; use newest)
~/bin/llama.cpp/build/bin/llama-server \
  -m /path/to/DeepSeek-V4-Flash-UD-Q2_K_XL-00001-of-00003.gguf \
  -ngl 99 \
  -fa on \
  -c 32768 \
  --temp 1.0 --top-p 1.0 --min-p 0.0 \
  --host 127.0.0.1 --port 8080
```

Notes:
- Point `-m` at shard 1 of 3; do not use the merged 97 GB file (redundant).
- Raise `-c` stepwise (65536 → 131072 → …) watching startup `llama_kv_cache ... MiB` log lines against free memory; Think Max wants ≥384K per the model card.
- Optional: `--cache-type-k q8_0` to shave KV memory — only on a current build (bug #25382 on June/early-July builds); f16 default is fine given the compressed cache.
- Optional: `--chat-template-kwargs '{"reasoning_effort":"max"}'` / `'{"enable_thinking":false}'` per Unsloth.
- Check the local build first: `~/bin/llama.cpp/build/bin/llama-server --version` — must report ≥ b9840; if older, rebuild from master.
