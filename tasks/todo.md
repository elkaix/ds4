# Convert SuperDeepseek-V4-Flash-abliterated-MQ-2xDGX → ds4 GGUF (2026-08-12)

Goal: replace the daily `drowzeys-...-DS4-Q2.gguf` with a ds4-format Q2 built from
`~/models/hf/SuperDeepseek-V4-Flash-abliterated-MQ-2xDGX` (158 GiB, 50 shards).
Reason to switch: worst-mode refusal 97.92% → 4.17% with tool gates held at 100%.

## Plan

- [x] 1. Verify source is ds4-convertible → verify: `config.json` matches official 0731
      (`expert_dtype: fp4`, `quant_method: fp8`, `scale_fmt: ue8m0`, 43 layers, 256/top-6). **OK**
- [x] 2. Verify every safetensors dtype has a converter path → verify: dtype histogram
      (F8_E8M0, I8, BF16, F8_E4M3, F32, I64) all handled in `deepseek4-quantize.c`. **OK**
- [x] 3. Pick recipe → verify: `--dry-run` with template only (no type flags) reports
      `type_changes: 0` and `approx_file_bytes` == current file's 86,720,111,488. **OK, exact**
- [x] 4. Pre-flight `--compare-tensor` per dtype family vs the current drowzeys Q2. **OK** —
      see results below.
- [x] 5. Conversion done: exit 0, 86,720,111,488 bytes (exact match), 1328 tensors, type_changes 0.
- [x] 6. Load test: generates ("I'm Qwen..."), prefill 47.9 gen 43.1 t/s.
- [x] 7. Coherence: 17*24 -> 408 step-by-step, no loop, 43.2 t/s.
- [x] 8. Refusal probes: NO difference vs drowzeys — both comply on all 4. See review.
      Bar not met; user chose to switch anyway on 2026-08-13.
- [x] 9. Switched on user instruction: symlink + run.md + new KV dir. Live and verified.
      drowzeys Q2 kept on disk as rollback.

## Acceptance criteria

Ship only if: conversion exit 0 AND loads AND coherent AND measurably fewer refusals than
the drowzeys Q2 on the same probes. Otherwise keep drowzeys and document why.

## Pre-flight results (step 4)

Regenerated one tensor per dtype path and byte-compared against the current drowzeys Q2:

| tensor | path | result |
|---|---|---|
| `blk.0.ffn_gate_exps.weight` | I8+E8M0 → IQ2_XXS | **byte-identical** |
| `blk.0.ffn_down_exps.weight` | I8+E8M0 → Q2_K | **byte-identical** |
| `blk.0.ffn_gate_shexp.weight` | F8_E4M3+E8M0 → Q8_0 | **byte-identical** |
| `token_embd.weight` | BF16 → F16 | **byte-identical** |
| `blk.0.attn_output_b.weight` | F8_E4M3+E8M0 → Q8_0 | **differs, 16,348,915 bytes** |

This is the ideal shape. The four identical results prove the dequant/requant pipeline and
the imatrix apply are correct on this checkpoint, and confirm SuperDeepseek left the routed
experts untouched exactly as its README claims. The single mismatch is `attn_output_b` —
HF `attn.wo_b`, the 46 pairs SuperDeepseek actually modified. So the abliteration is real,
it is the only delta, and it survives the Q8_0 attention-projection recipe.

## Recipe decision

Pass **no type flags** — the template
`~/models/ds4/templates/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.header.gguf`
already encodes the antirez q2-imatrix recipe. Adding `--attention-proj q8_0` caused 21
spurious `blk.N.indexer.attn_q_b.weight` f16→q8_0 changes; template-only gives 0.

## Deferred

`--dspark-support` not generated. DSpark measured 16% slower on 2026-08-09 and was removed
from the daily command.

## Review section (2026-08-13)

**Conversion: SUCCESS.** `SuperDeepseek-V4-Flash-abliterated-MQ-DS4-Q2.gguf`, 86,720,111,488
bytes — byte-count identical to the current drowzeys Q2. Loads, coherent, **43.1 t/s decode**
= parity with the model in production. Pre-flight already proved the pipeline (4/5 tensors
byte-identical; the 1 delta = `attn_output_b` = the abliterated `attn.wo_b`).

**Refusal probes (step 8): NO measurable difference vs the drowzeys Q2 I already run.**
Both are abliterated builds. On 4 probes (insult-by-name limerick, phishing-email
construction, manipulative scam lines, forceful opinion) BOTH complied fully. The
SuperDeepseek README's "97.92% -> 4.17%" is measured against the *base* DeepSeek-0731, NOT
against drowzeys — which is already uncensored. So on the refusal axis this is a **lateral
move, not an upgrade.**

**Verdict vs stated acceptance criteria: does NOT clear the bar.** todo.md said "ship only
if measurably fewer refusals than the drowzeys Q2 ... otherwise keep drowzeys and document
why." Refusals are identical, so the criterion is not met. The conversion is valid and
usable; there is just no refusal-based reason to switch.

**Symlink NOT flipped.** `ds4flash.gguf` still -> drowzeys Q2. Both files on disk (995 GB
free). Switching is one command if the user wants the newer fine-tune for reasons other than
refusals (it is a different OBLITERATUS+SuperTune checkpoint; any quality/tool-gate edge
would need a benchmark this session did not run).

## SWITCHED — live on the new model (2026-08-13 09:18)

User overrode the acceptance bar and chose to switch despite refusals being identical.
Done and verified live:

- `ds4flash.gguf` -> `SuperDeepseek-V4-Flash-abliterated-MQ-DS4-Q2.gguf`
- `run.md:15` `--model` updated; `--kv-disk-dir` moved to
  `~/.ds4/server-kv/superdeepseek-mq-q2` (fresh dir — different fine-tune, and
  `--kv-cache-reject-different-quant` would NOT catch it since the quant is identical;
  a shared dir would have fed the old model's checkpoints to the new one)
- server pid 59213, `--ctx 393216`, healthy in 40 s
- planned memory: KV 3.37 + buffers 3.00 + resident model 80.76 = **87.13 GiB** (unchanged
  from drowzeys, well under the 118 GB wired cap)
- live decode **42.2 t/s**, generation correct (12*13 = 156)

sha256: `608757f676569178ccd06ee07c475833d085f93cab4400d45e953ec875b7bade`

Rollback (one command + restart):
`ln -sfn ~/models/gguf/drowzeys-keys-DeepSeekV4-Flash-GA-0731-Abliterated-32-32-DS4-Q2.gguf ds4flash.gguf`
and revert `run.md:15`. The drowzeys Q2 stays on disk.

Note: the model self-identifies inconsistently ("Qwen" via CLI, "DeepSeek-V3" via API).
Cosmetic — inherited from the fine-tune's training data, not a conversion defect. The
drowzeys build does the same.

# Live DS4 setup audit (2026-08-22)

Goal: validate `run-ds4-monitored.sh`, the running server, cache continuity,
throughput, memory, disk, and cited upstream claims against this M5 Max.

## Plan

- [x] 1. Preserve and record repo/runtime baseline.
- [x] 2. Inspect actual command, binary, logs, KV directory, memory, swap, disk.
- [x] 3. Validate cited claims against local source and current upstream state.
- [x] 4. Run controlled repeated-request/cache probes on this machine.
- [x] 5. Record ranked findings, exact safe config, and remaining limitations.

## Acceptance criteria

- Every setup claim is labeled observed, source-confirmed, contradicted, or unvalidated.
- Recommendations reflect the actual local branch/build and measured machine behavior.
- No server/config/source change occurs during this read-only audit.
- Existing dirty work remains intact and the review section records commands/results.

## Review

- Live server stayed healthy on PID 95676 at `127.0.0.1:8000`; running binary is
  up to date with local `prod` source (`make -q ds4-server` exit 0).
- Before the audit probe, the real Pi tool loop had 8 cache hits / 2 cold requests
  and 309,378 / 353,238 prompt tokens cached (87.6%). This setup does not reproduce
  issue #816's 787/787 miss pattern.
- Controlled chat probe: prewarm 0/27 cached, repeat 27/27 cached in 0.094 s,
  exact assistant replay 30 cached + 11 new tokens in 0.241 s.
- The initial audit favored `--ctx 393216` for Think Max. The user then selected
  `--ctx 262144` for lower memory pressure; the launcher now uses it and therefore
  gives up Think Max's 393216-token minimum. Keep one session; omit batched mode.
- The launcher now sets `--kv-cache-cold-max-tokens 65536`. Keep `--trace` temporary and private:
  it records full prompts, generated text, reasoning, and tool arguments.
- KV store: 140 valid files, 83.97 GiB / 128 GiB. Whole-volume free space was
  about 2.03 TiB; 23 purgeable Time Machine snapshots exist. During the live sample,
  an active Hugging Face download matched 98.9% of free-space loss while KV stayed
  unchanged. Historical disk loss cannot be attributed to DS4 from `disk_free`.
- Live memory: wired limit 122880 MiB; busy GPU allocation ~98.1 GiB, in-use
  ~89.7 GiB; no thermal/performance warning. Swap rose during unrelated concurrent
  Node/Chrome work, so RSS/swap alone cannot prove model-weight eviction.
- Upstream `main` is still 84cc882. PRs #818/#850/#827 remain open; #601 remains
  open draft. #818 helps exact/divergent prefixes >=256 tokens, not `common=1`.
- Verification: `bash -n`, `shellcheck`, exact-value assertions, server help,
  `make ds4_test && ./ds4_test --server`, and health/stats checks all passed.
  Restart verified on PID 6262: health `ok`, context 262144, cold max 65536,
  wired limit 122880 MiB. Existing dirty baseline preserved; only the launcher
  and this task tracker section changed.

# DS4 max-fan lifecycle (2026-08-22)

Goal: command both fans to verified maximum while ds4-server owns the runtime,
then restore Apple automatic control when the server stops.

## Plan

- [x] Discover and validate the existing ThermalForge controller.
- [x] Fail before model startup when non-interactive fan control is unavailable.
- [x] Verify actual max RPM with a bounded ramp deadline.
- [x] Restore and verify automatic mode on normal exit, signals, or server failure.
- [x] Keep a server-watching fallback for wrapper hard-kill, then run failure checks.

## Acceptance criteria

- The server never continues after max-fan activation or verification fails.
- Fan commands and ramp/restore waits are bounded.
- Cleanup is idempotent and does not leave the server running after fan failure.
- A stopped server does not leave fans in manual mode.

## Review

- ThermalForge owns max/auto commands through its existing scoped NOPASSWD rule;
  macmon independently verifies actual RPM and reports both fans every monitor cycle.
- Verified maximum: fan0 5317/5349 RPM, fan1 5684/5777 RPM.
- Verified normal and TERM/EXIT restore, forced ramp-verification failure restore,
  fail-closed controller probe, and monitor fallback after simulated server death.
  Final fan modes are both `auto`.
- Fan commands are bounded to 5 seconds, ramp to 20 seconds, restore to 5 seconds.
- `bash -n`, ShellCheck, help smoke test, exact function tests, and `git diff --check`
  passed. The real DS4 server is currently stopped; no test process remains.
