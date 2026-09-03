# Run ds4-server (daily)

## 1. After every reboot — raise the Metal wired-memory ceiling

Resets to 0 on reboot. The Q2 model will not fit without it.

```bash
sudo sysctl iogpu.wired_limit_mb=118000
```

## 2. Start the server

```bash
cd ~/Projects/open-source/ds4 && ./ds4-server --chdir "$PWD" --metal \
  --model ~/models/gguf/DeepSeek-V4-Flash-Vision-Exp-Abliterated-IQ2XXS-w2Q2K-AProjQ4K-SExpQ8-OutQ4K.gguf \
  --vision ~/models/gguf/DeepSeek-V4-Flash-Vision-Encoder.gguf \
  --ctx 262144 --tokens 32768 \
  --warm-weights --power 100 \
  --host 127.0.0.1 --port 8000 \
  --kv-disk-dir ~/.ds4/server-kv/deepseek-v4-flash-vision-exp-ablit-q4k --kv-disk-space-mb 131072 \
  --kv-cache-min-tokens 2048 --kv-cache-reject-different-quant
```

`--ctx 262144` matches both clients (see §5) and the four `run-ds4-*.sh` wrappers.
**82.74 GiB planned** with the AProjQ4K build (KV 2.36 + buffers 2.00 + model 78.37),
well under the cap — and it needs no `iogpu.wired_limit_mb` bump to load.
Think Max needs `393216`; raise both the server and the clients together if you want it.

### The model (AProjQ4K, adopted 2026-09-02)

`DeepSeek-V4-Flash-Vision-Exp-Abliterated-IQ2XXS-w2Q2K-AProjQ4K-SExpQ8-OutQ4K.gguf`,
84,155,819,296 bytes, sha256 `22f7eeb2d3bd6142501fc54146e8ff9470ae03e2cae9a5dcfe8543a43d7ed4cb`. Built here: the audreyt Vision-Exp Abliterated Q8 GGUF with its
216 dense attention-projection + output-head tensors re-quantized `q8_0 -> q4_K`,
imatrix-guided by the merged routed-1p5m + dense-220k file, then **spliced** so the
other 1112 tensors keep the reference's exact bytes. Resident model **78.37 GiB**
(vs 80.76 for Q8). The 33 `blk.{10..42}.attn_output_b` refusal-direction edits are
preserved (they are inside the 216 and were re-quantized from the same abliterated
FP8 source). Needs `ds4-server` at `98e3101` or later — `--vision` only exists after
the 2026-09-02 upstream merge.

Measured against the Q8 build (interleaved A/B, cold-mmap discard arms dropped):
decode **+25% short / +36% at ~2.8K / +24% at 34K**, prefill flat-to-up. The 21
`blk.N.indexer.attn_q_b` tensors are pinned f16 — `--attention-proj` is a prefix glob
that would otherwise drop the lightning-indexer Q projection to 4 bit.

**`--vision` is not optional in spirit.** The GGUF is `sidecar_required=true`; the
encoder is a separate 932,857,760-byte file and must be the *unmodified* 316-tensor
`DeepSeek-V4-Flash-Vision-Encoder.gguf` from `antirez/deepseek-v4-gguf`. Omit it and
the server still answers text requests, but every image request fails.

**Do NOT attach a DSpark sidecar.** This is not Headroom128 0731, and the 0731
support GGUF does not match it — that pairing is antirez/ds4#949. `run-ds4-conf.sh`,
`run-ds4-strict.sh` and `run-ds4-dspark.sh` have had their `--mtp`/`--dspark` flags
removed and are now equivalent to `run-ds4-monitored.sh`; each header records what to
restore if a Vision-Exp sidecar ever ships. DSpark also measured 16% *slower* here
(2026-08-09), so re-run `tasks/dspark-ab.sh` before believing otherwise.

A **fresh `--kv-disk-dir`** is mandatory on every model swap here. Do not rely on
`--kv-cache-reject-different-quant`: the previous three builds shared one recipe
byte-for-byte, so it would not have caught those swaps at all. The dir above is
`...-q4k`; the Q8 build's checkpoints stay in `...-q2` and are not reused.

Rollback (the Q8 Vision-Exp build stays on disk, 86,720,111,776 bytes,
sha256 `66e47437ce7201546fa870a23ccdc094282660f1e52f3403ed5fb59604b2d894`):
```bash
ln -sfn ~/models/gguf/DeepSeek-V4-Flash-Vision-Exp-Abliterated-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8.gguf ds4flash.gguf
# and revert --model and --kv-disk-dir above (--vision is unchanged by this swap)
```

## 3. Health check

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/stats | python3 -m json.tool
```

Expect `resident model 80.76 GiB + KV 3.37 + buffers 3.00 = 87.13 GiB planned`,
`context buffers 6523.67 MiB`, `compressed_kv_rows=98306`.

`run-ds4-monitored.sh` uses the same `--ctx 262144`, which measures
`KV 2.36 + buffers 2.00 + resident model 80.76 = 85.13 GiB planned` (verified
2026-09-02). `resident model 80.76 GiB` is identical for both, so **the model fits
under the default Metal cap** — §1's sysctl is what `run-ds4-monitored.sh` insists on,
not what the weights actually require.

## 3b. Live dashboard

```bash
open http://127.0.0.1:8000/dashboard
```

Live decode/prefill t/s with sparklines, context utilization, queue and client counts,
KV cache hit rate and per-path breakdown, and a red anomaly row (queue rejections,
disconnect drops, cancelled prefills). Polls `/stats` once a second.

Served by ds4-server itself from `dashboard.html`, embedded into the binary at build
time — same-origin, so no `--cors`. Edit `dashboard.html` and `make ds4-server` to change
it. Poll interval: `/dashboard?ms=500`.

## 4. What speed to actually expect

**Decode decays with context depth. The headline number is a near-empty number.**
Do not read a mid-session figure against it — measured on this build:

| context depth | decode |
|---|---:|
| ~0 (fresh) | **44.5 t/s** |
| 8k | ~36 t/s |
| ~48k | **~33 t/s** |

~33 t/s at 48k is healthy, not a regression. For comparison the best MLX build reports
31.5 t/s at 32k, so this is still the fastest option for this model.

**Prefill, not decode, is what makes a coding agent feel slow.** A cold 47.7k-token
prompt costs ~88 s; the same prefix on the next turn costs ~0.7 s from the KV cache.
That 100× gap is why §5 matters more than any tokens/sec number.

## 5. Keep the KV disk cache healthy

The disk cache is what turns an 88 s cold prefill into a 0.7 s warm one. When it fills,
it thrashes — evicting entries mid-prefill that were never reused:

```text
kv cache evicted reason=disk-cache-full tokens=2048 hits=0 ...   ← repeated = cache full
```

Check it, and clear it if it is at budget (it is only a cache — costs one cold prefill):

```bash
du -sh ~/.ds4/server-kv/deepseek-v4-flash-ga-0731-q2     # vs --kv-disk-space-mb
rm -rf ~/.ds4/server-kv/deepseek-v4-flash-ga-0731-q2/*   # with the server stopped
```

Budget raised to **131072 MiB (128 GB)** on 2026-08-09 after it pinned at the old 32 GB.
There is 1.5 TB free, so headroom is cheap.

---

## What changed from the old command

`--mtp`, `--dspark`, and `--dspark-confidence 0.9` were **removed**.

DSpark (speculative "guess-ahead" decoding) was measured on a cooled machine on
2026-08-09 and is **16% slower** on this model and quant:

| config | decode mean | range | n |
|---|---:|---:|---:|
| DSpark ON | 37.49 t/s | 37.01 – 37.96 | 6 |
| DSpark OFF | **44.58 t/s** | 43.80 – 45.35 | 6 |

Every off-run beat every on-run; the distributions do not overlap. Wall clock for 512
tokens went 14.25 s → 11.82 s. DSpark's own accounting agreed independently
(`net_saved` ≈ −21% of target time).

Why it loses: it only drafts on **19% of decode cycles**, and although acceptance is good
when it does (88.2%), that averages just **0.164 accepted tokens per cycle** — not enough
to pay for propose+verify+replay running on every cycle.

Side benefit: the 6.0 GB DSpark support GGUF is no longer loaded, freeing that memory.

To re-test it later (e.g. after an upstream change), on an **idle, cooled** machine:

```bash
./tasks/dspark-ab.sh ~/Projects/open-source/ds4 test_off off
./tasks/dspark-ab.sh ~/Projects/open-source/ds4 test_on  on
```

Run the arms interleaved and only believe a result when every run of one arm beats every
run of the other — this Mac drifts up to 40% across windows when warm.

---

## Variants

**Think Max** — raise to `--ctx 393216` (costs ~1 GiB more KV) *and* raise both client limits in §5 to match.
Lower the client limits to match, or the client will budget past what the server holds.

**Client context limits must never exceed the server's `--ctx`.** Neither client reads
them from `/v1/models`; both are static files.

| client | file | keys | current |
|---|---|---|---|
| **Pythinker CLI** | `~/.pythinker-code/config.toml` → `[models."ds4/deepseek-v4-flash"]` | `max_context_size` / `max_output_size` | `262144` / `32768` ✓ |
| **Pi** | `~/.pi/agent/models.json` → `providers.ds4.models[0]` | `contextWindow` / `maxTokens` | `262144` / `32768` ✓ |

Both currently match `--ctx 262144 --tokens 32768` (verified on disk 2026-09-02 — an earlier
version of this table said 393216, which neither client ever used). If you change `--ctx`,
change both; a client must never budget more context than the server allocated.

Pythinker CLI also sets `reserved_context_size = 50000` under `[loop_control]`, which it
holds back from the window for its own bookkeeping.

**Reasoning effort** — Pythinker exposes `low` / `high` / `max`. `max` is the slow one:
it can spend 3,000+ thinking tokens (~90 s at 33 t/s) before the first useful output.
Use `high` for coding, `max` only for genuinely hard reasoning.

---

## Gotchas

- **Port 8000 first.** If startup fails with `Address already in use`, check
  `lsof -nP -iTCP:8000 -sTCP:LISTEN` before blaming the model — the 95 GB load succeeding
  tells you nothing about whether the bind will.
- **Keep `127.0.0.1`.** Do not bind `0.0.0.0` without auth or a firewall.
- `/v1/models` advertises **two** ids (`deepseek-v4-flash` and `deepseek-v4-pro`) for the
  one loaded GGUF. Expected upstream behaviour; it is why some benchmark tools refuse to
  auto-detect and need `--model deepseek-v4-flash` passed explicitly.
- **`--tokens 32768`** caps runaway generations when a client omits its own limit (server
  default is 393216).
- **`--kv-cache-min-tokens 2048`** avoids writing many tiny KV checkpoints to SSD.
- **`--chdir`** makes Metal shader resolution reliable when launched from a script or
  LaunchAgent.
