# Run ds4-server (daily)

## 1. After every reboot — raise the Metal wired-memory ceiling

Resets to 0 on reboot. The Q2 model will not fit without it.

```bash
sudo sysctl iogpu.wired_limit_mb=118000
```

## 2. Start the server

```bash
cd ~/Projects/open-source/ds4 && ./ds4-server --chdir "$PWD" --metal \
  --model ~/models/gguf/DeepSeek-V4-Flash-Vision-Exp-Abliterated-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8.gguf \
  --vision ~/models/gguf/DeepSeek-V4-Flash-Vision-Encoder.gguf \
  --ctx 393216 --tokens 32768 \
  --warm-weights --power 100 \
  --host 127.0.0.1 --port 8000 \
  --kv-disk-dir ~/.ds4/server-kv/deepseek-v4-flash-vision-exp-ablit-q2 --kv-disk-space-mb 131072 \
  --kv-cache-min-tokens 2048 --kv-cache-reject-different-quant
```

`--ctx 393216` matches the client config (see §5) and is the minimum for Think Max.
Costs ~1 GiB more KV than 262144: **87.13 GiB planned**, well under the 118 GiB cap.

### The model (switched 2026-09-02)

`audreyt/DeepSeek-V4-Flash-Vision-Exp-Abliterated-GGUF`, 86,720,111,776 bytes,
sha256 `66e47437ce7201546fa870a23ccdc094282660f1e52f3403ed5fb59604b2d894`. Official
Vision-Exp IQ2 recipe with the rank-1 refusal-direction edit baked into 33
`blk.{10..42}.attn_output_b` tensors. Needs `ds4-server` at `98e3101` or later —
`--vision` only exists after the 2026-09-02 upstream merge.

**`--vision` is not optional in spirit.** The GGUF is `sidecar_required=true`; the
encoder is a separate 932,857,760-byte file and must be the *unmodified* 316-tensor
`DeepSeek-V4-Flash-Vision-Encoder.gguf` from `antirez/deepseek-v4-gguf`. Omit it and
the server still answers text requests, but every image request fails.

**Do NOT attach a DSpark sidecar.** This is not Headroom128 0731, and the 0731
support GGUF does not match it — that pairing is antirez/ds4#949. `run-ds4-conf.sh`,
`run-ds4-strict.sh` and `run-ds4-dspark.sh` therefore stay on the drowzeys 0731 pair
and refuse to start if `MODEL` is repointed at a Vision-Exp build.

A **fresh `--kv-disk-dir`** is mandatory on this swap: the quant recipe is
byte-identical to the previous SuperDeepseek build, so
`--kv-cache-reject-different-quant` would **not** catch it and the old model's
checkpoints would be restored into this one.

Rollback (SuperDeepseek MQ stays on disk):
```bash
ln -sfn ~/models/gguf/SuperDeepseek-V4-Flash-abliterated-MQ-DS4-Q2.gguf ds4flash.gguf
# and revert --model, --vision and --kv-disk-dir above
```

## 3. Health check

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/stats | python3 -m json.tool
```

Expect `resident model 80.76 GiB + KV 3.37 + buffers 3.00 = 87.13 GiB planned`,
`context buffers 6523.67 MiB`, `compressed_kv_rows=98306`.

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

**Lower memory** — drop to `--ctx 262144` if you never use Think Max (saves ~1 GiB KV).
Lower the client limits to match, or the client will budget past what the server holds.

**Client context limits must never exceed the server's `--ctx`.** Neither client reads
them from `/v1/models`; both are static files.

| client | file | keys | current |
|---|---|---|---|
| **Pythinker CLI** | `~/.pythinker-code/config.toml` → `[models."ds4/deepseek-v4-flash"]` | `max_context_size` / `max_output_size` | `393216` / `32768` ✓ |
| **Pi** | `~/.pi/agent/models.json` → `providers.ds4.models[0]` | `contextWindow` / `maxTokens` | `393216` / `32768` ✓ |

Both currently match `--ctx 393216 --tokens 32768`. If you change `--ctx`, change both.

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
