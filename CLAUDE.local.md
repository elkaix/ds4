# CLAUDE.local.md — ds4 (DwarfStar) local setup notes

Personal working notes for this clone. **Keep this file updated whenever Claude works
in this repo** (setup changes, model swaps, fixes, benchmark results). Not for upstream —
never commit it.

## What this is

DwarfStar (`antirez/ds4`) — self-contained native inference engine for **DeepSeek V4
Flash**, Metal-first. This M5 Max 128 GB MacBook is the project's reference machine
(README benchmarks: ~87 t/s prefill, ~34 t/s gen at q2).

## Setup state (2026-07-01)

- Cloned to `~/Projects/open-source/ds4`, built with `make` (Metal). Binaries: `ds4`,
  `ds4-server`, `ds4-bench`, `ds4-eval`, `ds4-agent` — all compile clean.
- Models in `gguf/`:
  - **q2-q4-imatrix** (98 GB, `DeepSeek-V4-Flash-Layers37-42...-imatrix-fixed.gguf`) —
    **current default** (`ds4flash.gguf` points here since 2026-07-02). Upstream's
    higher-quality pick for 128 GB MacBooks (q2 routed experts, last 6 layers q4).
    Requires the raised GPU wired limit (see below) — OOMs at the default cap.
  - **q2-imatrix** (87 GB, `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-...-chat-v2-imatrix.gguf`) —
    fallback; fits under the default macOS GPU wired-memory cap with no sudo needed.
- Download note: repo's `download_model.sh` uses single-connection curl (~1.5 MB/s here).
  Use `hf download antirez/deepseek-v4-gguf <file> --local-dir ./gguf` instead
  (Xet backend, ~30-50 MB/s effective). Then `ln -sfn gguf/<file> ds4flash.gguf`.

## RESOLVED: Metal OOM on q2-q4-imatrix (98 GB)

`./ds4 -m gguf/DeepSeek-V4-Flash-Layers37-42...gguf` (93 GB mapped weights) failed
prefill with `kIOGPUCommandBufferCallbackErrorOutOfMemory` — macOS GPU wired limit
is ≈75% of RAM (~96 GB), and 93 GB weights + KV/scratch didn't fit under it.

Fixed 2026-07-02 with `sudo sysctl iogpu.wired_limit_mb=118000`. **Resets on
reboot** — re-run it before using q2-q4-imatrix after a restart, or fall back to
q2-imatrix (fits under the default cap) by repointing the symlink.

## Status: working (q2-q4-imatrix default since 2026-07-02)

Verified 2026-07-02 with `./ds4 -p "..." -n 32 --nothink` on q2-q4-imatrix
(wired limit raised): **prefill 68.59 t/s, generation 35.05 t/s** — roughly 2×
the q2-imatrix numbers (39.80 / 17.35 measured 2026-07-01).

## In-context perf under pi load (measured 2026-07-02, q2-q4-imatrix)

From live server logs at ~46K context: prefill ~320 t/s avg (25K-token chunk starting
at 20K depth), generation steady ~24.8 t/s (vs 35 t/s near-empty — normal depth decay).
Disk KV cache confirmed working: 20,480-token prefix restored in 55 ms. macmon during:
118 GB RAM used, zero swap, GPU ~77% @ 34 W, 78°C. Evictions (`reason=evict`) started
appearing in the 8 GB disk cache — raise `--kv-disk-space-mb` if cross-restart hits drop.

## Pi agent hookup (2026-07-02)

ds4-server is wired into pi as provider `ds4`, model `deepseek-v4-flash`
(`~/.pi/agent/models.json`, `contextWindow: 262144`; enabled in `settings.json`,
and set as pi's default provider/model). Pi expects the server at
`http://127.0.0.1:8000/v1`. Start it with:

```sh
cd ~/Projects/open-source/ds4 && ./ds4-server --ctx 262144 --port 8000
```

Keep `--ctx` in sync with the `contextWindow` in pi's models.json. 256K is the
safe ceiling with the 98 GB quant: 118 GB wired limit − 93 GB weights leaves
~25 GB, and 256K of compressed KV uses ~7 GB (model supports up to 1M ≈ 26 GB
KV, but that doesn't fit alongside this quant). At 256K, macmon shows 118.8/128
GB RAM used, zero swap.
Which quant pi gets is whatever `ds4flash.gguf` points to — the server exposes
one model id regardless. Remember the sysctl above after reboots (the server
OOMs on q2-q4-imatrix without it).

## Startup after reboot (current setup)

Both steps required — the sysctl resets on reboot and q2-q4-imatrix OOMs without it:

```sh
sudo sysctl iogpu.wired_limit_mb=118000
cd ~/Projects/open-source/ds4 && ./ds4-server --ctx 262144 --port 8000 \
  --kv-disk-dir ~/.ds4/server-kv --kv-disk-space-mb 8192
```

This is the pair that serves pi's default model (`ds4/deepseek-v4-flash`).
The disk KV cache (added 2026-07-02) persists prompt prefixes across server
restarts — without it, a restart re-prefills pi's full context from zero
(measured: ~10 min cold prefill for a 146K-token pi prompt at ~250 t/s avg;
warm same-session turns only prefill new tokens either way).

## Local branch: server-enhancements (2026-07-02)

Nine commits implementing the review findings below (all except: engine frontier
snapshots, incremental snapshot storage, eviction rescoring, Anthropic
stop_sequence stop_reason, stop-string rewind — the last two were judged too
invasive for the finish-string plumbing). Highlights: tool-visible live
checkpoint for chat/Anthropic tool turns (eliminates the per-turn 650 MiB
evict+disk-restore for pi), prefill cancellation on disconnect/Ctrl-C,
threshold-based snapshot cadence, deferred consume-unlink, GET /health +
/stats, --max-queue. Verified with `make` + `./ds4_test --server` +
`./ds4_agent_test` + `./ds4-eval --self-test-extractors` (full `make test`
long-context part needs the GPU the production server holds).
**Verified live 2026-07-02 17:05-17:10** after server restart: every pi turn
continues in memory (memory_token hits — the raw-DSML separator fix makes
replays token-exact, so the tool-visible key sits unused as safety net).
Turn overhead dropped from ~80-100 s (evict store + disk load + 25K re-prefill)
to ~2-3 s; decode 25-27.9 t/s; unaligned continued snapshots firing (73728).
/health + /stats live on port 8000. Remaining tune: snapshots are ~1 GB at
76K+ ctx, so raise --kv-disk-space-mb to 16-32 GB on next restart.

**ROOT CAUSE FOUND 17:23 (via --trace)**: the recurring token-mismatch at
33,126 is **pi's "pi-lens" automated-check feature**: it re-sends the whole
conversation with the user message at that turn replaced by
"[pi-lens automated check — not a...". Two interleaved clients (main + lens)
ping-pong the single live session; each lens check costs a ~69K-token prefill
(~4 min) since it genuinely diverges 56K tokens back. NOT tool-result trimming
(earlier hypothesis wrong), NOT a server bug. Fix in pi: disable pi-lens or
route it off this backend. Server-side mitigation already in place: 32 GB disk
budget keeps the main thread's 1.2 GB evict snapshots, so main-thread recovery
after a lens check is snapshot-load + tail prefill, not re-prefill. Still
worth doing: eviction-scoring fix (recent zero-hit waypoints were evicted
first under the old 8 GB budget, cost ~12K extra tokens once).

## Server review findings (2026-07-02, 4-agent code review)

Top enhancement candidates, verified with file:line evidence (full details in session notes):
1. Chat/completions lacks a visible-transcript live checkpoint (responses_live exists,
   ds4_server.c:7995; chat has none, :7719) — root cause of the per-turn 650 MiB
   evict-store + disk-load cycle with pi. Mirror responses_live for chat tool turns.
2. `ds4_session_set_cancel` never wired in server (sync sites :10227/:10251; pattern in
   ds4_agent.c:4179) — disconnected clients burn full prefills; Ctrl-C can't stop prefill.
3. Continued-snapshot cadence `% step == 0` never fires from unaligned resume positions
   (ds4_kvstore.c:741) — causes the 12–20K snapshot gaps; use threshold-crossing.
4. Consume-on-load unlinks snapshot before tail prefill succeeds (ds4_kvstore.c:1322);
   transient Metal OOM then loses both live state and snapshot.
5. No /health,/stats; unbounded queue; dead queued/non-stream clients not detected.
Real bug: parallel tool-call attach duplicates multi-invoke DSML blocks (:8147).

## Usage

```sh
cd ~/Projects/open-source/ds4
./ds4                          # interactive CLI chat
./ds4-server --ctx 262144 --port 8000 --kv-disk-dir ~/.ds4/server-kv --kv-disk-space-mb 8192   # API for pi (see above)
./ds4-agent                    # native coding agent (alpha); use --chdir if launched elsewhere
```

- Only works with the project's own GGUFs from `antirez/deepseek-v4-gguf` — not a
  general GGUF runner.
- Agent sessions persist in `~/.ds4/kvcache` (`/save`, `/list`, `/switch <sha>`).
- Use `--trace` when reporting bugs upstream.