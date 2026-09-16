# ds4 branch audit — DeepSeek V4, GLM 5.3, Qwen 3.8, dashboards

Date: 2026-09-15
Rechecked against `antirez/ds4` + open PRs the same day.
Scope: every local branch in the shared ds4 worktree set, `origin/main`, and open PRs that touch those three models or `/dashboard`/`/stats`.
Not done: did not start `run-ds4-monitored.sh`, `run-glm-ds4.sh`, or `run-qwen38-ds4.sh`. They share port 8000, `~/.ds4/monitor.log`, and `pkill -f "thermalforge watch"`.

Verified locally: DeepSeek dashboard `tsc` + eslint + 16 tests pass; GLM and Qwen `tsc --noEmit` pass; `./ds4_test --server` OK on `test/pr954`.

---

## Machine policy (acceptance gate)

This report’s **action** columns assume the daily box, not upstream’s headline SKUs:

- **M5 Max, 40 GPU cores, 128 GB, Metal**
- **Resident / GPU-focused inference** — weights in Metal working set, not SSD-streamed decode
- **Three production trees, not `origin/main`:**
  - DeepSeek V4 Flash Vision-Exp O2b on `test/pr954`
  - GLM-5.3 Flash on the `#964` lineage (`glm53-p4a`)
  - Ivan Qwen3.8 on `qwen38-ivan` (not upstream `#991`)
- A candidate enters a tree only after an A/B against **that tree’s current production SHA**, using the actual quant, context frontier, and long-context workload.
- Required metrics: prefill, steady decode, MTP acceptance, first-token latency, memory footprint, thermals, plus exact/quality parity.
- A PR measured only on M1 / M3 Ultra, or only under `--ssd-streaming`, **does not pass the gate**.

Watch filter: **hard-reject** any optimization whose performance or memory-capacity benefit depends on runtime SSD / offloaded weight access (`--ssd-streaming`, routed-expert streaming, SSD-dependent kernel/scheduling). Disk-backed metadata / n-gram packaging is **not** a performance candidate and must not be recommended unless separately shown not to touch the resident hot path. Flag M5/Metal **resident** prefill, resident decode, MTP, KV/cache, NAX, and correctness changes that apply to these three trees.

---

## Verdict

No upstream dashboard exists. The three local dashboards are not a three-model monitoring system: two `/stats` dialects, one port, one fan-watch killer.

Keep the three specialized trees. Do not collapse them onto `origin/main`. Do not run the three launchers together until C1 is fixed.

| Local model | Keep | Remote action |
|---|---|---|
| **DeepSeek V4 Flash / Vision-Exp O2b** `test/pr954` | yes | Fix dashboard C1/C2 first. Isolated A/B **`#758`**. Consider **`#778`** only if DSpark is actually on. **No V4.1. No SSD.** `#1044` is DeepSeek router correctness, not an M5 Flash speed patch. |
| **GLM-5.3 Flash** `glm53-p4a` | yes (`#964` base justified) | **Take `#1055`.** Take `#1037` if it applies cleanly. **Benchmark `#1051`** before adopting (local guard is already safe). Inspect `#1005` for session/prefix only. **`#1044` is not GLM.** |
| **Qwen3.8 Flash Next Ivan** `qwen38-ivan` | yes (not `#991`) | Agent/server correctness (`#1050`). **`#1047` / `#1056` are out of scope.** Single-GGUF/PLE packaging is **not** a speed item — port only if it keeps resident inference and does not add hot-path disk reads. Never `--ssd-streaming`. |

Live `/dashboard` is whatever is baked into `dashboard_html.h`. On DeepSeek and GLM that header is **clean** while `dashboard.html` is dirty — a running `ds4-server` still serves the last `make dashboard` embed, not the working-tree UI.

---

## Local branch map

One git repo, five worktrees. `origin` = `antirez/ds4`. Local `main` is **18 behind** `origin/main` (Qwen `#991` merge + DeepSeek v4.1 Metal/CUDA + CUDA SSD overlap).

| Branch | Worktree | SHA | vs origin/main | Dirty | Role |
|---|---|---|---|---|---|
| `test/pr954` | `ds4/` | `895fe47` | **81 ahead / 118 behind** | yes | Daily DeepSeek V4 Flash + strongest dashboard |
| `glm53-p4a` | `ds4-glm53/` | `d9a6895` | **11 ahead / 17 behind** | yes | Daily GLM 5.3 on PR `#964` engine |
| `qwen38-ivan` | `ds4-qwen38-ivan/` | `9b48358` | **88 ahead / 18 behind** | launcher/PLE untracked | Ivan M5 Qwen 3.8 + committed dashboard |
| `pr964-engine` | `ds4-glm53-964eng/` | `b3f48f9` | 9 / 17 | — | `#964` engine slice |
| `pr964-base` | `ds4-glm53-964base/` | `bd66c40` | 0 / 17 | — | Upstream “DeepSeek v4.1 Flash support for Metal” |
| `pr-1006` | (not checked out) | `4610665` | 1 / 18 | — | SSE keepalive while tool output withheld |
| `glm53-p4a-pre964` | (not checked out) | `1fa5580` | 77 / 118 | — | Pre-`#964` GLM tree; do not resume |
| `main` | (not checked out) | `6289c51` | 0 / 18 | — | Stale mirror of origin |

Merge-bases with `origin/main`:

- `test/pr954` → `110afdd` (2026-09-01, “Use random completion response IDs”)
- `glm53-p4a` → `bd66c40` (2026-09-12, v4.1 Metal)
- `qwen38-ivan` → `6289c51` (2026-09-08, agent hints)

`origin/main` has **no** `dashboard/` tree and no `/dashboard` HTML. Confirmed: `git ls-tree origin/main` has zero dashboard paths.

Upstream `docs/MODELS.md` (2026-09-15) confirms the model split this audit relies on:

- **V4 Flash ≠ V4.1 Flash** — separate GGUFs, graphs, footprints. V4.1 Q2 is 152 GiB of main weights; on one 128 GB Mac the documented path is `--ssd-streaming`. Not the daily O2b resident server.
- **Qwen3.8 official pack** is now **one 137.10 GiB GGUF**: 41.73 GiB resident main/MTP + **95.37 GiB BF16 n-grams kept on disk**, not mapped or preloaded. Docs: “no second file is needed”; they recommend a fast local SSD. That is **not** `--ssd-streaming` expert/weight streaming, but it is still disk-backed inference data. Local Ivan launcher still requires `--ple`.
- **GLM 5.3 Flash Q2** is the 128 GB Mac target; `#964` is already the local engine.

---

## What each local tree actually has

### DeepSeek — `test/pr954` `895fe47` + dirty worktree

Committed: React `/dashboard` (`3112db2`, `895fe47`, `960ad7d`), Vision-Exp O2b run config, local Metal indexer experiments, n-gram from `#846`.

Uncommitted (the real dashboard): `HealthStrip`, `RequestPanel`, `Chart`, `lib/{health,metrics,config,availability}.ts`, `test/` (16 tests), plus edits to `useStats.ts`, `api/stats.ts`, `ds4_server.c` `/stats` (+624 host/request-ring, **not in HEAD**), `run-ds4-monitored.sh`. `dashboard_html.h` is unmodified — the binary embed is stale vs this worktree.

Runtime: `./run-ds4-monitored.sh` → `127.0.0.1:8000`, O2b GGUF, `--vision`, KV under `~/.ds4/server-kv/...-o2b-...`. No `--ssd-streaming`.

`/stats` schema (DeepSeek-only): `identity`, `host`, `metal`, `process`, `kv`, `mtp`, `last_request`, `checkpoint`, per-path `cache.*`, plus `GET /stats/requests`. Client parses with `parseStats()` (`ds4/dashboard/src/api/stats.ts:309`).

Most relevant open Metal work for **this** tree is `#758` (M5 Max 128 GB indexed prefill, no SSD: attention stage −10.56% median latency, 8K prefill ~+2.94% mean, byte-identical full-prefill logits). Not the V4.1 family and not `#1044` as a speed patch.

### GLM — `glm53-p4a` `d9a6895` + dirty dashboard rewrite

11 commits on top of `#964` / v4.1 Metal, including Metal 4 shader fix (`d9a6895`) and engine cherry-picks. Dashboard **committed** in squash `775c162` as a 705-line instrumented UI (twin of Qwen; uses `ratioStr` / `usePersisted`). Then **rewritten uncommitted** (`App.tsx` +1209/−698, `styles.css` +1023) from ungitted `ds4-dashboard-design/`. That rewrite is a visual restyle that invents numbers. Keep HEAD App; discard the dirty rewrite. `dashboard_html.h` still clean, so the running binary has not picked up the rewrite either.

Runtime: `./run-glm-ds4.sh` → same `:8000`, `--mtp`, KV `~/.ds4/server-kv/glm-5.3-...`.

`#964` on this SKU: independent M5 Max 40-core / 128 GB Metal A/B on the PR reported **+17.6% median decode** (32/32 paired frontiers won) and **−1.9% median prefill** — unlike the larger M3 Ultra headline. Keep the engine; do not assume later GLM opts scale the same way on M5.

`/stats` schema (GLM/Qwen family): `mem`, `proc`, `totals`, `mtp`, `recent[]`, `slots`, `kv_disk`. `queue_rejected` and `queue_dropped_disconnected` are **hardcoded 0** (`ds4-glm53/ds4_server.c:14871-14872`). `send_stats` takes `inference_mu` for MTP (`:14825-14827`) — Qwen explicitly avoided that because it stalls `/stats` for a whole prefill (`ds4-qwen38-ivan/ds4_server.c:14731-14734`).

Engine already returns `glm-5.3-flash` from `server_model_id_from_engine` (`ds4-glm53/ds4_server.c:1190`) and aliases include `glm-5.3-flash` (`:1209`). `send_models()` still lists `glm-5.2` / `-chat` / `-reasoner` because it branches on `ds4_engine_is_glm_dsa` (`:15190-15195`). That is exactly `#1055` (list endpoint only; no inference change).

Padded-row decode: local already forces the checked kernel for GLM 5.3 (`!g->glm53` at `ds4-glm53/ds4.c:56266-56273`). `#1051` tracks a guaranteed-valid prefix instead of treating the whole selection as potentially padded. Correctness-safe architecture; **benchmark on `glm53-p4a` before adopting** (compile-only in the PR, no real-model GPU numbers).

### Qwen — `qwen38-ivan` `9b48358`

88-commit Ivan/M5 tensor-tile tree. Dashboard **committed and clean** (`8f94813`, `b70f312`, `9b48358`). Typecheck passes. No `test`/`lint` scripts in `dashboard/package.json`.

Untracked: `run-qwen38-ds4.sh`, `gguf-tools/qwen4_ple_sidecar.py`. Without the launcher the PLE sidecar is easy to miss; the model refuses to load without `--ple`.

Does **not** contain the origin `#991` merge commit (`f06d3eb`). Rebasing onto current `origin/main` is a conflict mine, not a fast-forward. Keep Ivan as the resident-performance baseline. Useful server/agent fixes (`#1050`) may land. **Do not adopt upstream single-GGUF/PLE packaging merely because it cuts residency by parking 95.37 GiB of n-grams on disk.** Port `e68bcc8` / `97ba327` / `9139e2a` only if they preserve the desired resident inference behavior and do not introduce performance-critical runtime SSD reads. Never enable `--ssd-streaming`, routed-expert streaming, or SSD-dependent kernel/scheduling paths.

**`#1047` / `#1056` are not on the resident roadmap.** `#1047` exists to make Qwen routed experts SSD-streamable so Q2 fits an M1 Max 32 GB. Its +72.7% MTP number compares two SSD-streaming implementations on that box, not resident vs optimized-resident. M5 NAX untested. `#1056` builds on `#1047`; auto dispatch is M1 Max–shaped; the PR says resident inference, other devices, large prefills and one-token decode retain the prior policy for those tuned paths. Do not A/B them on the 128 GB GPU-resident path unless someone extracts a **resident-only** kernel change and proves it on this M5.

---

## Independent review of the monitoring stack

### Critical

**C1 — Not parallel-safe.** All three launchers hardcode `PORT=8000`:

- `ds4/run-ds4-monitored.sh:32`
- `ds4-glm53/run-glm-ds4.sh:17`
- `ds4-qwen38-ivan/run-qwen38-ds4.sh:22`

Each refuses if the port is taken (so they will not bind concurrently), but on stop every monitor runs `pkill -INT -f "thermalforge watch"`:

- DeepSeek `:560`
- GLM `:545`
- Qwen `:606`

That kills **every** ThermalForge watch on the machine, including one owned by another launcher mid-teardown. All three append to the same `~/.ds4/monitor.log`. KV dirs are per-model (safe). Do not start two launchers at once until port/log/fan cleanup are parameterized. Extra copy of the same C1 pattern: `ds4/run-glm-ds4.sh:17` / `:537`.

**C2 — DeepSeek `/stats` mutates host-rate baselines unlocked (uncommitted `ds4_server.c`).** `send_stats` copies counters under `s->mu` then unlocks (`ds4/ds4_server.c:15843-15853`). `stats_json_append_process_and_host` (`:15680`) then writes `s->host_cpu_ns`, `s->host_pageins`, `s->host_disk_r/w`, `s->host_sample_at`, `s->peak_physical_footprint`, `s->swap_bytes_at_start` with no mutex (`:15717-15765`). Dashboard polls `/stats` at 1 s; the launcher monitor polls at 15 s. Two overlapping polls can both compute a rate against the same previous sample, then both publish, producing spike/zero `process_cpu_pct` / `pageins_per_s` / disk rates. Fix before committing that hunk: sample under a dedicated host-stats mutex, or compute rates in the client (GLM/Qwen already do this from cumulative `proc`).

### Warnings

**W1 — DeepSeek request rows are mouse-only.** `RequestPanel` opens the drawer with `onClick` on `<tr>` (`ds4/dashboard/src/components/RequestPanel.tsx:58-61`). Checkboxes are keyboard-reachable; row selection and compare-open are not (`tabIndex` / `onKeyDown` absent).

**W2 — DeepSeek working-tree `useStats` keeps `connection.state === 'live'` after poll failure.** On error, if any poll ever succeeded, dirty `useStats` sets `{ state: 'live', latencyMs: 0, staleS }` (`ds4/dashboard/src/hooks/useStats.ts:107-113`). `App.tsx:86` treats anything other than `'error'` as connected. `evaluateHealth` only adds a stale **warning** after 5 s (`lib/config.ts:26`, `lib/health.ts:37-42`). **HEAD `895fe47` still sets `state: 'error'`.** Do not commit this live-forever change.

**W3 — GLM and Qwen dashboards trust `/stats` JSON.** Both do `(await r.json()) as Stats` with no runtime parse (`ds4-glm53/dashboard/src/useStats.ts:49`, `ds4-qwen38-ivan/dashboard/src/useStats.ts:49`). DeepSeek’s `parseStats()` is the only schema gate.

**W4 — `ds4/run.md` still publishes AProjQ4K / 80.76 GiB numbers under an O2b header.** Header at `:3-5` says O2b. `:25-27` still says “82.74 GiB planned with the AProjQ4K build” and “needs no `iogpu.wired_limit_mb` bump”. Health-check `:77-84` still expects `resident model 80.76 GiB`. Launcher requires ≥118000 MiB (`run-ds4-monitored.sh:38,301`). The doc disagrees with itself and with the launcher.

**W5 — GLM dashboard rewrite lies when idle.** Uncommitted `ds4-glm53/dashboard/src/App.tsx`:

- Default model pill: `"DeepSeek V4 Flash Vision Experimental"` (`:207`)
- Cache table seeds `6` hits and `1` cold before stats arrive (`:157-166`)
- Sparklines fall back to hardcoded SVG paths when `data.length < 2` (`:762-766`)
- “Dropped on disconnect” is the string `"0"` (`:353`)
- Sidebar nav buttons do not change content (`:219-229`)
- `setState` during render for session baseline (`:92-97`) — Qwen has the same pattern (`ds4-qwen38-ivan/dashboard/src/App.tsx:149`)

**W6 — Two incompatible `/stats` dialects.** DeepSeek dashboard cannot consume GLM/Qwen `/stats` (no `cache.memory_token`, no `/stats/requests`). GLM/Qwen dashboards cannot consume DeepSeek `/stats` (no `recent[]`, `mem`, `proc`, `totals`).

Do **not** force GLM/Qwen servers onto the DeepSeek JSON schema. Put a **model adapter** in the dashboard client: DeepSeek `parseStats` + request-ring remain the observability standard; GLM/Qwen keep their tuned inference trees and `/stats` shapes. That keeps dashboard work off hot server paths.

### Refuted (previous note was stale)

| Claim | Now |
|---|---|
| GLM TypeScript fails on unused `ratioStr` / `usePersisted` in `ds4-glm53/dashboard/src/App.tsx:5` | **Stale.** Committed GLM App (705 lines) **uses** both, same as Qwen. Dirty rewrite dropped them. `tsc` passes on GLM, Qwen, and DeepSeek. |
| GLM already contains `#1051` | **Partial.** Local `!g->glm53` already refuses the unchecked padded path (`ds4.c:56273`). It does **not** have `last_indexer_guaranteed_prefix`. `#1051` is a finer, still-safe architecture; benchmark before adopting. |
| origin has a dashboard PR to reuse | **False.** Zero dashboard files on `origin/main`. Ancient `/health`/`ds4_dashboard` PRs are thinner than local. |
| `#1047`/`#1056` should be A/B’d on this Qwen path | **Rejected.** SSD-streaming / M1 Max 32 GB work. Out of scope for 128 GB resident Qwen. |
| `#1044` is an M5 performance take for daily Flash, or a GLM patch | **Rejected.** DeepSeek router softplus correctness; `test_deepseek41_*`; no model-level speed validation. |

---

## Remote PRs — classified for this machine

All listed PRs report `statusCheckRollup: []` (no CI). Author-run numbers. Columns below are **for the three production trees under the machine policy**, not “interesting on GitHub.”

### Take (low-risk, tree-specific)

| PR | Tree | Why |
|---|---|---|
| **#1055** advertise loaded GLM id in `/v1/models` | GLM | 7/3. `send_models` still emits `glm-5.2*` (`ds4-glm53/ds4_server.c:15190-15195`) while `server_model_id_from_engine` already returns `glm-5.3-flash` (`:1190`). List endpoint only. |
| **#1050** Qwen tool-call error names the tag | Qwen | 28/1, `ds4_agent.c`. Agent/server correctness, not kernels. |
| **#1037** Metal object header deps | GLM (and any Metal tree if the hunk applies) | 1/1 Makefile. `ds4.h`/`ds4_image.h` changes rebuild `ds4_metal.o`. No inference risk. |

### Evaluate on this M5, against the named tree’s SHA

| PR | Tree | Why / constraint |
|---|---|---|
| **#758** Metal M5 Max indexed prefill | DeepSeek | **Highest-priority speed candidate for daily Flash.** Author: M5 Max 128 GB, no SSD. Attention stage **−10.56% median latency**; 8K prefill mean **~+2.94%**; byte-identical full-prefill logits. Independent mixed Q2/Q4 M5 stack ~4% prefill. Isolated A/B on `test/pr954` O2b. Conflict risk with local pre-M5 cherry-picks (`8dc038d` merged then `a4308fa` reverted). |
| **#778** DSpark fold + M5 Metal decode | DeepSeek | Measured on **M5 Max 40-core / 128 GB**, not SSD: code 46.8→48.2 t/s, prose 44.8→46.6 t/s, much larger on predictable data. Measured with `ds4f-q2`, **not** Vision-Exp O2b. Isolated A/B **only if DSpark is actually enabled**. Not a cherry-pick. |
| **#1051** GLM padded-row mask | GLM | Prefix tracker through the bounds-checked kernel (`0xffffffff` pads). Local guard already safe. Compile-only in the PR. Benchmark on `glm53-p4a` before adopting. |
| **#1005** GLM live prefix rewind | GLM | Session/prefix behavior only. Read against `glm53-p4a` KV; not a speed patch. |
| **#1006** SSE keepalive while tool output withheld | per-tree | Local `pr-1006` is the commit; Qwen has same-message `0b212d4` (different SHA). Cherry-pick per tree for long tool streams. |

### Already local / do not cherry-pick

| PR | Status |
|---|---|
| **#964** Metal GLM 5.3 accel | `glm53-p4a` **is** this engine (`775c162`). M5 A/B on the PR: +17.6% decode, −1.9% prefill. Keep; do not re-merge the 14k dump. |
| **#991** Qwen3.8 Flash Next (merged 2026-09-14) | On `origin/main`. **Not** an ancestor of `qwen38-ivan`. Do not replace Ivan with origin Qwen. |
| **#990** experimental Qwen Metal port | Superseded by `#991`. Skip. |
| **#954** metal pre-M5 opts | Pieces landed on `test/pr954` then the full ivan merge was reverted (`8dc038d` / `a4308fa`). |

### N/A for the resident production path (do not evaluate)

SSD-streaming, V4.1, wrong model, or M1-32GB-shaped. Not “maybe later if we flip a flag.”

| PR | Why N/A |
|---|---|
| **#1047** Qwen SSD expert streaming | M1 Max 32 GB / Q2 / ctx 4096; compares two SSD implementations. M5 NAX untested. Resident 128 GB Qwen is out of scope. |
| **#1056** Qwen kernel/MTP/SSD overlap | Builds on `#1047`. Auto tiles are M1 Max–shaped. Resident / large prefill / one-token decode keep prior policy. |
| **#848 #849** packed expert SSD + router prefetch | DeepSeek V4 Flash but **SSD**. Daily launcher has no `--ssd-streaming`. |
| **#937** Vision-Exp image + SSD streaming | SSD. Daily Vision-Exp launcher does not stream. |
| **#1041 #1042 #1043** V4.1 decode queue/fuse/Q4 group-6 | Different model. V4.1 Q2 is 152 GiB main weights; 128 GB Mac path is SSD. `test/pr954` has no `ds4_engine_is_deepseek41`. |
| **#1033 #1034 #1035** V4.1 slab / stream decode / parallel engram | V4.1 + SSD/M2 Ultra. |
| **#1044** Metal router softplus | **DeepSeek correctness**, validated via `test_deepseek41_*`, no model-level speed numbers. Not GLM. Not an M5 Flash performance take. Optional later on `test/pr954` as a correctness patch only — do not put it in the GLM apply sequence. |
| **#892 #920** GLM MTP width / exact width-2 verify | `#892` is +19700 on a stale base; `glm53-p4a` has no `DS4_GLM_MTP_WIDTH`. `#920` overlaps local `--mtp`. Do not stack on `#964` blindly; not in the current apply order. |
| **#1036 #1024 #1011 #1052 #1028** | ROCm / Strix Halo / HIP. |
| **#718 #854 #808 #987 #326 #400 #81** | Chat UIs / welcome / ancient dashboard extracts. Local `/dashboard`+`/stats` already better. |
| **#419** expose only the loaded model | Fights GLM/Qwen alias lists. `#1055` is the GLM fix. |
| **#1058** KV checkpoint keys | Draft. Skip. |

### origin/main incoming (the 18 commits local `main` lacks)

Relevant if you fast-forward unused local `main` (not the daily worktrees):

- `f06d3eb` / `ccea768` — Qwen3.8 Flash Next (`#991`)
- `bd66c40` / `a04f46f` — DeepSeek v4.1 Flash Metal + CUDA
- `c2c3ce3` / `e9e1baa` / `6e4c285` — CUDA SSD expert overlap for v4.1
- Qwen n-gram packaging (`e68bcc8`, `97ba327`, `9139e2a`) and steering/tool-tag hardening — packaging is convenience, not a speed candidate (see below)

None of that is a dashboard. Fast-forwarding unused `main` is cheap. Fast-forwarding `test/pr954` or `qwen38-ivan` is not.

Material Qwen **packaging** gap vs origin (not an open PR, and not a reason to abandon Ivan): after `#991`, origin’s 137.10 GiB one-GGUF pack keeps **95.37 GiB of BF16 n-grams on disk** (not mapped/preloaded; docs recommend a fast local SSD). Local `run-qwen38-ds4.sh` still **requires** `--ple`. That is not `--ssd-streaming`, but it **is** disk-backed inference data. Do **not** port `e68bcc8` / `97ba327` / `9139e2a` just to drop the sidecar or shrink residency. Port only if separately shown to preserve resident hot-path behavior with no performance-critical runtime SSD reads.

---

## Recommendations

**D1.** Observability: DeepSeek uncommitted `parseStats` + request ring + 16 tests as the **client** standard. Qwen committed dashboard (`8f94813`, `b70f312`, `0aea700`, `9b48358`) stays. Revert GLM dirty App/styles/`dashboard.html`; keep the committed 705-line GLM dashboard from `775c162`. Unify via a **dashboard adapter**, not by rewriting GLM/Qwen `send_stats`.

**D2.** Before calling monitoring “ready”:

1. Parameterize `PORT`, monitor log path, and ThermalForge cleanup (C1). PID-scope the pkill.
2. Mutex or client-side host rates before committing DeepSeek `ds4_server.c` (C2).
3. Do not commit W2’s live-forever connection state; keep HEAD’s `'error'`.
4. Rebuild `dashboard_html.h` (`make dashboard`) after any UI you actually want served.
5. Commit Qwen’s untracked launcher + PLE sidecar, or the next session cannot start that model.

**D3.** Remote apply order **for this machine**:

1. **DeepSeek `test/pr954`:** fix C1/C2 (and do not commit W2). Isolated A/B `#758` vs current O2b SHA. `#778` only if DSpark is on. No V4.1, no SSD (`#848`/`#937`/`#1033`–`#1035`/`#1041`–`#1043` = N/A). `#1044` is optional DeepSeek correctness, not a speed item, not GLM.
2. **GLM `glm53-p4a`:** keep `#964`. Take `#1055`. Take `#1037` if the Makefile hunk applies. Benchmark `#1051` on this tree before adopting. Inspect `#1005` for prefix/session only.
3. **Qwen `qwen38-ivan`:** keep Ivan. **Do not test `#1047`/`#1056` on the resident path.** Take `#1050` if clean. **Qwen packaging:** do not adopt anything merely because it reduces residency by moving inference data to disk. Port single-GGUF/PLE packaging only if it preserves the desired resident inference behavior and does not introduce performance-critical runtime SSD reads. Never enable `--ssd-streaming`, routed-expert streaming, or SSD-dependent kernel/scheduling paths.

**D4.** Do not run the three launchers at once.

**D5.** Every performance candidate: M5 Max 128 GB + Metal + resident weights + this tree’s quant + same context frontier + exact/quality parity. Separate prefill, steady decode, MTP acceptance, TTFT, footprint, thermals. Benchmark against the tree’s own SHA, not `origin/main`, not short 4K M1 numbers.

Prior runtime context, not re-run: O2b promotion passed and O1 fallback remained; GLM FP8 download was incomplete at the last recorded check.
