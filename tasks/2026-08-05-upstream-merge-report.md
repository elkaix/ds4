# ds4 — upstream merge, verification, and A/B benchmark report

**Date:** 2026-08-05
**Repo:** `~/Projects/open-source/ds4` (fork of `antirez/ds4`)
**Machine:** MacBook Pro M5 Max, 128 GB unified RAM, macOS Darwin 25.5.0
**Outcome:** upstream merged, fully verified on GPU, `prod` moved onto it. Prefill up to **+71%**, decode at depth up to **+14%**.

---

## 1. Why we pulled

The session started as "check remote ds4 and pull fixes and enhancements." The check itself made the case:

- Local `main` was **86 commits behind** `origin/main` (`54b36ed` → `b030961`). The last fetch was 2026-07-31.
- `prod` (`a9ea2d0`) was **32 ahead / 122 behind** upstream.
- Two of those upstream commits are **Metal prefill kernels** — `532ec8b` (routed-MoE prefill) and `222b2cb` (indexed prefill) — which target exactly the path this machine runs.
- **Eleven of our 32 local commits had been superseded.** They were cherry-picks from other contributors' then-open PRs (double-free in the JSON parser, NaN in `json_int()`, O(n) tool-output validation, snapshot `token_count` bound, tokenizer merges bound, GGUF count bounds, agent KV read caps, bash temp-file unlink, Metal pread pool stats, Metal shader path resolution, unterminated reasoning → `reasoning_content`). Upstream had landed its own versions. Carrying local duplicates of already-landed work is pure future merge cost.
- Several new upstream commits land directly on the production configuration: `24fa85e` (keep disk KV checkpoints after successful loads), `7694112` (GLM live-prefix rewind), `7fb2830` and `af80694` (both on the `--dspark` path this machine serves with).

Doing nothing meant the gap keeps widening on a `ds4_server.c` that both sides keep rewriting. That file is the expensive part of every merge, and it gets more expensive the longer it waits.

The decisive argument arrived at the end, empirically — see §6.

---

## 2. Branch and safety setup

Per the user's instruction ("create a new branch, stash our changes"):

| action | result |
|---|---|
| `git stash push ds4.c` | the inert unsloth `vocab_size` fallback parked at `stash@{0}` |
| `git tag -f pre-upstream-merge-20260805 prod` | backup of `a9ea2d0` |
| `git checkout -b upstream-merge-20260805` | integration branch off `prod` |

`prod` and `server-enhancements` (PR #489 head, `b72593f`) were untouched during integration. Nothing was force-pushed anywhere; no remote was written to at any point in this session.

---

## 3. The merge

`git merge origin/main` produced **8 conflict hunks across 3 files** — far smaller than the 49-hunk merge of 2026-07-27, because upstream had absorbed most of what we were carrying.

| file | hunks |
|---|---:|
| `ds4_server.c` | 5 |
| `gguf-tools/deepseek4-quantize.c` | 2 |
| `ds4_kvstore.c` | 1 |

### 3.1 `ds4_kvstore.c` — take upstream, and remove our plumbing with it

Ours (`67a1c6b`) deferred the consume-unlink of a restored KV snapshot until the tail prefill succeeded, so a cancelled prefill could not lose the only recoverable copy. Upstream's `24fa85e` **never unlinks at all** — disk-budget eviction reclaims instead. That is strictly stronger, so ours retires with nothing lost.

The consequence was not optional: upstream **deleted the `consumed` field from `ds4_kvstore.h`**, and that header auto-merged to upstream's version. Every consumer in `ds4_server.c` had to go — `lr.consumed`, the `loaded_consumed_out` parameter through `kv_cache_try_load_text()` / `kv_cache_try_load()`, the `path_consumed` local and its `unlink()`, the `disk_cache_consume` transaction field and its unlink epilogue. Keeping any of it would not compile.

### 3.2 `ds4_server.c` — take upstream's logic, keep our transaction plumbing

The conflict was over unclosed-thinking handling. Both sides had rewritten it for different reasons.

- **Ours:** on a complete DSML block inside unclosed `<think>`, force a `</think>` close and make the model restart the call on the executable side.
- **Upstream (`51a1c14` + `9735ad8`):** a *complete* block inside unclosed reasoning is unambiguous enough to recover directly — no injected close marker, no restart.

Upstream's is better and is the version that will keep receiving upstream fixes. Taken, then rethreaded onto our transaction layer: `ctx_span` → `p->ctx_span`, `req_flags` → `p->req_flags`, `trace_event(s, trace_id, …)` → `production_observe_trace_event(p, …)`.

`chat_think_tool_recovery()` and its `DS4_SERVER_DISABLE_THINK_TOOL_RECOVERY` env guard were deleted with it — half-and-half would have left a dead static and broken the zero-warning bar. Checked first that no test drives that env var; none does.

**Kept ours:** the one-shot log line noting that thinking mode ignores request sampling parameters. It is orthogonal to the conflict and still true.

### 3.3 `gguf-tools/deepseek4-quantize.c` — take upstream

Upstream's `checked_shape_product()` / `checked_size_product()` overflow-safe bounds (`a968c08`) are a superset of our FP8/FP4 declared-shape cross-checks. Straight adoption.

### 3.4 The bug that only the compiler caught

Upstream's `7694112` (GLM live-prefix rewind) **auto-merged cleanly** and referenced a local `cache_diag`. Our tree had moved that diagnostic into the transaction struct. Result: `error: use of undeclared identifier 'cache_diag'`. Repointed at `p->trace_cache.rewind_to`.

This is the recurring lesson from both upstream merges: the dangerous damage is in the files that **do not** conflict.

### 3.5 Dropped commits

None dropped this round. (The 2026-07-27 merge dropped three #555 cherry-picks; that reconciliation was already complete.)

Final merge commit: **`ffc1cc3`**.

---

## 4. Two traps hit during the work

### 4.1 `git checkout --theirs` is whole-file, not per-hunk

`ds4_kvstore.c` had exactly one conflict hunk, so `git checkout --theirs ds4_kvstore.c` looked like a safe shortcut. It is not — it replaces the **entire file** with upstream's version, silently discarding every local change in the non-conflicting regions.

Two of our commits vanished this way:

- `8477b8c` — eviction freshness grace (a just-written snapshot scores like a once-hit file, so the live conversation's newest waypoints are not the first eviction victims while stale never-hit anchors survive on age alone).
- `cc4f3f1` — threshold-crossing continued-store cadence (prefills resumed from an unaligned cached position never land on exact multiples of the step, which used to leave arbitrarily large gaps between waypoint snapshots).

**Symptom:** 9 assertion failures in `./ds4_test --server`. Both were re-applied by hand on top of upstream's version of the file.

**Rule going forward:** resolve hunks in place. Never `--theirs` / `--ours` a file that also carries local changes outside the conflict.

### 4.2 `make` does not rebuild `ds4_test`

After re-applying the kvstore changes, `./ds4_test --server` reported the *same* 9 failures. The changes were correct; the test binary was stale. `make` builds the default target only — test binaries are built by `make test`.

**Rule:** after touching a `.c`, run `make ds4_test` before `./ds4_test --server`, or you will re-read stale failures and start debugging code that is already fixed.

---

## 5. Verification

Everything below was run on this machine, on the merged binary. The GPU was free (no server holding it) so nothing was skipped for contention.

### 5.1 Build

```
make clean && make -j10   → exit 0, ZERO warnings, all five binaries
cd gguf-tools && make     → exit 0, zero warnings
```

`gguf-tools` was built separately and deliberately: `make test` never enters that directory, and `deepseek4-quantize.c` was one of the three conflicted files. It is also the converter that produced the model currently being served.

### 5.2 `make test` — full suite, exit 0

| suite | result | note |
|---|---|---|
| `long-context` | OK | GPU |
| `metal-tensor-equivalence` | **OK** | top1 match on all 5 cases, top5 overlap 5/5, worst `max_abs` 2.90 on the two long cases |
| `metal-kernels` | OK | |
| `metal-short-prefill` | OK | |
| `metal-ssd-streaming-cache-pressure` | OK | |
| `streaming-decode-prefill-correctness` | OK | skipped by env flag (needs `DS4_TEST_SSD_STREAMING=1`) |
| `mtp-verify-depth` | OK | skipped by env flag (needs `DS4_TEST_MTP`) |
| `dspark-verify-depth` | OK | skipped by env flag (needs `DS4_TEST_DSPARK`) |
| `tool-call-quality` | OK | |
| `think-tool-recovery` | OK | exercises the §3.2 resolution |
| `logprob-vectors` | OK | |
| `local-golden-vectors` | OK | |
| `server` | OK | our transaction + kvstore tests, after the §4.1 fix |
| `./ds4_agent_test` | OK | |
| `./ds4-eval --self-test-extractors` | OK | |
| `test_gpu_args_cli` | **44 / 44** | |
| `test_q4k_dot` | **4 / 4** | |
| `test_mxfp4_dot` | **4 / 4** | |

`metal-tensor-equivalence` mattered more than usual this round — upstream added two Metal prefill kernels, so this suite was load-bearing rather than a formality.

Three suites report OK-by-skip (`streaming-decode-prefill-correctness`, `mtp-verify-depth`, `dspark-verify-depth`); they are gated behind env vars pointing at extra GGUFs and were not enabled. Stated here rather than counted as coverage.

### 5.3 Batched-mode serving — the test `make test` does not run

`tests/test_server_batching.py` is what actually drives `slot_worker_main → server_txn_run`, i.e. upstream's batched dispatch running through our transaction state machine. `make test` never invokes it.

Live server on `:8010`, `--ctx 8192 --batched-session 3`:

| mode | shape | result | throughput |
|---|---|---|---|
| non-streaming | 4 pairs / 8 concurrent requests | **PASS** | 13.98 tok/s aggregate, p50 14.7 s, p95 29.2 s |
| streaming | 3 pairs / 6 concurrent requests | **PASS** | 13.14 tok/s aggregate, p50 20.2 s, p95 25.9 s |

Each "pair" sends the same request twice and requires **identical output** — that is the actual assertion. 14 requests, zero aborts.

---

## 6. Benchmarks — why the merge is worth adopting

### 6.1 Method

`llama-benchy` (`~/Projects/BenchMarks/llama-benchy`, run via `uvx --from .`) against a live `ds4-server`.

Both runs used **byte-identical** server flags and benchmark shapes. Only the binary differed.

```
./ds4-server --chdir "$PWD" --metal \
  --model ~/models/gguf/drowzeys-…-DS4-Q2.gguf \
  --ctx 65536 --tokens 32768 \
  --mtp ~/models/gguf/drowzeys-…-DS4-DSpark-support.gguf \
  --dspark --dspark-confidence 0.9 \
  --host 127.0.0.1 --port 8000

llama-benchy --base-url http://127.0.0.1:8000/v1 \
  --model deepseek-v4-flash --served-model-name deepseek-v4-flash \
  --tokenizer deepseek-ai/DeepSeek-V4-Flash-0731 \
  --pp 2048 --tg 128 --depth 0 8192 32768 --runs 2 \
  --extra-body '{"temperature":0}'
```

Model: `drowzeys-keys-DeepSeekV4-Flash-GA-0731-Abliterated-32-32-DS4-Q2.gguf` (87 GB).

### 6.2 Results

| test | prod `a9ea2d0` (before) | merged `ffc1cc3` (after) | delta |
|---|---:|---:|---:|
| pp2048 | 374.66 ± 1.93 | **642.14 ± 11.93** | **+71%** |
| pp2048 @ d8192 | 430.82 ± 0.00 | **584.59 ± 10.67** | **+36%** |
| pp2048 @ d32768 | 377.80 ± 1.87 | **523.17 ± 10.50** | **+38%** |
| tg128 | 39.56 ± 0.00 | 39.32 ± 0.09 | −0.6% (noise) |
| tg128 @ d8192 | 32.23 ± 0.00 | **36.66 ± 0.37** | **+14%** |
| tg128 @ d32768 | 28.08 ± 0.02 | **30.33 ± 0.18** | **+8%** |

Units are tokens/second. `±` is the spread over 2 runs.

The prefill win is the headline and it maps cleanly onto what landed: `532ec8b` (routed-MoE prefill) and `222b2cb` (indexed prefill). Decode at depth improves as a second-order effect. Decode near-empty is flat, which is the expected result — it was never the bottleneck those kernels touch, and the flatness is itself evidence nothing regressed.

For context, decode near-empty was measured at 38.0 t/s on 2026-07-27; 39.3 t/s here is consistent, so the merged binary is at or above the previous best on every axis.

### 6.3 Machine state during the runs

| | merged run | prod run |
|---|---|---|
| RAM used | 116.1 GB | 116.5 GB |
| swap | **0.00 GB** | **0.00 GB** |
| GPU | 86.6% @ 61 W | 100% @ 79.8 W |

Measured with `macmon pipe`. **Neither run needed `iogpu.wired_limit_mb` raised** — the sysctl was `0` throughout, and the 87 GB quant at `--ctx 65536` fits under the default cap with no sudo.

### 6.4 Caveat — these are NOT DSpark numbers

Both servers were started with `--dspark`, but the server ran every request in `THINKING` mode (`chat ctx=… gen=128 THINKING finish=length` in both logs). Thinking mode **discards the request's sampling parameters**, so the `temperature: 0` that DSpark requires never reached the sampler, and no DSpark decode line appears in either log.

Read the table as ordinary greedy-mode throughput. The comparison is unaffected — both sides are non-DSpark for the same reason, so it remains like-for-like. Measuring DSpark properly requires a request that reaches the sampler at temperature 0, which means not being in thinking mode.

### 6.5 llama-benchy invocation notes

- `--model deepseek-v4-flash` must be passed **explicitly**. `ds4-server` advertises two ids (`deepseek-v4-flash` and `deepseek-v4-pro`), so auto-detect refuses with "Multiple models available."
- `--tokenizer deepseek-ai/DeepSeek-V4-Flash-0731` is needed; the served id is not an HF repo name.
- The `ttfr` / `e2e_ttft` columns at depth include the **full depth prefill** because prefix caching was not enabled for these runs. They are not latency-to-first-token for a warm session.

---

## 7. Pythinker — ds4-server added as a provider

Separate request handled in the same session. Pythinker has **no config file on this machine** (`~/.pythinker-ai/` does not exist; the runtime is Docker), so this is a code registry entry, not a JSON edit.

Two files changed in `~/Projects/active/Pythinker`, **uncommitted**:

- `pythinker/providers/registry.py` — new `ProviderSpec` in the local-deployment block: `is_local=True`, `is_direct=True`, `default_api_base="http://127.0.0.1:8000/v1"`, display name `ds4 (DeepSeek V4 — M5 Max)`.
- `pythinker/config/schema.py` — matching `ProvidersConfig` field.

### Two non-obvious decisions

**The provider is named `dwarfstar`, not `ds4`.** `find_by_name()` normalizes through pydantic's `to_snake()`, which turns `"ds4"` into `"ds_4"` and then never matches the spec. A provider literally named `ds4` cannot be selected as a forced default. Verified directly:

```
to_snake('ds4')        -> 'ds_4'
to_snake('ds4_server') -> 'ds_4_server'
to_snake('dwarfstar')  -> 'dwarfstar'
```

**`"deepseek-v4-flash"` is deliberately not a keyword.** The cloud `deepseek` spec sits earlier in `PROVIDERS`, and keyword matching walks that tuple in order — it would claim the model string whenever a DeepSeek API key is configured. Keywords are `("ds4", "dwarfstar")`.

Address the model as **`dwarfstar/deepseek-v4-flash`**.

Verified: `get_provider_name("dwarfstar/deepseek-v4-flash")` → `dwarfstar`; `find_by_name("dwarfstar")` returns the spec with the right base URL and both local flags set.

The pre-existing unrelated modifications in that repo (`.coderabbit.yaml`, `.gitignore`, `.graphifyignore`, `pythinker/web/dist/.gitkeep`) were **not** touched or committed.

---

## 8. Adoption and final state

After the benchmark result in §6, `prod` was moved onto the merge:

```sh
git checkout prod && git reset --hard upstream-merge-20260805
```

This is **not** a fast-forward — the old tip is reachable only via the tag.

| branch / ref | at | role |
|---|---|---|
| `main` | `54b36ed` | upstream mirror (now 86 behind `origin/main`; not advanced this session) |
| **`prod`** | **`ffc1cc3`** | **what we build and serve here** |
| `upstream-merge-20260805` | `ffc1cc3` | duplicate ref of the merge |
| `server-enhancements` | `b72593f` | PR #489 head, in sync with `fork`, untouched |
| `pre-upstream-merge-20260805` | `a9ea2d0` | **rollback point** |

Rebuilt on `prod`: exit 0, zero warnings, all five binaries.

**Rollback:** `git reset --hard pre-upstream-merge-20260805`.

---

## 9. Side effects worth knowing about

- **`ds4flash.gguf` exists again**, symlinked to `~/models/gguf/drowzeys-…-DS4-Q2.gguf`. `make test`'s long-context and Metal suites hard-require it. This supersedes the 2026-07-31 note stating there is intentionally no such symlink.
- **`stash@{0}`** holds the inert unsloth `vocab_size` fallback patch to `ds4.c`. It was never re-applied; the unsloth path is dead. Drop it whenever convenient.
- **`/v1/models` advertises two ids** (`deepseek-v4-flash`, `deepseek-v4-pro`). Pre-existing upstream behaviour at `ds4_server.c:14201`, not new — the server serves whichever GGUF is loaded under either alias. It is why llama-benchy auto-detect refuses, and it means Pythinker's local-model picker will list both.
- **Untracked, not committed:** `gguf_download.log`, `tasks/research-unsloth-ud-q2xl.md`, `CLAUDE.local.md` (the last is in `.git/info/exclude` by design and must never reach `.gitignore`).

---

## 10. Open items

1. **`--ctx 262144` at the default wired limit is untested.** Only `65536` was exercised this session, and it fit with no sudo. The daily-use command uses `262144`; run `sudo sysctl iogpu.wired_limit_mb=118000` first, or verify the smaller allocation holds.
2. **Two batched-mode caveats from the 2026-07-27 merge still stand** — the `worker_bound` `die()` guard in `production_restore` is inert in batched mode, and `production_statistics_record` mutates server-global counters that concurrent slots interleave (`/stats` `busy` is meaningless in batched mode). Cosmetic; would need per-slot stats.
3. **Nothing was pushed.** PR #489 has not been updated against the new upstream and may now be `CONFLICTING`. Reconciling it is the same `ds4_server.c` work, done a second time on a different base — deliberately deferred.
4. **DSpark has never been measured on this machine** (§6.4). Worth a dedicated run outside thinking mode.
5. **Local `main` is still `54b36ed`.** Fast-forwarding it to `origin/main` is zero-risk and was simply not part of the ask.
