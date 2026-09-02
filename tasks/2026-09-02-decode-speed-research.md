# Enhancing DeepSeek V4 Flash decode speed on `ds4-server`, M5 Max 128 GB

Research date **2026-09-02**. Upstream `origin/main` = `110afdd`; local `prod` = `98e3101`
(2026-09-02 merge + tool-schema cherry-pick). Active model =
`DeepSeek-V4-Flash-Vision-Exp-Abliterated-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8` + the
932 MB vision encoder.

Every external claim is cited. Local numbers are cited to `CLAUDE.local.md` / `tasks/`.

---

## 0. Where we actually stand

| measurement | value | source |
|---|---:|---|
| our decode, ctx 262144, short prompt | **41.21 t/s** | local, 2026-09-02 |
| our decode, SuperDeepseek MQ Q2, API | 42.2 t/s | local, 2026-08-13 |
| our decode, Q4_K attention build (unadopted) | **46.6 t/s** mean | local, 2026-08-27 |
| upstream M5 Max reference, Flash 0731 q2, 128-token code prompt | **44.49 t/s ordinary median; 48.19 DSpark median** | [qa] |
| upstream M5 Max reference, 11,707-token prompt | 463.44 prefill / **25.90** decode | [qa] |
| PR #755 headline (merged, we have it) | 39.4 → **45.3 t/s median** | [pr755] |

So we are ~7% under upstream's own short-prompt reference and the gap is plausibly
context depth (our 41.21 was at `--ctx 262144`) plus quant differences. The largest local win we have ever measured on DeepSeek decode came from **moving fewer
bytes per token** (L1, +13.2%), not from kernels — so weight the recipe levers above the
kernel levers. (Do not import the GLM-5.3 "62% of bus" roofline here: that figure is
dominated by KDA projections DeepSeek does not have.)

Depth decay is the largest single loss in real agent use: upstream measures 25.90 t/s at
11.7K tokens against ~44 at short prompts [qa], and PR #755's own author reports the
fusion gain shrinking from +15-22% on toy prompts to +12-13% on 11-24K agent contexts
[pr755]. **Bench at your working depth, not at 2K.**

---

## 1. Ranked levers

### L1 — Q8_0 → Q4_K attention projections + output head (already built, gate unrun)

The single largest measured win available to us, and it is a **model recipe change, not
kernel work**: Ivan Fioravanti's own `perf/decode-50tps` roadmap states the bit-exact
kernel track "landed no e2e decode gain this round" — the +12.6% he reported lives in the
GGUF.

Reproduced locally 2026-08-27: **+13.2% decode** (46.61 vs 41.17 t/s, zero overlap across
8 interleaved arms) and **−11.3% prefill**. `ds4.c`'s
`tensor_expect_glm_dense_quant_layout` already accepts `q4_K` for every attention
projection and the head, so **no code change is needed**.

Blocked on one thing only: **quality**. Ivan reported a CompSec regression at this recipe,
and our imatrix covers routed MoE only — nothing calibrates the 216 tensors the recipe
cuts to 4 bits. The `ds4-eval` baseline exists (74/92, COMPSEC 13/17); arm 2 was never run.

**But it cannot currently be applied to the daily model.** `deepseek4-quantize` takes
`--hf <safetensors dir>` and dequantizes FP8/FP4 safetensors; it cannot requantize a GGUF.
Vision-Exp Abliterated arrived here as a *downloaded GGUF* from `audreyt/…-GGUF`, and
`~/models/hf/` no longer exists (verified 2026-09-02) — so there is no conversion input for
a Vision-Exp Q4_K build unless an upstream safetensors source for that checkpoint exists.

So L1 splits in two:

- **L1a (executable now):** the Q4_K build on disk and the 74/92 `ds4-eval` baseline are
  *both* SuperDeepseek MQ, so the arm-2 A/B is internally valid. Run it —
  `./ds4-eval --plain -m …-MQ-DS4-Q2-AProjQ4K.gguf --trace …` — and it settles whether the
  recipe costs accuracy. **Adopting the result, however, means reverting the daily model
  off Vision-Exp**, giving up vision and the abliteration edit for +13.2% decode. That is a
  product decision, not a perf one.
- **L1b (blocked):** transferring the recipe to Vision-Exp needs a safetensors source for
  `DeepSeek-V4-Flash-Vision-Exp-Abliterated`. Find one first; the 158 GB SuperDeepseek
  source dir is gone too.

**Whenever it is run: guard `--attention-proj` with the 21
`blk.N.indexer.attn_q_b.weight=f16` overrides** — it is a prefix glob and silently demotes
the lightning indexer otherwise. A guarded `--dry-run` must report exactly 216 changes.

### L2 — Speculative decoding, done correctly (blocked by a missing sidecar)

Our own A/B retired `--dspark`: **−15.9% decode**, `net_saved` ≈ −21%, mechanism visible
in the counters (`no_draft=787` of 864 cycles, `scheduler_skips=615`, avg_accept 0.164).

**Upstream has since fixed exactly that mechanism.** PR #833 turns off the immediate
no-draft skip that "treats the first confidence prune as a failed cycle and throws away
later 5-token accepts", and routes bare 1-token drafts through ordinary decode. Measured
on an M5 Max 128 GiB, `--temp 0 --nothink`, n=128 [pr833]:

| prompt | main `84cc882` | #833 |
|---|---:|---:|
| C `add` | 48.63 t/s, skip=7, avg 1.385 | **53.24 t/s**, skip=0, avg 2.875 |
| Redis prose | 41.61 t/s | **43.56 t/s** |
| JSON | 56.87 | 55.73 (wash) |

Companions: **#778** folds the first-token forward into the verify batch and adds
`DS4_DSPARK_CONFIDENCE_GRID` / `_SCHEDULER_BACKOFF` / `_MARGIN_GATE` env knobs [pr778];
**#915** (MERGEABLE) fixes a real indexer sparse-threshold mismatch between decode and the
verifier — `worst_argmax_gap` 0.939 → 0.000 on real weights [pr915]; **#776** (MERGEABLE)
scopes the pause defaults by backend [pr776].

**The blocker is ours, not upstream's.** Vision-Exp ships no matching DSpark support GGUF
and the 0731 sidecar must not be attached — that is open issue **#949**, and #948 asks for
the same file. Until a Vision-Exp sidecar exists, DSpark is unavailable on the daily model
regardless of how good #833 is.

> **Action:** track #949/#948. If a sidecar lands, re-run `tasks/dspark-ab.sh` on a cooled
> machine **with #833 + #915 applied**, not against stock DSpark — our −16% measurement is
> a measurement of the *unfixed* scheduler.

### L3 — n-gram (prompt-lookup) speculation — the speculation that needs no sidecar

PR **#846** adds "prompt-lookup (n-gram) speculative decoding with no support model —
**+16% on repetition-heavy output** (code, edits), free when idle, wired into CLI, server,
and the native agent" [pr846]. Because it drafts from the prompt rather than a draft
model, it **sidesteps #949 entirely** and applies to Vision-Exp today.

Same PR carries a per-device defaults layer (M1/M2 only; M3/M4/M5 keep upstream defaults)
and two default-on wins. The M1-Ultra numbers do not transfer, but the n-gram mechanism
does. CONFLICTING against main, so expect the same `ds4_server.c` reconciliation we do
every merge.

> **Action:** highest-value experiment on this list after L1. Coding-agent output is
> exactly the repetition-heavy shape n-gram speculation wins on.

### L4 — Depth, not peak: cut what you pay per token at 30-60K

Decode at our real working depth is where the tokens are lost (44 → 25.9 t/s by 11.7K
[qa]). Levers:

- **`DS4_METAL_DECODE_INDEXER_SPARSE_THRESHOLD` — a live, untested knob in our tree.**
  `metal_graph_decode_indexer_sparse_threshold()` (`ds4.c:20527`) accepts
  64/128/256/512/1024/2048/4096 and **defaults to 1024**, deliberately keeping dense
  attention past the legacy 512-row window because "around the 2K frontier the sparse
  path's score/top-k setup dominates the smaller attention scan, while larger contexts
  benefit from sparse indexed attention." It changes only *which implementation consumes
  the compressed rows* — it cannot lower the 512-row selection — so **it is bit-exact and
  costs nothing to try**. Nobody has swept it on this machine. At 30-60K working depth,
  512 is the obvious candidate; at short prompts, 2048.
- **PR #169's actual lever is a different, absent, and non-exact one.** #169 cut decode's
  *selection* top-k from the model's 512 to 8 (`DS4_METAL_DECODE_INDEXER_TOP_K`), which is
  an approximation, not a fusion. Verified 2026-09-02: that symbol does not exist anywhere
  in our tree; only #169's `#149` prefill prerequisite reached main. Its **+18.2% / +16.8%
  generation** numbers [pr169] were measured against `613e9b2` where main decoded
  **28.72 t/s** — we are at 41.21 after #149, #755 and `0ad494e`, so treat that percentage
  as a May-2026 figure against a path that has since changed, **not** as headroom we still
  have. Its eval evidence is also thin (a 12-question slice). Worth porting only after the
  free threshold sweep above says the sparse/dense boundary actually matters here.
- **KV disk cache** removes *prefill*, not decode. Do not expect it to move t/s.
- Keep `--ctx` no larger than needed. Our own note: 512K only costs ~1.4 GB more KV than
  256K, but every GB of pre-allocated KV is a GB of pressure, and the local memory
  `ds4-long-context-decay-was-memory-pressure` attributes ~7-21% of observed decay to
  memory pressure/drift rather than context itself.

### L5 — Prefill (TTFT), not decode, but it dominates agent wall-clock

Not decode t/s, but it is what an agent session actually waits on:

- **#873** (MERGEABLE) — `bpe_emit_piece` O(n²) → O(n log n), byte-identical token ids
  [pr873]. An independent tracker measured ds4's BPE loop burning **175-250 s of CPU on a
  24K-token prompt** before inference starts [tracker]. If our prompts are ASCII-heavy the
  effect is smaller (the pathology is no-space/CJK pieces), but the fix is free and safe.
- **#942** detach tokenizer storage from model mmap; **#850** scale default prefill chunk
  with prompt length; **#864** IQ2_XXS MoE prefill half-LUT; **#830/#831/#832** the
  indexer prefill scorer series (#830 alone: bit-exact **+4.7% prefill at 64K**) [pr832].
- Note L1's **−11.3% prefill** cost partially cancels these. Decide by workload: L1 is a
  net win for interactive decode, a net loss for long-prompt ingest.

### L6 — Multi-session aggregate throughput (`--batched-session`)

PR **#799** (draft): per-stream Metal command queues for batched decode, **~1.7x
measured** aggregate, "best paired with `DS4_METAL_MODEL_UNTRACKED=1` (measured **+63%**
from that flag alone under multi-stream)" [pr799]. Its framing matters even if you never
merge it: *"Batch-1 decode on Apple silicon is latency-bound, not bandwidth-bound: an M3
Ultra (2x cores, 1.5x bandwidth of an M5 Max) decodes only ~7% faster."*

That is a genuine tension with the bandwidth-roofline framing in L0/L1. Resolution: the
roofline bounds *aggregate* bytes moved; single-stream decode leaves GPU capacity idle
between dependent skinny kernels, which is why concurrency scales far better than
single-stream speed. Relevant only if you run >1 client. Upstream's own reference shows
GLM-5.3 at **54.49 / 73.96 / 86.17 aggregate rows/s for 2 / 4 / 8 resident sessions** [qa].

### L7 — System hygiene (free, and it is where our A/Bs kept going wrong)

- `sudo sysctl iogpu.wired_limit_mb=118000` after every reboot.
- `--warm-weights`: a cold ~85 GB mmap costs **~25% of decode** and does not shed in one
  512-token warmup — the reason CLI cold-swap A/Bs read +37.7% then +23.5% here.
- Thermals: this machine drifted **21%** between benchmark windows; an independent M5 Max
  report saw ~10% decode decline as the GPU went 51 °C → 98 °C [kingy]. **Interleave every
  A/B.**
- `--power 100`; keep other GPU consumers off; check `lsof -nP -iTCP:8000 -sTCP:LISTEN`
  before blaming the model for a failed start.
- Clear `~/.ds4/server-kv` between benchmark campaigns — it grows to 128 GiB and silently
  warms "cold" runs.

---

## 2. Closed — do not reopen

| lever | verdict | evidence |
|---|---|---|
| `--dspark` **as currently shipped** | −15.9% decode, `net_saved` −21% | local A/B 2026-08-09, 12 samples, zero overlap |
| MTP width-3 (GLM-5.3) | net loss ~0.95x; the marginal *draft* step closes it | local, `DS4_GLM_VERIFY_SCAN` |
| `--cpu-moe` for Q4 on 128 GB | 3.5 t/s vs 26.8 for Q2 Metal | [pr172] |
| `--ssd-streaming` on a resident model | ~4.8 t/s vs ~16.8 resident | [readme] — and #877: cannot combine with `--mtp` |
| llama.cpp as a faster path | 12.9 t/s here vs ~41 in ds4 (DSV4 indexer/HC ops have no Metal kernels) | local 2026-07-31 |
| oMLX | its own residency-thrash bug held decode to 4-17 t/s; fixed 2026-08-28, no figure published since | [tracker] |

---

## 3. Recommended order

1. **Sweep `DS4_METAL_DECODE_INDEXER_SPARSE_THRESHOLD`** (512 / 1024 / 2048) interleaved
   at 2K and at 30K. Bit-exact, already in the binary, one env var. Free.
2. **Run the L1a eval** on the SuperDeepseek Q4_K build already on disk. It answers "does
   4-bit attention cost accuracy" for ~3 h of GPU — and that answer is what unblocks L1b,
   the version that could apply to Vision-Exp.
3. **Find a Vision-Exp Abliterated safetensors source** (or accept that the daily model is
   frozen at IQ2XXS/AProjQ8). Without it L1's +13.2% is unreachable on the current model.
4. **Take #873 + #942 + #832** on the next upstream merge — all MERGEABLE, all
   token-identical or bit-exact, all prefill/TTFT. Cheapest real merge work.
5. **Then #846's n-gram speculation.** +16% on code-shaped output and it sidesteps #949 —
   but it is CONFLICTING and wired into `ds4_server.c`, the file that has cost us 8, 20,
   35 and 49 conflict hunks on four successive merges. Budget it as a merge, not a
   cherry-pick.
6. **Watch #949/#948.** A Vision-Exp DSpark sidecar plus #833/#915 is the path back to
   upstream's 48.19 t/s DSpark median.

Every step: interleaved ABAB on a cooled machine, `--warm-weights`, assert the toggle
actually engaged, and record the arm in the log.

---

## References

[qa]: https://github.com/antirez/ds4/blob/110afdd8886586f18fc9b28bc5533152dd10e728/QA_BEFORE_RELEASES.md
[pr755]: https://github.com/antirez/ds4/pull/755
[pr833]: https://github.com/antirez/ds4/pull/833
[pr778]: https://github.com/antirez/ds4/pull/778
[pr915]: https://github.com/antirez/ds4/pull/915
[pr776]: https://github.com/antirez/ds4/pull/776
[pr846]: https://github.com/antirez/ds4/pull/846
[pr799]: https://github.com/antirez/ds4/pull/799
[pr169]: https://github.com/antirez/ds4/pull/169
[pr873]: https://github.com/antirez/ds4/pull/873
[pr832]: https://github.com/antirez/ds4/pull/832
[pr172]: https://github.com/antirez/ds4/issues/172
[readme]: https://github.com/antirez/ds4
[tracker]: https://dreamingwell.github.io/apple-llm-performance
[kingy]: https://kingy.ai/blog/local-ai-hardware-guide

- QA reference numbers: [qa]
- PR #755 merged Metal fusions: [pr755]
- DSpark scheduler fix: [pr833] · fold+env knobs: [pr778] · threshold fix: [pr915] · backend scoping: [pr776]
- n-gram speculation: [pr846]
- Batched per-stream queues: [pr799]
- Decode indexer top-k: [pr169]
- Tokenizer O(n²): [pr873] · indexer prefill top-k: [pr832]
- cpu-moe: [pr172] · SSD streaming and session table: [readme]
- Third-party trackers: [tracker], [kingy]
