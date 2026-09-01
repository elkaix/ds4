# S55-200 Baseline

Status: **measured failure; telemetry completion still pending**

This ledger reports only generated tokens divided by server-reported decode
time. Prefill, cache, page-in, free RAM, and wall latency receive zero S55
credit.

## Provenance

- Source head recorded with the run: `535d6bd230a679dd801cef4a09dd772860097d53`.
  The binary has since been replaced, so the source-to-binary match cannot be
  re-attested; the binary hash below is the exact executable provenance.
- Runtime binary SHA-256: `33214637a2c2876eebdc801c2208a65619a12e6e51f364c76a4af0bfe3e52b13`
- Model: `GLM-5.3-Flash-UNCEN-Q2.gguf`, 96,505,818,432 bytes, inode
  `273408410`, mtime `2026-08-31T18:40:44-0400`; full SHA-256 is pending the
  idle-I/O lease and must be frozen before this becomes a complete Phase 0.
- Hardware/OS: Apple M5 Max, 40-core GPU, 128 GiB; macOS 26.6.2 (`25G83`),
  `arm64`; `iogpu.wired_limit_mb=118000`.
- Runtime: Metal, native width-2 MTP, context ceiling 262,144, one slot,
  `--tokens 32768`, `--power 100`, disk KV budget 131,072 MiB, minimum 2,048
  tokens, cold maximum 65,536 tokens, reject-different-quant enabled.
- Current host toolchain record: Xcode 26.6 (`17F113`), Apple clang 21.0.0,
  Apple Metal 32023.883. The old binary does not embed a verifiable compiler
  identity, so this is environment evidence rather than binary attestation.
- Sampling: temperature 0, `ignore_eos=true`, 512 generated tokens per rung
- Prompt corpus: 1,157,096 bytes, SHA-256
  `78493835239cb7a3b35228bf7304f72d3f293d9b99c4a7ef10e0aca206c7150b`;
  actual prompt-token count comes from the server response. No sliding or
  truncation is accepted.

`python3 speed-bench/build_s55_corpus.py OUTPUT` reproduces the corpus
byte-for-byte from the frozen `b265e22` Git objects. It preserves the original
`en_US.UTF-8` path order and excludes the later untracked `competition` scaffold
and `roadmap.md`. The second copy of `README.md` is part of the frozen input and
is retained for exactness; the content itself is natural repository prose/source
rather than repeated synthetic filler.

## Palindromic Run

| Target | Actual context | Pass 1 t/s | Pass 2 t/s | Scored t/s | ms/token |
|---:|---:|---:|---:|---:|---:|
| 2K | 2,033 | 21.38 cold | 33.01 | 33.01 | 30.29 |
| 32K | 32,150 | 26.64 | 32.38 | 26.64 | 37.54 |
| 64K | 65,093 | 24.15 | 27.19 | 24.15 | 41.41 |
| 128K | 130,506 | 23.67 | 24.84 | 23.67 | 42.25 |
| 200K | 197,395 | 22.75 | 22.48 | 22.48 | 44.48 |

The 2K arm incurred cold model page-in and is not a trustworthy context rung.
It is discarded rather than averaged. Every other rung uses the lower pass,
because S55-200 is a minimum sustained rate, not a mean.

```text
S55-200 observed minimum: 22.48 t/s
Target deficit:           32.52 t/s
2K to 200K decay:         31.90%
Paired 200K drift:         1.19%
```

The benchmark script's printed `22.62 t/s` is the mean of the two 200K passes.
It is not the contract score and is rejected. The large order deltas at 32K,
64K, and 128K also require later ABBA controls; they cannot be described as
context-independent stability.

## Missing Canonical Evidence

- Full model SHA-256 and reproducible source-to-binary attestation
- Width-2 acceptance, committed tokens/cycle, cycle/draft/verify milliseconds at 200K
- Whole-command-buffer GPU span with profiler ON/OFF equivalence
- GPU power/temperature and wired/compressor/swap telemetry
- Four-workload 200K suite and 60-minute drift test

No optimization may use this baseline as proof of a speedup. It establishes
that a real ~200K request runs and that the present configuration fails
S55-200. Candidate gains still require matched A/B measurement.

## Supplementary MTP Timing Diagnostic

A separate logged run over the same deterministic continuation measured 214
accepted and 83 rejected width-2 cycles for 511 accounted tokens at both 2K
and 32K: acceptance `0.720539`, `1.720539` committed tokens/cycle. The timing
logger perturbs throughput, so these counts do not replace the decode ledger.
They also cannot be reused at 200K. At this acceptance, the illustrative S55
cycle budget is 31.2825 ms; the canonical budget remains pending `a200`.
