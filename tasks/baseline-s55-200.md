# baseline-s55-200 — the first trustworthy 200K measurement

Phase 1 of the S55-200 contract. No optimisation is attempted here; the point is
to stop extrapolating 60K to 200K.

## Provenance (Phase 0)

```text
binary        ds4-server built 2026-09-01, tree at commit 535d6bd
              (run-glm-ds4.sh carries uncommitted local edits; ds4.c does not)
model         ~/models/gguf/GLM-5.3-Flash-UNCEN-Q2.gguf   97 GB
server args   --metal --mtp --ctx 262144 --tokens 32768 --power 100
              --kv-disk-dir ~/.ds4/server-kv/glm-5.3-flash-uncen-q2
              --kv-disk-space-mb 131072 --kv-cache-min-tokens 2048
              --kv-cache-cold-max-tokens 65536 --kv-cache-reject-different-quant
machine       M5 Max, 128 GB, fans forced to maximum (fan0 5349, fan1 5777 RPM)
prompt        natural text: agent notes + engine README + ds4_server.c source
              (competition/… corpus200.md, 1.157 M chars, no repeated filler)
generation    512 tokens per point, temperature 0, ignore_eos, reasoning none
order         palindrome 2K 32K 64K 128K 200K | 200K 128K 64K 32K 2K, one process
```

## Result

| context | ascending pass | descending pass | ms/token (descending) | page-ins, ascending |
|--------:|---------------:|----------------:|----------------------:|--------------------:|
|   2,033 |          21.38 |       **33.01** |                 30.29 |              55,415 |
|  32,150 |          26.64 |       **32.38** |                 30.88 |             290,009 |
|  65,093 |          24.15 |       **27.19** |                 36.78 |             510,463 |
| 130,506 |          23.67 |       **24.84** |                 40.26 |             836,544 |
| 197,395 |          22.75 |       **22.48** |                 44.48 |             837,226 |

```text
S55-200  =  min over the ladder  =  22.48 t/s
target                              55.00 t/s
deficit                             32.52 t/s   (2.45x)
```

**The two passes are not equally trustworthy and must not be averaged.** The
ascending pass climbs while the machine is still faulting the 97 GB checkpoint
in: it logs 55 k page-ins at 2K rising to 837 k at 200K, and 171,916 swap-ins in
a single 64K request. The descending pass runs on a machine that has already
touched everything, and its page-in counts fall monotonically (836 k → 11 k).
The ascending 2K point in particular is a cold-mmap first request, which is a
measurement of the page cache, not of context.

So the descending column is the context curve and the ascending column is a
memory-pressure trace. Reported together because the difference between them is
itself a finding: **at 2K, page-in pressure alone costs 35% of decode
throughput** (21.38 vs 33.01).

## Context decay

On the clean descending pass:

```text
2K   -> 200K     33.01 -> 22.48     -31.9%
2K   -> 32K                          -1.9%    within the contract's 2%
32K  -> 64K                         -16.0%
64K  -> 128K                         -8.6%
128K -> 200K                         -9.5%
```

The gate allows 2%. The knee is between 32K and 64K, and it is not a smooth
1/N curve: 2K to 32K is essentially free, then every doubling costs ~9%.

That shape matters. A cost that grew with the number of attended tokens would
show up between 2K and 32K too. A cost that switches on at a threshold would
look exactly like this — and there is a known threshold in that interval:
`glm_graph_verify_rows_eligible` refuses any position past
`dense_limit = 4096`, after which MTP verification falls back from the
decode-style row pass to `glm_graph_forward_indexed_tokens` (ledger F23). The
32K point already sits above that boundary, so the boundary alone does not
explain the 32K→64K step; something in the sparse path is also scaling.

## Cycle budget at 200K

At the measured width-2 acceptance a = 0.721 (measured at short context; **a200
has not yet been measured and must be**, per contract §5):

```text
tokens per cycle              1 + a          = 1.721
observed cycle at 200K        1.721 / 22.48  = 76.56 ms
observed cycle at 2K          1.721 / 33.01  = 52.14 ms
context penalty                              = 24.42 ms/cycle

cycle required for 55 t/s     1.721 / 55     = 31.29 ms
reduction required at 200K    76.56 - 31.29  = 45.27 ms   (59%)
```

For scale: the whole measured forward pass at short context is 44.6 ms, and the
bandwidth roofline for one token is 8.35 ms. **S55-200 requires removing more
time than a whole short-context forward pass takes.** It cannot come from
kernel tuning.

## What is now open

1. `a200` — acceptance at 200K. Every number above that uses 0.721 is provisional.
2. The 24.42 ms context penalty, attributed. DSA sparse path and the verify-path
   fallback are the two candidates and they are separable.
3. Whether decode at 200K is swap-limited. The descending pass says largely not
   (218 to 543 swap-ins at 64K/128K), but the ascending pass says it can be.
