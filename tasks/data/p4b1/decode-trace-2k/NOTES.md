# P4b1-C — 2K decode F32 traces (2026-09-15)

## Result: PASS

256 greedy decode frames at ctx 2048, `DS4_METAL_DISABLE_METAL4=1`, MTP off,
hybrid `GLM-5.3-Flash-UNCEN-d21b-L17-18-19Q4KExperts-Q2`.

| | A (base) | B (constructed #964) |
|---|---|---|
| commit | `bd66c40` | `b3f48f9` |
| steps | 256 | 256 |
| tokens | identical | identical |
| F32 frames | byte-identical | byte-identical |
| vocab | 154880 | 154880 |

`worst_abs=0`, `n_diff_elements=0`. This is the decode-step gate #964's own
history requires (frontier logits alone are insufficient). 32K and 200K traces
not yet run.

Artifacts: `logits-2048-A/`, `logits-2048-B/`, `compare.txt`, `arms.txt`.
Harness: `tasks/p4b1_decode_trace.sh` + `DS4_BENCH_DUMP_GEN_LOGITS_DIR` in both
frozen `ds4-bench` binaries.
