# P4b1-A3 phase1 — FLASH_TUNING aggregate (2026-09-15)

Frozen B binary `ds4-bench.p4b1-frozen` (`c5b572279c468844`), 2K, GEN=64,
Metal4-off, ABBA.

| label | n | pref_med | gen_med | first_ms |
|---|---:|---:|---:|---:|
| default (B) | 2 | 261.28 | 34.81 | 28.297 |
| flash_off | 2 | 262.17 | 30.01 | 32.993 |

Prefill gain vs default: **+0.34%** — below the 5% fan-out gate. Family
rollbacks skipped.

`DS4_METAL_DISABLE_GLM53_FLASH_TUNING=1` does **not** explain the A→B prefill
regression. It does cost decode (~14% gen). The prefill regression lives
outside this aggregate switch (or those prefill opts are not selected on this
hybrid/Metal4-off shape). Next: confirm A↔B still shows the regression on
frozen binaries, then isolate elsewhere.