# P4b1 #1051 reachability — PASS (2026-09-15)

Instrumented `ds4-bench` only (`DS4_GLM_DISPATCH_COUNT=1`). Timing binaries
`ds4-bench.p4b1-frozen` were not used.

| tag | exact | sg8_checked | sg8_unchecked | generic | verdict |
|---|---:|---:|---:|---:|---|
| 2048-A | 0 | 0 | 0 | 5632 | generic only |
| 2048-B | 5632 | 0 | 0 | 0 | exact, no sg8 |
| 32768-A | 0 | 0 | 0 | 5632 | generic only |
| 32768-B | 5632 | 0 | 0 | 0 | exact, no sg8 |

5632 = 2 frontiers (1024 warm-up + measured) × 256 gen × 11 DSA layers.

Conclusion: do not restart P4b1 for #1051.