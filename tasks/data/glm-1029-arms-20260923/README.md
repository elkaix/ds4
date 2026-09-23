# Issue #1029 attribution on glm53-m5-prod stage 2 (2026-09-23)

Harness `tasks/glm_1029_arms.sh`, summary `tasks/glm_1029_arms_summary.py`.
Base f935161c; arm diffs in `arms.txt`. Arms B and C are inexact, speed only.
Stopped by the user after 2 of 3 cycles; partial cycle 3 is in `partial-cycle3/`.

- Continued 2K on 14K: neither revert recovers anything (B/A x0.974, C/A x0.994).
  The issue's -11% does not reproduce here. Ledger: glm.perf.1029-continued-null-stage2.
- Cold: both reverts +3-4% at 8K/16K, -2-4% at 24K, n=2. Open.
  Ledger: glm.perf.1029-cold-open-stage2.
- Decode steady: 30.9-37.9 t/s, one low run (cont14k-B-2-1, 30.90).
