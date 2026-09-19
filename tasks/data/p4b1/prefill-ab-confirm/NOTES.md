# Quiet-machine A/B confirm at 2K (2026-09-15)

Frozen timing binaries, Metal4-off, MTP off, GEN=64, ABBA n=2.

| arm | pref_med | gen_med |
|---|---:|---:|
| A `bd66c40` | 261.85 | 29.94 |
| B `b3f48f9` | 262.25 | 34.94 |

```text
paired prefill %  [+0.30, +0.00]  mean +0.15%
paired gen %      [+17.11, +16.29]
```

The original P4b1 −13.9% prefill / inconclusive decode was measured under
concurrent-merge noise. On a quiet machine the constructed #964 treatment
shows **flat prefill** and a **~16–17% decode gain**, consistent with the
independent M5 Max official-Q2 median (+17.6%). Still n=2 / GEN=64 /
2K-only — escalate to n=4 GEN=256 before adoption.