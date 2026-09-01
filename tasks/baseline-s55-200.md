# S55-200 Baseline — Superseded Analysis

The raw Phase 1 measurements remain preserved in:

- `tasks/data/baseline-s55-200.json`
- `tasks/data/baseline-s55-200.out`

Use the canonical interpretation in `../baseline-s55-200.md`.

This earlier note is superseded because it reused short-context width-2 MTP
acceptance to calculate a 200K cycle budget and promoted a 32K-knee hypothesis.
The S55 contract forbids both: `a200` must be measured at a real 200K context,
and later direct profiling closed the 32K knee as a special-boundary cause.
Until `a200` exists, the 200K current cycle, target cycle, and remaining
millisecond deficit are **unknown**.
