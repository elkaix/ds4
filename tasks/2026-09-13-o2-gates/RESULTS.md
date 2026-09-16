# O2b promotion gates — 2026-09-13

| Gate | Result |
|---|---|
| 1 tensors | PASS — 1328 tensors, 30 upgraded (L33-42 gate/up/down → Q4_K); subsequent byte comparison verifies all 1298 base payloads, 30 donor payloads, and 18 O1-shared payloads |
| 2 ppl (paired, same binary, ctx 32768) | PASS — O1 3.699577 → O2b 3.684018 (−0.42%) |
| 3 official continuation (100 cases, 0731 fixture, relative) | PASS — avg_nll 0.4865 → 0.4658 (−4.3%); first_match 51 → 57; greedy_lcp 4.19 → 4.53; top1 83.57% → 84.13%; per-case NLL better 64/100 |
| 4 prompts (7, temp 0) | PASS — 0 flags (no empty/U+FFFD/repeat/length); all finish=stop; think prompt identical (400) |
| 5 decode/prefill A/B | PASS — all 8 runs complete; deep decode medians −5.72% / −2.11%, deep prefill −7.45% / −3.75% (passes 1 / 2) |
| 6/7 262K + memory | PASS — original 256363-token prompt completed without swapout growth; follow-up fills exactly 262144 total tokens with zero startup-through-shutdown swapouts, pressure normal, no observed decode stalls |
| Daily promotion | PASS — canonical launcher and `ds4flash.gguf` select O2b; :8000 health/models/inference verified, O1 retained as fallback |

O2b planned memory: 97.63 GiB resident; 102.00 GiB at 262144 ctx (KV 2.36 + buffers 2.00).

## Final speed medians

Four observations per arm in each row. These are median comparisons, not a claim
that every individual observation met the budget.

| Window | O1 decode | O2b decode | Change | O1 prefill | O2b prefill | Change |
|---|---:|---:|---:|---:|---:|---:|
| Mid, pass 1 | 39.88 | 39.55 | −0.82% | 528.45 | 475.80 | −9.96% |
| Mid, pass 2 | 34.28 | 32.35 | −5.62% | 269.55 | 247.60 | −8.14% |
| Deep, pass 1 | 31.57 | 29.77 | −5.72% | 495.45 | 458.55 | −7.45% |
| Deep, pass 2 | 27.95 | 27.36 | −2.11% | 381.55 | 367.25 | −3.75% |

## Full-context follow-up

`fill-followup-result.json`: 261888 prompt + 256 completion = **262144** tokens.
The saved O2b cache supplied 249856 tokens; 12032 were prefilled. This supplements
the original long prefill run; it is not a second cold full-prefill benchmark.
The original downloader was stopped entirely before this follow-up. Existing desktop
apps were left open and recorded in `fill-followup-processes.txt`.

- 83 one-second samples span before startup through clean shutdown.
- Swapouts delta **0**; `kern.memorystatus_vm_pressure_level` always **1 (normal)**.
- Peak server RSS **99.77 GiB**; decode **21.55 t/s** at the ceiling.
- 256 streamed content events: median gap **45.8 ms**, p99 **47.4 ms**, maximum **178.7 ms**.
- Existing swap allocation was retained; it did not grow through new swapouts.

Payload proof is in `payload-verification.json` and `payload-verification.log`.
The old `gate1.log` alone verifies metadata, not payload equality. Likewise,
`o2-gates.sh` can print COMPLETE after failures; this verdict uses the recorded
results and explicit follow-up assertions rather than that shell exit status.

## Promotion and cleanup

- Canonical daily server started 2026-09-13 at 14:19 EDT using O2b and its own KV directory.
  `daily-smoke.json` records healthy endpoints, 262144 advertised context, and exact `READY` output.
- O1's model file and separate KV directory remain available via `DS4_ARM=o1`.
- Deleted only approved R7 Q2 base and R6 donor: **178.40 GiB**. The receipt records
  identity checks, no-open-file checks, both deletions, and unchanged O1/O2 identities.
  These files were unlinked, not trashed; rebuilding is required to recover them.
- The HF source checkpoint remains on disk; its deletion was not included in this cleanup.
- GLM download relaunched at the same pinned revision and destination; current PID/log
  and progress evidence are in `download-resume.json`. Original PID 82440 is gone.
- Prior launcher/source changes were preserved. Local note backups are recorded in
  `daily-launch.json`; no commit or push performed.
