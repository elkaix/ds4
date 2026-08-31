# Abliterated GLM-5.3 Flash local DS4 evaluation

Date: 2026-08-29 (America/New_York)

Status: conversion, scoring, and HTTP serving verified locally at 4K context;
long-context validation remains open

Scope: Apple M5 Max, 128 GB unified memory, Metal, DS4, GGUF, and the
OpenAI-compatible local server

## Decision

Use `GLM-5.3-Flash-ABLIT-Q2.gguf` for resident local serving. It is the best
resident abliterated GLM-5.3 artifact measured here.

Use `GLM-5.3-Flash-ABLIT-Q4_K.gguf --ssd-streaming` when continuation quality
matters more than latency. It has the best score among the artifacts measured
here, but its 177.77-GiB size prevents full residency on this host.

Do not use `GLM-5.3-Flash-ABLIT-Q2-imatrix.gguf` as the default. The calibrated
build improved first-token match and greedy-prefix length, but worsened the
primary average-NLL metric by 0.672% versus the uncalibrated abliterated Q2.

The previous recommendation to keep SuperDeepseek because no exact GLM-5.3
artifact was usable is superseded. The exact GLM-5.3 FP8 checkpoint has now
been converted to GGUF; its derived Q2 has been validated, scored, and served
through DS4 locally.

## Verified artifacts

| Artifact | Local path | Exact bytes | Verification |
| --- | --- | ---: | --- |
| Abliterated FP8 source | `/Users/panda/models/hf/GLM-5.3-Flash-ABLITERATED-FP8/` | 62 safetensor shards; 306 GB on disk | Complete source layout |
| Resident Q2 | `/Users/panda/models/gguf/GLM-5.3-Flash-ABLIT-Q2.gguf` | 96,505,818,432 | 1,412 tensors; validator passed |
| Calibrated Q2 | `/Users/panda/models/gguf/GLM-5.3-Flash-ABLIT-Q2-imatrix.gguf` | 96,505,818,432 | 1,412 tensors; validator passed |
| Q4_K | `/Users/panda/models/gguf/GLM-5.3-Flash-ABLIT-Q4_K.gguf` | 190,875,528,512 | 1,412 tensors; validator passed |
| Uncensored FP8 source | `/Users/panda/models/hf/GLM-5.3-Flash-Uncensored-FP8/` | 62 shards; 328,337,455,904 | Complete; tokenizer byte-identical to base |
| Uncensored Q2 | `/Users/panda/models/gguf/GLM-5.3-Flash-UNCEN-Q2.gguf` | 96,505,818,432 | 1,412 tensors; 67.8m conversion |
| Routed-MoE imatrix | `/Users/panda/models/gguf/glm53-ablit-routed-moe-ds4-400k.dat` | 507,253,623 | 364 prompts; 400,000 tokens; 97.67% expert coverage |

The calibrated and uncalibrated Q2 files have identical quantization-type and
role byte totals but different payload bytes, as expected.

## Quality evidence

All five rows use the same official Z.ai FP8 continuation fixture: 100 cases
and 11,559 target tokens, scored through the identical `score_official`
invocation at `--ctx 4096 --quality`. Lower average NLL is better.

| Artifact | Average NLL | First-token match | Average greedy LCP | Result |
| --- | ---: | ---: | ---: | --- |
| Stock Q2 reference | 0.458949357 | 89/100 | 7.450 | Reference |
| Abliterated Q2, no imatrix | 0.475749935 | 87/100 | 7.740 | **Resident default** |
| Abliterated Q2, 400K imatrix | 0.478948951 | 89/100 | 8.090 | NLL regressed 0.672% versus no-imatrix Q2 |
| **Uncensored (orcarouter) Q2** | 0.469144034 | 85/100 | 8.820 | 1.39% better NLL than abliterated Q2, but not significant (t = -1.90) |
| Abliterated Q4_K | **0.299178196** | **91/100** | **10.870** | **Best measured quality; SSD streaming required** |

Raw receipts:

- `/Users/panda/models/gguf/qscores/stock-q2.tsv`
- `/Users/panda/models/gguf/qscores/ablit-q2-noimx.tsv`
- `/Users/panda/models/gguf/qscores/ablit-q2-imx.tsv`
- `/Users/panda/models/gguf/qscores/ablit-q4k.tsv`
- `/Users/panda/models/gguf/qscores/uncen-q2.tsv`

The uncalibrated abliterated Q2 is 3.661% worse than stock Q2 by average NLL.
The calibrated Q2 is 4.358% worse than stock Q2. Q4_K is 34.812% better than
stock Q2 on this fixture, but that comparison combines model-weight changes
with a much higher-precision quantization; it does not isolate abliteration
drift.

## Runtime evidence

The resident Q2 was served through `ds4-server` at `--ctx 4096` and returned
the exact requested text `RESIDENT_OK` through `/v1/chat/completions`:

- planned memory: 96.67 GiB;
- prefill: 10.04 tokens/s;
- decode: 30.49 tokens/s;
- response: 19 prompt tokens and 4 completion tokens;
- shutdown: clean after the request completed.

The Q4_K quality run used Metal SSD streaming. Its score proves that the model
loads and completes the full fixture in streaming mode; it is not evidence of
resident or low-latency HTTP serving.

## Imatrix conclusion

DS4 collected 134,400,000 routed-expert observations from 364 rendered prompts
and stopped at 400,000 tokens. Coverage reached 12,096 of 12,384 expert slots,
or 97.67%.

That calibration is not a quality win under the selected metric. Keep the
imatrix and calibrated GGUF as experiment receipts, but select the
uncalibrated Q2 for resident service.

## Uncensored (orcarouter) Q2 comparison

`GLM-5.3-Flash-UNCEN-Q2.gguf` converts from the orcarouter FP8 checkpoint under
the same plan as the abliterated Q2 and lands at the identical
96,505,818,432 bytes, so the two are directly comparable within one
quantization class.

Against the abliterated Q2 it is **directionally better but not separable on
this fixture**:

- paired per-case mean `delta_nll` = -0.007662, SE 0.004024, **t = -1.90**;
  better on 61 of 100 cases
- first-token match 85 vs 87 is **not** a real regression: 5 discordant
  abliterated-only against 3 uncensored-only, **McNemar exact p = 0.727**
- average greedy LCP 8.820 is the highest of the three Q2 rows and 18% above
  stock's 7.450

For scale, abliterated-vs-stock is t = +1.98 on the same test, so both
intervention effects sit below the resolution of a 100-case fixture.

**Conclusion: the uncensored Q2 neither demonstrably recovers capability
relative to the abliterated Q2 nor degrades it.** The 39% apparent closure of
the abliterated-to-stock NLL gap is not statistically supported at n = 100. The
resident default is unchanged pending a larger fixture or a Q4_K-class
comparison where quantization noise is not dominant.

Weight-level mechanism analysis of the two interventions:
`glm53-abliteration-forensics-2026-08-31.md`.

## Reproduce the comparison

```sh
./gguf-tools/quality-testing/score_official \
  /Users/panda/models/gguf/GLM-5.3-Flash-ABLIT-Q2.gguf \
  gguf-tools/quality-testing/data/glm53-flash-openrouter-zai-fp8-100/manifest.tsv \
  /Users/panda/models/gguf/qscores/ablit-q2-noimx.tsv \
  4096 --quality

python3 gguf-tools/quality-testing/compare_scores.py \
  /Users/panda/models/gguf/qscores/ablit-q2-noimx.tsv \
  /Users/panda/models/gguf/qscores/ablit-q2-imx.tsv
```

## Open validation limits

- Only 4K context was exercised through HTTP during this audit. Do not claim
  32K or longer request support until a prompt exceeding 4,096 tokens succeeds.
- Q4_K was scored through SSD streaming, not tested through `ds4-server`.
- Q4_K plus the collected imatrix has not been built or scored.
- The uploader reports MMLU 87.33% versus 86.74% base and HarmBench compliance
  320/320. Those are provenance claims, not independent local measurements.
- Local refusal evidence remains a smoke probe, not a representative refusal
  suite.
- The uncensored-vs-abliterated NLL comparison is underpowered at 100 cases
  (t = -1.90). Do not report it as a capability win without a larger fixture.

## Sources

- [Abliterated FP8 release](https://huggingface.co/dealignai/GLM-5.3-Flash-ABLITERATED-FP8)
- [Official GLM-5.3 Flash source](https://huggingface.co/zai-org/GLM-5.3-Flash)
- `gguf-tools/glm53_quantize.py`
- `gguf-tools/glm53_validate_gguf.py`
- `gguf-tools/quality-testing/README.md`
- `gguf-tools/quality-testing/compare_scores.py`
