# GLM-5.3-Flash abliteration forensics: Dealign vs OrcaRouter

Date: 2026-08-31. Weight-level comparison of two uncensored GLM-5.3-Flash
checkpoints against a stock baseline. All conclusions are about **weights**, not
behaviour; no activations were measured.

## Method

- Checkpoints: `dealignai/GLM-5.3-Flash-ABLITERATED-FP8` (D),
  `orcarouter/GLM-5.3-Flash-Uncensored-FP8` (O), both block-FP8 e4m3 128x128.
- Stock baseline: `GLM-5.3-Flash-Q2.gguf`, whose `kda_output` / `attn_output`
  tensors are **Q8_0**. Empirical Q8_0 comparison floor `n = 0.00555` relative
  Frobenius, estimated from L0-10, where D-vs-stock residuals show no
  recoverable low-rank structure and are consistent with quantization error.
- Expert tensors are IQ2_XXS in that GGUF, so no stock baseline exists for them.
  Worked around via `P_{r_g^E}^\perp`: Orca's expert edit is `r_g^E a^T`, so the
  `r_g^E`-orthogonal part of the pairwise delta isolates Dealign's component.
  For any direction `s` orthogonal to `r_g^E`, `W_O^T s == W_stock^T s` exactly,
  which supplies a stock proxy for the projection-form test.
- Layer typing per `glm53_quantize.py:698`: trunk L0-44, `L%4==3` are DSA,
  L45 = MTP.
- **Source dtype differs by layer type**, and this, not the baseline, governs
  which conclusions hold where. In both HF checkpoints, KDA `o_proj` is stored
  **BF16** while DSA `o_proj` is stored **block-FP8** (`F8_E4M3`,
  `weight_scale_inv` [32,128]). The stock GGUF tensor is Q8_0 for both, so a
  Q8_0 baseline does not remove DSA noise: the FP8 error lives in D and O
  themselves. Their `scale_inv` blocks differ from each other by up to 6.4%,
  i.e. independent requantization.
- Confirmed empirically by residual sharing. After removing each model's
  leading rank-1 component, `cos(res_D, res_O)` = 0.88-0.92 at KDA layers with
  both residuals at 0.0057 (the shared Q8_0 baseline error, as designed), but
  only 0.12-0.19 at DSA layers, where `|res_O|/|W|` = 0.0234-0.0246 against
  Dealign's 0.0055.
- Consequence: `rho1` and projection residual `eps` are comparable across KDA
  layers only. At DSA layers, `alpha` and `|v.q|` remain trustworthy **on the
  Dealign side** (see below), and no rank conclusions are drawn.

## Mechanism classification

Projection form is tested by whether the rank-1 delta's right singular vector
satisfies `v_l || W_l^T u_l`, then fitting `dW_l = -alpha_l u_l (u_l^T W_l)`.

### OrcaRouter: fixed direction within each writer family, alpha ~ 1

- `|u_i . u_j| = 1.0000` across all 33 clean `o_proj` layers; 1.0000 at L44,
  0.9983 at MTP.
- `|v . q| = 0.9999` and `alpha = 0.9982-0.9994` at every layer. Flat.
- Routed experts carry the same construction but a **measurably different**
  vector: `|r_g^E . r_g^A| = 0.9568`. The expert-delta covariance contains a
  dominant rank-1 spike: the leading eigenvalue accounts for 29.6% of the trace,
  while the next four individually account for only 0.26%, 0.17%, 0.13%
  and 0.12%.

Orca additionally **requantizes the DSA `o_proj` tensors**; Dealign does not.
At L7 and L11, which neither model edits, Dealign's residual against stock is
exactly the Q8_0 floor (0.00553) while Orca's is 0.0246. Dealign preserves the
stock FP8 payload on untouched DSA layers; Orca re-saved them. This asymmetry
is what leaves the Dealign side of the DSA layers measurable.

Orca uses the same projection-form construction with a fixed direction *within*
each measured writer family and near-unit strength. The routed-expert direction
is highly aligned with, but measurably distinct from, the attention direction.
Unless the 0.9568 gap can be attributed to quantization or estimation error, a
single identical `r_g` across all writer families is not supported.

### Dealign: layer-local projection families with strength schedules

Attention `o_proj` family `A_l`, L12-42:

| L | alpha_A | \|v.q\| | rho1 corrected | cos(A_l, A_prev) |
|---|--------:|--------:|---------------:|-----------------:|
| 12 | 1.000 | 0.9998 | 0.9962 | - |
| 18 | 1.224 | 0.9998 | 0.9974 | 0.434 |
| 22 | 1.356 | 0.9998 | 0.9981 | 0.235 |
| 26 | **1.469** | 0.9998 | 0.9981 | **0.0004** |
| 32 | 1.244 | 0.9997 | 0.9972 | 0.012 |
| 37 | 0.718 | 0.9997 | 0.9931 | 0.015 |
| 42 | 0.389 | 0.9944 | 0.9711 | 0.218 |

Shared-expert `down_proj` family `S_l`, L20-42, peak alpha 1.324 at L26,
`|v.q|` up to 0.988, within 0.01 of the unconstrained rank-1 bound.
Cross-family `|S_l . A_l| <= 0.108`.

DSA `o_proj` family, L15-43. A fourth writer family with its own onset and the
same unimodal schedule shape:

| L | alpha_D | \|v.q\| |
|---|--------:|--------:|
| 15 | 0.967 | 0.982 |
| 19 | 1.094 | 0.981 |
| 23 | 1.255 | 0.987 |
| 27 | **1.360** | 0.987 |
| 31 | 1.154 | 0.984 |
| 35 | 0.777 | 0.970 |
| 39 | 0.421 | 0.963 |
| 43 | 0.218 | 0.930 |

Peak at L27, against L26 for both the KDA and shared-expert families. Only
`alpha` and `|v.q|` are reported here; `rho1` and `eps` are withheld because
Dealign's DSA deltas also carry FP8 error (see Method).

L43 is an **ordinary** late-trunk projection, not a precursor to L44: spectrum
sigma/sigma1 = 1.000, 0.114, 0.093, 0.078, and `|u_43 . u_44,1| = 0.108`, so it
is rank-1 dominant and not inside L44's subspace. L44 remains an isolated
exception. `|u_43 . u_MTP| = 0.387` places L43 with L38-42 in leaning toward
MTP geometry.

Routed experts: after removing `r_g^E`, `rho1 = 0.036` and `F(A_l)` sits on the
random-orthogonal control (2.4e-4). No detectable Dealign low-rank edit in the
16 expert x layer cells sampled.

L44: neither projection-form (`|v.q| = 0.753` vs Orca's 0.995) nor explained by
rank <= 8 down to the noise floor (eps 0.312 vs 0.12 noise).
MTP L45: dominated by a rank-1 component (eps 0.316 vs ~0.23 FP8 floor) but
**not** projection-form (`|v.q| = 0.220`, projection-constrained eps 0.978).

## Principal result: separable strength and direction control

The full sweep covers all 33 ordinary clean KDA layers. Within the detectable
Dealign projection region, L12-42, `alpha_A` follows a smooth unimodal depth
schedule: approximately 1.000 at L12, rising continuously to approximately
1.469 at L26, then declining to approximately 0.389 by L42. Outside that
region, L0-10, the delta sits at the comparison floor and `alpha_A` is not
meaningfully measurable as a projection strength; those fitted values are
degenerate and are not used.

In contrast, adjacent `A_l` direction similarity undergoes an abrupt transition
at L25->L26, falling from approximately 0.226 to 0.0004 while `alpha_A` remains
continuous through its maximum.

**The weight-edit pattern therefore exhibits separable strength and direction
control.** The smooth scalar gain schedule and the discontinuous direction
geometry vary independently with depth. This rules out explanations in which
the observed direction transition is merely a consequence of changing edit
magnitude. It does not identify the procedure that generated either quantity.

This is consistent with separate underlying mechanisms or controls, but does
not establish their implementation.

### L26 is an aligned structural transition

Three independently measured properties change character at L26:

- `alpha_A` maximum, approximately 1.469
- `cos(A_26, A_25)` = 0.0004, a 500x drop in one layer step
- `alpha_S` maximum, approximately 1.324

Before L26 the attention direction family rotates with correlation while
strength rises; after it, directions are largely orthogonal while the scalar
schedule decays. This identifies L26 as a common transition depth in the
weight-edit geometry, but does not by itself establish a common generator or
causal coupling.

### Onset is a step, at full layer resolution

L11 is directly measured and is **below the detectable floor**: `dD = 0.0055`
(the Q8_0 comparison floor exactly), `|v.q| = 0.192`, `alpha = 0.0014`,
`rho1 = 0.0005` -- identical to the unedited L3 and L7. L12 already exhibits
essentially exact projection form with `|v.q| = 0.9998` and `alpha = 1.000`.

With both bracketing layers measured, there is no ramp: the KDA intervention
switches on between L11 and L12 at full strength. The DSA family switches on
separately and later, between L11 and L15.

### Estimator control

Orca measured through the identical pipeline gives flat `|v.q| = 0.9999` and
`alpha = 0.9982-0.9994` across the same 33 layers. The Dealign curve and the
L26 transition are therefore not estimator artifacts.

## Summary

**Orca:** projection-form edits, `alpha ~ 1`, fixed direction within each
measured writer family; `r_g^A` and `r_g^E` highly aligned but not identical.

**Dealign, ordinary trunk:** writer-family-specific, layer-local projection
edits; smooth depth-conditioned `alpha_l`; independently varying direction
field; aligned structural transition at L26. Three distinct writer families
with distinct onsets and a common schedule shape: KDA `o_proj` L12-42
(alpha peak 1.469 at L26), DSA `o_proj` L15-43 (peak 1.360 at L27),
shared-expert `down_proj` L20-42 (peak 1.324 at L26).

**L44 and MTP:** separate non-projection-form regimes.

The most significant result is not that `alpha` peaks at L26. It is that a
continuous scalar gain field passes smoothly through L26 while the direction
field undergoes an abrupt geometric transition there, giving weight-level
evidence that edit magnitude and edit direction are distinct degrees of freedom
in Dealign's construction.

## Behavioural / capability measurement

Both checkpoints were converted to Q2 under the identical plan (both
96,505,818,432 bytes) and scored on the same 100-case Z.ai continuation
fixture. Average NLL: stock 0.458949, abliterated 0.475750, uncensored
0.469144, abliterated Q4_K 0.299178.

The uncensored Q2 is 1.39% better than the abliterated Q2 but the paired test
gives t = -1.90 (SE 0.004024, better on 61/100), and its 85/100 first-token
match is not a real regression (McNemar exact p = 0.727). **Neither weight-level
mechanism difference documented above produces a capability difference
resolvable at n = 100.** Full numbers and receipts:
`abliterated-ds4-alternative-2026-08-28.md`.

This is a capability measurement only. No refusal or compliance behaviour was
measured, and none of the structural findings above should be read as
predicting behavioural aggressiveness.

## Limits

- `o_proj` (KDA and DSA) and shared/routed expert `down_proj` only.
- DSA-layer conclusions are restricted to `alpha` and `|v.q|` on the Dealign
  side, because DSA `o_proj` is block-FP8 at source in both checkpoints and
  Orca requantized those tensors.
- Expert-family attribution is conditional on the validated fixed-direction
  rank-1 Orca model *within the expert writer family*, using the empirically
  estimated `r_g^E`.
- L0-10 states no edit **above the empirical Q8_0 floor**; a smaller edit is
  not excluded.
- drowzeys' reported 0.74 L44 delta is not reproducible under Frobenius-relative
  normalization (measured 0.0476); recorded as a metric-definition discrepancy.
- The capability comparison is underpowered at 100 cases and cannot separate
  the two interventions.
