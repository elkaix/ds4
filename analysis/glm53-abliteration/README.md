# GLM-5.3-Flash abliteration forensics — analysis code and frozen artifacts

Reproduction material for
`docs/research/glm53-abliteration-forensics-2026-08-31.md`.

## Layout

```
lib/glmlib.py    safetensors header reader; BF16 and block-FP8 (OCP e4m3) dequant
lib/gguf.py      minimal GGUF reader; Q8_0 / BF16 / F32 tensor loads
scripts/         one script per experiment, in the order they were run
data/            frozen direction vectors and sampled results
PROVENANCE.md    checkpoint hashes, sizes, and source tensor dtypes
```

## Running

Scripts resolve `lib/` relative to their own location, so they run from
anywhere. They need `numpy` and read the two FP8 checkpoints plus the stock
GGUF from the absolute paths in `lib/glmlib.py` and each script's `GG`
constant — edit those if your layout differs.

```sh
python3 scripts/sweep.py      # the headline alpha / |v.q| sweep over clean KDA layers
```

Requires roughly 700 GB of checkpoints on disk; nothing here is self-contained.

## Scripts, in order

| script | produces |
|---|---|
| `baseline2.py` | first three-way D / O / stock table; establishes the 0.00555 Q8_0 floor |
| `lvec.py` | left singular vectors per layer; Orca global-direction control; L44/MTP subspaces |
| `projform.py` | projection-form discriminator `\|v.q\|`, fitted `alpha`, residual `eps` |
| `experts.py` | first routed-expert probe against `r_g^A` |
| `shexp.py` | shared-expert sweep; locates the L20-42 band |
| `perp.py` | `P_perp` isolation of the Dealign component in expert deltas |
| `sfam.py` | `r_g^E` re-estimation; `S_l` projection-form test |
| `sweep.py` | full KDA sweep, all clean layers (the F38 result) |
| `dsa.py` | DSA-layer sweep; L11 onset bracket; L43 vs L44 |
| `a2.py` | 192-cell routed-expert negative with held-out `r_g^E` |
| `freeze.py` | consolidates `data/directions.npz` |

## `data/directions.npz`

| key | shape | meaning |
|---|---|---|
| `trunk_layers` | (24,) | the clean KDA trunk layers analysed |
| `A` | (24, 4096) | Dealign attention `o_proj` direction per layer |
| `S` | (24, 4096) | Dealign shared-expert `down_proj` direction per layer |
| `rgA` | (4096,) | Orca global attention direction |
| `rgE` | (4096,) | Orca expert-family direction (held-out estimate) |
| `orca_per_layer` | (24, 4096) | Orca's per-layer estimate; all mutually 1.0000 |
| `U44`, `s44` | (4096,8), (8,) | Dealign L44 left subspace and spectrum |
| `U45`, `s45` | (4096,8), (8,) | Dealign MTP left subspace and spectrum |
| `U44_orca`, `U45_orca` | (4096,2) | Orca controls at the same tensors |

Signs of singular vectors are arbitrary; compare with `abs(u_i @ u_j)`.

## Caveats carried from the write-up

- Weight-space only. No refusal or compliance behaviour was measured.
- DSA-layer results are limited to `alpha` and `|v.q|` on the Dealign side.
- The routed-expert negative covers 192 of 12,384 cells.
- Expert-family attribution is conditional on Orca's fixed-direction model
  within that family.
