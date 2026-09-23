# P2 weight-reconstruction sensitivity proxy

Not activation sensitivity. No imatrix. Ranking is ΔNMSE of Q4_K vs Q8_0
on the uncensored BF16 gate/up tensors; gate and up are normalized separately, then summed.

- source: `/Users/panda/models/hf/Qwen3.8-Flash-Next-Uncensored`
- revision: `8336e613ea508b13c2159bd0f68965d97a606b95`
- Q4_K encoder: `libds4quants Q4_K (/Users/panda/Projects/open-source/ds4-qwen38-ivan/gguf-tools/libds4quants.dylib) — production convert path`
- Q8_0 encoder: `gguf-py Q8_0 (bit-exact ggml-quants.c; production convert path — convert does not patch Q8_0)`
- dequant: `llama.cpp gguf-py dequantize_blocks (ggml decoder)`
- L24 Q8 byte match: **True**
- Q4_K control byte match (blk.0): **True**
- layers: 48/48  tensors: 96/96  finite: True

## Rankings

### Overall top 16

| rank | layer | type | combined ΔNMSE | gate ΔNMSE | up ΔNMSE |
| ---: | ---: | --- | ---: | ---: | ---: |
| 1 | 0 | GDN | 0.0112391 | 0.00568098 | 0.00555813 |
| 2 | 2 | GDN | 0.0107044 | 0.00538677 | 0.00531764 |
| 3 | 1 | GDN | 0.0106777 | 0.00537188 | 0.00530582 |
| 4 | 3 | QSA | 0.0105742 | 0.00530852 | 0.0052657 |
| 5 | 5 | GDN | 0.0105638 | 0.00529638 | 0.00526744 |
| 6 | 4 | GDN | 0.010552 | 0.00529967 | 0.00525235 |
| 7 | 38 | GDN | 0.0105178 | 0.00528308 | 0.00523475 |
| 8 | 47 | QSA | 0.0105122 | 0.0052715 | 0.00524071 |
| 9 | 6 | GDN | 0.0104823 | 0.00526877 | 0.00521354 |
| 10 | 9 | GDN | 0.0104812 | 0.0052474 | 0.00523376 |
| 11 | 46 | GDN | 0.0104691 | 0.00526865 | 0.00520041 |
| 12 | 25 | GDN | 0.0104381 | 0.0052241 | 0.00521398 |
| 13 | 14 | GDN | 0.0104192 | 0.00521501 | 0.00520421 |
| 14 | 15 | QSA | 0.0104146 | 0.00520359 | 0.00521096 |
| 15 | 34 | GDN | 0.0104004 | 0.00521717 | 0.00518324 |
| 16 | 22 | GDN | 0.0103971 | 0.00521815 | 0.00517897 |

### Top GDN

| rank | layer | combined ΔNMSE |
| ---: | ---: | ---: |
| 1 | 0 | 0.0112391 |
| 2 | 2 | 0.0107044 |
| 3 | 1 | 0.0106777 |
| 5 | 5 | 0.0105638 |
| 6 | 4 | 0.010552 |
| 7 | 38 | 0.0105178 |
| 9 | 6 | 0.0104823 |
| 10 | 9 | 0.0104812 |
| 11 | 46 | 0.0104691 |
| 12 | 25 | 0.0104381 |
| 13 | 14 | 0.0104192 |
| 15 | 34 | 0.0104004 |
| 16 | 22 | 0.0103971 |
| 17 | 16 | 0.0103951 |
| 18 | 45 | 0.0103944 |
| 19 | 30 | 0.0103898 |

### Top QSA

| rank | layer | combined ΔNMSE |
| ---: | ---: | ---: |
| 4 | 3 | 0.0105742 |
| 8 | 47 | 0.0105122 |
| 14 | 15 | 0.0104146 |
| 21 | 31 | 0.010384 |
| 22 | 7 | 0.0103797 |
| 28 | 39 | 0.0103549 |
| 34 | 27 | 0.0103283 |
| 37 | 35 | 0.0103202 |
| 40 | 11 | 0.0103131 |
| 41 | 23 | 0.0103119 |
| 44 | 43 | 0.010297 |
| 45 | 19 | 0.0102912 |

## Pattern

- GDN n=36 median ΔNMSE 0.0103716
- QSA n=12 median ΔNMSE 0.0103416
- overall min/median/max 0.010285 / 0.0103659 / 0.0112391
- top-16 composition: {'GDN': 13, 'QSA': 3}

## Knee

There is **no sharp sensitivity knee**. Combined ΔNMSE spans only **0.010285–0.011239** (max/min = 1.093). Layer **0** is a mild outlier (+0.000535 above rank 2); after that the ranked list is almost flat (drops ~1e-5). GDN and QSA medians match to ~0.3%. Top-16 is 13 GDN / 3 QSA, close to the 36/12 base rate (would expect 12/4). Layer 24 plumbing sits at rank **31** — not a quality-sensitive layer under this proxy.

P3 must **not** build 16 full GGUFs. Suggested empirical set (one-at-a-time, generate→test→delete):

| role | layer | type | rank | why |
| --- | ---: | --- | ---: | --- |
| top outlier | 0 | GDN | 1 | only layer clearly above the pack |
| early-GDN cluster | 2 | GDN | 2 | next-best, still early trunk |
| top QSA | 3 | QSA | 4 | best QSA, matched-architecture control vs L0/L2 |
| last QSA | 47 | QSA | 8 | opposite end of QSA stack |
| late GDN | 38 | GDN | 7 | late-trunk GDN vs early GDN |
| plumbing | 24 | GDN | 31 | already built; mid-pack proxy |
| median | 8 | GDN | 24 | middle of the flat bulk |
| bottom | 13 | GDN | 48 | least ΔNMSE |

If L0 vs L13 shows no downstream quality gap, the reconstruction proxy does not predict quality and P3/P4 should stop guessing from this ranking.
