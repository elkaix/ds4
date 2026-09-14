# Bandwidth, arithmetic intensity, and where the 55 t/s has to come from

Every number below is either measured on this machine (M5 Max, 128 GB, DS4 at
`535d6bd`, `GLM-5.3-Flash-UNCEN-Q2.gguf`) or derived from the GGUF header.
Nothing here is a vendor figure except the 546 GB/s bus peak, and that one is
used only as an upper bound, never as an achieved rate.

## 1. What a token actually costs

From the GGUF header:

```text
total parameters              309.5 B
active parameters / token      14.6 B      4.7%
bytes per parameter             0.312      2.49 bits
active bytes / token            4.56 GB
```

Roofline:

```text
4.56 GB / 546 GB/s  =  8.35 ms/token   ->  120 t/s   (bus limit, unreachable)
```

Measured decode at 2K, warm, MTP width-2 on: **~34 t/s**, i.e. a 50.7 ms cycle
producing 1.721 tokens. Against the roofline that is **23% of peak bandwidth**.
So the deficit is not "not enough bandwidth"; it is 4.3x of something else.

## 2. Arithmetic intensity of the three kernel families

$I = \dfrac{\text{FLOPs}}{\text{bytes transferred}}$

| Kernel | FLOPs | Bytes | $I$ | Regime |
|---|---|---|---|---|
| W4A16 GEMV, K=4096 | $2K$ per row | $0.625K$ per row | **3.2** | memory |
| W2A16 GEMV (this checkpoint) | $2K$ | $0.3125K$ | **6.4** | memory |
| Fused SwiGLU expert, I=2048 | $6IK$ | $1.875IK/K$ … see below | **3.2** | memory |
| Paged flash-decode, L tokens | $4LD$ | $4LD$ (half K+V) | **1.0** | memory |

The M5 Max sits at roughly 200 FLOP/byte of balance. Every one of these is two
orders of magnitude below it. **There is no compute-bound kernel in this
model at batch 1.** Any optimization that trades bytes for arithmetic wins; any
that adds a byte of traffic loses, however clever the math.

That is the entire justification for the register-resident dequantization in
`fused_gemv_w4a16.metal`: writing dequantized FP16 weights back to DRAM would
turn 0.625 bytes/weight into 2.625 bytes/weight, a 4.2x traffic increase, in
exchange for saving an unpack that is free in a memory-bound kernel.

### Fused SwiGLU traffic reduction

Unfused, per expert per token:

```text
read x         H * 2          =   8 KB
read Wg, Wu    2 * I*H * 0.625 = 10.5 MB
write g, u     2 * I * 2       =   8 KB
read g, u      2 * I * 2       =   8 KB
read Wd        H*I * 0.625     =  5.25 MB
write out      H * 2           =   8 KB
                                 -------
                                 15.78 MB
```

Fused (`fused_swiglu_moe.metal`), the intermediate never leaves threadgroup
memory and x is read once for both gate and up:

```text
15.78 MB - 24 KB  =  15.76 MB      ratio 0.9985
```

**A 0.15% traffic reduction.** This is the honest number, and it is why the
fused FFN is not the S55 lever. The intermediate is 2048 halfs against 15.75 MB
of weights; fusing it saves essentially nothing on bandwidth. What it saves is
*dispatches* — and DS4 already collects that win: `kernel_mul_mv_id_iq2_xxs_
pair_swiglu_pack2_overlap_f32` already pairs gate and up, fuses SwiGLU, and
packs two experts per dispatch, and `kernel_mul_mv_id_q2_k_sum6_f32` already
sums six experts in one. Measured routed FFN cost is 3 dispatches/layer,
126 per pass.

Corroborated by direct measurement (ledger F22): a **dense** 151 M-parameter
layer costs 0.804 ms and a **routed** 226 M-parameter layer costs 0.72–0.83 ms.
If the MoE path were inefficient those would differ. They do not.

## 3. Where the pass actually goes

Per-layer cost from the step-1 verify sweep at pos = 2651 (ledger F22):

```text
DSA layers (il % 4 == 3), 11 of 45   2.110 ms each
KDA layers,               34 of 45   0.826 ms each
embed + 154,880-vocab head + host    0.35 ms      (<1% of the pass)

whole pass                          44.6 ms
  of which DSA premium              14.1 ms      32%
```

Rank separation between the two groups is perfect — every DSA layer is more
expensive than every KDA layer. Removing the DSA premium entirely, which is not
achievable but bounds the lever, gives a 30.5 ms pass, a 37 ms cycle, and
**46 t/s**. Still short of 55.

## 4. The structural statement

At the measured acceptance rate $a = 0.721$:

```text
tokens per cycle          1 + a          = 1.721
cycle budget for 55 t/s   1.721 / 55     = 31.29 ms
current cycle                              50.70 ms
required reduction                         19.41 ms   (38.3%)
```

Verification alone is 43.70 ms of that cycle — **already above the entire
budget**. Even at perfect width-2 acceptance the budget is only
$2/55 = 36.36$ ms, still under the current verify cost.

So: the current schedule cannot reach 55 t/s by tuning any single kernel. The
hardware can (8.35 ms roofline). What has to change is the schedule.

## 5. Occupancy and register budget

`fused_gemv_w4a16.metal`, `kernel_gemv_w4a16`:

```text
threads/threadgroup      256   (8 simdgroups, one output row each)
threadgroup memory         0
registers/thread         ~40   (16 unpacked bytes + 2 accumulators + addressing)
occupancy limit          register-bound at ~6 concurrent threadgroups/core
bytes/lane/iteration      16   (ulong2, 16-byte aligned)
bytes/simdgroup/iter     512   (4 whole 128-byte cache lines, no gaps)
```

`fused_swiglu_moe.metal`, `kernel_swiglu_ffn_fused`:

```text
threads/threadgroup      256
threadgroup memory      4 KB   (I = 2048 halfs)
concurrent TGs/core        8   (32 KB limit / 4 KB)
```

`paged_hybrid_attention.metal`, `paged_hybrid_attention_d128`:

```text
threads/threadgroup      128   (4 simdgroups splitting the KV axis)
registers/thread          VPT = D/32 = 4 for q + 4 for the accumulator, plus
                          the running (m, l) pair: ~24 registers
threadgroup memory      2.1 KB (4 simdgroups x 128 floats + 8 scalars)
```

Every one of these is comfortably inside the 32 KB threadgroup limit, so
occupancy is register-bound rather than memory-bound, which is the correct side
to be bound on for a bandwidth-limited kernel.

## 6. Porting the 4-bit unpack to this checkpoint's real formats

The expert tensors here are **not** W4A16:

```text
blk.3..45.ffn_gate_exps   43 x IQ2_XXS   (4096, 2048, 288)
blk.3..45.ffn_up_exps     43 x IQ2_XXS   (4096, 2048, 288)
blk.3..45.ffn_down_exps   43 x Q2_K      (2048, 4096, 288)
```

`dot_group_q2` in `quant_common.metalh` implements the straightforward 2-bit
unpack (4 values per byte, one 16-byte lane load covering 64 weights). The two
real formats differ from it as follows, and both differences matter:

**Q2_K** is a 256-weight super-block: 16 sub-blocks of 16 weights, each with a
4-bit scale and a 4-bit min packed into one byte, plus an FP16 super-block scale
and an FP16 super-block min. The effective weight is
$w = d_{\text{super}} \cdot s_{\text{sub}} \cdot q - m_{\text{super}} \cdot m_{\text{sub}}$.
The lane structure above is unchanged; only the scale lookup gains one level of
indirection, resolved once per 16 weights rather than per weight.

**IQ2_XXS** is not a scalar quantization at all. It is a codebook format: each
32-weight block stores 4 uint16 indices into a 256-entry table of eight-weight
sign-magnitude patterns, plus a 4-bit scale. Unpacking is a table lookup, not a
shift-and-mask, so the register-resident argument gets *stronger* (the codebook
is 2 KB and belongs in threadgroup memory or constant space, read once per
threadgroup and reused by every lane), while the "eliminate the multiply"
optimization in `dot_group_q4` does not apply.

`ds4_metal.m` already implements both, fused with the pairing and SwiGLU, so
the port target here is the *scheduling* of those kernels, not their inner loop.

## 7. What this says about the submission

Ranked by measured whole-pass scope:

| # | Change | Scope | Status |
|---|---|---|---|
| 1 | Overlap host encoding with GPU execution | whole pass | **measuring** — the pass is encoded into one command buffer and committed only at the end, so encode does not overlap execution. `DS4_METAL_GPU_BUSY_PROFILE=1` on a decode-only (appending) request gives wall-minus-busy directly. |
| 2 | DSA verification chain | 14.1 ms of 44.6 | bounded at ~46 t/s, needs a second lever |
| 3 | Row verifier above the 4096 dense window | long context only | required for the 32K/64K/128K/200K rungs; today `verify_rows` refuses past `dense_limit` and MTP falls back to the indexed batch path |
| 4 | Width-3 MTP | +7.1% | real but small |
| 5 | Fused expert FFN kernels | 0.15% of traffic | already shipped in DS4; closed |
