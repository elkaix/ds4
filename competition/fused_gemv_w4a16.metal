// fused_gemv_w4a16.metal — fused 4-bit dequantization + GEMV for Apple Silicon.
//
// y[M] = W[M,K] * x[K]      W: 4-bit group-quantized, x/y: half (W4A16)
//
// Design rules this kernel is built on, and why:
//
//  1. Weights are never written back to DRAM in FP16. Nibbles are unpacked in
//     SIMD registers and consumed immediately. A GEMV at batch 1 is purely
//     memory bound, so any dequantize-to-scratch pass would exactly double the
//     bytes moved and halve throughput. Arithmetic intensity is derived in
//     PERFORMANCE.md; the short version is I = 2*K / (0.5*K + eps) ~ 4 FLOP per
//     byte, versus ~200 FLOP/byte of available compute, so bytes are the only
//     thing that matters here.
//
//  2. Structure of arrays, not blocked structs. The usual `{half d; half m;
//     uchar qs[16];}` block struct is 20 bytes, so consecutive blocks straddle
//     cache lines and every load is misaligned. Quants, scales and zeros live
//     in three separate arrays instead: the quant stream is then contiguous and
//     128-byte aligned per row, which is what saturates the bus.
//
//  3. One SIMD group per output row, lanes striding the K axis by 16 bytes
//     (ulong2 = 32 nibbles = one quant group). Each lane's loads are 16-byte
//     aligned and the 32 lanes of a SIMD group cover 512 contiguous bytes, i.e.
//     four full 128-byte cache lines per step, with no gaps.
//
//  4. No threadgroup_barrier and no threadgroup memory on the hot path. The
//     only cross-lane communication is the final reduction, done with
//     simd_shuffle_down (and simd_sum where the target supports it), which is
//     register-to-register inside one SIMD group and needs no barrier at all.
//     The activation vector is read straight from device memory: it is shared
//     by every simdgroup in the threadgroup, so it is served from L2 after the
//     first touch, and staging it would cost a barrier to save nothing.
//
// Integration point in DS4: this is the shape of kernel that
// `kernel_mul_mv_*` in ds4_metal.m provides for q4_K but does not provide for
// IQ2_XXS or Q2_K, which is what this repository's GLM-5.3 checkpoint actually
// uses (86 IQ2_XXS + 43 Q2_K expert tensors). See the note at the end of this
// file for the 2-bit variant.

#include "quant_common.metalh"

struct GemvW4A16Params {
    uint M;              // output rows
    uint K;              // reduction length, multiple of QK4
    uint row_stride_q;   // bytes between rows of `quants`   (>= K/2)
    uint row_stride_g;   // groups between rows of scales/zeros (>= K/QK4)
    uint x_stride;       // elements between columns of x (1 for a plain GEMV)
    uint n_cols;         // number of activation columns (1 = GEMV, >1 = thin GEMM)
    uint has_bias;
    uint fuse_silu;      // apply x*sigmoid(x) to the result before storing
};


// ---------------------------------------------------------------------------
// kernel_gemv_w4a16
//
// grid:  (M rounded up to rows-per-threadgroup) x n_cols
// tg:    SIMD_WIDTH * ROWS_PER_TG threads
//
// Each SIMD group owns one output row; ROWS_PER_TG simdgroups share the
// threadgroup so that the activation cache lines they all read are fetched
// once into L2 and reused ROWS_PER_TG times.
// ---------------------------------------------------------------------------

constant constexpr int ROWS_PER_TG = 8;   // 8 * 32 = 256 threads

kernel void kernel_gemv_w4a16(
        device const uchar            *quants   [[buffer(0)]],
        device const half             *scales   [[buffer(1)]],
        device const half             *zeros    [[buffer(2)]],
        device const half             *x        [[buffer(3)]],
        device const half             *bias     [[buffer(4)]],
        device       half             *y        [[buffer(5)]],
        constant     GemvW4A16Params  &p        [[buffer(6)]],
        uint3  tgpig   [[threadgroup_position_in_grid]],
        uint   sgitg   [[simdgroup_index_in_threadgroup]],
        uint   tiisg   [[thread_index_in_simdgroup]]) {

    const uint row = tgpig.x * ROWS_PER_TG + sgitg;
    if (row >= p.M) return;

    const uint col = tgpig.y;
    const device half *a_base = x + (ulong)col * p.x_stride * p.K;

    const device uchar *qrow = quants + (ulong)row * p.row_stride_q;
    const device half  *srow = scales + (ulong)row * p.row_stride_g;
    const device half  *zrow = zeros  + (ulong)row * p.row_stride_g;

    const uint n_groups = p.K / QK4;

    float acc = 0.0f;

    // Lane `tiisg` takes groups tiisg, tiisg+32, tiisg+64, ... Each iteration is
    // one 16-byte aligned load per lane; the SIMD group sweeps 512 contiguous
    // bytes, four whole cache lines, per iteration.
    for (uint g = tiisg; g < n_groups; g += SIMD_WIDTH) {
        const device ulong2 *qptr =
            reinterpret_cast<const device ulong2 *>(qrow + (ulong)g * BYTES_PER_GROUP);
        const ulong2 packed = *qptr;

        const float scale = (float)srow[g];
        const float zero  = (float)zrow[g];

        acc += dot_group_q4(packed, a_base + (ulong)g * QK4, scale, zero);
    }

    // Register-to-register butterfly reduction across the 32 lanes. No
    // threadgroup memory, no barrier: SIMD groups execute in lockstep, so the
    // shuffles need no synchronization of any kind.
    acc = simd_reduce_add(acc);

    if (tiisg == 0) {
        if (p.has_bias) acc += (float)bias[row];
        if (p.fuse_silu) acc = silu(acc);
        y[(ulong)col * p.M + row] = (half)acc;
    }
}

// ---------------------------------------------------------------------------
// kernel_gemv_w4a16_4row
//
// Bandwidth-optimal variant for the case that actually dominates decode: the
// same activation vector against several weight matrices (q/k/v, or gate/up).
// One pass over x, four independent weight streams, four outputs. Saves three
// reads of x and, more importantly, three kernel launches.
// ---------------------------------------------------------------------------

kernel void kernel_gemv_w4a16_x4(
        device const uchar           *quants0  [[buffer(0)]],
        device const half            *scales0  [[buffer(1)]],
        device const half            *zeros0   [[buffer(2)]],
        device const uchar           *quants1  [[buffer(3)]],
        device const half            *scales1  [[buffer(4)]],
        device const half            *zeros1   [[buffer(5)]],
        device const half            *x        [[buffer(6)]],
        device       half            *y0       [[buffer(7)]],
        device       half            *y1       [[buffer(8)]],
        constant     GemvW4A16Params &p        [[buffer(9)]],
        uint3  tgpig   [[threadgroup_position_in_grid]],
        uint   sgitg   [[simdgroup_index_in_threadgroup]],
        uint   tiisg   [[thread_index_in_simdgroup]]) {

    const uint row = tgpig.x * ROWS_PER_TG + sgitg;
    if (row >= p.M) return;

    const uint n_groups = p.K / QK4;
    float acc0 = 0.0f, acc1 = 0.0f;

    const device uchar *q0 = quants0 + (ulong)row * p.row_stride_q;
    const device uchar *q1 = quants1 + (ulong)row * p.row_stride_q;
    const device half  *s0 = scales0 + (ulong)row * p.row_stride_g;
    const device half  *s1 = scales1 + (ulong)row * p.row_stride_g;
    const device half  *z0 = zeros0  + (ulong)row * p.row_stride_g;
    const device half  *z1 = zeros1  + (ulong)row * p.row_stride_g;

    for (uint g = tiisg; g < n_groups; g += SIMD_WIDTH) {
        const device half *a = x + (ulong)g * QK4;

        const ulong2 p0 = *reinterpret_cast<const device ulong2 *>(q0 + (ulong)g * BYTES_PER_GROUP);
        const ulong2 p1 = *reinterpret_cast<const device ulong2 *>(q1 + (ulong)g * BYTES_PER_GROUP);

        acc0 += dot_group_q4(p0, a, (float)s0[g], (float)z0[g]);
        acc1 += dot_group_q4(p1, a, (float)s1[g], (float)z1[g]);
    }

    acc0 = simd_reduce_add(acc0);
    acc1 = simd_reduce_add(acc1);

    if (tiisg == 0) {
        // gate/up fusion: y0 holds silu(gate) * up when fuse_silu is set.
        if (p.fuse_silu) {
            y0[row] = (half)(silu(acc0) * acc1);
        } else {
            y0[row] = (half)acc0;
            y1[row] = (half)acc1;
        }
    }
}

// ---------------------------------------------------------------------------
// 2-bit variant — the format this repository's checkpoint actually uses
//
// GLM-5.3-Flash-UNCEN-Q2 stores its expert tensors as 86 IQ2_XXS + 43 Q2_K
// tensors. The structure above carries over unchanged; only the unpack does
// not. Q2_K packs four 2-bit values per byte with a 16-element sub-block scale
// and a super-block (256-element) scale, so a lane's 16-byte load covers 64
// weights rather than 32, and the group scale becomes a product of two
// quantities. That halves the bytes per weight again, which is why the same
// SIMD-group structure gets closer to the bus limit here than it does at 4 bit.
// ---------------------------------------------------------------------------


kernel void kernel_gemv_w2a16(
        device const uchar           *quants  [[buffer(0)]],
        device const half            *scales  [[buffer(1)]],
        device const half            *zeros   [[buffer(2)]],
        device const half            *x       [[buffer(3)]],
        device       half            *y       [[buffer(4)]],
        constant     GemvW4A16Params &p       [[buffer(5)]],
        uint3  tgpig   [[threadgroup_position_in_grid]],
        uint   sgitg   [[simdgroup_index_in_threadgroup]],
        uint   tiisg   [[thread_index_in_simdgroup]]) {

    const uint row = tgpig.x * ROWS_PER_TG + sgitg;
    if (row >= p.M) return;

    const uint n_groups = p.K / QK2;
    const device uchar *qrow = quants + (ulong)row * p.row_stride_q;
    const device half  *srow = scales + (ulong)row * p.row_stride_g;
    const device half  *zrow = zeros  + (ulong)row * p.row_stride_g;

    float acc = 0.0f;
    for (uint g = tiisg; g < n_groups; g += SIMD_WIDTH) {
        const ulong2 packed =
            *reinterpret_cast<const device ulong2 *>(qrow + (ulong)g * BYTES_PER_GROUP_Q2);
        acc += dot_group_q2(packed, x + (ulong)g * QK2, (float)srow[g], (float)zrow[g]);
    }

    acc = simd_reduce_add(acc);
    if (tiisg == 0) y[row] = (half)acc;
}
