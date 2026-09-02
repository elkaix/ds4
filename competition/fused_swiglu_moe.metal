// swiglu_fused.metal — the whole expert FFN in one dispatch.
//
//   h    = silu(Wg . x) * (Wu . x)        [I]      I = 2048 for GLM-5.3 experts
//   out  = Wd . h                         [H]      H = 4096
//
// The naive form is three GEMVs and three round trips of the intermediate
// vector through DRAM, plus three kernel launches per expert. With 8 routed
// experts and a shared expert per token across 42 MoE layers, that is 1134
// dispatches per forward pass just for the FFN.
//
// This kernel does all of it in one dispatch per (token, expert):
//
//   phase 1  gate and up are projected together, one pass over x, and the
//            SiLU product is written to threadgroup memory. x is read once.
//   phase 2  the down projection reads h out of threadgroup memory. The
//            intermediate never touches DRAM.
//
// Threadgroup budget: h is I halfs = 4 KB at I = 2048, well inside the 32 KB
// limit, so several threadgroups stay resident per core.
//
// One barrier separates the two phases. There is no other synchronization:
// the reductions are simd_shuffle_down butterflies inside a SIMD group.
//
// Integration point in DS4: `glm_graph_verify_rows` reuses the batch FFN
// encoder (routed split + big-gate combine). The measurement in tasks/ledger.md
// (F22) found the routed MoE FFN is *not* the anomaly in the forward pass -- a
// dense layer with 151 M parameters costs the same 0.80 ms as a routed layer
// with 226 M -- so this kernel is a dispatch-count and intermediate-traffic
// win, not the answer to the S55-200 deficit. It is priced honestly in
// PERFORMANCE.md.

#include "quant_common.metalh"

constant constexpr int SG_PER_TG_FFN = 8;                             // 8 simdgroups
constant constexpr int TG_THREADS_FFN = SIMD_WIDTH * SG_PER_TG_FFN;   // 256
constant constexpr int MAX_INTERMEDIATE = 4096;                       // I upper bound

struct SwiGLUParams {
    uint  hidden;         // H, reduction length of gate/up and output width of down
    uint  intermediate;   // I, output width of gate/up and reduction length of down
    uint  n_tokens;
    uint  gate_row_stride_q;   // bytes  between rows of the gate/up quant streams
    uint  gate_row_stride_g;   // groups between rows of the gate/up scale streams
    uint  down_row_stride_q;
    uint  down_row_stride_g;
    float expert_weight;       // routing weight, folded into the store
    uint  accumulate;          // 1 = out += w * result (MoE combine), 0 = out = result
};

// ---------------------------------------------------------------------------
// kernel_swiglu_ffn_fused
//
// grid:        (n_tokens, 1, 1)
// threadgroup: TG_THREADS_FFN
//
// buffers:
//   0  x            half  [n_tokens][H]
//   1  gate_q/s/z   (2,3)
//   4  up_q/s/z     (5,6)
//   7  down_q/s/z   (8,9)
//   10 out          half  [n_tokens][H]
//   11 params
// ---------------------------------------------------------------------------
kernel void kernel_swiglu_ffn_fused(
        device const half         *x        [[buffer(0)]],
        device const uchar        *gate_q   [[buffer(1)]],
        device const half         *gate_s   [[buffer(2)]],
        device const half         *gate_z   [[buffer(3)]],
        device const uchar        *up_q     [[buffer(4)]],
        device const half         *up_s     [[buffer(5)]],
        device const half         *up_z     [[buffer(6)]],
        device const uchar        *down_q   [[buffer(7)]],
        device const half         *down_s   [[buffer(8)]],
        device const half         *down_z   [[buffer(9)]],
        device       half         *out      [[buffer(10)]],
        constant     SwiGLUParams &p        [[buffer(11)]],
        uint  tgpig [[threadgroup_position_in_grid]],
        uint  sgitg [[simdgroup_index_in_threadgroup]],
        uint  tiisg [[thread_index_in_simdgroup]]) {

    const uint tok = tgpig;
    if (tok >= p.n_tokens) return;

    const uint H = p.hidden;
    const uint I = p.intermediate;
    const device half *xr = x + (ulong)tok * H;

    threadgroup half h[MAX_INTERMEDIATE];

    // ---- phase 1: gate and up in one pass over x, SiLU applied in register --
    //
    // Both weight rows are streamed against the same activation window, so x's
    // cache lines are touched once and serve both. This is the fusion that
    // matters at batch 1: the alternative reads x twice and writes gate and up
    // to DRAM only to read them back.
    const uint n_groups_h = H / QK4;

    for (uint row = sgitg; row < I; row += SG_PER_TG_FFN) {
        const device uchar *gq = gate_q + (ulong)row * p.gate_row_stride_q;
        const device uchar *uq = up_q   + (ulong)row * p.gate_row_stride_q;
        const device half  *gs = gate_s + (ulong)row * p.gate_row_stride_g;
        const device half  *us = up_s   + (ulong)row * p.gate_row_stride_g;
        const device half  *gz = gate_z + (ulong)row * p.gate_row_stride_g;
        const device half  *uz = up_z   + (ulong)row * p.gate_row_stride_g;

        float acc_g = 0.0f, acc_u = 0.0f;
        for (uint g = tiisg; g < n_groups_h; g += SIMD_WIDTH) {
            const device half *a = xr + (ulong)g * QK4;
            const ulong2 pg = *reinterpret_cast<const device ulong2 *>(gq + (ulong)g * BYTES_PER_GROUP);
            const ulong2 pu = *reinterpret_cast<const device ulong2 *>(uq + (ulong)g * BYTES_PER_GROUP);
            acc_g += dot_group_q4(pg, a, (float)gs[g], (float)gz[g]);
            acc_u += dot_group_q4(pu, a, (float)us[g], (float)uz[g]);
        }

        acc_g = simd_reduce_add(acc_g);
        acc_u = simd_reduce_add(acc_u);
        if (tiisg == 0) h[row] = (half)(silu(acc_g) * acc_u);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- phase 2: down projection straight out of threadgroup memory -------
    const uint n_groups_i = I / QK4;
    const float w = p.expert_weight;

    for (uint row = sgitg; row < H; row += SG_PER_TG_FFN) {
        const device uchar *dq = down_q + (ulong)row * p.down_row_stride_q;
        const device half  *ds = down_s + (ulong)row * p.down_row_stride_g;
        const device half  *dz = down_z + (ulong)row * p.down_row_stride_g;

        float acc = 0.0f;
        for (uint g = tiisg; g < n_groups_i; g += SIMD_WIDTH) {
            const ulong2 pd = *reinterpret_cast<const device ulong2 *>(dq + (ulong)g * BYTES_PER_GROUP);
            acc += dot_group_q4_tg(pd, h + (ulong)g * QK4, (float)ds[g], (float)dz[g]);
        }
        acc = simd_reduce_add(acc);

        if (tiisg == 0) {
            device half *o = out + (ulong)tok * H + row;
            *o = p.accumulate ? (half)((float)(*o) + w * acc) : (half)(w * acc);
        }
    }
}

// ---------------------------------------------------------------------------
// kernel_moe_topk_route — top-k routing with minimal thread divergence.
//
// The usual formulation sorts, or runs k passes of an argmax with a mask, both
// of which put a data-dependent branch in the inner loop. This one keeps the
// running top-k in registers as an insertion list of fixed length K: every lane
// walks the same number of experts, executes the same instructions, and the
// only divergence is a predicated swap. For K = 8 over 288 experts that is 288
// iterations of straight-line code with an 8-deep compare chain, all unrolled.
//
// grid:        (n_tokens, 1, 1)     threadgroup: SIMD_WIDTH
// ---------------------------------------------------------------------------

struct RouteParams {
    uint  n_experts;      // 288 for GLM-5.3
    uint  top_k;          // 8 routed
    uint  n_tokens;
    uint  norm_weights;   // 1 = renormalize the k selected weights to sum to 1
};

constant constexpr int MAX_TOPK = 16;

kernel void kernel_moe_topk_route(
        device const half        *logits   [[buffer(0)]],  // [n_tokens][n_experts]
        device       uint        *ids_out  [[buffer(1)]],  // [n_tokens][top_k]
        device       half        *w_out    [[buffer(2)]],  // [n_tokens][top_k]
        constant     RouteParams &p        [[buffer(3)]],
        uint  tgpig [[threadgroup_position_in_grid]],
        uint  tiisg [[thread_index_in_simdgroup]]) {

    const uint tok = tgpig;
    if (tok >= p.n_tokens) return;
    const device half *lr = logits + (ulong)tok * p.n_experts;
    const uint K = min(p.top_k, (uint)MAX_TOPK);

    // Each lane keeps a private top-K over its strided slice, then the 32
    // per-lane lists are merged through simd_shuffle. No threadgroup memory.
    float best_v[MAX_TOPK];
    uint  best_i[MAX_TOPK];
#pragma clang loop unroll(full)
    for (int i = 0; i < MAX_TOPK; ++i) { best_v[i] = -INFINITY; best_i[i] = 0u; }

    for (uint e = tiisg; e < p.n_experts; e += SIMD_WIDTH) {
        float v = (float)lr[e];
        uint  id = e;
        // Predicated insertion: no branch, constant trip count.
#pragma clang loop unroll(full)
        for (int i = 0; i < MAX_TOPK; ++i) {
            const bool swap = v > best_v[i];
            const float tv = swap ? best_v[i] : v;
            const uint  ti = swap ? best_i[i] : id;
            best_v[i] = swap ? v  : best_v[i];
            best_i[i] = swap ? id : best_i[i];
            v = tv; id = ti;
        }
    }

    // Merge the 32 lane-local lists: K rounds of "global max, then retire it".
    // Each round is one simd_max plus one simd_shuffle to recover the owner,
    // so the whole merge is 2K cross-lane ops with no memory traffic.
    float sel_v[MAX_TOPK];
    uint  sel_i[MAX_TOPK];
    uint  head = 0;   // how many of this lane's entries are already consumed

    float sum = 0.0f;
    for (uint r = 0; r < K; ++r) {
        const float cand = best_v[head];
        const float gmax = simd_max(cand);
        // Lowest lane holding the max wins, so ties are resolved deterministically.
        const uint  owner = simd_min(cand == gmax ? tiisg : SIMD_WIDTH);
        const uint  gid   = simd_shuffle(best_i[head], owner);
        sel_v[r] = gmax;
        sel_i[r] = gid;
        if (tiisg == owner) {
#pragma clang loop unroll(full)
            for (int i = 0; i < MAX_TOPK - 1; ++i) {
                best_v[i] = best_v[i + 1];
                best_i[i] = best_i[i + 1];
            }
            best_v[MAX_TOPK - 1] = -INFINITY;
        }
        sum += exp(gmax - sel_v[0]);
    }

    if (tiisg != 0) return;
    const float inv = p.norm_weights && sum > 0.0f ? 1.0f / sum : 1.0f;
    for (uint r = 0; r < K; ++r) {
        ids_out[(ulong)tok * p.top_k + r] = sel_i[r];
        w_out[(ulong)tok * p.top_k + r]   = (half)(exp(sel_v[r] - sel_v[0]) * inv);
    }
}
