// paged_flash_decode.metal — Paged Flash-Decode with streaming online softmax.
//
// Decode-time attention for continuous batching: one query position per
// sequence, KV history scattered across fixed-size physical pages addressed
// through a per-sequence block table.
//
//   out[h, :] = sum_t softmax_t( q[h,:] . k[t,:] * scale + mask ) * v[t,:]
//
// The S x S score matrix is never materialized. Each simdgroup keeps a running
// (m, l, acc) triple and folds every new score into it immediately, which is
// the standard online-softmax recurrence:
//
//   m' = max(m, s)
//   l' = l * exp(m - m') + exp(s - m')
//   a' = a * exp(m - m') + exp(s - m') * v
//
// so the only state is one scalar pair plus the D-vector accumulator, all in
// registers. Global memory traffic is exactly the KV bytes read once.
//
// Work decomposition, and why it is this one:
//
//   * Lanes split the head dimension, not the KV axis. Lane i owns elements
//     [i*D/32, (i+1)*D/32) of q, of every k it touches, and of the output
//     accumulator. Consequence: the p*v accumulation is entirely lane-local and
//     needs no cross-lane communication at all -- the expensive part of
//     attention has zero reductions. Only the q.k dot needs a reduction, and
//     that is one simd_sum per KV token, done four tokens at a time so the
//     shuffle latency is pipelined behind the next loads.
//
//   * Simdgroups split the KV axis (a "split-K" over history). Four simdgroups
//     per threadgroup each sweep a strided quarter of the pages, then combine
//     their partial (m, l, acc) once at the end through threadgroup memory.
//     That is one barrier for the whole kernel.
//
//   * Pages are visited in block-table order and each page is contiguous, so
//     the K and V streams are sequential 128-byte-aligned reads.
//
// Integration point in DS4: this is the shape the GLM-5.3 DSA layers need if
// the MTP row verifier is ever to run above the 4096-token dense window. Today
// `glm_graph_verify_rows_eligible` (ds4.c:47170) refuses any position past
// dense_limit precisely because the row pass can only attend over the dense
// compact cache; a paged decode kernel that indexes through a block table is
// what removes that restriction.

#include <metal_stdlib>
using namespace metal;

constant constexpr int SIMD_WIDTH   = 32;
constant constexpr int SG_PER_TG    = 4;                       // simdgroups per threadgroup
constant constexpr int TG_THREADS   = SIMD_WIDTH * SG_PER_TG;  // 128
constant constexpr int UNROLL       = 4;                       // KV tokens per inner step

struct PagedAttnParams {
    uint  n_heads;          // query heads
    uint  n_kv_heads;       // key/value heads (GQA: n_heads % n_kv_heads == 0)
    uint  head_dim;         // D
    uint  page_size;        // KV tokens per physical page
    uint  max_pages;        // stride of block_tables, in page ids
    uint  kv_page_stride;   // elements between consecutive pages of one kv head
    uint  kv_head_stride;   // elements between kv heads inside one page
    uint  q_stride;         // elements between query heads
    uint  o_stride;         // elements between output heads
    float scale;            // 1/sqrt(D), applied to the score
    float logit_softcap;    // 0 disables
    uint  n_seqs;
};

inline float apply_softcap(float s, float cap) {
    return cap > 0.0f ? cap * precise::tanh(s / cap) : s;
}

// ---------------------------------------------------------------------------
// kernel_paged_flash_decode<D>
//
// grid:        (n_heads, n_seqs, 1)   threadgroups
// threadgroup: TG_THREADS
//
// buffers:
//   0 q             half [n_seqs][n_heads][D]
//   1 k_cache       half [n_pages][n_kv_heads][page_size][D]
//   2 v_cache       half [n_pages][n_kv_heads][page_size][D]
//   3 block_tables  uint [n_seqs][max_pages]
//   4 seq_lens      uint [n_seqs]              KV length, inclusive of this step
//   5 out           half [n_seqs][n_heads][D]
//   6 params
// ---------------------------------------------------------------------------

template <int D>
kernel void paged_flash_decode(
        device const half            *q            [[buffer(0)]],
        device const half            *k_cache      [[buffer(1)]],
        device const half            *v_cache      [[buffer(2)]],
        device const uint            *block_tables [[buffer(3)]],
        device const uint            *seq_lens     [[buffer(4)]],
        device       half            *out          [[buffer(5)]],
        constant     PagedAttnParams &p            [[buffer(6)]],
        uint3 tgpig  [[threadgroup_position_in_grid]],
        uint  sgitg  [[simdgroup_index_in_threadgroup]],
        uint  tiisg  [[thread_index_in_simdgroup]]) {

    constexpr int VPT = D / SIMD_WIDTH;      // vector elements per thread
    static_assert(D % SIMD_WIDTH == 0, "head_dim must be a multiple of the SIMD width");

    const uint head = tgpig.x;
    const uint seq  = tgpig.y;
    if (head >= p.n_heads || seq >= p.n_seqs) return;

    const uint kv_head  = head / (p.n_heads / p.n_kv_heads);
    const uint kv_len   = seq_lens[seq];
    const uint n_pages  = (kv_len + p.page_size - 1) / p.page_size;
    const device uint *table = block_tables + (ulong)seq * p.max_pages;

    // --- load this lane's slice of q, once, into registers -----------------
    const device half *qh = q + ((ulong)seq * p.n_heads + head) * p.q_stride;
    float q_reg[VPT];
#pragma clang loop unroll(full)
    for (int i = 0; i < VPT; ++i) q_reg[i] = (float)qh[tiisg * VPT + i];

    // --- running online-softmax state --------------------------------------
    float m_run = -INFINITY;
    float l_run = 0.0f;
    float acc[VPT];
#pragma clang loop unroll(full)
    for (int i = 0; i < VPT; ++i) acc[i] = 0.0f;

    // --- sweep this simdgroup's stride of the page list --------------------
    for (uint pg = sgitg; pg < n_pages; pg += SG_PER_TG) {
        const uint page_id  = table[pg];
        const uint base_tok = pg * p.page_size;
        const uint n_tok    = min(p.page_size, kv_len - base_tok);

        const device half *kp = k_cache + (ulong)page_id * p.kv_page_stride
                                        + (ulong)kv_head * p.kv_head_stride;
        const device half *vp = v_cache + (ulong)page_id * p.kv_page_stride
                                        + (ulong)kv_head * p.kv_head_stride;

        uint t = 0;
        for (; t + UNROLL <= n_tok; t += UNROLL) {
            // Four independent partial dots. Issuing them together lets the
            // four dependent simd_sum reductions overlap with each other's
            // shuffle latency instead of serializing one per token.
            float part[UNROLL];
#pragma clang loop unroll(full)
            for (int u = 0; u < UNROLL; ++u) {
                const device half *kr = kp + (ulong)(t + u) * D;
                float s = 0.0f;
#pragma clang loop unroll(full)
                for (int i = 0; i < VPT; ++i) {
                    s = fma(q_reg[i], (float)kr[tiisg * VPT + i], s);
                }
                part[u] = s;
            }

            float score[UNROLL];
#pragma clang loop unroll(full)
            for (int u = 0; u < UNROLL; ++u) {
                score[u] = apply_softcap(simd_sum(part[u]) * p.scale, p.logit_softcap);
            }

            // Fold all four into the running state. Doing the max over the
            // group first means one exp() of the correction factor per four
            // tokens instead of four.
            float m_blk = max(max(score[0], score[1]), max(score[2], score[3]));
            const float m_new = max(m_run, m_blk);
            const float corr  = exp(m_run - m_new);

            float w[UNROLL];
            float l_blk = 0.0f;
#pragma clang loop unroll(full)
            for (int u = 0; u < UNROLL; ++u) {
                w[u] = exp(score[u] - m_new);
                l_blk += w[u];
            }

#pragma clang loop unroll(full)
            for (int i = 0; i < VPT; ++i) acc[i] *= corr;
#pragma clang loop unroll(full)
            for (int u = 0; u < UNROLL; ++u) {
                const device half *vr = vp + (ulong)(t + u) * D;
#pragma clang loop unroll(full)
                for (int i = 0; i < VPT; ++i) {
                    acc[i] = fma(w[u], (float)vr[tiisg * VPT + i], acc[i]);
                }
            }
            l_run = l_run * corr + l_blk;
            m_run = m_new;
        }

        // Tail of a partially filled page.
        for (; t < n_tok; ++t) {
            const device half *kr = kp + (ulong)t * D;
            float s = 0.0f;
#pragma clang loop unroll(full)
            for (int i = 0; i < VPT; ++i) s = fma(q_reg[i], (float)kr[tiisg * VPT + i], s);
            const float score = apply_softcap(simd_sum(s) * p.scale, p.logit_softcap);

            const float m_new = max(m_run, score);
            const float corr  = exp(m_run - m_new);
            const float w     = exp(score - m_new);
            const device half *vr = vp + (ulong)t * D;
#pragma clang loop unroll(full)
            for (int i = 0; i < VPT; ++i) {
                acc[i] = fma(w, (float)vr[tiisg * VPT + i], acc[i] * corr);
            }
            l_run = l_run * corr + w;
            m_run = m_new;
        }
    }

    // --- combine the SG_PER_TG partial softmaxes ---------------------------
    //
    // Rescaling by exp(m_i - m*) before summing is what makes the split-K
    // combine exact: each partial is a correctly normalized softmax over its
    // own token subset, and the merge is the same recurrence applied once more.
    threadgroup float sh_m[SG_PER_TG];
    threadgroup float sh_l[SG_PER_TG];
    threadgroup float sh_a[SG_PER_TG][D];

    if (tiisg == 0) { sh_m[sgitg] = m_run; sh_l[sgitg] = l_run; }
#pragma clang loop unroll(full)
    for (int i = 0; i < VPT; ++i) sh_a[sgitg][tiisg * VPT + i] = acc[i];

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg != 0) return;

    float m_all = sh_m[0];
#pragma clang loop unroll(full)
    for (int g = 1; g < SG_PER_TG; ++g) m_all = max(m_all, sh_m[g]);

    float l_all = 0.0f;
    float w_g[SG_PER_TG];
#pragma clang loop unroll(full)
    for (int g = 0; g < SG_PER_TG; ++g) {
        w_g[g] = (sh_l[g] > 0.0f) ? exp(sh_m[g] - m_all) : 0.0f;
        l_all += sh_l[g] * w_g[g];
    }

    const float inv_l = l_all > 0.0f ? 1.0f / l_all : 0.0f;
    device half *o = out + ((ulong)seq * p.n_heads + head) * p.o_stride;

#pragma clang loop unroll(full)
    for (int i = 0; i < VPT; ++i) {
        const uint d = tiisg * VPT + i;
        float v = 0.0f;
#pragma clang loop unroll(full)
        for (int g = 0; g < SG_PER_TG; ++g) v = fma(w_g[g], sh_a[g][d], v);
        o[d] = (half)(v * inv_l);
    }
}

template [[host_name("paged_flash_decode_d64")]]
kernel void paged_flash_decode<64>(
        device const half *, device const half *, device const half *,
        device const uint *, device const uint *, device half *,
        constant PagedAttnParams &, uint3, uint, uint);

template [[host_name("paged_flash_decode_d128")]]
kernel void paged_flash_decode<128>(
        device const half *, device const half *, device const half *,
        device const uint *, device const uint *, device half *,
        constant PagedAttnParams &, uint3, uint, uint);

// ---------------------------------------------------------------------------
// kernel_rmsnorm_rope_qkv — RMSNorm + QKV projection + RoPE + paged KV store,
// in one dispatch.
//
// The pre-attention chain is four passes over the same 4096-wide hidden state
// in a naive implementation, three of which write it back to DRAM. Here the
// normalized activation is computed once into threadgroup memory, all three
// projections read it from there, RoPE is applied to q and k in registers
// before they are ever stored, and k/v land directly in their physical page.
// The hidden state crosses the bus once.
//
// grid:        (n_seqs, 1, 1)
// threadgroup: TG_THREADS
// ---------------------------------------------------------------------------

struct QKVRoPEParams {
    uint  hidden;         // K of the projections
    uint  n_heads;
    uint  n_kv_heads;
    uint  head_dim;
    uint  page_size;
    uint  max_pages;
    uint  kv_page_stride;
    uint  kv_head_stride;
    float eps;            // RMSNorm epsilon
    float rope_theta;
    uint  rope_dim;       // rotated prefix of head_dim (may be < head_dim)
};

constant constexpr int MAX_HIDDEN   = 4096;   // GLM-5.3 embedding_length
constant constexpr int MAX_QK_ROWS  = 6144;   // n_heads*D + n_kv_heads*D, staged for RoPE

kernel void kernel_rmsnorm_rope_qkv(
        device const half          *x        [[buffer(0)]],  // [n_seqs][hidden]
        device const half          *norm_w   [[buffer(1)]],  // [hidden]
        device const half          *wq       [[buffer(2)]],  // [n_heads*D][hidden]
        device const half          *wk       [[buffer(3)]],
        device const half          *wv       [[buffer(4)]],
        device       half          *q_out    [[buffer(5)]],
        device       half          *k_cache  [[buffer(6)]],
        device       half          *v_cache  [[buffer(7)]],
        device const uint          *block_tables [[buffer(8)]],
        device const uint          *positions    [[buffer(9)]],
        constant     QKVRoPEParams &p        [[buffer(10)]],
        uint  tgpig  [[threadgroup_position_in_grid]],
        uint  tpitg  [[thread_position_in_threadgroup]],
        uint  sgitg  [[simdgroup_index_in_threadgroup]],
        uint  tiisg  [[thread_index_in_simdgroup]]) {

    const uint seq = tgpig;
    const uint H   = p.hidden;
    const uint D   = p.head_dim;

    // Staged in half, not float: 4096 floats + 6144 floats would be 40 KB and
    // Apple GPUs cap threadgroup memory at 32 KB per threadgroup. In half the
    // pair is 8 KB + 12 KB = 20 KB, which also leaves room for two resident
    // threadgroups per core. Accumulation stays in float; only the staged value
    // is narrowed, and it is an activation that is about to be cast to half
    // anyway.
    threadgroup half xn[MAX_HIDDEN];

    // --- RMSNorm: one pass to the sum of squares, one to normalize ---------
    const device half *xr = x + (ulong)seq * H;
    float ss = 0.0f;
    for (uint i = tpitg; i < H; i += TG_THREADS) {
        const float v = (float)xr[i];
        ss = fma(v, v, ss);
    }
    ss = simd_sum(ss);

    threadgroup float sg_ss[SG_PER_TG];
    if (tiisg == 0) sg_ss[sgitg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total = 0.0f;
#pragma clang loop unroll(full)
    for (int g = 0; g < SG_PER_TG; ++g) total += sg_ss[g];
    const float inv_rms = rsqrt(total / (float)H + p.eps);

    for (uint i = tpitg; i < H; i += TG_THREADS) {
        xn[i] = (half)((float)xr[i] * inv_rms * (float)norm_w[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- pass 1: project every q and k row into threadgroup staging --------
    //
    // q and k are staged rather than written straight out because RoPE mixes
    // element d with element d +/- rope_dim/2 of the same head, and that
    // partner is a different output row. Staging costs 2 * (q_rows + kv_rows)
    // floats of threadgroup memory and one barrier; recomputing the partner
    // row would cost a second full projection of the hidden state.
    const uint pos      = positions[seq];
    const uint page     = pos / p.page_size;
    const uint slot     = pos % p.page_size;
    const uint page_id  = block_tables[(ulong)seq * p.max_pages + page];

    const uint q_rows  = p.n_heads    * D;
    const uint kv_rows = p.n_kv_heads * D;

    threadgroup half stage[MAX_QK_ROWS];

    for (uint row = sgitg; row < q_rows + kv_rows; row += SG_PER_TG) {
        const device half *wrow = (row < q_rows) ? (wq + (ulong)row * H)
                                                 : (wk + (ulong)(row - q_rows) * H);
        float acc = 0.0f;
        for (uint i = tiisg; i < H; i += SIMD_WIDTH) acc = fma((float)wrow[i], (float)xn[i], acc);
#pragma clang loop unroll(full)
        for (uint off = SIMD_WIDTH / 2; off > 0; off >>= 1) acc += simd_shuffle_down(acc, off);
        if (tiisg == 0) stage[row] = (half)acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- pass 2: rotate and store q and k ----------------------------------
    for (uint row = tpitg; row < q_rows + kv_rows; row += TG_THREADS) {
        const bool is_q  = row < q_rows;
        const uint local = is_q ? row : row - q_rows;
        const uint h     = local / D;
        const uint d     = local % D;

        float outv = (float)stage[row];
        if (d < p.rope_dim) {
            const uint  half_dim = p.rope_dim / 2;
            const bool  lower    = d < half_dim;
            const uint  pair_d   = lower ? d + half_dim : d - half_dim;
            const float partner  = (float)stage[row - d + pair_d];
            const float freq     = pow(p.rope_theta,
                                       -(float)(lower ? d : pair_d) * 2.0f / (float)p.rope_dim);
            const float ang = (float)pos * freq;
            const float c = cos(ang), s = sin(ang);
            outv = lower ? (outv * c - partner * s)
                         : (partner * s + outv * c);
        }

        if (is_q) {
            q_out[(ulong)seq * q_rows + local] = (half)outv;
        } else {
            k_cache[(ulong)page_id * p.kv_page_stride + (ulong)h * p.kv_head_stride
                    + (ulong)slot * D + d] = (half)outv;
        }
    }

    // --- pass 3: v needs no rotation, so it streams straight to its page ----
    for (uint row = sgitg; row < kv_rows; row += SG_PER_TG) {
        const device half *wrow = wv + (ulong)row * H;
        float acc = 0.0f;
        for (uint i = tiisg; i < H; i += SIMD_WIDTH) acc = fma((float)wrow[i], (float)xn[i], acc);
#pragma clang loop unroll(full)
        for (uint off = SIMD_WIDTH / 2; off > 0; off >>= 1) acc += simd_shuffle_down(acc, off);
        if (tiisg == 0) {
            const uint h = row / D, d = row % D;
            v_cache[(ulong)page_id * p.kv_page_stride + (ulong)h * p.kv_head_stride
                    + (ulong)slot * D + d] = (half)acc;
        }
    }
}
