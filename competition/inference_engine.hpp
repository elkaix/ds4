// inference_engine.h — Metal dispatch harness for the decode loop.
//
// Pure C++ interface: no Metal, Foundation or Objective-C type crosses this
// header, so callers compile as plain C++17. The implementation
// (inference_engine.cpp) is compiled as Objective-C++ (-x objective-c++),
// which is what gives it direct access to the Metal API on this machine
// without the metal-cpp single-header distribution.
//
// Design commitments, all of which are load-bearing for decode throughput:
//
//   * Zero copy at the host boundary. Weights are mmapped by the caller and
//     wrapped with newBufferWithBytesNoCopy, so a 97 GB checkpoint is never
//     duplicated into a Metal allocation. Activations use
//     MTLResourceStorageModeShared, which on a UMA part means the CPU and GPU
//     address the same physical pages; there is no staging buffer and no blit.
//
//   * One command buffer per token, many dispatches inside it. A decode step
//     for GLM-5.3 is ~1100 dispatches; committing each one separately would
//     pay a kernel-launch round trip per dispatch. Encoding them into a single
//     buffer costs one commit.
//
//   * Non-blocking submission with a fixed number of frames in flight. The
//     host encodes token n+1 while the GPU runs token n. waitUntilCompleted is
//     never called on the hot path; a counting semaphore released from a
//     completion handler provides back pressure instead.
//
//   * Residency is declared once. Expert weights are placed in a residency set
//     so the driver keeps them mapped, rather than re-validating page
//     residency on every command buffer.
//
// What this harness deliberately does not do: offload dense projections to the
// ANE. That is not an oversight. The GLM-5.3-Flash checkpoint in this tree
// stores its expert tensors as 86 IQ2_XXS + 43 Q2_K -- 2-bit formats. CoreML
// accepts FP16 and INT8 weights; there is no 2-bit path into the ANE, so an
// ANE partition would require dequantizing to INT8 first, which more than
// doubles the bytes moved for the one thing that is already bandwidth bound.
// The interface below keeps a backend seam (IComputeBackend) so an ANE
// implementation can be added for an FP16 checkpoint, but shipping a CoreML
// path for a 2-bit model would be shipping a regression.

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace ds4c {

// ---------------------------------------------------------------------------
// Buffers
// ---------------------------------------------------------------------------

class Buffer {
public:
    virtual ~Buffer() = default;
    virtual void       *contents()      = 0;   // host pointer, same pages as the GPU sees
    virtual std::size_t length() const  = 0;
};

using BufferPtr = std::shared_ptr<Buffer>;

// ---------------------------------------------------------------------------
// Kernel arguments
//
// A dispatch is described by data, not by a virtual call per kernel, so the
// encode loop is a flat walk over a vector rather than a chain of indirect
// calls. `bytes` entries are inlined with setBytes (<= 4 KB), which avoids
// allocating a buffer per token for parameter structs.
// ---------------------------------------------------------------------------

struct Binding {
    enum class Kind { kBuffer, kBytes };
    Kind        kind   = Kind::kBuffer;
    unsigned    index  = 0;
    BufferPtr   buffer;            // kBuffer
    std::size_t offset = 0;        // kBuffer
    const void *data   = nullptr;  // kBytes
    std::size_t size   = 0;        // kBytes

    static Binding Buf(unsigned i, BufferPtr b, std::size_t off = 0) {
        Binding x; x.kind = Kind::kBuffer; x.index = i; x.buffer = std::move(b); x.offset = off; return x;
    }
    template <typename T>
    static Binding Bytes(unsigned i, const T &v) {
        Binding x; x.kind = Kind::kBytes; x.index = i; x.data = &v; x.size = sizeof(T); return x;
    }
};

struct Grid {
    unsigned threadgroups_x = 1, threadgroups_y = 1, threadgroups_z = 1;
    unsigned threads_x      = 1, threads_y      = 1, threads_z      = 1;
    unsigned threadgroup_memory = 0;   // dynamic threadgroup allocation, bytes
};

// Opaque handle to a compiled compute pipeline state.
using PipelineId = std::uint32_t;
constexpr PipelineId kInvalidPipeline = 0xFFFFFFFFu;

// ---------------------------------------------------------------------------
// Backend
// ---------------------------------------------------------------------------

class IComputeBackend {
public:
    virtual ~IComputeBackend() = default;

    // Compile a .metal source tree (or load a prebuilt .metallib) once at
    // startup. Returns false and fills `error` on failure.
    virtual bool LoadLibrary(const std::string &path, std::string *error) = 0;
    virtual PipelineId Pipeline(const std::string &function_name, std::string *error) = 0;

    // Wrap host memory the caller already owns. `bytes` must be page aligned
    // and `length` a multiple of the page size; nothing is copied.
    virtual BufferPtr WrapNoCopy(void *bytes, std::size_t length) = 0;
    // Allocate shared (CPU+GPU visible) memory.
    virtual BufferPtr Allocate(std::size_t length) = 0;

    // Declare a set of buffers permanently resident. Call once for the weights.
    virtual void MakeResident(const std::vector<BufferPtr> &buffers) = 0;

    // --- frame lifecycle ---------------------------------------------------
    // BeginFrame blocks only when kMaxFramesInFlight frames are already
    // outstanding; that is the back pressure, and it is the only place the host
    // can wait.
    virtual void BeginFrame() = 0;
    virtual void Encode(PipelineId pipeline, const Grid &grid,
                        const std::vector<Binding> &bindings) = 0;
    // Insert a barrier between dispatches that have a real data dependency.
    // Dispatches without one are left free to overlap.
    virtual void Barrier() = 0;
    virtual void EndFrame(std::function<void()> on_complete = {}) = 0;
    virtual void DrainAll() = 0;

    // Wall-clock GPU time of the last completed frame, in milliseconds, taken
    // from the command buffer's own timestamps. This is the only profiling hook
    // on the hot path: it reads two scalars the driver already recorded and
    // does not drain the queue, so leaving it on is free.
    virtual double LastFrameGpuMs() const = 0;
};

std::unique_ptr<IComputeBackend> CreateMetalBackend(std::string *error);

// ---------------------------------------------------------------------------
// Decode step
//
// A thin layer over the backend that knows the shape of one GLM-style decoder
// layer. It owns no weights: every tensor is a Buffer the caller mmapped.
// ---------------------------------------------------------------------------

struct ModelGeometry {
    unsigned hidden          = 4096;
    unsigned n_layers        = 46;
    unsigned n_heads         = 32;
    unsigned n_kv_heads      = 4;
    unsigned head_dim        = 128;
    unsigned intermediate    = 12288;   // dense FFN
    unsigned expert_inter    = 2048;    // per-expert FFN
    unsigned n_experts       = 288;
    unsigned n_experts_used  = 8;
    unsigned dense_layers    = 3;       // leading_dense_block_count
    unsigned page_size       = 256;     // KV page, tokens
    float    rms_eps         = 1e-5f;
    float    rope_theta      = 10000.0f;
    unsigned rope_dim        = 128;
};

struct LayerWeights {
    BufferPtr attn_norm, ffn_norm;
    BufferPtr wq, wk, wv, wo;
    // Quantized MoE streams, structure of arrays (see quant_common.metalh).
    BufferPtr gate_q, gate_s, gate_z;
    BufferPtr up_q,   up_s,   up_z;
    BufferPtr down_q, down_s, down_z;
    BufferPtr router;
    bool      is_moe = true;
};

struct DecodeIO {
    BufferPtr hidden;         // [n_seqs][hidden]
    BufferPtr k_cache, v_cache;
    BufferPtr block_tables;   // [n_seqs][max_pages]
    BufferPtr seq_lens;       // [n_seqs]
    BufferPtr positions;      // [n_seqs]
    BufferPtr logits;
    unsigned  n_seqs = 1;
    unsigned  max_pages = 0;
};

class DecodeStep {
public:
    DecodeStep(IComputeBackend &backend, const ModelGeometry &geo);
    bool Prepare(std::string *error);

    // Encodes every layer of one decode step into a single command buffer and
    // submits it without blocking. `on_complete` runs on a driver thread.
    void Run(const std::vector<LayerWeights> &layers, const DecodeIO &io,
             std::function<void()> on_complete = {});

    // Bytes the step reads from DRAM, computed from the geometry and the
    // quantization, for the roofline in PERFORMANCE.md.
    std::size_t ActiveBytesPerToken() const;

private:
    IComputeBackend &be_;
    ModelGeometry    geo_;
    PipelineId       p_qkv_    = kInvalidPipeline;
    PipelineId       p_attn_   = kInvalidPipeline;
    PipelineId       p_gemv_   = kInvalidPipeline;
    PipelineId       p_route_  = kInvalidPipeline;
    PipelineId       p_ffn_    = kInvalidPipeline;
    BufferPtr        q_, attn_out_, route_ids_, route_w_, router_logits_;
};

constexpr unsigned kMaxFramesInFlight = 3;

}  // namespace ds4c
