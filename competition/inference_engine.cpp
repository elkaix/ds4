// inference_engine.cpp — metal-cpp implementation of the decode harness.
//
// Build (this is the exact line used to verify it):
//
//   clang++ -std=c++17 -O2 -c inference_engine.cpp -I third_party/metal-cpp \
//           -framework Metal -framework Foundation -framework QuartzCore
//
// metal-cpp is header only and needs its implementation emitted in exactly one
// translation unit; the three defines below do that, and must appear before the
// headers. No other file in the project may define them.

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION

#include "inference_engine.hpp"

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include <cassert>
#include <condition_variable>
#include <cstring>
#include <fstream>
#include <mutex>
#include <sstream>
#include <unordered_map>

namespace ds4c {
namespace {

// ---------------------------------------------------------------------------
// A counting semaphore, C++17 (std::counting_semaphore is C++20).
//
// This is the whole back-pressure mechanism: the host is allowed to be
// kMaxFramesInFlight tokens ahead of the GPU and no further. Acquire blocks
// only when the GPU has fallen that far behind, which on a saturated decode
// loop is the steady state; the point is that the host never calls
// waitUntilCompleted and therefore never idles the queue between tokens.
// ---------------------------------------------------------------------------
class Semaphore {
public:
    explicit Semaphore(unsigned count) : count_(count) {}
    void Acquire() {
        std::unique_lock<std::mutex> lk(m_);
        cv_.wait(lk, [this] { return count_ > 0; });
        --count_;
    }
    void Release() {
        {
            std::lock_guard<std::mutex> lk(m_);
            ++count_;
        }
        cv_.notify_one();
    }
    void WaitFor(unsigned target) {
        std::unique_lock<std::mutex> lk(m_);
        cv_.wait(lk, [this, target] { return count_ >= target; });
    }

private:
    std::mutex              m_;
    std::condition_variable cv_;
    unsigned                count_;
};

class MetalBuffer final : public Buffer {
public:
    MetalBuffer(MTL::Buffer *b, bool owns) : buf_(b), owns_(owns) {}
    ~MetalBuffer() override {
        if (owns_ && buf_) buf_->release();
    }
    void       *contents() override      { return buf_->contents(); }
    std::size_t length() const override  { return buf_->length(); }
    MTL::Buffer *mtl() const             { return buf_; }

private:
    MTL::Buffer *buf_  = nullptr;
    bool         owns_ = true;
};

std::string ReadFile(const std::string &path, bool *ok) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { *ok = false; return {}; }
    std::ostringstream ss;
    ss << f.rdbuf();
    *ok = true;
    return ss.str();
}

// The page size the no-copy path must align to. Apple silicon is 16 KB.
std::size_t HostPageSize() {
    static const std::size_t p = static_cast<std::size_t>(::getpagesize());
    return p;
}

// ---------------------------------------------------------------------------
// MetalBackend
// ---------------------------------------------------------------------------
class MetalBackend final : public IComputeBackend {
public:
    explicit MetalBackend(MTL::Device *dev)
        : dev_(dev),
          queue_(dev->newCommandQueue()),
          sem_(kMaxFramesInFlight) {}

    ~MetalBackend() override {
        DrainAll();
        for (auto *p : pipelines_) if (p) p->release();
        if (residency_) residency_->release();
        if (lib_)   lib_->release();
        if (queue_) queue_->release();
        if (dev_)   dev_->release();
    }

    bool LoadLibrary(const std::string &path, std::string *error) override {
        NS::Error *err = nullptr;
        if (path.size() > 9 && path.compare(path.size() - 9, 9, ".metallib") == 0) {
            NS::String *p = NS::String::string(path.c_str(), NS::UTF8StringEncoding);
            NS::URL *url = NS::URL::fileURLWithPath(p);
            lib_ = dev_->newLibrary(url, &err);
        } else {
            bool ok = false;
            const std::string src = ReadFile(path, &ok);
            if (!ok) { if (error) *error = "cannot read " + path; return false; }
            NS::String *s = NS::String::string(src.c_str(), NS::UTF8StringEncoding);
            MTL::CompileOptions *opts = MTL::CompileOptions::alloc()->init();
            opts->setLanguageVersion(MTL::LanguageVersion3_1);
            // Fast math is safe for everything here except the online softmax,
            // which is written so that it never relies on strict ordering; the
            // running max keeps every exponent argument <= 0 regardless.
            opts->setFastMathEnabled(true);
            lib_ = dev_->newLibrary(s, opts, &err);
            opts->release();
        }
        if (!lib_) {
            if (error) {
                *error = err ? err->localizedDescription()->utf8String()
                             : "library creation failed";
            }
            return false;
        }
        return true;
    }

    PipelineId Pipeline(const std::string &name, std::string *error) override {
        auto it = by_name_.find(name);
        if (it != by_name_.end()) return it->second;
        if (!lib_) { if (error) *error = "no library loaded"; return kInvalidPipeline; }

        NS::String *n = NS::String::string(name.c_str(), NS::UTF8StringEncoding);
        MTL::Function *fn = lib_->newFunction(n);
        if (!fn) { if (error) *error = "no such kernel: " + name; return kInvalidPipeline; }

        NS::Error *err = nullptr;
        MTL::ComputePipelineState *ps = dev_->newComputePipelineState(fn, &err);
        fn->release();
        if (!ps) {
            if (error) {
                *error = err ? err->localizedDescription()->utf8String()
                             : "pipeline creation failed";
            }
            return kInvalidPipeline;
        }
        const PipelineId id = static_cast<PipelineId>(pipelines_.size());
        pipelines_.push_back(ps);
        by_name_.emplace(name, id);
        return id;
    }

    // Wrap caller-owned memory. The deallocator is null because the caller (who
    // mmapped the checkpoint) still owns the pages; Metal only maps them into
    // the GPU address space. This is what keeps a 97 GB checkpoint from being
    // duplicated into a Metal allocation.
    BufferPtr WrapNoCopy(void *bytes, std::size_t length) override {
        const std::size_t page = HostPageSize();
        if (reinterpret_cast<std::uintptr_t>(bytes) % page != 0) return nullptr;
        const std::size_t rounded = (length + page - 1) / page * page;
        MTL::Buffer *b = dev_->newBuffer(bytes, rounded,
                                         MTL::ResourceStorageModeShared, nullptr);
        if (!b) return nullptr;
        return std::make_shared<MetalBuffer>(b, /*owns=*/true);
    }

    BufferPtr Allocate(std::size_t length) override {
        MTL::Buffer *b = dev_->newBuffer(length, MTL::ResourceStorageModeShared);
        if (!b) return nullptr;
        std::memset(b->contents(), 0, length);
        return std::make_shared<MetalBuffer>(b, /*owns=*/true);
    }

    // Declaring the weights resident once removes per-command-buffer residency
    // validation. With ~1100 dispatches per token over hundreds of distinct
    // expert buffers, that validation is not free.
    void MakeResident(const std::vector<BufferPtr> &buffers) override {
        if (buffers.empty()) return;
        if (!residency_) {
            MTL::ResidencySetDescriptor *d = MTL::ResidencySetDescriptor::alloc()->init();
            d->setInitialCapacity(buffers.size());
            NS::Error *err = nullptr;
            residency_ = dev_->newResidencySet(d, &err);
            d->release();
            if (!residency_) return;   // older driver: fall back to implicit residency
            queue_->addResidencySet(residency_);
        }
        for (const auto &b : buffers) {
            auto *mb = static_cast<MetalBuffer *>(b.get());
            if (mb) residency_->addAllocation(mb->mtl());
        }
        residency_->commit();
        residency_->requestResidency();
    }

    void BeginFrame() override {
        sem_.Acquire();
        cmd_ = queue_->commandBuffer();
        cmd_->retain();
        enc_ = cmd_->computeCommandEncoder();
    }

    void Encode(PipelineId pipeline, const Grid &grid,
                const std::vector<Binding> &bindings) override {
        assert(enc_ && pipeline < pipelines_.size());
        enc_->setComputePipelineState(pipelines_[pipeline]);
        for (const auto &b : bindings) {
            if (b.kind == Binding::Kind::kBuffer) {
                auto *mb = static_cast<MetalBuffer *>(b.buffer.get());
                enc_->setBuffer(mb ? mb->mtl() : nullptr, b.offset, b.index);
            } else {
                // Inlined parameter structs. setBytes has a 4 KB limit and
                // avoids allocating a buffer per token for a 48-byte struct.
                assert(b.size <= 4096);
                enc_->setBytes(b.data, b.size, b.index);
            }
        }
        if (grid.threadgroup_memory) {
            enc_->setThreadgroupMemoryLength(grid.threadgroup_memory, 0);
        }
        enc_->dispatchThreadgroups(
            MTL::Size(grid.threadgroups_x, grid.threadgroups_y, grid.threadgroups_z),
            MTL::Size(grid.threads_x, grid.threads_y, grid.threads_z));
    }

    // Only between dispatches that actually depend on each other. Independent
    // expert FFNs are left unbarriered so the scheduler can overlap them.
    void Barrier() override {
        if (enc_) enc_->memoryBarrier(MTL::BarrierScopeBuffers);
    }

    void EndFrame(std::function<void()> on_complete) override {
        if (!cmd_) return;
        enc_->endEncoding();
        enc_ = nullptr;

        MTL::CommandBuffer *cb = cmd_;
        auto *self = this;
        auto cb_done = [self, cb, on_complete](MTL::CommandBuffer *done) {
            self->last_gpu_ms_.store(
                (done->GPUEndTime() - done->GPUStartTime()) * 1000.0,
                std::memory_order_relaxed);
            if (on_complete) on_complete();
            self->sem_.Release();
            cb->release();
        };
        cmd_->addCompletedHandler(cb_done);
        cmd_->commit();
        cmd_ = nullptr;
    }

    void DrainAll() override { sem_.WaitFor(kMaxFramesInFlight); }

    double LastFrameGpuMs() const override {
        return last_gpu_ms_.load(std::memory_order_relaxed);
    }

private:
    MTL::Device            *dev_   = nullptr;
    MTL::CommandQueue      *queue_ = nullptr;
    MTL::Library           *lib_   = nullptr;
    MTL::ResidencySet      *residency_ = nullptr;
    MTL::CommandBuffer     *cmd_   = nullptr;
    MTL::ComputeCommandEncoder *enc_ = nullptr;

    std::vector<MTL::ComputePipelineState *>       pipelines_;
    std::unordered_map<std::string, PipelineId>    by_name_;

    Semaphore                sem_;
    std::atomic<double>      last_gpu_ms_{0.0};
};

}  // namespace

std::unique_ptr<IComputeBackend> CreateMetalBackend(std::string *error) {
    MTL::Device *dev = MTL::CreateSystemDefaultDevice();
    if (!dev) { if (error) *error = "no Metal device"; return nullptr; }
    return std::unique_ptr<IComputeBackend>(new MetalBackend(dev));
}

// ---------------------------------------------------------------------------
// DecodeStep
// ---------------------------------------------------------------------------

DecodeStep::DecodeStep(IComputeBackend &backend, const ModelGeometry &geo)
    : be_(backend), geo_(geo) {}

bool DecodeStep::Prepare(std::string *error) {
    p_qkv_   = be_.Pipeline("kernel_rmsnorm_rope_qkv",   error);
    p_attn_  = be_.Pipeline("paged_hybrid_attention_d128", error);
    p_gemv_  = be_.Pipeline("kernel_gemv_w4a16",         error);
    p_route_ = be_.Pipeline("kernel_moe_topk_route",     error);
    p_ffn_   = be_.Pipeline("kernel_swiglu_ffn_fused",   error);
    if (p_qkv_ == kInvalidPipeline || p_attn_ == kInvalidPipeline ||
        p_gemv_ == kInvalidPipeline || p_route_ == kInvalidPipeline ||
        p_ffn_ == kInvalidPipeline) {
        return false;
    }

    const std::size_t h = geo_.hidden;
    q_             = be_.Allocate(geo_.n_heads * geo_.head_dim * sizeof(std::uint16_t));
    attn_out_      = be_.Allocate(h * sizeof(std::uint16_t));
    router_logits_ = be_.Allocate(geo_.n_experts * sizeof(std::uint16_t));
    route_ids_     = be_.Allocate(geo_.n_experts_used * sizeof(std::uint32_t));
    route_w_       = be_.Allocate(geo_.n_experts_used * sizeof(std::uint16_t));
    return q_ && attn_out_ && router_logits_ && route_ids_ && route_w_;
}

std::size_t DecodeStep::ActiveBytesPerToken() const {
    // Dense per layer: attention projections at 4 bit plus their scales/zeros.
    const std::size_t qkv_out = (std::size_t)(geo_.n_heads + 2 * geo_.n_kv_heads) * geo_.head_dim;
    const std::size_t attn_params = (qkv_out + (std::size_t)geo_.hidden) * geo_.hidden;

    // Routed experts touched per token: (n_experts_used + 1 shared) x 3 matrices.
    const std::size_t expert_params =
        (std::size_t)(geo_.n_experts_used + 1) * 3 * geo_.expert_inter * geo_.hidden;
    const std::size_t dense_ffn_params = (std::size_t)3 * geo_.intermediate * geo_.hidden;

    const std::size_t moe_layers   = geo_.n_layers - geo_.dense_layers;
    const std::size_t total_params =
        (std::size_t)geo_.n_layers * attn_params +
        moe_layers * expert_params +
        (std::size_t)geo_.dense_layers * dense_ffn_params;

    // 4 bits of quant + one half scale and one half zero per QK4=32 weights
    // = 0.5 + 4/32 = 0.625 bytes per weight.
    return total_params * 5 / 8;
}

void DecodeStep::Run(const std::vector<LayerWeights> &layers, const DecodeIO &io,
                     std::function<void()> on_complete) {
    struct QKVRoPEParams {
        std::uint32_t hidden, n_heads, n_kv_heads, head_dim, page_size, max_pages;
        std::uint32_t kv_page_stride, kv_head_stride;
        float eps, rope_theta;
        std::uint32_t rope_dim;
    };
    struct PagedAttnParams {
        std::uint32_t n_heads, n_kv_heads, head_dim, page_size, max_pages;
        std::uint32_t kv_page_stride, kv_head_stride, q_stride, o_stride;
        float scale, logit_softcap;
        std::uint32_t n_seqs, layer_kind, dsa_top_k;
    };
    struct GemvParams {
        std::uint32_t M, K, row_stride_q, row_stride_g, x_stride, n_cols, has_bias, fuse_silu;
    };
    struct RouteParams {
        std::uint32_t n_experts, top_k, n_tokens, norm_weights;
    };
    struct SwiGLUParams {
        std::uint32_t hidden, intermediate, n_tokens;
        std::uint32_t gate_row_stride_q, gate_row_stride_g;
        std::uint32_t down_row_stride_q, down_row_stride_g;
        float         expert_weight;
        std::uint32_t accumulate;
    };

    const std::uint32_t kv_head_stride = geo_.page_size * geo_.head_dim;
    const std::uint32_t kv_page_stride = geo_.n_kv_heads * kv_head_stride;

    QKVRoPEParams qp{geo_.hidden, geo_.n_heads, geo_.n_kv_heads, geo_.head_dim,
                     geo_.page_size, io.max_pages, kv_page_stride, kv_head_stride,
                     geo_.rms_eps, geo_.rope_theta, geo_.rope_dim};

    PagedAttnParams ap{geo_.n_heads, geo_.n_kv_heads, geo_.head_dim, geo_.page_size,
                       io.max_pages, kv_page_stride, kv_head_stride,
                       geo_.head_dim, geo_.head_dim,
                       1.0f / std::sqrt((float)geo_.head_dim), 0.0f,
                       io.n_seqs, /*layer_kind=*/0, /*dsa_top_k=*/2048};

    RouteParams rp{geo_.n_experts, geo_.n_experts_used, io.n_seqs, 1};

    be_.BeginFrame();

    for (std::size_t l = 0; l < layers.size(); ++l) {
        const LayerWeights &w = layers[l];

        // --- RMSNorm + QKV + RoPE + paged KV store, one dispatch ------------
        be_.Encode(p_qkv_, Grid{io.n_seqs, 1, 1, 128, 1, 1, 0}, {
            Binding::Buf(0, io.hidden), Binding::Buf(1, w.attn_norm),
            Binding::Buf(2, w.wq), Binding::Buf(3, w.wk), Binding::Buf(4, w.wv),
            Binding::Buf(5, q_), Binding::Buf(6, io.k_cache), Binding::Buf(7, io.v_cache),
            Binding::Buf(8, io.block_tables), Binding::Buf(9, io.positions),
            Binding::Bytes(10, qp)});
        be_.Barrier();

        // --- attention ------------------------------------------------------
        // layer_kind selects the hybrid branch: GLM-5.3 runs KDA on layers
        // where il % 4 != 3 and DSA on il % 4 == 3 (11 of 45 trunk layers).
        ap.layer_kind = (l % 4 == 3) ? 1u : 0u;
        be_.Encode(p_attn_, Grid{geo_.n_heads, io.n_seqs, 1, 128, 1, 1, 0}, {
            Binding::Buf(0, q_), Binding::Buf(1, io.k_cache), Binding::Buf(2, io.v_cache),
            Binding::Buf(3, io.block_tables), Binding::Buf(4, io.seq_lens),
            Binding::Buf(5, attn_out_), Binding::Bytes(6, ap)});
        be_.Barrier();

        if (!w.is_moe) {
            // Dense leading blocks: a single fused SwiGLU over the wide FFN.
            SwiGLUParams sp{geo_.hidden, geo_.intermediate, io.n_seqs,
                            geo_.hidden / 2, geo_.hidden / 32,
                            geo_.intermediate / 2, geo_.intermediate / 32,
                            1.0f, 0u};
            be_.Encode(p_ffn_, Grid{io.n_seqs, 1, 1, 256, 1, 1, 0}, {
                Binding::Buf(0, attn_out_),
                Binding::Buf(1, w.gate_q), Binding::Buf(2, w.gate_s), Binding::Buf(3, w.gate_z),
                Binding::Buf(4, w.up_q),   Binding::Buf(5, w.up_s),   Binding::Buf(6, w.up_z),
                Binding::Buf(7, w.down_q), Binding::Buf(8, w.down_s), Binding::Buf(9, w.down_z),
                Binding::Buf(10, io.hidden), Binding::Bytes(11, sp)});
            be_.Barrier();
            continue;
        }

        // --- router ---------------------------------------------------------
        GemvParams gp{geo_.n_experts, geo_.hidden, geo_.hidden / 2, geo_.hidden / 32,
                      1, 1, 0, 0};
        be_.Encode(p_gemv_, Grid{(geo_.n_experts + 7) / 8, 1, 1, 256, 1, 1, 0}, {
            Binding::Buf(0, w.router), Binding::Buf(1, w.gate_s), Binding::Buf(2, w.gate_z),
            Binding::Buf(3, attn_out_), Binding::Buf(4, nullptr),
            Binding::Buf(5, router_logits_), Binding::Bytes(6, gp)});
        be_.Barrier();

        be_.Encode(p_route_, Grid{io.n_seqs, 1, 1, 32, 1, 1, 0}, {
            Binding::Buf(0, router_logits_), Binding::Buf(1, route_ids_),
            Binding::Buf(2, route_w_), Binding::Bytes(3, rp)});
        be_.Barrier();

        // --- experts --------------------------------------------------------
        // No barrier between experts: they accumulate into disjoint... they do
        // not, they accumulate into the same output, so the accumulate flag
        // makes each one a read-modify-write. The barrier that matters is the
        // one after the group, not between members, because Metal orders
        // dispatches within an encoder unless told otherwise; the memoryBarrier
        // calls above are for the cross-encoder cases the driver cannot infer.
        for (unsigned e = 0; e < geo_.n_experts_used + 1; ++e) {
            SwiGLUParams sp{geo_.hidden, geo_.expert_inter, io.n_seqs,
                            geo_.hidden / 2, geo_.hidden / 32,
                            geo_.expert_inter / 2, geo_.expert_inter / 32,
                            1.0f, e == 0 ? 0u : 1u};
            be_.Encode(p_ffn_, Grid{io.n_seqs, 1, 1, 256, 1, 1, 0}, {
                Binding::Buf(0, attn_out_),
                Binding::Buf(1, w.gate_q), Binding::Buf(2, w.gate_s), Binding::Buf(3, w.gate_z),
                Binding::Buf(4, w.up_q),   Binding::Buf(5, w.up_s),   Binding::Buf(6, w.up_z),
                Binding::Buf(7, w.down_q), Binding::Buf(8, w.down_s), Binding::Buf(9, w.down_z),
                Binding::Buf(10, io.hidden), Binding::Bytes(11, sp)});
        }
        be_.Barrier();
    }

    be_.EndFrame(std::move(on_complete));
}

}  // namespace ds4c

// ---------------------------------------------------------------------------
// C bridge, for linking straight into the DS4 runtime (ds4.c / ds4_server.c),
// which is C and cannot see the C++ types above.
// ---------------------------------------------------------------------------

extern "C" {

struct ds4c_engine {
    std::unique_ptr<ds4c::IComputeBackend> backend;
    std::unique_ptr<ds4c::DecodeStep>      step;
    ds4c::ModelGeometry                    geo;
    char                                   error[256];
};

ds4c_engine *ds4c_engine_create(const char *library_path) {
    auto *e = new ds4c_engine();
    std::string err;
    e->backend = ds4c::CreateMetalBackend(&err);
    if (!e->backend || !e->backend->LoadLibrary(library_path, &err)) {
        std::snprintf(e->error, sizeof(e->error), "%s", err.c_str());
        return e;   // caller inspects ds4c_engine_error
    }
    e->step = std::unique_ptr<ds4c::DecodeStep>(new ds4c::DecodeStep(*e->backend, e->geo));
    if (!e->step->Prepare(&err)) {
        std::snprintf(e->error, sizeof(e->error), "%s", err.c_str());
    }
    return e;
}

const char *ds4c_engine_error(const ds4c_engine *e) {
    return (e && e->error[0]) ? e->error : nullptr;
}

double ds4c_engine_last_gpu_ms(const ds4c_engine *e) {
    return (e && e->backend) ? e->backend->LastFrameGpuMs() : 0.0;
}

unsigned long long ds4c_engine_active_bytes_per_token(const ds4c_engine *e) {
    return (e && e->step) ? (unsigned long long)e->step->ActiveBytesPerToken() : 0ull;
}

void ds4c_engine_drain(ds4c_engine *e) {
    if (e && e->backend) e->backend->DrainAll();
}

void ds4c_engine_destroy(ds4c_engine *e) { delete e; }

}  // extern "C"
