/* Exercise private session bookkeeping without loading a model.
 * Ported from upstream 233eeb8 (test_rewind, test_session_memory). The TP
 * payload and GPU spec-state sections are not ported: this tree keeps the
 * PR #920 KDA transaction and the retain-all-rows indexer pool kernel
 * (tests/test_glm53_kda.c) instead of a whole-state backup. */
#include "../ds4.c"
#include <assert.h>

static void test_rewind(void) {
    ds4_engine e = { .backend = DS4_BACKEND_CPU };
    ds4_session *s = calloc(1, sizeof(*s));
    assert(s);
    s->engine = &e;
    s->ctx_size = 1024;
    for (int i = 0; i < 260; i++) ds4_tokens_push(&s->checkpoint, i);
    s->checkpoint_valid = true;
    s->mtp_draft_valid = true;
    s->checkpoint_images = calloc(1, sizeof(*s->checkpoint_images));
    assert(s->checkpoint_images);
    s->checkpoint_image_count = 1;
    ds4_vision_identity *images = s->checkpoint_images;

    ds4_session_rewind(s, 260);
    assert(s->checkpoint_valid && s->mtp_draft_valid);
    ds4_session_rewind(s, 300);
    assert(s->checkpoint.len == 260 && s->checkpoint_valid);
    const int boundaries[] = {259, 256, 255, 128, 127, 4, 3, 0};
    for (size_t i = 0; i < sizeof(boundaries) / sizeof(*boundaries); i++) {
        s->checkpoint_valid = true;
        ds4_session_rewind(s, boundaries[i]);
        assert(s->checkpoint.len == boundaries[i]);
        assert(!s->checkpoint_valid && !s->mtp_draft_valid);
        assert(s->checkpoint_images == images && s->checkpoint_image_count == 1);
        assert(ds4_session_common_prefix(s, &s->checkpoint) == 0);
        assert(ds4_session_argmax(s) == -1);
        assert(ds4_session_argmax_excluding(s, 1) == -1);
        assert(ds4_session_argmax_ignoring_eos(s, DS4_THINK_NONE) == -1);
        uint64_t rng = 1;
        assert(ds4_session_sample(s, 1, 0, 1, 0, &rng) == -1);
        char err[256] = "";
        int accepted[2];
        assert(ds4_session_eval(s, 1, err, sizeof(err)) != 0);
        assert(strstr(err, "synchronized checkpoint"));
        assert(ds4_session_eval_speculative(s, 1, 2, -1, 1, 0, 1, 0,
                    &rng, accepted, 2, err, sizeof(err)) == -1);
    }
    ds4_session_rewind(s, -1);
    ds4_session_rewind(NULL, 0);
    ds4_session_free(s);
}

static void test_session_memory(void) {
    const uint64_t gib = UINT64_C(1) << 30;
    ds4_engine e = { .backend = DS4_BACKEND_METAL,
                     .placement_session_count_hint = 4 };
    assert(ds4_engine_glm_graph_budget(&e, 2*gib) == 8*gib);
    e.glm_session_count = 1;
    e.glm_session_graph_bytes = 3*gib;
    assert(ds4_engine_glm_graph_budget(&e, 2*gib) == 9*gib);
    e.glm_session_count = 4;
    e.glm_session_graph_bytes = 8*gib;
    assert(ds4_engine_glm_graph_budget(&e, 3*gib) == 11*gib);
    e.placement_session_count_hint = 0;
    assert(ds4_engine_glm_graph_budget(&e, 3*gib) == 11*gib);
    assert(ds4_engine_glm_graph_budget(&e, UINT64_MAX) == UINT64_MAX);
    e.backend = DS4_BACKEND_CUDA;
    assert(ds4_engine_glm_graph_budget(&e, 3*gib) == 3*gib);

    ds4_session *s = calloc(1, sizeof(*s));
    assert(s);
    e.backend = DS4_BACKEND_CPU;
    s->engine = &e;
    s->glm_reserved_graph_bytes = 2*gib;
    ds4_session_free(s);
    assert(e.glm_session_count == 3 && e.glm_session_graph_bytes == 6*gib);
    e.backend = DS4_BACKEND_METAL;
    e.placement_session_count_hint = 4;
    assert(ds4_engine_glm_graph_budget(&e, 2*gib) == 8*gib);
}

#ifndef DS4_NO_GPU
static void test_glm_attention_budget(void) {
    const ds4_shape saved_shape = g_ds4_shape;
    g_ds4_shape = DS4_SHAPE_GLM53;
    ds4_glm_gpu_graph *g = calloc(1, sizeof(*g));
    assert(g);
    g->glm53 = true;
    g->compact_cache_cap = 16384;
    const uint32_t capacities[] = {1024, 4096, 8192, 16384};
    for (size_t i = 0; i < sizeof(capacities) / sizeof(*capacities); i++) {
        g->ctx_cap = capacities[i];
        assert(glm_graph_dense_compact_attention_limit(g) == 2051);
        g->full_kv_cache = true;
        assert(glm_graph_dense_compact_attention_limit(g) == 2051);
        g->full_kv_cache = false;
    }
    free(g);
    g_ds4_shape = saved_shape;
}
#endif

static void test_snapshot_bytes(void) {
    const ds4_shape saved_shape = g_ds4_shape;
    const uint32_t saved_ratio = g_ds4_compress_ratios[0];
    g_ds4_shape = DS4_SHAPE_FLASH;
    g_ds4_shape.n_layer = 1;
    g_ds4_shape.n_head_dim = 4;
    g_ds4_shape.n_vocab = 8;
    g_ds4_compress_ratios[0] = 0;
    ds4_engine e = {.backend = DS4_BACKEND_CPU};
    ds4_session *s = calloc(1, sizeof(*s));
    assert(s);
    s->engine = &e;
    s->ctx_size = 8;
    s->prefill_cap = 1;
    s->checkpoint_valid = true;
    s->logits = calloc(DS4_N_VOCAB, sizeof(float));
    assert(s->logits);
    kv_cache_init(&s->cpu_cache, 8, 0);
    for (int i = 0; i < 3; i++) ds4_tokens_push(&s->checkpoint, i);
    /* Every byte of the final float is nonzero, including the last byte. */
    const uint32_t bits = UINT32_C(0x3f9e0652);
    for (int i = 0; i < 3 * 4; i++)
        memcpy(s->cpu_cache.layer[0].raw_kv + i, &bits, sizeof(bits));
    ds4_session_snapshot snapshot = {0};
    const int lengths[] = {1, 3, 2, 3};
    for (size_t i = 0; i < sizeof(lengths) / sizeof(*lengths); i++) {
        s->checkpoint.len = lengths[i];
        s->cpu_cache.layer[0].n_raw = lengths[i];
        char err[192] = {0};
        FILE *fp = tmpfile();
        assert(fp);
        assert(ds4_session_save_payload(s, fp, err, sizeof(err)) == 0);
        const long bytes = ftell(fp);
        assert(bytes > 0);
        assert(ds4_session_save_snapshot(s, &snapshot, err, sizeof(err)) == 0);
        assert(snapshot.len == (uint64_t)bytes);
        rewind(fp);
        for (long j = 0; j < bytes; j++) {
            const int expected = fgetc(fp);
            if (expected != snapshot.ptr[j])
                fprintf(stderr, "snapshot byte %ld/%ld: expected=%d actual=%d\n",
                        j, bytes, expected, snapshot.ptr[j]);
            assert(expected == snapshot.ptr[j]);
        }
        assert(fgetc(fp) == EOF);
        fclose(fp);
    }
    ds4_session_snapshot_free(&snapshot);
    ds4_session_free(s);
    g_ds4_shape = saved_shape;
    g_ds4_compress_ratios[0] = saved_ratio;
}

int main(void) {
    test_rewind();
    test_session_memory();
    test_snapshot_bytes();
#ifndef DS4_NO_GPU
    test_glm_attention_budget();
#endif
    puts("session state tests: ok");
    return 0;
}
