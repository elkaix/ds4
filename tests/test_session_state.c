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

int main(void) {
    test_rewind();
    test_session_memory();
    puts("session state tests: ok");
    return 0;
}
