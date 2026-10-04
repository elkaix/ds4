/* Predictor-draft acceptance for the Qwen3.8 speculative cycle: pure CPU,
 * no engine, model, or GPU.  The env is read once into a static, so the
 * binary runs twice (see the test-qwen4-sampled-drafts Makefile target):
 * without DS4_QWEN4_SAMPLED_DRAFTS the predictor rule stays the opportunistic
 * argmax (qwen4_draft_accepted); with DS4_QWEN4_SAMPLED_DRAFTS=1
 * (temperature > 0) qwen4_predictor_row_accepted routes predictor rows
 * through qwen4_copy_row_accepted instead — a draft commits only when the
 * request's own sampler draws it from the row, consuming exactly that one
 * draw, rolled back on a miss so the caller redraws that same token.  Greedy
 * decoding stays argmax in both modes. */
#include "../ds4.c"
#include <assert.h>

static float *test_row(uint32_t seed) {
    float *row = xmalloc((size_t)DS4_N_VOCAB * sizeof(row[0]));
    for (uint32_t i = 0; i < DS4_N_VOCAB; i++) {
        uint32_t h = (i * 2654435761u) ^ (seed * 40503u);
        h ^= h >> 13; h *= 2246822519u; h ^= h >> 16;
        row[i] = ((float)(h & 0xFFFFu) / 65535.0f - 0.5f) * 12.0f;
    }
    return row;
}

int main(void) {
    const bool sampled = qwen4_sampled_drafts();
    ds4_engine *e = xcalloc(1, sizeof(*e));   /* qwen4_rows_invariant stays false */
    ds4_session *s = xcalloc(1, sizeof(*s));
    s->engine = e;
    bool saw_split = false;   /* some draw that is not the row argmax */

    for (uint32_t seed = 1; seed <= 6; seed++) {
        float *row = test_row(seed);
        /* full-vocab and top_k sampler branches */
        for (int p = 0; p < 2; p++) {
            const float temperature = p ? 1.3f : 0.7f;
            const int top_k = p ? 40 : 0;
            const float top_p = p ? 1.0f : 0.95f;
            const float min_p = p ? 0.0f : 0.05f;
            for (uint64_t k = 1; k <= 8; k++) {
                /* plain decoding: one ds4_session_sample draw from this row */
                const uint64_t plain0 = k * 0x9E3779B97F4A7C15ull;
                s->logits = row;
                s->checkpoint_valid = true;
                uint64_t plain = plain0;
                const int drawn = ds4_session_sample(s, temperature, top_k, top_p, min_p, &plain);
                assert(drawn >= 0 && (uint32_t)drawn < DS4_N_VOCAB);
                assert(plain != plain0);   /* a real draw advanced the rng */
                const int am = sample_argmax(row, DS4_N_VOCAB);
                if (drawn != am) saw_split = true;

                uint64_t rng = plain0;
                if (sampled) {
                    /* the drawn token is accepted and consumes that same single draw */
                    assert(qwen4_predictor_row_accepted(s, row, drawn, temperature, top_k, top_p, min_p, &rng));
                    assert(rng == plain);
                    /* any other token is rejected with the rng restored, so the
                     * caller redraws this row and gets `drawn` again */
                    const int other = (drawn + 1) % (int)DS4_N_VOCAB;
                    rng = plain0;
                    assert(!qwen4_predictor_row_accepted(s, row, other, temperature, top_k, top_p, min_p, &rng));
                    assert(rng == plain0);
                    s->logits = row;
                    assert(ds4_session_sample(s, temperature, top_k, top_p, min_p, &rng) == drawn);
                    assert(rng == plain);
                } else {
                    /* default: opportunistic argmax, no rng draw at all */
                    assert(qwen4_predictor_row_accepted(s, row, am, temperature, top_k, top_p, min_p, &rng));
                    assert(rng == plain0);
                    if (drawn != am)
                        assert(!qwen4_predictor_row_accepted(s, row, drawn, temperature, top_k, top_p, min_p, &rng));
                    assert(rng == plain0);
                }

                /* greedy: deterministic argmax rule, rng untouched, both modes */
                rng = plain0;
                assert(qwen4_predictor_row_accepted(s, row, am, 0.0f, 0, 0.9f, 0.0f, &rng));
                assert(!qwen4_predictor_row_accepted(s, row, (am + 1) % (int)DS4_N_VOCAB, 0.0f, 0, 0.9f, 0.0f, &rng));
                assert(rng == plain0);
            }
        }
        free(row);
    }
    /* the draw-vs-argmax split must actually occur, or the sampled-mode
     * assertions above could not tell the draw rule from argmax */
    assert(saw_split);

    /* a row-invariant engine samples regardless of the switch (unchanged) */
    e->qwen4_rows_invariant = true;
    float *row = test_row(77);
    const uint64_t plain0 = 0x1234abcd5678ef01ull;
    s->logits = row;
    s->checkpoint_valid = true;
    uint64_t plain = plain0;
    const int drawn = ds4_session_sample(s, 0.7f, 0, 0.95f, 0.05f, &plain);
    assert(drawn >= 0 && plain != plain0);
    uint64_t rng = plain0;
    assert(qwen4_draft_accepted(s, row, drawn, 0.7f, 0, 0.95f, 0.05f, &rng));
    assert(rng == plain);
    free(row);

    free(s->sample_probs);
    free(s);
    free(e);
    printf("Qwen3.8 predictor-draft acceptance (%s): %s OK\n",
           sampled ? "DS4_QWEN4_SAMPLED_DRAFTS=1" : "default",
           sampled ? "sampler-drawn accept, rng rollback, greedy argmax"
                   : "opportunistic argmax, greedy argmax");
    return 0;
}
