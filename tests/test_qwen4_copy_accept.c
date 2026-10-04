/* Copied-row acceptance for the Qwen3.8 speculative cycle
 * (qwen4_copy_row_accepted): pure CPU, no engine, model, or GPU.  The
 * contract under test is the plain-decoding invariant — a copied draft is
 * accepted only when the request's own sampler draws it from that row,
 * consuming exactly the draw plain decoding would consume, and a miss
 * leaves the rng untouched so the caller redraws that same token. */
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

static void check_params(ds4_session *s, const float *row, float temperature,
                         int top_k, float top_p, float min_p) {
    for (uint64_t seed = 1; seed <= 8; seed++) {
        /* plain decoding: one ds4_session_sample draw from this row */
        s->logits = (float *)row;
        s->checkpoint_valid = true;
        uint64_t plain = seed * 0x9E3779B97F4A7C15ull;
        const int drawn = ds4_session_sample(s, temperature, top_k, top_p, min_p, &plain);
        assert(drawn >= 0 && (uint32_t)drawn < DS4_N_VOCAB);
        assert(plain != seed * 0x9E3779B97F4A7C15ull);   /* a real draw advanced the rng */

        /* the drawn token is accepted and consumes that same single draw */
        uint64_t rng = seed * 0x9E3779B97F4A7C15ull;
        assert(qwen4_copy_row_accepted(s, row, drawn, temperature, top_k, top_p, min_p, &rng));
        assert(rng == plain);

        /* any other token is rejected with the rng restored, so the caller
         * redraws this row and gets `drawn` again */
        rng = seed * 0x9E3779B97F4A7C15ull;
        const int other = (drawn + 1) % (int)DS4_N_VOCAB;
        assert(!qwen4_copy_row_accepted(s, row, other, temperature, top_k, top_p, min_p, &rng));
        assert(rng == seed * 0x9E3779B97F4A7C15ull);
    }
}

int main(void) {
    ds4_session *s = xcalloc(1, sizeof(*s));

    for (uint32_t seed = 1; seed <= 3; seed++) {
        float *row = test_row(seed);
        /* full-vocab and top_k sampler branches */
        check_params(s, row, 0.7f, 0, 0.95f, 0.05f);
        check_params(s, row, 1.3f, 40, 1.0f, 0.0f);
        check_params(s, row, 0.05f, 0, 0.0f, 0.0f);

        /* greedy: deterministic argmax rule, rng untouched either way */
        const int am = sample_argmax(row, DS4_N_VOCAB);
        uint64_t rng = 12345;
        assert(qwen4_copy_row_accepted(s, row, am, 0.0f, 0, 0.9f, 0.0f, &rng));
        assert(!qwen4_copy_row_accepted(s, row, (am + 1) % (int)DS4_N_VOCAB, 0.0f, 0, 0.9f, 0.0f, &rng));
        assert(rng == 12345);
        free(row);
    }

    free(s->sample_probs);
    free(s);
    printf("Qwen3.8 copy-row acceptance: sampler-drawn accept, rng rollback, greedy argmax OK\n");
    return 0;
}
