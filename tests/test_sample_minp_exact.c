/* Exactness gate for the fast min-p draw in the top_p >= 1.0 branch of
 * sample_full_vocab. Pure CPU: no engine, model, or GPU.
 *
 * For every case the default (fast) path and the DS4_SAMPLE_LEGACY
 * reference path must return the same token and leave the rng in the same
 * state - including the cases where no draw is consumed (greedy fallbacks,
 * degenerate sums, min_p > 1). The sweep covers the Qwen3.8 vocabulary
 * size and small vocabularies, the production temperatures and min_p
 * values, and adversarial logit shapes: sharp peaks, flat rows, exact
 * ties, clusters a few ulp above/below the min-p rejection boundary,
 * non-finite entries, all-non-finite rows, all-equal rows and rows whose
 * finite values never exceed the -1e30 scan sentinel.
 *
 * Non-finite floats are built from bit patterns (not the INFINITY/NAN
 * macros) so the file compiles warning-free under -ffast-math. */
#include "../ds4.c"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define TEST_VOCAB (248320u)
#define DRAW_SEEDS 14

static float *g_scratch;
static uint64_t g_cases;

static void fail_case(const char *what, uint32_t V, float T, float mu,
                      uint64_t seed, int fast, int leg,
                      uint64_t rng_fast, uint64_t rng_leg) {
    fprintf(stderr,
            "MISMATCH: %s V=%u T=%g min_p=%g seed=%llu "
            "fast=%d leg=%d rng_fast=%llu rng_leg=%llu\n",
            what, V, (double)T, (double)mu, (unsigned long long)seed,
            fast, leg, (unsigned long long)rng_fast,
            (unsigned long long)rng_leg);
    exit(1);
}

/* ---- data rng (row construction only; draws use the engine rng) ---- */
static uint64_t t_rng_next(uint64_t *s) {
    uint64_t x = *s;
    if (x == 0) x = 0x9e3779b97f4a7c15ull;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *s = x;
    return x * 0x2545f4914f6cdd1dull;
}

static double t_rng_uniform(uint64_t *s) {
    return (double)(t_rng_next(s) >> 11) / 9007199254740992.0;
}

static float t_rng_gauss(uint64_t *s) {
    double u1 = t_rng_uniform(s);
    const double u2 = t_rng_uniform(s);
    if (u1 < 1e-300) u1 = 1e-300;
    return (float)(sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2));
}

/* ---- bit-exact float helpers (no INFINITY/NAN macros: -ffast-math) ---- */
static float bits_f32(uint32_t b) {
    float f;
    memcpy(&f, &b, sizeof(f));
    return f;
}

static float t_pos_inf(void) { return bits_f32(0x7f800000u); }
static float t_neg_inf(void) { return bits_f32(0xff800000u); }
static float t_nan(void) { return bits_f32(0x7fc00000u); }

/* ---- logit shapes ---- */
enum {
    SHAPE_PEAKED,
    SHAPE_NEAR_UNIFORM,
    SHAPE_EXACT_TIES,
    SHAPE_BOUNDARY_LADDER,
    SHAPE_BOUNDARY_CLUSTERS,
    SHAPE_SPRINKLE_NONFINITE,
    SHAPE_ALL_NONFINITE,
    SHAPE_ALL_EQUAL,
    SHAPE_BELOW_SENTINEL,
    SHAPE_GAUSS_OUTLIERS,
    SHAPE_RANDOM_WIDE,
    SHAPE_ZEROS,
    SHAPE_COUNT
};

static const char *shape_name(int shape) {
    static const char *names[] = {
        "peaked", "near-uniform", "exact-ties", "boundary-ladder",
        "boundary-clusters", "sprinkle-nonfinite", "all-nonfinite",
        "all-equal", "below-sentinel", "gauss-outliers", "random-wide",
        "zeros"
    };
    return names[shape];
}

/* Ladder of scaled values a few ulp either side of the min-p rejection
 * boundary (and of the expf(x) >= min_rel crossing), materialized as
 * logits max_logit + T*scaled. Rounded back through the division in the
 * sampler, these land within a couple of ulp of both knife edges. */
static void t_place_boundary_ladder(float *row, uint32_t V, uint64_t *rng,
                                    uint32_t m0, float max_logit,
                                    float T, float mu, int span) {
    if (!(mu > 0.0f) || !isfinite(mu) || V < 8u) return;
    float cutoff = logf(mu);
    for (int k = 0; k < 64 && isfinite(cutoff); k++) {
        cutoff = nextafterf(cutoff, -FLT_MAX);
        if (expf(cutoff) < mu) break;
    }
    if (!isfinite(cutoff)) return;
    uint32_t placed = 0;
    for (int k = -span; k <= span; k++) {
        float s = cutoff;
        const int down = k < 0 ? -k : 0;
        const int up = k > 0 ? k : 0;
        for (int j = 0; j < down; j++) s = nextafterf(s, -FLT_MAX);
        for (int j = 0; j < up; j++) s = nextafterf(s, FLT_MAX);
        for (int rep = 0; rep < 3 && placed + 4u < V; rep++) {
            uint32_t idx = 1u + (uint32_t)(t_rng_uniform(rng) * (V - 2.0));
            if (idx >= V - 1u) idx = V - 2u;
            if (idx == m0) continue;
            row[idx] = max_logit + T * s;
            placed++;
        }
    }
}

static void build_row(float *row, uint32_t V, int shape, uint64_t seed,
                      float T, float mu) {
    uint64_t rng = seed * 0x2545f4914f6cdd1dull + 0x1234u;
    const uint32_t m0 = V > 3u ? (uint32_t)(t_rng_uniform(&rng) * (V - 2.0)) + 1u
                               : 0u;

    for (uint32_t i = 0; i < V; i++) row[i] = t_rng_gauss(&rng) * 4.0f;

    switch (shape) {
    case SHAPE_PEAKED:
        row[m0] = 30.0f;
        for (int k = 1; k <= 4 && (uint32_t)k < V; k++) {
            uint32_t idx = (m0 * (uint32_t)(k + 1) + 7u) % V;
            if (idx != m0) row[idx] = 22.0f + 6.0f * (float)k;
        }
        break;
    case SHAPE_NEAR_UNIFORM:
        for (uint32_t i = 0; i < V; i++) {
            row[i] = 1.0f + (float)(i % 5) * 1.2e-7f;
        }
        break;
    case SHAPE_EXACT_TIES:
        for (uint32_t i = 0; i < V; i++) {
            row[i] = (i & 1u) ? 0.3f : 2.5f;   /* half exact ties at max */
        }
        break;
    case SHAPE_BOUNDARY_LADDER:
        row[m0] = 12.0f;
        t_place_boundary_ladder(row, V, &rng, m0, 12.0f, T, mu, 12);
        break;
    case SHAPE_BOUNDARY_CLUSTERS: {
        row[m0] = 12.0f;
        t_place_boundary_ladder(row, V, &rng, m0, 12.0f, T, mu, 3);
        /* a second cluster: just above, just below and exactly on the
         * expf(x) >= min_rel crossing, plus a mass far above the bulk */
        if (mu > 0.0f && isfinite(mu) && V >= 64u) {
            float x = logf(mu);
            while (isfinite(x) && expf(x) < mu) {
                x = nextafterf(x, FLT_MAX);
            }
            for (int k = 0; k < 12 && V > 12u; k++) {
                const uint32_t idx = 1u + ((uint32_t)t_rng_next(&rng) % (V - 2u));
                float s = x;
                for (int j = 0; j < (k % 3); j++) {
                    s = nextafterf(s, (k & 4) ? -FLT_MAX : FLT_MAX);
                }
                if (idx != m0) row[idx] = 12.0f + T * s;
            }
        }
        break;
    }
    case SHAPE_SPRINKLE_NONFINITE:
        row[m0] = 25.0f;
        if (V > 64u) {
            const uint32_t n_bad = V / 512u + 3u;
            for (uint32_t k = 0; k < n_bad && V > 8u; k++) {
                const uint32_t idx = (uint32_t)(t_rng_uniform(&rng) * V);
                if (idx == m0) continue;
                switch (k % 3) {
                case 0: row[idx] = t_neg_inf(); break;
                case 1: row[idx] = t_pos_inf(); break;
                default: row[idx] = t_nan(); break;
                }
            }
        }
        break;
    case SHAPE_ALL_NONFINITE:
        for (uint32_t i = 0; i < V; i++) {
            switch ((i + (uint32_t)(seed & 3u)) % 3u) {
            case 0: row[i] = t_neg_inf(); break;
            case 1: row[i] = t_pos_inf(); break;
            default: row[i] = t_nan(); break;
            }
        }
        break;
    case SHAPE_ALL_EQUAL:
        for (uint32_t i = 0; i < V; i++) {
            row[i] = -2.75f + 0.5f * (float)(seed % 3u);
        }
        break;
    case SHAPE_BELOW_SENTINEL:
        /* finite values far below the -1e30 scan sentinel: the sentinel
         * itself must remain the max and best must stay 0 */
        for (uint32_t i = 0; i < V; i++) {
            row[i] = -2.0e38f - (float)(t_rng_uniform(&rng) * 1.0e38);
        }
        if ((seed & 1u) && V > 2u) row[0] = t_nan();
        if ((seed & 2u) && V > 5u) row[5] = -1.0e30f;   /* == sentinel */
        if ((seed & 4u) && V > 9u) row[9] = -1.0e30f;
        break;
    case SHAPE_GAUSS_OUTLIERS: {
        float base_max = -1.0e30f;
        for (uint32_t i = 0; i < V; i++) {
            row[i] = t_rng_gauss(&rng) * 3.0f;
            if (row[i] > base_max) base_max = row[i];
        }
        for (int k = 0; k < 8 && V > 16u; k++) {
            uint32_t idx = (uint32_t)(t_rng_uniform(&rng) * V);
            if (idx == m0) continue;
            row[idx] = base_max + 0.5f + (float)k * 0.7f;
        }
        row[m0] = base_max + 6.5f;
        break;
    }
    case SHAPE_RANDOM_WIDE:
        for (uint32_t i = 0; i < V; i++) {
            row[i] = (float)(t_rng_uniform(&rng) * 80.0 - 40.0);
        }
        break;
    case SHAPE_ZEROS:
        for (uint32_t i = 0; i < V; i++) {
            row[i] = (i & 1u) ? -0.0f : +0.0f;
        }
        if (V > 3u) row[V - 1u] = bits_f32(0x00000001u); /* denormal */
        break;
    default:
        break;
    }
}

/* ---- exactness comparison ---- */
static void compare_case(const float *row, uint32_t V, float T, float mu,
                         uint64_t seed, const char *tag) {
    uint64_t rng_fast = seed * 0x9E3779B97F4A7C15ull + 0x123456789abcdef1ull;
    if (rng_fast == 0) rng_fast = 42;
    uint64_t rng_leg = rng_fast;

    g_ds4_sample_legacy_env = 0;
    const int fast = sample_full_vocab(row, V, T, 1.0f, mu,
                                       &rng_fast, g_scratch);
    g_ds4_sample_legacy_env = 1;
    const int leg = sample_full_vocab(row, V, T, 1.0f, mu,
                                      &rng_leg, g_scratch);
    if (fast != leg || rng_fast != rng_leg) {
        fail_case(tag, V, T, mu, seed, fast, leg, rng_fast, rng_leg);
    }
    g_cases++;
}

/* Wrapper-level comparison: normalization of top_p > 1 and min_p < 0 in
 * sample_top_p_min_p still lands in the same branch. */
static void compare_wrapper_case(const float *row, uint32_t V, float T,
                                 float top_p, float mu, uint64_t seed) {
    uint64_t rng_fast = seed * 0xD1B54A32D192ED03ull + 7u;
    uint64_t rng_leg = rng_fast;

    g_ds4_sample_legacy_env = 0;
    const int fast = sample_top_p_min_p(row, V, T, 0, top_p, mu,
                                        &rng_fast, g_scratch);
    g_ds4_sample_legacy_env = 1;
    const int leg = sample_top_p_min_p(row, V, T, 0, top_p, mu,
                                       &rng_leg, g_scratch);
    if (fast != leg || rng_fast != rng_leg) {
        fail_case("wrapper", V, T, mu, seed, fast, leg, rng_fast, rng_leg);
    }
    g_cases++;
}

/* ---- microbenchmark ---- */
static double now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

static double bench_path(const float *row, uint32_t V, float T, float mu,
                         int legacy, int draws, uint64_t *tokens_out) {
    uint64_t rng = 0xC0FFEE123456789ull;
    uint64_t acc = 0;
    g_ds4_sample_legacy_env = legacy;
    for (int i = 0; i < 50; i++) {
        acc += (uint64_t)sample_full_vocab(row, V, T, 1.0f, mu,
                                           &rng, g_scratch);
    }
    const double t0 = now_ns();
    for (int i = 0; i < draws; i++) {
        acc += (uint64_t)sample_full_vocab(row, V, T, 1.0f, mu,
                                           &rng, g_scratch);
    }
    const double dt = now_ns() - t0;
    *tokens_out = acc;
    return dt / (double)draws;
}

static void run_microbench(const float *row, const char *label) {
    enum { BENCH_DRAWS = 3000 };
    uint64_t acc_fast = 0, acc_leg = 0;
    const double ns_leg = bench_path(row, TEST_VOCAB, 1.0f, 0.05f, 1,
                                     BENCH_DRAWS, &acc_leg);
    const double ns_fast = bench_path(row, TEST_VOCAB, 1.0f, 0.05f, 0,
                                      BENCH_DRAWS, &acc_fast);
    if (acc_fast != acc_leg) {
        fprintf(stderr, "MISMATCH: microbench %s token stream "
                        "(fast=%llu legacy=%llu)\n", label,
                (unsigned long long)acc_fast,
                (unsigned long long)acc_leg);
        exit(1);
    }
    printf("  %-22s legacy %10.0f ns/draw   fast %10.0f ns/draw   %.2fx\n",
           label, ns_leg, ns_fast,
           ns_fast > 0.0 ? ns_leg / ns_fast : 0.0);
}

int main(void) {
    g_scratch = malloc((size_t)TEST_VOCAB * sizeof(g_scratch[0]));
    float *row = malloc((size_t)TEST_VOCAB * sizeof(row[0]));
    if (!g_scratch || !row) {
        fprintf(stderr, "out of memory\n");
        return 1;
    }

    /* ---- runtime selection of the legacy path ---- */
    if (ds4_sample_legacy_env_value("1") != 1 ||
        ds4_sample_legacy_env_value("0") != 0 ||
        ds4_sample_legacy_env_value("") != 0 ||
        ds4_sample_legacy_env_value("yes") != 0 ||
        ds4_sample_legacy_env_value("1x") != 0 ||
        ds4_sample_legacy_env_value(NULL) != 0) {
        fprintf(stderr, "FAIL: DS4_SAMPLE_LEGACY env parsing\n");
        return 1;
    }
    g_ds4_sample_legacy_env = -1;
    setenv("DS4_SAMPLE_LEGACY", "1", 1);
    if (!ds4_sample_legacy_selected() || g_ds4_sample_legacy_env != 1) {
        fprintf(stderr, "FAIL: DS4_SAMPLE_LEGACY=1 not honored\n");
        return 1;
    }
    g_ds4_sample_legacy_env = -1;
    setenv("DS4_SAMPLE_LEGACY", "0", 1);
    if (ds4_sample_legacy_selected()) {
        fprintf(stderr, "FAIL: DS4_SAMPLE_LEGACY=0 not honored\n");
        return 1;
    }
    unsetenv("DS4_SAMPLE_LEGACY");
    g_ds4_sample_legacy_env = -1;
    if (ds4_sample_legacy_selected()) {
        fprintf(stderr, "FAIL: default is not the fast path\n");
        return 1;
    }

    /* ---- exactness sweep ---- */
    static const uint32_t Vs[] = { 1u, 2u, 7u, 1000u, TEST_VOCAB };
    static const float Ts[] = { 0.3f, 0.7f, 1.0f, 1.5f };
    static const float Ms[] = { 0.0f, 0.01f, 0.05f, 0.5f, 1.0f, 1.5f };

    for (int shape = 0; shape < SHAPE_COUNT; shape++) {
        for (size_t iv = 0; iv < sizeof(Vs) / sizeof(Vs[0]); iv++) {
            const uint32_t V = Vs[iv];
            for (size_t it = 0; it < sizeof(Ts) / sizeof(Ts[0]); it++) {
                const float T = Ts[it];
                for (size_t im = 0; im < sizeof(Ms) / sizeof(Ms[0]); im++) {
                    const float mu = Ms[im];
                    build_row(row, V, shape, 1000u + (uint64_t)shape * 131u
                              + (uint64_t)iv * 17u + (uint64_t)it * 7u
                              + (uint64_t)im * 3u, T, mu);
                    for (int s = 0; s < DRAW_SEEDS; s++) {
                        compare_case(row, V, T, mu,
                                     (uint64_t)s * 7919u + 101u,
                                     shape_name(shape));
                    }
                    if (V == 1000u) {
                        compare_wrapper_case(row, V, T, 1.0f, mu, 555u);
                        compare_wrapper_case(row, V, T, 1.5f, mu, 556u);
                        if (mu > 0.0f) {
                            compare_wrapper_case(row, V, T, 1.0f, -mu, 557u);
                        }
                    }
                }
            }
        }
        printf("shape %-19s done (%llu cases so far)\n",
               shape_name(shape), (unsigned long long)g_cases);
    }

    /* ---- extra corners outside the matrix ---- */
    {
        static const float corner_Ts[] = { 0.05f, 1e-6f, 1e-30f, 100.0f };
        static const float corner_Ms[] = { 0.0f, 0.9999999f, 1.0f,
                                           1.0000001f, 2.0f };
        const float corner_nan_min_p[] = { 0.05f, t_nan() };
        for (size_t it = 0; it < sizeof(corner_Ts) / sizeof(corner_Ts[0]);
             it++) {
            for (size_t im = 0; im < sizeof(corner_Ms) / sizeof(corner_Ms[0]);
                 im++) {
                for (int shape = SHAPE_BOUNDARY_LADDER;
                     shape <= SHAPE_BOUNDARY_CLUSTERS; shape++) {
                    build_row(row, 1000u, shape, 777u + it * 13u + im,
                              corner_Ts[it], corner_Ms[im]);
                    for (int s = 0; s < DRAW_SEEDS; s++) {
                        compare_case(row, 1000u, corner_Ts[it],
                                     corner_Ms[im], (uint64_t)s * 7919u + 3u,
                                     "corner");
                    }
                }
            }
        }
        /* NaN min_p normalizes to min_rel == 0 (no boundary): compare a
         * few rows anyway, V small enough that the fast path collects */
        for (size_t k = 0; k < sizeof(corner_nan_min_p) /
                           sizeof(corner_nan_min_p[0]); k++) {
            for (int shape = SHAPE_PEAKED; shape <= SHAPE_SPRINKLE_NONFINITE;
                 shape++) {
                build_row(row, 1000u, shape, 909u + k, 1.0f,
                          corner_nan_min_p[k]);
                for (int s = 0; s < DRAW_SEEDS; s++) {
                    compare_case(row, 1000u, 1.0f, corner_nan_min_p[k],
                                 (uint64_t)s * 6007u + 11u, "corner-nan-minp");
                }
            }
        }
        printf("corner cases done (%llu cases so far)\n",
               (unsigned long long)g_cases);
    }

    if (g_cases < 20000u) {
        fprintf(stderr, "FAIL: only %llu cases ran, need >= 20000\n",
                (unsigned long long)g_cases);
        return 1;
    }
    printf("exactness: %llu cases, fast == legacy (token and rng state)\n",
           (unsigned long long)g_cases);

    /* ---- microbenchmark at the Qwen3.8 vocabulary ---- */
    printf("min-p draw microbenchmark  V=%u  temperature=1.00  min_p=0.05  "
           "draws=3000 per path\n", TEST_VOCAB);
    build_row(row, TEST_VOCAB, SHAPE_PEAKED, 424242u, 1.0f, 0.05f);
    run_microbench(row, "peaked row");
    build_row(row, TEST_VOCAB, SHAPE_GAUSS_OUTLIERS, 171717u, 1.0f, 0.05f);
    run_microbench(row, "realistic row");

    free(row);
    free(g_scratch);
    printf("sample min-p exact: all cases passed\n");
    return 0;
}
