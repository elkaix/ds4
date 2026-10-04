/* Adaptive copy-draft width of the Qwen3.8 speculative cycle
 * (qwen4_copy_width_update and friends): pure CPU, no engine, model, or
 * GPU.  ds4.c is included whole so the static policy functions are
 * reachable; the session is a zeroed struct driven the way
 * ds4_session_qwen4_spec_cycle_inner drives it.  The contract under test:
 * the width starts at (and 0 means) the cap, a run of full accepts stays
 * there, a partial accept drops to max(floor, a+1) and full accepts climb
 * back one per cycle, the width never leaves [floor, cap], ADAPT=0 pins it
 * at the cap, DS4_QWEN4_COPY_FLOOR overrides the floor, and replaced
 * checkpoint content (the generation bump that clears the copy index)
 * resets it to the cap while a plain rewind does not. */
#include "../ds4.c"
#include <assert.h>

static const uint32_t CAP = 7u;   /* QWEN4_COPY_MAX_LIMIT, the widest cap */

static void reset_policy_env(void) {
    qwen4_copy_adapt_env = -1;
    qwen4_copy_floor_env = -1;
}

int main(void) {
    char *saved_adapt = getenv("DS4_QWEN4_COPY_ADAPT");
    char *saved_floor = getenv("DS4_QWEN4_COPY_FLOOR");
    char *saved_max = getenv("DS4_QWEN4_COPY_MAX");
    char *saved_t4 = getenv("DS4_QWEN4_COPY_T4");
    if (saved_adapt) saved_adapt = strdup(saved_adapt);
    if (saved_floor) saved_floor = strdup(saved_floor);
    if (saved_max) saved_max = strdup(saved_max);
    if (saved_t4) saved_t4 = strdup(saved_t4);
    unsetenv("DS4_QWEN4_COPY_ADAPT");
    unsetenv("DS4_QWEN4_COPY_FLOOR");
    /* the widest cap so the rule is exercised over its full range */
    setenv("DS4_QWEN4_COPY_MAX", "7", 1);
    unsetenv("DS4_QWEN4_COPY_T4");
    reset_policy_env();

    /* --- the pure rule: inputs (width, cap, floor, n, a) -> new width --- */

    /* no copies verified: the width is untouched */
    for (uint32_t w = 0; w <= CAP; w++)
        assert(qwen4_copy_width_update(w, CAP, 3u, 0u, 0u) == w);

    /* all accepted: +1 per cycle, pinned at the cap */
    assert(qwen4_copy_width_update(1u, CAP, 2u, 1u, 1u) == 2u);
    assert(qwen4_copy_width_update(4u, CAP, 3u, 4u, 4u) == 5u);
    assert(qwen4_copy_width_update(CAP, CAP, 3u, CAP, CAP) == CAP);

    /* partial accept: max(floor, a + 1) */
    assert(qwen4_copy_width_update(7u, CAP, 3u, 7u, 1u) == 3u);   /* floor wins */
    assert(qwen4_copy_width_update(7u, CAP, 3u, 7u, 2u) == 3u);
    assert(qwen4_copy_width_update(7u, CAP, 3u, 7u, 4u) == 5u);   /* a + 1 wins */
    assert(qwen4_copy_width_update(7u, CAP, 3u, 7u, 6u) == CAP);
    assert(qwen4_copy_width_update(7u, CAP, 2u, 7u, 1u) == 2u);   /* floor 2 */
    assert(qwen4_copy_width_update(3u, CAP, 3u, 2u, 0u) == 3u);   /* break at row 0 */

    /* exhaustive over session-reachable states (width in [floor, cap]:
     * width starts at the cap and the drop path clamps at the floor, so
     * the +1 widen never starts below it): the rule never leaves
     * [floor, cap] and matches the spec (a == n -> +1; else
     * max(floor, a+1)) for every reachable state */
    for (uint32_t floor = 2u; floor <= CAP; floor++) {
        for (uint32_t width = floor; width <= CAP; width++) {
            for (uint32_t n = 1u; n <= CAP; n++) {
                for (uint32_t a = 0u; a <= n; a++) {
                    const uint32_t next = qwen4_copy_width_update(width, CAP, floor, n, a);
                    assert(next >= floor && next <= CAP);
                    if (a == n) assert(next == (width < CAP ? width + 1u : CAP));
                    else {
                        const uint32_t want = a + 1u;
                        assert(next == (want > floor ? want : floor));
                    }
                }
            }
        }
    }

    /* --- session semantics: offer + note, default env --- */
    const uint32_t cap = qwen4_copy_cap();      /* 7 in this environment */
    const uint32_t floor = qwen4_copy_floor(cap);
    assert(cap >= 2u && floor >= 2u && floor <= cap);

    /* fresh session: width 0 means the cap, so copy-heavy text starts at
     * today's full width from the first cycle */
    ds4_session *s = xcalloc(1, sizeof(*s));
    assert(qwen4_copy_offer(s, cap) == cap);

    /* a run of full accepts stays at the cap */
    qwen4_copy_note_verified(s, cap, cap);
    qwen4_copy_note_verified(s, cap, cap);
    qwen4_copy_note_verified(s, 2u, 2u);
    assert(s->qwen4_copy_width == 0u || s->qwen4_copy_width == cap);
    assert(qwen4_copy_offer(s, cap) == cap);

    /* one partial accept drops to max(floor, a+1): the agent-shaped case
     * (cap offered, 1-2 accepted).  A fixed-cap replacement fails here. */
    qwen4_copy_note_verified(s, cap, 1u);
    assert(s->qwen4_copy_width == (floor > 2u ? floor : 2u));
    assert(qwen4_copy_offer(s, cap) == (floor > 2u ? floor : 2u));
    qwen4_copy_note_verified(s, cap, 2u);
    assert(s->qwen4_copy_width == (floor > 3u ? floor : 3u));

    /* full accepts climb back one per cycle to the cap, then stay */
    while (s->qwen4_copy_width < cap) {
        const uint32_t w = s->qwen4_copy_width;
        qwen4_copy_note_verified(s, w, w);
        assert(s->qwen4_copy_width == w + 1u);
    }
    qwen4_copy_note_verified(s, cap, cap);
    assert(s->qwen4_copy_width == cap);
    assert(qwen4_copy_offer(s, cap) == cap);

    /* the offer clamps a width past the cap back to it */
    s->qwen4_copy_width = cap + 2u;
    assert(qwen4_copy_offer(s, cap) == cap);
    free(s);

    /* --- DS4_QWEN4_COPY_FLOOR override --- */
    ds4_session *f = xcalloc(1, sizeof(*f));
    setenv("DS4_QWEN4_COPY_FLOOR", "5", 1);
    reset_policy_env();
    assert(qwen4_copy_floor(CAP) == 5u);
    qwen4_copy_note_verified(f, cap, 1u);
    assert(f->qwen4_copy_width == (cap > 5u ? 5u : cap));   /* floor 5 beats a+1 */
    free(f);

    f = xcalloc(1, sizeof(*f));
    setenv("DS4_QWEN4_COPY_FLOOR", "2", 1);
    reset_policy_env();
    assert(qwen4_copy_floor(CAP) == 2u);
    qwen4_copy_note_verified(f, cap, 1u);
    assert(f->qwen4_copy_width == 2u);           /* a + 1 = 2 */
    free(f);

    /* out-of-range floors fall back to the default 3, clamped to the cap */
    setenv("DS4_QWEN4_COPY_FLOOR", "8", 1);
    reset_policy_env();
    assert(qwen4_copy_floor(CAP) == 3u);
    setenv("DS4_QWEN4_COPY_FLOOR", "1", 1);
    reset_policy_env();
    assert(qwen4_copy_floor(CAP) == 3u);
    setenv("DS4_QWEN4_COPY_FLOOR", "9", 1);
    reset_policy_env();
    assert(qwen4_copy_floor(2u) == 2u);          /* cap 2: floor clamped under */
    unsetenv("DS4_QWEN4_COPY_FLOOR");
    reset_policy_env();
    assert(qwen4_copy_floor(2u) == 2u);

    /* --- DS4_QWEN4_COPY_ADAPT=0: fixed cap, today's behaviour --- */
    setenv("DS4_QWEN4_COPY_ADAPT", "0", 1);
    reset_policy_env();
    assert(!qwen4_copy_adapt());
    ds4_session *a0 = xcalloc(1, sizeof(*a0));
    qwen4_copy_note_verified(a0, cap, 1u);       /* a brutal partial accept */
    qwen4_copy_note_verified(a0, cap, 0u);
    assert(a0->qwen4_copy_width == 0u);          /* never moved: still "cap" */
    assert(qwen4_copy_offer(a0, cap) == cap);
    free(a0);
    setenv("DS4_QWEN4_COPY_ADAPT", "1", 1);
    reset_policy_env();
    assert(qwen4_copy_adapt());
    unsetenv("DS4_QWEN4_COPY_ADAPT");
    reset_policy_env();
    assert(qwen4_copy_adapt());

    /* --- replaced content resets the width; a rewind does not --- */
    ds4_session *r = xcalloc(1, sizeof(*r));
    r->ctx_size = 4096;
    for (int i = 0; i < 32; i++) token_vec_push(&r->checkpoint, 100 + (i % 7));
    assert(qwen4_copy_index_sync(r));            /* builds the index at this gen */
    assert(r->qwen4_copy_idx.gen == r->checkpoint_gen);
    r->qwen4_copy_width = 4u;
    r->checkpoint_gen++;                         /* new prompt in a reused slot */
    assert(qwen4_copy_index_sync(r));            /* clears the index... */
    assert(r->qwen4_copy_ra_len == 0u);
    assert(r->qwen4_copy_idx.upto == (uint32_t)r->checkpoint.len);   /* rebuilt */
    assert(r->qwen4_copy_width == 0u);           /* ...and resets the width */
    assert(qwen4_copy_offer(r, cap) == cap);
    r->qwen4_copy_width = 4u;
    r->checkpoint.len = 16;                      /* plain rewind, same content */
    for (int i = 0; i < 8; i++) token_vec_push(&r->checkpoint, 200 + i);
    assert(qwen4_copy_index_sync(r));
    assert(r->qwen4_copy_width == 4u);           /* untouched by a rewind */
    qwen4_copy_index_free(&r->qwen4_copy_idx);
    ds4_session_free_test_checkpoint(r);

    if (saved_adapt) setenv("DS4_QWEN4_COPY_ADAPT", saved_adapt, 1);
    else unsetenv("DS4_QWEN4_COPY_ADAPT");
    if (saved_floor) setenv("DS4_QWEN4_COPY_FLOOR", saved_floor, 1);
    else unsetenv("DS4_QWEN4_COPY_FLOOR");
    if (saved_max) setenv("DS4_QWEN4_COPY_MAX", saved_max, 1);
    else unsetenv("DS4_QWEN4_COPY_MAX");
    if (saved_t4) setenv("DS4_QWEN4_COPY_T4", saved_t4, 1);
    else unsetenv("DS4_QWEN4_COPY_T4");
    reset_policy_env();
    free(saved_adapt);
    free(saved_floor);
    free(saved_max);
    free(saved_t4);

    printf("Qwen3.8 copy width: start at cap, hold, drop to max(floor,a+1), climb, "
           "bounds, ADAPT=0, floor override, replace-reset OK\n");
    return 0;
}
