/* Adaptive depth policy of the Qwen3.8 speculative cycle (qwen4_spec_depth)
 * as a pure state machine: no engine, model, or GPU is created.  ds4.c is
 * included whole (without DS4_NO_GPU — the spec section is GPU-guarded at
 * compile time) so the static policy functions are reachable; the session is
 * a zeroed struct exactly like ds4_session_create produces, driven the way
 * ds4_session_qwen4_spec_cycle_inner drives it. */
#include "../ds4.c"
#include <assert.h>

static void note_window(ds4_session *s, int accepts, int rejects) {
    for (int i = 0; i < accepts; i++) qwen4_spec_note_first_draft(s, true);
    for (int i = 0; i < rejects; i++) qwen4_spec_note_first_draft(s, false);
}

int main(void) {
    char *saved_depth = getenv("DS4_QWEN4_MTP_DEPTH");
    if (saved_depth) saved_depth = strdup(saved_depth);
    unsetenv("DS4_QWEN4_MTP_DEPTH");

    /* fresh session: no window, no warmup -> depth 2 */
    ds4_session *s = xcalloc(1, sizeof(*s));
    assert(qwen4_spec_depth(s) == 2);
    assert(!s->qwen4_depth3_engaged);

    /* warmup with a perfect window engages depth 3 */
    note_window(s, 8, 0);
    assert(s->qwen4_depth_cycles == 8);
    assert(qwen4_spec_depth(s) == 3);
    assert(s->qwen4_depth3_engaged);

    /* seven of eight accepted keeps depth 3 (bits >= 6) */
    note_window(s, 7, 1);
    assert(qwen4_spec_depth(s) == 3);

    /* weak window disengages and clears the streak */
    s->qwen4_reject2_streak = 1;
    note_window(s, 0, 8);
    assert(qwen4_spec_depth(s) == 2);
    assert(!s->qwen4_depth3_engaged);
    assert(s->qwen4_reject2_streak == 0);

    /* a fresh perfect window re-engages after a weak-window disengage */
    note_window(s, 8, 0);
    assert(qwen4_spec_depth(s) == 3);

    /* second-draft rejections disengage at streak >= 2 and must clear the
     * streak there too: it is only updated inside deep cycles, so a stale
     * streak would latch depth 3 off for the life of the session */
    s->qwen4_reject2_streak = 1;
    assert(qwen4_spec_depth(s) == 3);   /* one rejection is not enough */
    s->qwen4_reject2_streak = 2;
    assert(qwen4_spec_depth(s) == 2);
    assert(!s->qwen4_depth3_engaged);
    assert(s->qwen4_reject2_streak == 0);

    /* the regression: a new perfect window re-engages depth 3 */
    note_window(s, 8, 0);
    assert(qwen4_spec_depth(s) == 3);
    assert(s->qwen4_depth3_engaged);

    /* an accept2 inside the next deep cycle clears the streak itself */
    s->qwen4_reject2_streak = 1;
    assert(qwen4_spec_depth(s) == 3);

    /* env override wins over the adaptive policy */
    setenv("DS4_QWEN4_MTP_DEPTH", "2", 1);
    assert(qwen4_spec_depth(s) == 2);
    setenv("DS4_QWEN4_MTP_DEPTH", "3", 1);
    ds4_session *fresh = xcalloc(1, sizeof(*fresh));
    assert(qwen4_spec_depth(fresh) == 3);
    assert(!fresh->qwen4_depth3_engaged);   /* forced, not learned */
    free(fresh);
    if (saved_depth) setenv("DS4_QWEN4_MTP_DEPTH", saved_depth, 1);
    else unsetenv("DS4_QWEN4_MTP_DEPTH");
    free(saved_depth);

    free(s);
    printf("Qwen3.8 spec depth policy: warmup, engage, disengage, streak latch, env override OK\n");
    return 0;
}
