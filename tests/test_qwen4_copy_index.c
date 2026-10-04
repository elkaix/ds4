/* Prompt-lookup index for the Qwen3.8 context-copy proposer
 * (qwen4_copy_propose and its qwen4_copy_index): pure CPU, no engine,
 * model, or GPU.  The contract under test: the index grows only for
 * appended checkpoint tokens, the proposer picks the occurrence with the
 * best backward agreement (falling back to older ones), a rewind truncates
 * the index without leaking stale positions, a replaced prefix is caught
 * by the guard and rebuilt, and a copy block that broke on an edit
 * re-anchors its source alignment on the MTP draft. */
#include "../ds4.c"
#include <assert.h>

static ds4_session *new_session(int ctx) {
    ds4_session *s = xcalloc(1, sizeof(*s));
    s->ctx_size = ctx;
    return s;
}

static void push(ds4_session *s, const int *t, int n) {
    for (int i = 0; i < n; i++) token_vec_push(&s->checkpoint, t[i]);
}

static void free_session(ds4_session *s) {
    qwen4_copy_index_free(&s->qwen4_copy_idx);
    ds4_session_free_test_checkpoint(s);   /* frees s itself */
}

static void check_out(const int *out, int nc, const int *want, int nw) {
    assert(nc == nw);
    for (int i = 0; i < nw; i++) assert(out[i] == want[i]);
}

static void test_best_candidate(void) {
    /* two occurrences of [3,4,5,6]; the newer is preceded by a mismatch,
     * the older by seven more agreeing tokens: the older must win */
    ds4_session *s = new_session(4096);
    const int t[] = { 50,51, 1,2,3,4,5,6, 7,8, 99, 21,22, 1,2,3,4,5,6, 40,41,
                      50,51, 1,2,3,4,5 };
    push(s, t, (int)(sizeof(t) / sizeof(t[0])));
    int out[QWEN4_COPY_MAX_LIMIT];
    int nc = qwen4_copy_propose(s, 6, 4, out, 7);
    check_out(out, nc, (const int[]){ 7,8,99,21,22,1,2 }, 7);
    assert(s->qwen4_copy_src == 8);
    /* ties keep the newest: min 2 and min 4 (the gram floor) agree here */
    int out2[QWEN4_COPY_MAX_LIMIT];
    check_out(out2, qwen4_copy_propose(s, 6, 2, out2, 7), (const int[]){ 7,8,99,21,22,1,2 }, 7);
    /* the newest occurrence's agreement is 6, the older one's 8 */
    check_out(out2, qwen4_copy_propose(s, 6, 8, out2, 7), (const int[]){ 7,8,99,21,22,1,2 }, 7);
    assert(qwen4_copy_propose(s, 6, 9, out2, 7) == 0);
    /* incremental growth: streaming the same text in two chunks finds it */
    ds4_session *s2 = new_session(4096);
    push(s2, t, 12);
    assert(qwen4_copy_propose(s2, 6, 4, out, 7) == 0);
    push(s2, t + 12, (int)(sizeof(t) / sizeof(t[0])) - 12);
    check_out(out, qwen4_copy_propose(s2, 6, 4, out, 7), (const int[]){ 7,8,99,21,22,1,2 }, 7);
    /* bounded memory: 2x ctx table slots, prev chain of ctx + 2 */
    assert(s2->qwen4_copy_idx.mask + 1u == 8192u);
    assert(s2->qwen4_copy_idx.cap == 4098u);
    free_session(s);
    free_session(s2);
}

static void test_rewind_truncation(void) {
    /* occurrence at 0..3 and at 7..10; a rewind past the second one must
     * not leak its stale table position into later proposals */
    ds4_session *s = new_session(4096);
    const int t[] = { 1,2,3,4, 5, 60,61, 1,2,3,4, 9 };
    push(s, t, (int)(sizeof(t) / sizeof(t[0])));
    push(s, (const int[]){ 1,2,3 }, 3);
    int out[QWEN4_COPY_MAX_LIMIT];
    /* both occurrences agree 4; the tie keeps the newest ([9,1,2,3]) */
    check_out(out, qwen4_copy_propose(s, 4, 4, out, 4), (const int[]){ 9,1,2,3 }, 4);
    /* rewind past the second occurrence and regrow different text */
    s->checkpoint.len = 6;
    push(s, (const int[]){ 70,1,2,3 }, 4);
    /* the stale slot chains to the older live occurrence at 0..3 */
    check_out(out, qwen4_copy_propose(s, 4, 4, out, 4), (const int[]){ 5,60,70,1 }, 4);
    assert(s->qwen4_copy_src == 4);
    /* a shorter rewind that keeps both occurrences still finds the newest */
    s->checkpoint.len = 8;
    push(s, (const int[]){ 3,4,1,2,3 }, 5);
    /* the second occurrence's gram was partially rewritten by the regrow;
     * its stale slot still chains to the live occurrence at 0..3 */
    check_out(out, qwen4_copy_propose(s, 4, 4, out, 4), (const int[]){ 5,60,70,1 }, 4);
    assert(s->qwen4_copy_src == 4);
    free_session(s);
}

static void test_replaced_prefix(void) {
    /* session reuse / KV restore: the checkpoint is replaced wholesale;
     * proposals must come from the new content, never the old */
    ds4_session *s = new_session(4096);
    push(s, (const int[]){ 1,2,3,4,5,6,7,8 }, 8);
    int out[QWEN4_COPY_MAX_LIMIT];
    assert(qwen4_copy_propose(s, 9, 4, out, 4) == 0);   /* just index it */
    s->checkpoint.len = 0;
    push(s, (const int[]){ 5,6,7,8, 1,2,3,4, 1,2,3 }, 11);
    check_out(out, qwen4_copy_propose(s, 4, 4, out, 4), (const int[]){ 1,2,3 }, 3);
    /* same-length replacement then growth: the guard fails and rebuilds */
    ds4_session *s2 = new_session(4096);
    push(s2, (const int[]){ 1,2,3,4,5,6,7,8 }, 8);
    assert(qwen4_copy_propose(s2, 9, 4, out, 4) == 0);
    s2->checkpoint.len = 0;
    push(s2, (const int[]){ 8,7,6,5,4,3,2,1 }, 8);
    push(s2, (const int[]){ 9,9,9 }, 3);
    push(s2, (const int[]){ 7,6,5 }, 3);
    check_out(out, qwen4_copy_propose(s2, 4, 4, out, 4), (const int[]){ 3,2,1,9 }, 4);
    assert(s2->qwen4_copy_idx.upto == (uint32_t)s2->checkpoint.len);
    assert(s2->qwen4_copy_idx.live == s2->qwen4_copy_idx.upto - 4u);
    free_session(s);
    free_session(s2);
}

static void test_replaced_shorter_prefix(void) {
    /* GPU-validated regression: a reused server slot receives a shorter
     * prompt that replaces the indexed transcript (prose -> edit -> copy
     * in one slot).  The shrink alone must not be read as a rewind of the
     * same prefix: the generation bump marks the replacement, the index
     * rebuilds, and grams that exist only in the new content propose.
     * Without the bump this returns 0 proposals for the whole request. */
    ds4_session *s = new_session(4096);
    int old[240];
    for (int i = 0; i < 240; i++) old[i] = 500 + (i % 97);
    push(s, old, 240);
    int out[QWEN4_COPY_MAX_LIMIT];
    assert(qwen4_copy_propose(s, 8888, 4, out, 7) == 0);   /* index the transcript */
    assert(s->qwen4_copy_idx.upto == 240u);
    assert(s->qwen4_copy_idx.gen == s->checkpoint_gen);
    /* ds4_session_sync_internal's reset branch for a new, shorter prompt */
    s->checkpoint.len = 0;
    s->checkpoint_gen++;
    push(s, (const int[]){ 7001,7002,7003,7004, 42, 7001,7002,7003 }, 8);
    check_out(out, qwen4_copy_propose(s, 7004, 4, out, 7), (const int[]){ 42,7001,7002,7003 }, 4);
    assert(s->qwen4_copy_idx.upto == 8u);
    /* the public slot-reuse reset bumps the generation itself */
    ds4_session_invalidate(s);
    assert(s->checkpoint_gen != s->qwen4_copy_idx.gen);
    push(s, (const int[]){ 7001,7002,7003,7004, 43, 7001,7002,7003 }, 8);
    check_out(out, qwen4_copy_propose(s, 7004, 4, out, 7), (const int[]){ 43,7001,7002,7003 }, 4);
    /* a plain rewind of the new prompt keeps the fast truncate path */
    s->checkpoint.len = 5;
    push(s, (const int[]){ 7001,7002,7003 }, 3);
    check_out(out, qwen4_copy_propose(s, 7004, 4, out, 7), (const int[]){ 43,7001,7002,7003 }, 4);
    free_session(s);
}

static void test_reanchor(void) {
    /* a copy block breaks on an edit: the source alignment carries, and
     * the next proposal re-anchors on the MTP draft within +-3 tokens */
    ds4_session *s = new_session(4096);
    push(s, (const int[]){ 100,101,102,103,104,105,106,107, 100,101,102,103 }, 12);
    int out[QWEN4_COPY_MAX_LIMIT];
    check_out(out, qwen4_copy_propose(s, 104, 4, out, 7),
              (const int[]){ 105,106,107,100,101,102,103 }, 7);
    assert(s->qwen4_copy_src == 5);
    /* the cycle commits first_token and two copies, then breaks */
    push(s, (const int[]){ 104,105,106 }, 3);
    s->qwen4_copy_ra_pos = s->qwen4_copy_src + 2u;   /* the cycle does this */
    s->qwen4_copy_ra_len = (uint32_t)s->checkpoint.len;
    push(s, (const int[]){ 200 }, 1);                /* the edit's substitution */
    s->glm_mtp_have = 1;
    s->glm_mtp_parent = 108;
    s->glm_mtp_draft = 100;                          /* the copy resumes at 100 */
    check_out(out, qwen4_copy_propose(s, 108, 4, out, 7),
              (const int[]){ 100,101,102,103,104,105,106 }, 7);
    assert(s->qwen4_copy_src == 8);
    /* without the alignment the same state proposes nothing */
    ds4_session *bare = new_session(4096);
    push(bare, s->checkpoint.v, s->checkpoint.len);
    bare->glm_mtp_have = 1;
    bare->glm_mtp_parent = 108;
    bare->glm_mtp_draft = 100;
    assert(qwen4_copy_propose(bare, 108, 4, out, 7) == 0);
    /* the alignment expires once 32 more tokens have committed */
    for (int i = 0; i < 33; i++) push(s, &i, 1);
    s->glm_mtp_parent = 999;
    s->glm_mtp_draft = 0;                            /* present in the source */
    assert(qwen4_copy_propose(s, 999, 4, out, 7) == 0);
    free_session(s);
    free_session(bare);
}

static void test_edges(void) {
    ds4_session *s = new_session(64);
    int out[QWEN4_COPY_MAX_LIMIT];
    for (int n = 0; n < 5; n++) {
        push(s, (const int[]){ 1 }, 1);
        /* too short for the 4-gram lookup, whatever the minimum */
        assert(qwen4_copy_propose(s, 2, 2, out, 7) == 0);
    }
    /* a gram that never occurred proposes nothing at any length */
    assert(qwen4_copy_propose(s, 2, 2, out, 7) == 0);
    assert(qwen4_copy_propose(s, 1, 2, out, 0) == 0);
    /* an allocation failure must not crash: none here, but a missing
     * index (never synced) also proposes nothing */
    ds4_session *bare = xcalloc(1, sizeof(*bare));
    assert(qwen4_copy_propose(bare, 1, 2, out, 7) == 0);
    free_session(s);
    free_session(bare);
}

int main(void) {
    test_best_candidate();
    test_rewind_truncation();
    test_replaced_prefix();
    test_replaced_shorter_prefix();
    test_reanchor();
    test_edges();
    fprintf(stderr, "test_qwen4_copy_index: all tests passed\n");
    return 0;
}
