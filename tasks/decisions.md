# S55-200 Decisions

## D1 — Preserve the production stack

Use only the existing DS4 C / Objective-C / Metal engine and harness. Reject
the untracked C++/`metal-cpp` competition scaffold, MLX, and llama.cpp.

## D2 — Profile with native tooling first

Try `xctrace` Metal System Trace and its export before adding runtime code. The
M5 Max exposes stage-boundary but not dispatch-boundary counter sampling, so the
fallback is timestamped compute-encoder regions inside the existing command
buffer with one normal completion wait. Accept either only at <=3% overhead.

## D3 — No optimization before a measured ceiling

The first retained source change must target the largest measured 200K macro
component and have a plausible >=5% whole-cycle ceiling. The 2K routed ablation
is diagnostic, not authorization for a kernel rewrite.

## D4 — Static tensor bytes do not establish a decode roofline

Keep the reproducible 9.635 GB active-trunk inventory as a hypothesis input.
Reject its theoretical-bandwidth ceiling and the claim that S55 is impossible:
the width-2 verifier, routed expert union, nextn draft calls, and output heads
have different execution multiplicities. M1 measured bandwidth plus macro GPU
time must price them. Park KDA requantization until that proof and a recurrent-
state quality gate exist.

## D5 — Keep alternative speculation research-only

Tree/block speculation is not the closed linear width-3 design, but it requires
new branch-aware KDA state and DSA causal verification. Do not implement it from
theoretical acceptance math. Promote it only after actual `a200`, n-row timing,
route-union data, and M1 show a >=5% whole-cycle path with exact output and
adaptive fallback.

## D6 — Retain H19 as a small cleanup, not an S55 result

The discarded accepted-cycle draft head is logically dead and its 2K/65K
ABBA is positive, but it earns zero S55 credit before paired observable-state
and 200K tests. Reuse the existing native DS4 paired harness with
`DS4_GLM_MTP_DISCARDED_HEAD` supplied explicitly; do not add another harness.
Remove the unmeasured byte-share claim and correct the rollback comment when
the shared source lease is released.

## D7 — Build M1 inside DS4, not beside it

Reuse the existing graph-dump hooks to capture real 200K width-2 FFN inputs,
the existing GGUF parser for tensor offsets, and the production one-row/batch
routed-MoE APIs for arms C/D. Add only one diagnostic Metal checksum reader and
thin DS4 adapter for equal-byte sequential/pattern arms A/B. Run all four arms
over the same 4,756,340,736 logical bytes across 42 routed layers, with one
ordinary completion per arm and balanced order. Report logical and unique bytes
separately; the benchmark is diagnostic and earns zero S55 credit by itself.

## D8 — Never bypass the high-context verifier gate

Do not force `glm_graph_verify_rows` beyond its dense compact-attention window.
That would reduce effective DSA history and fail the real-200K semantic gate.
A high-context N=2 specialization is admissible only if it retains indexed DSA
selection over full physical history plus the existing KDA base/prefix
transaction, with exact output evidence.
