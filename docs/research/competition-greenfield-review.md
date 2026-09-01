# Competition Greenfield Artifact Review

Date: 2026-09-01

Scope: read-only review of the seven top-level untracked files under
`ds4-glm53/competition/`, frozen at combined SHA-256
`c18aa7652130332eb46d5e2ceffc7407f2540aba9d07504b07b812daa6f7ab17`.
The vendored `third_party/metal-cpp` tree was identified but not treated as
original implementation. Claude's live server, model, and test were untouched.

## Verdict

**REJECT and do not integrate.** These files are a separate C++/`metal-cpp`
harness, not an optimization of antirez/ds4. They neither implement the exact
GLM-5.3 execution graph nor form a runnable engine. Their theoretical numbers
receive zero S55 credit.

The related untracked `roadmap.md`, frozen at SHA-256
`1e6d845f25d519dcd56e602142e22417d8eb045fc586a14e2705e8cf7034b2f5`, is
also not the execution contract. It scores only through 85K, reuses 2K/32K
acceptance, and prioritizes rewind/TTFT work that contributes zero decode t/s.
The user's S55-200 contract and `tasks/todo.md` supersede it.

## Critical Findings

- **G1 — Wrong stack and no integration.** `inference_engine.cpp` includes the
  vendored `metal-cpp` API and implements a new C++ backend. DS4's `Makefile`
  contains no reference to any `competition/` source or kernel. This directly
  violates the native C99/Objective-C, existing-harness contract.
- **G2 — Startup fails before inference.** `DecodeStep::Prepare()` requests
  `paged_hybrid_attention_d128`; the Metal file exports
  `paged_flash_decode_d128`. No requested attention pipeline exists.
- **G3 — KDA and DSA are absent.** The host appends `layer_kind` and
  `dsa_top_k` to its parameter struct, but the Metal struct ends at `n_seqs`
  and never reads either field. The kernel always scans the full paged KV
  history. It has no KDA recurrent state transition and no DSA sparse routing
  table or top-k physical-block traversal.
- **G4 — The checkpoint formats are not implemented.** `dot_group_q2()` is a
  flat affine two-bit unpack. It is neither the IQ2_XXS codebook/sign layout nor
  the Q2_K 256-weight super-block layout. The fused FFN itself calls only the
  generic Q4 unpack. Exact GLM weights therefore cannot be consumed correctly.
- **G5 — The decode graph is incomplete and unsafe.** Router IDs and weights
  are produced but never used to address or weight experts; the loop dispatches
  the same tensor pointers nine times with weight `1.0`. Dense FFN geometry is
  12,288 while the fused kernel allocates space for 4,096 intermediates. Output
  projection, logits, residual/mHC, real KDA/DSA state, sampling, and a C run
  entrypoint are absent. Cross-token frames-in-flight also cannot overlap the
  batch-one autoregressive dependency claimed by the header.

## Evidence Gate

No compile or GPU test was run because Claude still holds the machine lease.
Compilation cannot rescue G1-G5; the pipeline-name mismatch alone is a static
startup failure. Preserve these untracked files as another agent's work, but do
not merge, benchmark, or use them to choose a production optimization.

The valid path remains the existing DS4 graph: measure MTP-on `a200` and cycle
time, accept a non-serializing profiler, run M1, then edit only the largest
measured component with a plausible >=5% whole-cycle ceiling.
