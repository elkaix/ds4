# M5 clean-engine curve — PARTIAL (2 of 44 frontier measurements)

Arm A = glm53-m5-prod default (GLM top-k fast OFF, opt-in upstream).
Arm B = DS4_GLM_ENABLE_TOPK_FAST=1. ABBA planned: A1 B1 B2 A2.
Stopped by operator after 2 frontiers of A-1 to return the GPU to the
production server; re-run with tasks/m5_clean_curve.sh when the GPU is free.

Results captured (teacher-forced 256 tokens, ctx-alloc 262144, chunk 2048):
  ctx 32768  prefill 417.2 t/s  decode 33.79 t/s (steady 33.88)
  ctx 49152  prefill 394.6 t/s  decode 33.41 t/s (steady 33.45)

For scale: the old glm53-p4a organic agent session measured 28.64 t/s
token-weighted below 70K (MTP off past 32K, stale binary). Not apples to
apples (bench vs agent session), but the clean engine at 32-48K sits ~5 t/s
above that range with the M3-Ultra-gated GLM53 kernels now enabled.
