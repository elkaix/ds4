# 32K/48K tensor-off sanity — ABORTED, rerun pending

13:12:53 attempt refused: operator started the production server one
second earlier (organic session underway on the corrected route).
Single sweep planned: --ctx-start 32768 --step-incr 16384 --ctx-max 49152,
teacher-forced 256, power 100, fans max, DS4_METAL_DISABLE_TENSOR_API=1.
Purpose: cost of disabling the non-equivalent tensor route (~0/2/10%+),
to reset the 33.8/33.4 t/s attribution (those numbers ran tensor ON).

Note: the running organic session's early generations at ~33K will
provide a first organic tensor-off decode figure anyway.
