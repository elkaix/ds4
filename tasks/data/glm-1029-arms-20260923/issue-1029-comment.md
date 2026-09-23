Data point from a second M5 Max (128 GB). This is a different arm from the report, so it does not overturn it.

**Setup:** GLM 5.3 Flash with a hybrid quant: the Q2 body, KDA v/output at Q4_K, and routed experts in layers 17–23 at Q4_K. The engine is the base f935161c (`0e9cc2d` plus the #1090 GLM kernels), Metal, Tensor API off, resident, `ds4-bench --ctx-alloc 102400`, teacher-forced 256-token decode, powermode 2, fans pinned at max, 30 s gaps.

`0e9cc2d` makes two separable prefill changes, and I built one binary per change:
- **B:** dense-attention limit set back to `ctx_cap` (4096) instead of `top_k + pool - 1` (2051).
- **C:** sparse prefill slices on the `_valid` pipeline instead of the bounds-checked one.

Both are inexact and were built only to measure speed. Order ABC CBA, 2 cycles (the third was stopped).

| prefill t/s | A base | B dense | C valid | B/A | C/A |
|---|---|---|---|---|---|
| cold 8K | 267.8 | 279.4 | 279.3 | ×1.044 | ×1.043 |
| cold 16K | 258.3 | 266.8 | 266.6 | ×1.033 | ×1.033 |
| cold 24K | 268.3 | 258.3 | 262.9 | ×0.963 | ×0.979 |
| 2K continued on a 14K prefix | 256.1 | 249.5 | 254.6 | ×0.974 | ×0.994 |

- **Continued prefill:** neither half gets back the reported −11%. The dense-limit change can't affect it anyway: every `dense_limit` use only applies below position 4096.
- **Cold prefill:** both reverts are about 3–4% faster at 8K and 16K, but slower at 24K. That's n=2, so it's unresolved. I can't tell whether it's the same effect at a smaller size.
- **Decode:** unchanged.

The regression may depend on the plain resident Q2 layout or on the build before #1090. If someone can reproduce it, a `_valid` fast path for slices with no padded IDs would keep exactness. Reverting would not.
