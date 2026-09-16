# Gates: Apple Silicon observability dashboard

OWNS: dashboard/**, dashboard.html, ds4_server.c, ds4_gpu.h, ds4_metal.m, ds4_cuda.cu, rocm/ds4_rocm_current_api_compat.cuh

Summary: ds4 /dashboard diagnoses inference from runtime primitives without hardcoded machine or model assumptions

- [x] G1: derived metric formulas match the spec on known fixtures
  CHECK: npm test
  EXPECT: /^metrics tests passed$/m
  CWD: dashboard
  EVIDENCE: automatic-evidence=v1; definition-sha256=13e57c386db367481df35a09c0283e39b7a299db84d6c01d16472606c11d905a; exit=0; EXPECT=matched; output-sha256=7c34feb0a0995518609d8faa8a18fd4ed45b117f23f5f656a51ab4ba171ec7ff; output-bytes=1083; shell=/bin/sh; cwd=dashboard; path=7eb9a1746060/24 entries

- [x] G2: missing telemetry renders as Not instrumented and zero stays 0
  CHECK: npm test
  EXPECT: /^availability tests passed$/m
  CWD: dashboard
  EVIDENCE: automatic-evidence=v1; definition-sha256=364d5cce2b302c8a9976012b45a99f65b21b5ad8d04b49140b6bfe71f9cbb0e1; exit=0; EXPECT=matched; output-sha256=3ee36464b99a4a44294f309498edfaf3d13f0206f9c713fab6e3da34d7b32a50; output-bytes=1083; shell=/bin/sh; cwd=dashboard; path=7eb9a1746060/24 entries

- [x] G3: dashboard typechecks
  CHECK: npx tsc --noEmit && printf 'TYPECHECK_OK\n'
  EXPECT: /^TYPECHECK_OK$/m
  CWD: dashboard
  EVIDENCE: automatic-evidence=v1; definition-sha256=ec984b94c9d3d66fb41177c58f81f66fb5589ca02755f50ed5f48392126baacc; exit=0; EXPECT=matched; output-sha256=0d31cf08e125020004c508c562e62037cd1809c414d62a869bc1452c072c96f0; output-bytes=13; shell=/bin/sh; cwd=dashboard; path=7eb9a1746060/24 entries

- [x] G4: dashboard eslint is clean
  CHECK: npx eslint src && printf 'LINT_OK\n'
  EXPECT: /^LINT_OK$/m
  CWD: dashboard
  EVIDENCE: automatic-evidence=v1; definition-sha256=60f3b1a22d15fc6cc6620e76c48498c9b24413a92e6a98433503a1c7e212a25f; exit=0; EXPECT=matched; output-sha256=00f8027709ba969248ef1de221d355e95291fe77a63021b526a7db106e9f1f09; output-bytes=8; shell=/bin/sh; cwd=dashboard; path=7eb9a1746060/24 entries

- [x] G5: dashboard production bundle is a single self-contained HTML file
  CHECK: npm run build && python3 -c "import pathlib; p=pathlib.Path('../dashboard.html'); t=p.read_text(); assert p.stat().st_size>10000; assert '<div id=\"root\">' in t; assert 'http://' not in t.split('<script')[0] or 'color-scheme' in t; print('DASHBOARD_BUILT')"
  EXPECT: /^DASHBOARD_BUILT$/m
  CWD: dashboard
  EVIDENCE: automatic-evidence=v1; definition-sha256=dd4ddb7ff02d93fe90a2a22b4029d0ab66262ea213d44debcfc0692fa75a4207; exit=0; EXPECT=matched; output-sha256=111b13455a5e49a7d6c1c267210a4fae024818db3b5b7377af278ced7af05d9e; output-bytes=417; shell=/bin/sh; cwd=dashboard; path=7eb9a1746060/24 entries

- [x] G6: server unit tests still accept the published /stats counters
  CHECK: make ds4_test && ./ds4_test --server
  EXPECT: server: OK
  EVIDENCE: automatic-evidence=v1; definition-sha256=7ce23940d658b682fa55a03a45ab2f856fec523ac712a0b57bb127442505b988; exit=0; EXPECT=matched; output-sha256=dd2c0f7c75fae1914be2edb8fcb79e49c53ecf5fb132ece835271e08cc08cbf1; output-bytes=291; shell=/bin/sh; cwd=.; path=7eb9a1746060/24 entries; inputs=ds4_test@14f706ccfbe9
