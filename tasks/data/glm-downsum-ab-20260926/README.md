# GLM down+sum n8 experiment (DS4_METAL_ENABLE_DOWNSUM8), 2026-09-26
- patch: ds4_metal.m:41538 admitted n8 single-device into q2_k sum6 down+sum (encoder nei0<=8 OK, shader loops nei0); REVERTED after NULL verdict
- equivalence: greedy 48/48 identical w/ route active; avg_nll 2.242623864 off vs 2.241643279 on (d=9.8e-4 abs, 4.4e-4 rel, << 0.005 fixture noise); ppl runs bit-reproducible
- A/B: 4 ABBA cycles 2K teacher-forced, route asserted sum8=1 on every B run, paired B/A x1.0102 (1.0036/1.0041/1.0208/1.0124), inside 8% drift band -> NULL
- generation path confirmed to reach the same generic dispatcher (route line in greedy-on log)
- ideal dispatch saving ~0.4 ms of 26.9 ms/token matches measured ~0.3 ms
