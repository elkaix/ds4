import assert from 'node:assert/strict';
import { test } from 'node:test';
import {
  checkpointAsyncNs,
  comparableDecodeRegression,
  contextBucket,
  decodeStats,
  decodeTokS,
  freshPrefillTokS,
  freshTokens,
  metalHeadroom,
  metalUtilizationPct,
  msPerToken,
  mtpEconomics,
  reusePct,
  requestCacheHitRate,
  showEffectivePromptThroughput,
  swapDelta,
  waterfallReconcile,
} from '../src/lib/metrics.ts';
import { displayNumber } from '../src/lib/availability.ts';
import { parseRequestList, parseStats } from '../src/api/stats.ts';

test('decode tok/s and ms/token', () => {
  const ns = 4.7916666667 * 1e9;
  const tokS = decodeTokS(184, ns);
  const ms = msPerToken(ns, 184);
  assert.ok(tokS != null);
  assert.ok(ms != null);
  assert.ok(Math.abs(tokS - 38.4) < 0.05);
  assert.ok(Math.abs(ms - 26.0417) < 0.05);
  assert.equal(decodeTokS(0, ns), undefined);
  assert.equal(decodeTokS(10, 0), undefined);
});

test('fresh prefill tok/s uses compute time not logical wall', () => {
  const ns = 7.41 * 1e9;
  const tokS = freshPrefillTokS(1121, ns);
  assert.ok(tokS != null);
  assert.ok(Math.abs(tokS - 151.282) < 0.05);
  assert.equal(freshPrefillTokS(0, ns), undefined);
});

test('reuse pct and fresh tokens', () => {
  const pct = reusePct(91220, 92341);
  assert.ok(pct != null);
  assert.ok(Math.abs(pct - 98.786) < 0.02);
  assert.equal(freshTokens(92341, 91220), 1121);
  assert.equal(reusePct(10, 0), undefined);
});

test('request cache hit rate is distinct from token reuse', () => {
  const hit = requestCacheHitRate(53, 2);
  assert.ok(hit != null);
  assert.ok(Math.abs(hit - 96.3636) < 0.02);
  assert.equal(requestCacheHitRate(0, 0), undefined);
});

test('metal headroom and utilization', () => {
  const allocated = 101.8 * 1024 ** 3;
  const recommended = 115.2 * 1024 ** 3;
  const head = metalHeadroom(recommended, allocated);
  const util = metalUtilizationPct(allocated, recommended);
  assert.ok(head != null);
  assert.ok(util != null);
  assert.ok(Math.abs(head / 1024 ** 3 - 13.4) < 0.05);
  assert.ok(Math.abs(util - 88.368) < 0.05);
  assert.equal(metalHeadroom(0, allocated), undefined);
});

test('checkpoint async split', () => {
  assert.equal(checkpointAsyncNs(780e6, 19e6), 761e6);
  assert.equal(checkpointAsyncNs(10, 40), 0);
});

test('MTP profitability C/S < 1+a', () => {
  const e = mtpEconomics(61, 100, 1.42e6, 1e6);
  assert.ok(e);
  assert.ok(Math.abs(e.acceptance - 0.61) < 1e-9);
  assert.ok(Math.abs(e.onePlusA - 1.61) < 1e-9);
  assert.ok(Math.abs(e.cycleOverStep - 1.42) < 1e-9);
  assert.equal(e.profitable, true);
  const unprof = mtpEconomics(10, 100, 2e6, 1e6);
  assert.ok(unprof);
  assert.equal(unprof.profitable, false);
});

test('waterfall unclassified remainder', () => {
  const r = waterfallReconcile(
    [
      { name: 'prefill', ns: 3.21e9 },
      { name: 'decode', ns: 9.04e9 },
      { name: 'lookup', ns: 8e6 },
      { name: 'tok', ns: 23e6 },
      { name: 'ttft', ns: 61e6 },
    ],
    12.34e9,
  );
  assert.ok(Math.abs(r.measuredNs - 12.342e9) < 1e6);
  assert.equal(r.unclassifiedNs, 0);
  const short = waterfallReconcile([{ name: 'decode', ns: 12.13e9 }], 12.34e9);
  assert.ok(Math.abs(short.unclassifiedNs - 0.21e9) < 1e6);
});

test('percentiles suppressed below sample thresholds', () => {
  const nine = decodeStats([1, 2, 3, 4, 5, 6, 7, 8, 9]);
  assert.equal(nine.n, 9);
  assert.equal(nine.median, undefined);
  assert.equal(nine.p95, undefined);
  assert.equal(nine.min, 1);
  assert.equal(nine.max, 9);
  const ten = decodeStats([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
  assert.equal(ten.median, 5.5);
  assert.equal(ten.p95, undefined);
  const twenty = decodeStats(Array.from({ length: 20 }, (_, i) => i + 1));
  assert.ok(twenty.p95 != null);
  assert.ok(twenty.p90 != null);
});

test('effective prompt throughput hidden under 5% gap', () => {
  assert.equal(showEffectivePromptThroughput(151, 150), false);
  assert.equal(showEffectivePromptThroughput(151, 200), true);
  assert.equal(showEffectivePromptThroughput(undefined, 200), false);
});

test('context buckets adapt to capacity', () => {
  assert.equal(contextBucket(4000, 262144), '0–8K');
  assert.equal(contextBucket(90000, 262144), '64K–128K');
  assert.equal(contextBucket(200000, 131072), '128K+');
});

test('regression is relative to comparable baseline', () => {
  const d = comparableDecodeRegression(39.1, 44.8);
  assert.ok(d != null);
  assert.ok(Math.abs(d - -12.723) < 0.02);
});

test('swap delta is current minus session start', () => {
  assert.equal(swapDelta(880 * 1024 ** 2, 880 * 1024 ** 2), 0);
  assert.equal(swapDelta(1492 * 1024 ** 2, 880 * 1024 ** 2), 612 * 1024 ** 2);
});

test('availability: missing is not instrumented, zero is 0', () => {
  assert.equal(displayNumber(undefined, (v) => String(v)), 'Not instrumented');
  assert.equal(displayNumber(null, (v) => String(v)), 'Not instrumented');
  assert.equal(displayNumber(Number.NaN, (v) => String(v)), 'Not instrumented');
  assert.equal(displayNumber(0, (v) => String(v)), '0');
  assert.equal(displayNumber(0, (v) => `${v.toFixed(0)} MiB`), '0 MiB');
  assert.equal(displayNumber(13.4, (v) => v.toFixed(1)), '13.4');
  console.log('availability tests passed');
});

test('parseStats keeps legacy /stats and treats absent nests as missing', () => {
  const s = parseStats({
    uptime_s: 10,
    busy: false,
    queue_depth: 0,
    clients: 1,
    live_tokens: 0,
    ctx_size: 4096,
    requests: 0,
    queue_rejected: 0,
    queue_dropped_disconnected: 0,
    prefill_cancelled: 0,
    prompt_tokens: 0,
    cached_tokens: 0,
    generated_tokens: 0,
    last_prefill_tps: 0,
    last_decode_tps: 0,
    cache: {
      hits: 0,
      cold: 0,
      memory_token: 0,
      memory_text: 0,
      responses_visible: 0,
      responses_tool_output: 0,
      anthropic_tool_output: 0,
      thinking_visible: 0,
      tool_visible: 0,
      disk_text: 0,
    },
  });
  assert.equal(s.ctx_size, 4096);
  assert.equal(s.host, undefined);
  assert.equal(s.metal, undefined);
  assert.equal(s.last_request, undefined);
  const rich = parseStats({
    ...{
      uptime_s: 10,
      busy: false,
      queue_depth: 0,
      clients: 1,
      live_tokens: 0,
      ctx_size: 4096,
      requests: 0,
      queue_rejected: 0,
      queue_dropped_disconnected: 0,
      prefill_cancelled: 0,
      prompt_tokens: 0,
      cached_tokens: 0,
      generated_tokens: 0,
      last_prefill_tps: 0,
      last_decode_tps: 0,
      cache: s.cache,
    },
    metal: { allocated_bytes: 0, recommended_max_bytes: 100 },
    host: { unified_memory_bytes: 128 * 1024 ** 3, swap_bytes: 0 },
  });
  assert.equal(rich.metal?.allocated_bytes, 0);
  assert.equal(rich.host?.swap_bytes, 0);
});

test('parseStats and parseRequestList support new telemetry fields', () => {
  const s = parseStats({
    uptime_s: 10,
    busy: false,
    queue_depth: 0,
    clients: 1,
    live_tokens: 0,
    ctx_size: 4096,
    requests: 0,
    queue_rejected: 0,
    queue_dropped_disconnected: 0,
    prefill_cancelled: 0,
    prompt_tokens: 0,
    cached_tokens: 0,
    generated_tokens: 0,
    last_prefill_tps: 0,
    last_decode_tps: 0,
    cache: { hits: 0, cold: 0, memory_token: 0, memory_text: 0, responses_visible: 0, responses_tool_output: 0, anthropic_tool_output: 0, thinking_visible: 0, tool_visible: 0, disk_text: 0 },
    host: { gpu_util_pct: 73.5, gpu_cores: 40, thermal_state: 'nominal' },
    checkpoint: { saves: 5, restores: 1, last_blocking_s: 0.015, last_write_s: 0.05 },
  });
  assert.equal(s.host?.gpu_util_pct, 73.5);
  assert.equal(s.host?.gpu_cores, 40);
  assert.equal(s.checkpoint?.saves, 5);
  assert.equal(s.checkpoint?.last_blocking_s, 0.015);

  const reqs = parseRequestList({
    requests: [
      {
        id: 1,
        started_at_s: 100,
        completed_at_s: 102,
        logical_prompt_tokens: 50,
        first_token_s: 0.25,
        prefill_compute_s: 0.18,
        mtp_state: 'active',
      },
    ],
  });
  assert.equal(reqs.length, 1);
  assert.equal(reqs[0]?.firstTokenNs, 0.25 * 1e9);
  assert.equal(reqs[0]?.prefillComputeNs, 0.18 * 1e9);
  assert.equal(reqs[0]?.mtpState, 'active');
  console.log('metrics tests passed');
});
