/** Presentation-layer formulas. Server owns primitives; this file owns rates. */

export const NS_PER_S = 1e9;
export const NS_PER_MS = 1e6;
export const BYTES_PER_KIB = 1024;
export const BYTES_PER_MIB = 1024 ** 2;
export const BYTES_PER_GIB = 1024 ** 3;

export type TelemetryState = 'measured' | 'zero' | 'inactive' | 'missing';

export type Telemetry<T> =
  | { state: 'measured'; value: T }
  | { state: 'zero'; value: T }
  | { state: 'inactive' }
  | { state: 'missing' };

export function measured<T>(value: T | null | undefined, isZero: (v: T) => boolean): Telemetry<T> {
  if (value == null) return { state: 'missing' };
  if (isZero(value)) return { state: 'zero', value };
  return { state: 'measured', value };
}

export function measuredNumber(value: number | null | undefined): Telemetry<number> {
  if (value == null || !Number.isFinite(value)) return { state: 'missing' };
  if (value === 0) return { state: 'zero', value };
  return { state: 'measured', value };
}

export function decodeTokS(decodeTokens: number, decodeNs: number): number | undefined {
  if (!(decodeTokens > 0) || !(decodeNs > 0)) return undefined;
  return decodeTokens / (decodeNs / NS_PER_S);
}

export function msPerToken(decodeNs: number, decodeTokens: number): number | undefined {
  if (!(decodeTokens > 0) || !(decodeNs > 0)) return undefined;
  return decodeNs / NS_PER_MS / decodeTokens;
}

export function freshPrefillTokS(freshPromptTokens: number, prefillComputeNs: number): number | undefined {
  if (!(freshPromptTokens > 0) || !(prefillComputeNs > 0)) return undefined;
  return freshPromptTokens / (prefillComputeNs / NS_PER_S);
}

export function reusePct(cachedPromptTokens: number, logicalPromptTokens: number): number | undefined {
  if (!(logicalPromptTokens > 0)) return undefined;
  return (cachedPromptTokens / logicalPromptTokens) * 100;
}

export function requestCacheHitRate(hits: number, cold: number): number | undefined {
  const total = hits + cold;
  if (!(total > 0)) return undefined;
  return (hits / total) * 100;
}

export function metalHeadroom(recommendedMaxBytes: number, allocatedBytes: number): number | undefined {
  if (!(recommendedMaxBytes > 0) || allocatedBytes < 0) return undefined;
  return recommendedMaxBytes - allocatedBytes;
}

export function metalUtilizationPct(allocatedBytes: number, recommendedMaxBytes: number): number | undefined {
  if (!(recommendedMaxBytes > 0) || allocatedBytes < 0) return undefined;
  return (allocatedBytes / recommendedMaxBytes) * 100;
}

export function checkpointAsyncNs(writeNs: number, blockingNs: number): number | undefined {
  if (writeNs < 0 || blockingNs < 0) return undefined;
  return Math.max(0, writeNs - blockingNs);
}

export function freshTokens(logicalPromptTokens: number, cachedPromptTokens: number): number {
  return Math.max(0, logicalPromptTokens - cachedPromptTokens);
}

export function effectivePromptTokS(logicalPromptTokens: number, promptWallNs: number): number | undefined {
  if (!(logicalPromptTokens > 0) || !(promptWallNs > 0)) return undefined;
  return logicalPromptTokens / (promptWallNs / NS_PER_S);
}

/** Show effective throughput only when it differs from compute throughput by >= 5%. */
export function showEffectivePromptThroughput(
  computeTokS: number | undefined,
  effectiveTokS: number | undefined,
): boolean {
  if (computeTokS == null || effectiveTokS == null || computeTokS === 0) return false;
  return Math.abs(effectiveTokS - computeTokS) / computeTokS >= 0.05;
}

export interface MtpEconomics {
  acceptance: number;
  tokensPerCycle: number;
  cycleOverStep: number;
  onePlusA: number;
  profitable: boolean;
}

export function mtpEconomics(acceptedTokens: number, cycles: number, cycleNs: number, plainStepNs: number): MtpEconomics | undefined {
  if (!(cycles > 0) || !(plainStepNs > 0) || !(cycleNs > 0)) return undefined;
  const a = acceptedTokens / cycles;
  const cOverS = cycleNs / plainStepNs;
  const onePlusA = 1 + a;
  return {
    acceptance: a,
    tokensPerCycle: a + 1, /* draft-verify cycle yields 1 verified + accepted extras */
    cycleOverStep: cOverS,
    onePlusA,
    profitable: cOverS < onePlusA,
  };
}

/** Spec profitability uses C/S < 1+a where a is accepted extras per cycle. */
export function mtpTokensPerCycle(acceptedTokens: number, cycles: number): number | undefined {
  if (!(cycles > 0)) return undefined;
  return acceptedTokens / cycles;
}

export interface WaterfallStage { name: string; ns: number }

export function waterfallReconcile(
  stages: readonly WaterfallStage[],
  totalNs: number,
): { measuredNs: number; unclassifiedNs: number; totalNs: number } {
  const measuredNs = stages.reduce((s, st) => s + st.ns, 0);
  return {
    measuredNs,
    unclassifiedNs: Math.max(0, totalNs - measuredNs),
    totalNs,
  };
}

export function percentile(sortedAsc: readonly number[], p: number): number | undefined {
  if (sortedAsc.length === 0) return undefined;
  if (p <= 0) return sortedAsc[0];
  if (p >= 100) return sortedAsc[sortedAsc.length - 1];
  const idx = (p / 100) * (sortedAsc.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  const a = sortedAsc[lo];
  const b = sortedAsc[hi];
  if (a == null || b == null) return undefined;
  const t = idx - lo;
  return a + (b - a) * t;
}

export function median(values: readonly number[]): number | undefined {
  if (values.length === 0) return undefined;
  const s = [...values].sort((a, b) => a - b);
  return percentile(s, 50);
}

export function mean(values: readonly number[]): number | undefined {
  if (values.length === 0) return undefined;
  return values.reduce((a, b) => a + b, 0) / values.length;
}

export function ema(values: readonly number[], alpha = 0.3): number | undefined {
  const first = values[0];
  if (first == null) return undefined;
  let v = first;
  for (let i = 1; i < values.length; i++) {
    const next = values[i];
    if (next == null) continue;
    v = alpha * next + (1 - alpha) * v;
  }
  return v;
}

/** Percentiles only when n is large enough: p90/p95 need n>=20; median/distribution n>=10. */
export function decodeStats(msPerTok: readonly number[]): {
  n: number;
  min?: number;
  max?: number;
  median?: number;
  mean?: number;
  p90?: number;
  p95?: number;
} {
  const n = msPerTok.length;
  if (n === 0) return { n: 0 };
  const s = [...msPerTok].sort((a, b) => a - b);
  const out: ReturnType<typeof decodeStats> = { n };
  const lo = s[0];
  const hi = s[n - 1];
  const avg = mean(s);
  if (lo != null) out.min = lo;
  if (hi != null) out.max = hi;
  if (avg != null) out.mean = avg;
  if (n >= 10) {
    const med = percentile(s, 50);
    if (med != null) out.median = med;
  }
  if (n >= 20) {
    const p90 = percentile(s, 90);
    const p95 = percentile(s, 95);
    if (p90 != null) out.p90 = p90;
    if (p95 != null) out.p95 = p95;
  }
  return out;
}

export function contextBucket(tokens: number, capacity: number): string {
  const edges = [8_192, 32_768, 65_536, 131_072, 262_144];
  const scaled = capacity > 0
    ? edges.filter((e) => e <= capacity * 1.01)
    : edges;
  const bounds = [...scaled, Number.POSITIVE_INFINITY];
  let prev = 0;
  for (const b of bounds) {
    if (tokens < b) {
      const hi = Number.isFinite(b) ? formatK(b) : `${formatK(prev)}+`;
      const lo = formatK(prev);
      return Number.isFinite(b) ? `${lo}–${hi}` : hi;
    }
    prev = b;
  }
  return `${formatK(prev)}+`;
}

function formatK(n: number): string {
  if (n >= 1024) return `${(n / 1024).toFixed(0)}K`;
  return String(n);
}

export function bytesToGiB(bytes: number): number {
  return bytes / BYTES_PER_GIB;
}

export function bytesToMiB(bytes: number): number {
  return bytes / BYTES_PER_MIB;
}

export function nsToMs(ns: number): number {
  return ns / NS_PER_MS;
}

export function nsToS(ns: number): number {
  return ns / NS_PER_S;
}

export function secondsToNs(s: number): number {
  return s * NS_PER_S;
}

export function comparableDecodeRegression(
  observedTokS: number,
  baselineTokS: number,
): number | undefined {
  if (!(baselineTokS > 0)) return undefined;
  return ((observedTokS - baselineTokS) / baselineTokS) * 100;
}

export function swapDelta(currentBytes: number, sessionStartBytes: number): number {
  return currentBytes - sessionStartBytes;
}
