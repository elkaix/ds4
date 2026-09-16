/** Overridable warning thresholds. Query params win; nothing is machine-specific. */

export interface Thresholds {
  metalHeadroomWarnGiB: number;
  metalUtilizationWarnPct: number;
  swapGrowthWarnMiB: number;
  decodeRegressionWarnPct: number;
  queueWarnDepth: number;
  queueWarnDurationS: number;
  checkpointBlockWarnMs: number;
  ctxWarnPct: number;
  ctxCritPct: number;
  staleWarnS: number;
}

const defaults: Thresholds = {
  metalHeadroomWarnGiB: 8,
  metalUtilizationWarnPct: 90,
  swapGrowthWarnMiB: 256,
  decodeRegressionWarnPct: 15,
  queueWarnDepth: 1,
  queueWarnDurationS: 2,
  checkpointBlockWarnMs: 50,
  ctxWarnPct: 85,
  ctxCritPct: 95,
  staleWarnS: 5,
};

function qnum(p: URLSearchParams, key: string, fallback: number): number {
  const v = Number(p.get(key));
  return Number.isFinite(v) && v > 0 ? v : fallback;
}

export function resolveThresholds(): Thresholds {
  const p = new URLSearchParams(location.search);
  return {
    metalHeadroomWarnGiB: qnum(p, 'headroom', defaults.metalHeadroomWarnGiB),
    metalUtilizationWarnPct: qnum(p, 'metalpct', defaults.metalUtilizationWarnPct),
    swapGrowthWarnMiB: qnum(p, 'swap', defaults.swapGrowthWarnMiB),
    decodeRegressionWarnPct: qnum(p, 'regress', defaults.decodeRegressionWarnPct),
    queueWarnDepth: qnum(p, 'queue', defaults.queueWarnDepth),
    queueWarnDurationS: qnum(p, 'queuewait', defaults.queueWarnDurationS),
    checkpointBlockWarnMs: qnum(p, 'ckpt', defaults.checkpointBlockWarnMs),
    ctxWarnPct: qnum(p, 'ctxwarn', defaults.ctxWarnPct),
    ctxCritPct: qnum(p, 'ctxcrit', defaults.ctxCritPct),
    staleWarnS: qnum(p, 'stale', defaults.staleWarnS),
  };
}
