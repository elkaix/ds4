import type { Stats } from '../api/stats.ts';
import { bytesToGiB, bytesToMiB, metalHeadroom, metalUtilizationPct } from './metrics.ts';
import type { Thresholds } from './config.ts';

export type HealthLevel = 'healthy' | 'warning' | 'critical';

export interface Alert {
  level: HealthLevel;
  title: string;
  detail: string;
  target?: string;
}

export interface Health {
  level: HealthLevel;
  label: string;
  alerts: Alert[];
}

export function evaluateHealth(
  stats: Stats | null,
  opts: {
    connected: boolean;
    staleS: number | null;
    thresholds: Thresholds;
    decodeRegressionPct?: number;
  },
): Health {
  const alerts: Alert[] = [];
  if (!opts.connected) {
    return {
      level: 'critical',
      label: 'UNAVAILABLE',
      alerts: [{ level: 'critical', title: 'ds4-server unavailable', detail: 'Cannot reach /stats' }],
    };
  }
  if (opts.staleS != null && opts.staleS > opts.thresholds.staleWarnS) {
    alerts.push({
      level: 'warning',
      title: 'Telemetry stale',
      detail: `last update ${opts.staleS.toFixed(1)} s ago`,
    });
  }
  if (!stats) {
    return { level: alerts.length ? 'warning' : 'healthy', label: alerts.length ? 'WARNING' : 'STARTING', alerts };
  }

  if (stats.queue_rejected > 0) {
    alerts.push({
      level: 'warning',
      title: 'Queue rejected requests',
      detail: `${String(stats.queue_rejected)} since process start`,
      target: 'queue',
    });
  }
  if (stats.queue_depth >= opts.thresholds.queueWarnDepth) {
    alerts.push({
      level: 'warning',
      title: 'Requests queued',
      detail: `depth ${String(stats.queue_depth)}`,
      target: 'queue',
    });
  }

  const ctxPct = stats.ctx_size > 0 ? (stats.live_tokens / stats.ctx_size) * 100 : 0;
  if (ctxPct >= opts.thresholds.ctxCritPct) {
    alerts.push({
      level: 'critical',
      title: 'Context near capacity',
      detail: `${ctxPct.toFixed(1)}% of runtime window`,
      target: 'context',
    });
  } else if (ctxPct >= opts.thresholds.ctxWarnPct) {
    alerts.push({
      level: 'warning',
      title: 'Context high',
      detail: `${ctxPct.toFixed(1)}% of runtime window`,
      target: 'context',
    });
  }

  const metal = stats.metal;
  if (metal?.allocated_bytes != null && metal.recommended_max_bytes != null) {
    const head = metalHeadroom(metal.recommended_max_bytes, metal.allocated_bytes);
    const util = metalUtilizationPct(metal.allocated_bytes, metal.recommended_max_bytes);
    if (head != null && bytesToGiB(head) < opts.thresholds.metalHeadroomWarnGiB) {
      alerts.push({
        level: head <= 0 ? 'critical' : 'warning',
        title: 'Low Metal headroom',
        detail: `${bytesToGiB(Math.max(0, head)).toFixed(1)} GiB remaining`,
        target: 'metal',
      });
    }
    if (util != null && util >= opts.thresholds.metalUtilizationWarnPct) {
      alerts.push({
        level: 'warning',
        title: 'Metal working set high',
        detail: `${util.toFixed(1)}% of recommended max`,
        target: 'metal',
      });
    }
  }

  const swapDelta = stats.host?.swap_delta_bytes;
  if (swapDelta != null && bytesToMiB(swapDelta) >= opts.thresholds.swapGrowthWarnMiB) {
    alerts.push({
      level: 'warning',
      title: 'Swap growth detected',
      detail: `+${bytesToMiB(swapDelta).toFixed(0)} MiB this session`,
      target: 'host',
    });
  }
  const pressure = stats.host?.memory_pressure;
  if (pressure === 'critical') {
    alerts.push({
      level: 'critical',
      title: 'Memory pressure critical',
      detail: pressure,
      target: 'host',
    });
  } else if (pressure === 'warn' || pressure === 'warning') {
    alerts.push({
      level: 'warning',
      title: 'Memory pressure elevated',
      detail: pressure,
      target: 'host',
    });
  }

  const thermal = stats.host?.thermal_state;
  if (thermal === 'critical') {
    alerts.push({
      level: 'critical',
      title: 'Thermal critical',
      detail: thermal,
      target: 'host',
    });
  } else if (thermal === 'serious' || thermal === 'fair') {
    alerts.push({
      level: 'warning',
      title: 'Thermal pressure elevated',
      detail: 'Performance may be constrained',
      target: 'host',
    });
  }

  if (opts.decodeRegressionPct != null && opts.decodeRegressionPct <= -opts.thresholds.decodeRegressionWarnPct) {
    alerts.push({
      level: 'warning',
      title: 'Decode regression',
      detail: `${opts.decodeRegressionPct.toFixed(1)}% vs comparable requests`,
      target: 'decode',
    });
  }

  const ckpt = stats.checkpoint?.last_blocking_s;
  if (ckpt != null && ckpt * 1000 >= opts.thresholds.checkpointBlockWarnMs) {
    alerts.push({
      level: 'warning',
      title: 'Checkpoint affecting latency',
      detail: `${(ckpt * 1000).toFixed(0)} ms request-path blocking`,
      target: 'checkpoint',
    });
  }

  const level: HealthLevel = alerts.some((a) => a.level === 'critical')
    ? 'critical'
    : alerts.some((a) => a.level === 'warning')
      ? 'warning'
      : 'healthy';
  const label = level === 'healthy' ? 'HEALTHY' : level === 'warning' ? 'WARNING' : 'CRITICAL';
  return { level, label, alerts };
}
