import type { Health } from '../lib/health.ts';
import { displayNumber } from '../lib/availability.ts';
import { fmtBytes, fmtTokens } from '../lib/format.ts';

interface Props {
  health: Health;
  busy: boolean;
  decodeTokS: number | undefined;
  liveTokens: number | undefined;
  ctxSize: number | undefined;
  metalAllocated: number | undefined;
  metalLimit: number | undefined;
  metalHeadroom: number | undefined;
  swapDelta: number | undefined;
  pressure: string | undefined;
  thermal: string | undefined;
  queue: number | undefined;
}

export function HealthStrip({
  health, busy, decodeTokS, liveTokens, ctxSize,
  metalAllocated, metalLimit, metalHeadroom,
  swapDelta, pressure, thermal, queue,
}: Props) {
  const cls = health.level === 'critical' ? 'crit' : health.level === 'warning' ? 'warn' : busy ? 'busy' : 'ok';
  const label = busy && health.level === 'healthy' ? 'BUSY' : health.label;
  const ctx = liveTokens != null && ctxSize != null
    ? `${fmtTokens(liveTokens)} / ${fmtTokens(ctxSize)}`
    : displayNumber(undefined, String);
  const metal = metalAllocated != null && metalLimit != null
    ? `${fmtBytes(metalAllocated)} / ${fmtBytes(metalLimit)}`
    : displayNumber(metalAllocated, (n) => fmtBytes(n));
  const head = metalHeadroom != null ? fmtBytes(metalHeadroom) : 'Not instrumented';
  const swap = swapDelta != null ? `${swapDelta >= 0 ? '+' : ''}${fmtBytes(swapDelta)}` : 'Not instrumented';
  return (
    <div className={`strip strip--${cls}`} role="status" aria-live="polite">
      <b><i className="strip__dot" aria-hidden="true" />{label}</b>
      <span><span className="strip__label">Decode</span><strong>{displayNumber(decodeTokS, (n) => n.toFixed(1))} tok/s</strong></span>
      <span><span className="strip__label">Context</span><strong>{ctx}</strong></span>
      <span><span className="strip__label">Metal</span><strong>{metal}</strong></span>
      <span><span className="strip__label">Headroom</span><strong>{head}</strong></span>
      <span><span className="strip__label">Swap Δ</span><strong>{swap}</strong></span>
      <span><span className="strip__label">Pressure</span><strong>{pressure ?? 'Not instrumented'}</strong></span>
      <span><span className="strip__label">Thermal</span><strong>{thermal ?? 'Not instrumented'}</strong></span>
      <span><span className="strip__label">Queue</span><strong>{displayNumber(queue, String)}</strong></span>
    </div>
  );
}
