import type { Stats } from '../api/stats';
import { fmtInt } from '../lib/format';

const ITEMS: readonly { key: keyof Stats & ('queue_rejected' | 'queue_dropped_disconnected' | 'prefill_cancelled'); label: string; hint: string }[] = [
  { key: 'queue_rejected', label: 'Queue rejected', hint: '429 — queue was full (--max-queue)' },
  { key: 'queue_dropped_disconnected', label: 'Dropped on disconnect', hint: 'client left while queued' },
  { key: 'prefill_cancelled', label: 'Prefill cancelled', hint: 'client left mid-prefill' },
];

/** Counters that should stay at zero. Non-zero rows get the critical status + icon, never color alone. */
export function Anomalies({ stats }: { stats: Stats }) {
  const any = ITEMS.some((i) => stats[i.key] > 0);
  return (
    <ul className="anom" aria-live="polite">
      {ITEMS.map((i) => {
        const v = stats[i.key];
        return (
          <li key={i.key} className={v > 0 ? 'anom__row anom__row--hit' : 'anom__row'}>
            <span className="anom__icon" aria-hidden="true">{v > 0 ? '⚠' : '✓'}</span>
            <span className="anom__label">
              {i.label}
              <small>{i.hint}</small>
            </span>
            <span className="anom__count">{fmtInt(v)}</span>
          </li>
        );
      })}
      {!any && <li className="anom__none tone-muted">No anomalies since start</li>}
    </ul>
  );
}
