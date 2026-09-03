export type Severity = 'ok' | 'warn' | 'critical';

interface Props {
  /** 0..100 */
  pct: number;
  severity: Severity;
  label: string;
}

/** A ratio against a limit. Fill carries severity; the track is a lighter step of the same ramp. */
export function Meter({ pct, severity, label }: Props) {
  const clamped = Math.max(0, Math.min(100, pct));
  return (
    <div
      className={`meter meter--${severity}`}
      role="meter"
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={Math.round(clamped)}
    >
      <div className="meter__fill" style={{ width: `${String(clamped)}%` }} />
    </div>
  );
}
