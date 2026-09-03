import { useId, useState, type PointerEvent } from 'react';
import { fmtAgo, fmtFixed } from '../lib/format';

interface Props {
  /** Chronological values; the last one is "now". */
  values: readonly number[];
  /** Timestamps parallel to `values`, for the hover tooltip. */
  times: readonly number[];
  unit: string;
  digits?: number;
  /** CSS color of the line; area fill derives from it. */
  color: string;
  height?: number;
  title: string;
}

const W = 100;

/**
 * Single-series area sparkline with a crosshair tooltip. The y-domain hugs the
 * data (padded, never below zero) so a 5-minute throughput trend keeps its
 * shape; absolute values live in the tooltip and the facts row. `vector-effect`
 * keeps strokes at 1.5px when the viewBox is stretched to the tile width.
 */
export function Sparkline({ values, times, unit, digits = 1, color, height = 56, title }: Props) {
  const [hover, setHover] = useState<number | null>(null);
  const gradId = useId();
  const n = values.length;
  if (n < 2) {
    return (
      <div className="spark spark--empty" style={{ height }} aria-hidden="true">
        collecting samples…
      </div>
    );
  }
  const dataMax = Math.max(...values);
  const dataMin = Math.min(...values);
  const range = dataMax - dataMin;
  const lo = range > 0 ? Math.max(0, dataMin - range * 0.6) : 0;
  const hi = range > 0 ? dataMax + range * 0.2 : Math.max(dataMax * 1.1, 1e-9);
  const x = (i: number) => (i / (n - 1)) * W;
  const y = (v: number) => 2 + (1 - (v - lo) / (hi - lo)) * (height - 4);
  const pts = values.map((v, i) => `${x(i).toFixed(2)},${y(v).toFixed(2)}`);
  const line = pts.join(' ');
  const area = `0,${String(height)} ${line} ${String(W)},${String(height)}`;

  const onMove = (e: PointerEvent<SVGSVGElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const i = Math.round(((e.clientX - r.left) / r.width) * (n - 1));
    setHover(Math.max(0, Math.min(n - 1, i)));
  };

  const hv = hover == null ? undefined : values[hover];
  const ht = hover == null ? undefined : times[hover];
  const newest = times[n - 1] ?? 0;

  return (
    <div className="spark" style={{ height }}>
      <svg
        viewBox={`0 0 ${String(W)} ${String(height)}`}
        preserveAspectRatio="none"
        role="img"
        aria-label={`${title}, last ${String(n)} samples, range ${fmtFixed(dataMin, digits)} to ${fmtFixed(dataMax, digits)} ${unit}`}
        onPointerMove={onMove}
        onPointerLeave={() => { setHover(null); }}
      >
        <defs>
          <linearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity="0.22" />
            <stop offset="100%" stopColor={color} stopOpacity="0.02" />
          </linearGradient>
        </defs>
        <polygon points={area} fill={`url(#${gradId})`} />
        <polyline points={line} fill="none" stroke={color} strokeWidth="1.5" strokeLinejoin="round" vectorEffect="non-scaling-stroke" />
        {hover != null && hv != null && (
          <>
            <line x1={x(hover)} x2={x(hover)} y1="0" y2={height} className="spark__cross" vectorEffect="non-scaling-stroke" />
            <circle cx={x(hover)} cy={y(hv)} r="3" fill={color} className="spark__dot" />
          </>
        )}
      </svg>
      {hover != null && hv != null && ht != null && (
        <div className="spark__tip" style={{ left: `${String((hover / (n - 1)) * 100)}%` }} role="status">
          <b>{fmtFixed(hv, digits)}</b> {unit}
          <span>{fmtAgo(newest - ht)}</span>
        </div>
      )}
    </div>
  );
}
