import { useState, type PointerEvent } from 'react';
import { fmtFixed } from '../lib/format.ts';

export interface Series {
  values: readonly number[];
  color: string;
}

interface Props {
  series: readonly Series[];
  labels: readonly string[];
  unit: string;
  digits?: number;
  title: string;
  height?: number;
}

/** Quiet line chart. No area fill. Needs ≥2 points. */
export function Chart({ series, labels, unit, digits = 1, title, height = 88 }: Props) {
  const [hover, setHover] = useState<number | null>(null);
  const n = Math.max(0, ...series.map((s) => s.values.length));
  if (n < 2) {
    return <div className="chart chart--empty">Need more samples</div>;
  }
  const all = series.flatMap((s) => [...s.values]);
  const dataMax = Math.max(...all);
  const dataMin = Math.min(...all);
  const range = dataMax - dataMin;
  const lo = range > 0 ? Math.max(0, dataMin - range * 0.08) : 0;
  const hi = range > 0 ? dataMax + range * 0.08 : Math.max(dataMax * 1.1, 1e-9);
  const W = 100;
  const x = (i: number) => (n === 1 ? 0 : (i / (n - 1)) * W);
  const y = (v: number) => 2 + (1 - (v - lo) / (hi - lo)) * (height - 4);
  const path = (values: readonly number[]) =>
    values.map((v, i) => `${i === 0 ? 'M' : 'L'}${x(i).toFixed(2)} ${y(v).toFixed(2)}`).join(' ');

  const onMove = (e: PointerEvent<SVGSVGElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const i = Math.round(((e.clientX - r.left) / r.width) * (n - 1));
    setHover(Math.max(0, Math.min(n - 1, i)));
  };

  return (
    <div className="chart" style={{ height, position: 'relative' }}>
      <svg
        viewBox={`0 0 ${String(W)} ${String(height)}`}
        preserveAspectRatio="none"
        role="img"
        aria-label={title}
        onPointerMove={onMove}
        onPointerLeave={() => { setHover(null); }}
      >
        <line x1="0" x2={W} y1={height - 1} y2={height - 1} className="chart__axis" />
        {series.map((s, si) => (
          <path key={si} d={path(s.values)} className="chart__line" stroke={s.color} />
        ))}
        {hover != null && (
          <line x1={x(hover)} x2={x(hover)} y1="0" y2={height} className="chart__axis" />
        )}
      </svg>
      {hover != null && (
        <div className="chart__tip" style={{ left: `${String((hover / (n - 1)) * 100)}%` }}>
          {series.map((s, si) => (
            <span key={si}>{fmtFixed(s.values[hover], digits)} {unit}</span>
          ))}
          {labels[hover] ? <span> · {labels[hover]}</span> : null}
        </div>
      )}
    </div>
  );
}
