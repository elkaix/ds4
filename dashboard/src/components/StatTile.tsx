import type { ReactNode } from 'react';

interface Props {
  label: string;
  value: string;
  unit?: string;
  /** Small secondary facts rendered under the value, e.g. peak / average. */
  facts?: readonly { k: string; v: string }[];
  children?: ReactNode;
  className?: string;
}

/** A stat tile: label, hero value in proportional figures, optional trend. */
export function StatTile({ label, value, unit, facts, children, className }: Props) {
  return (
    <section className={`tile ${className ?? ''}`} aria-label={label}>
      <h2 className="tile__label">{label}</h2>
      <p className="tile__value">
        {value}
        {unit && <span className="tile__unit">{unit}</span>}
      </p>
      {children}
      {facts && facts.length > 0 && (
        <dl className="facts">
          {facts.map((f) => (
            <div key={f.k}>
              <dt>{f.k}</dt>
              <dd>{f.v}</dd>
            </div>
          ))}
        </dl>
      )}
    </section>
  );
}
