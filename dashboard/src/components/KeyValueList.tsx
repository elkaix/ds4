import type { ReactNode } from 'react';

export interface KV {
  k: string;
  v: ReactNode;
  /** Optional tone for the value text. */
  tone?: 'ok' | 'warn' | 'critical' | 'muted' | undefined;
}

export function KeyValueList({ rows }: { rows: readonly KV[] }) {
  return (
    <dl className="kv">
      {rows.map((r) => (
        <div key={r.k} className="kv__row">
          <dt>{r.k}</dt>
          <dd className={r.tone ? `tone-${r.tone}` : undefined}>{r.v}</dd>
        </div>
      ))}
    </dl>
  );
}
