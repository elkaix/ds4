import { CACHE_PATHS, type CacheStats } from '../api/stats';
import { fmtInt, fmtPct } from '../lib/format';

/** Per-path cache hit counts as a table with a share bar — sortable by count, cold listed last. */
export function CachePaths({ cache }: { cache: CacheStats }) {
  const total = cache.hits + cache.cold;
  const rows = CACHE_PATHS.map((p) => ({ label: p.label, count: cache[p.key], cold: false }))
    .sort((a, b) => b.count - a.count)
    .concat([{ label: 'Cold (no cache)', count: cache.cold, cold: true }]);
  const max = Math.max(1, ...rows.map((r) => r.count));

  return (
    <table className="paths">
      <thead>
        <tr>
          <th scope="col">Path</th>
          <th scope="col" className="num">Requests</th>
          <th scope="col" className="num">Share</th>
          <th scope="col" className="bar-col"><span className="sr-only">Relative volume</span></th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r.label} className={r.cold ? 'paths__cold' : undefined}>
            <th scope="row">{r.label}</th>
            <td className="num">{fmtInt(r.count)}</td>
            <td className="num tone-muted">{total ? fmtPct((r.count / total) * 100, 0) : '—'}</td>
            <td className="bar-col">
              <div className="paths__bar" aria-hidden="true">
                <i style={{ width: `${String((r.count / max) * 100)}%` }} />
              </div>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
