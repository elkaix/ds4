const grouped = new Intl.NumberFormat('en-US');
const compact = new Intl.NumberFormat('en-US', { notation: 'compact', maximumFractionDigits: 1 });

export const fmtInt = (v: number | undefined): string => (v == null ? '—' : grouped.format(v));
export const fmtCompact = (v: number | undefined): string => (v == null ? '—' : compact.format(v));
export const fmtFixed = (v: number | undefined, digits = 1): string =>
  v == null || !Number.isFinite(v) ? '—' : v.toFixed(digits);
export const fmtPct = (v: number | undefined, digits = 1): string =>
  v == null || !Number.isFinite(v) ? '—' : `${v.toFixed(digits)}%`;

export function fmtDuration(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds));
  const d = Math.floor(s / 86400);
  const h = Math.floor((s % 86400) / 3600);
  const m = Math.floor((s % 3600) / 60);
  if (d) return `${String(d)}d ${String(h)}h`;
  if (h) return `${String(h)}h ${String(m)}m`;
  if (m) return `${String(m)}m ${String(s % 60)}s`;
  return `${String(s)}s`;
}

export function fmtAgo(ms: number): string {
  const s = Math.round(ms / 1000);
  return s < 60 ? `${String(s)}s ago` : `${String(Math.floor(s / 60))}m ${String(s % 60)}s ago`;
}
