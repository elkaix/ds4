import { BYTES_PER_GIB, BYTES_PER_KIB, BYTES_PER_MIB } from './metrics.ts';
import { displayNumber } from './availability.ts';

const grouped = new Intl.NumberFormat('en-US');
const compact = new Intl.NumberFormat('en-US', { notation: 'compact', maximumFractionDigits: 1 });

export const fmtInt = (v: number | undefined): string =>
  displayNumber(v, (n) => grouped.format(n));
export const fmtCompact = (v: number | undefined): string =>
  displayNumber(v, (n) => compact.format(n));
export const fmtFixed = (v: number | undefined, digits = 1): string =>
  displayNumber(v, (n) => n.toFixed(digits));
export const fmtPct = (v: number | undefined, digits = 1): string =>
  displayNumber(v, (n) => `${n.toFixed(digits)}%`);

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

export function fmtBytes(bytes: number | undefined, digits = 1): string {
  return displayNumber(bytes, (n) => {
    const abs = Math.abs(n);
    if (abs >= BYTES_PER_GIB) return `${(n / BYTES_PER_GIB).toFixed(digits)} GiB`;
    if (abs >= BYTES_PER_MIB) return `${(n / BYTES_PER_MIB).toFixed(digits)} MiB`;
    if (abs >= BYTES_PER_KIB) return `${(n / BYTES_PER_KIB).toFixed(0)} KiB`;
    return `${n.toFixed(0)} B`;
  });
}

export function fmtTokens(n: number | undefined): string {
  return displayNumber(n, (v) => {
    const abs = Math.abs(v);
    if (abs >= 100_000) return `${(v / 1000).toFixed(1)}K`;
    if (abs >= 10_000) return `${(v / 1000).toFixed(1)}K`;
    return grouped.format(v);
  });
}

export function fmtTokS(n: number | undefined, digits = 1): string {
  return displayNumber(n, (v) => `${v.toFixed(digits)} tok/s`);
}

export function fmtMs(ms: number | undefined, digits = 0): string {
  return displayNumber(ms, (v) => {
    if (Math.abs(v) >= 1000) return `${(v / 1000).toFixed(2)} s`;
    return `${v.toFixed(digits)} ms`;
  });
}

export function fmtSeconds(s: number | undefined, digits = 2): string {
  return displayNumber(s, (v) => {
    if (Math.abs(v) < 1) return `${(v * 1000).toFixed(0)} ms`;
    return `${v.toFixed(digits)} s`;
  });
}

export function fmtClock(ms: number): string {
  const d = new Date(ms);
  const hh = String(d.getHours()).padStart(2, '0');
  const mm = String(d.getMinutes()).padStart(2, '0');
  const ss = String(d.getSeconds()).padStart(2, '0');
  return `${hh}:${mm}:${ss}`;
}
