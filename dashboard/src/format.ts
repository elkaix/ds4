export const num = (n: number): string => Math.round(n).toLocaleString("en-US");
export const tps = (n: number): string => `${(n || 0).toFixed(1)} tok/s`;
export const pct = (part: number, whole: number): number => (whole > 0 ? (100 * part) / whole : 0);
export const gb = (mb: number): string =>
  mb >= 1024 ? `${(mb / 1024).toFixed(1)} GB` : `${mb.toFixed(0)} MB`;
export function duration(s: number): string {
  s = Math.max(0, Math.floor(s));
  const d = Math.floor(s / 86400), h = Math.floor((s % 86400) / 3600), m = Math.floor((s % 3600) / 60);
  if (d) return `${d}d ${h}h`;
  if (h) return `${h}h ${m}m`;
  return `${m}m ${s % 60}s`;
}
