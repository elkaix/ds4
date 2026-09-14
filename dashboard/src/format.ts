/** Every formatter below accepts null/undefined/NaN and renders an em dash, so a
 *  missing field on an older server degrades to "—" instead of "0.0" or a crash. */
export const DASH = "—";

export type Maybe = number | null | undefined;

const ok = (n: Maybe): n is number => typeof n === "number" && Number.isFinite(n);

/** Ratio that yields null (→ "—") instead of NaN/Infinity when the divisor is empty. */
export function safeDiv(a: Maybe, b: Maybe): number | null {
  if (!ok(a) || !ok(b) || b <= 0) return null;
  const v = a / b;
  return Number.isFinite(v) ? v : null;
}

/** tokens over a nanosecond duration → tokens per second. */
export const rate = (tokens: Maybe, ns: Maybe): number | null => {
  const r = safeDiv(tokens, ns);
  return r === null ? null : r * 1e9;
};

/** 63938 → "63.9K"; used by the health strip where space is tight. */
export const compact = (n: Maybe): string => {
  if (!ok(n)) return DASH;
  const a = Math.abs(n);
  if (a >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (a >= 1000) return `${(n / 1000).toFixed(a >= 100_000 ? 0 : 1)}K`;
  return `${Math.round(n)}`;
};

export const num = (n: Maybe): string => (ok(n) ? Math.round(n).toLocaleString("en-US") : DASH);

export const tps = (n: Maybe): string => (ok(n) ? `${n.toFixed(n < 100 ? 1 : 0)} tok/s` : DASH);

export const pct = (part: Maybe, whole: Maybe): number => {
  const r = safeDiv(part, whole);
  return r === null ? 0 : r * 100;
};

/** Percentage of a ratio already in 0..1, or of a part/whole pair. */
export const pctStr = (part: Maybe, whole: Maybe, digits = 1): string => {
  const r = safeDiv(part, whole);
  return r === null ? DASH : `${(r * 100).toFixed(digits)}%`;
};

export const ratioStr = (r: Maybe, digits = 1): string => (ok(r) ? `${(r * 100).toFixed(digits)}%` : DASH);

/** Signed percentage, for "net vs plain". */
export const signedPct = (r: Maybe, digits = 1): string =>
  ok(r) ? `${r >= 0 ? "+" : ""}${(r * 100).toFixed(digits)}%` : DASH;

/** Nanoseconds → an adaptively scaled duration. */
export const ms = (ns: Maybe): string => {
  if (!ok(ns)) return DASH;
  const m = ns / 1e6;
  if (m >= 1000) return `${(m / 1000).toFixed(2)} s`;
  if (m >= 100) return `${m.toFixed(0)} ms`;
  if (m >= 1) return `${m.toFixed(1)} ms`;
  return `${m.toFixed(2)} ms`;
};

/** Nanoseconds → plain milliseconds (no unit switching), for tables. */
export const msNum = (ns: Maybe): string => (ok(ns) ? (ns / 1e6).toFixed(ns / 1e6 >= 100 ? 0 : 1) : DASH);

/** Nanoseconds → seconds. */
export const sec = (ns: Maybe, digits = 2): string => (ok(ns) ? `${(ns / 1e9).toFixed(digits)} s` : DASH);

/** Megabytes → GiB (the server reports MiB-scaled megabytes). */
export const gib = (mb: Maybe, digits = 2): string => (ok(mb) ? `${(mb / 1024).toFixed(digits)} GiB` : DASH);

/** Signed GiB delta. */
export const gibDelta = (mb: Maybe): string => {
  if (!ok(mb)) return DASH;
  const g = mb / 1024;
  if (Math.abs(g) < 0.05) return "+0";
  return `${g >= 0 ? "+" : ""}${g.toFixed(2)} GiB`;
};

export const mb = (bytes: Maybe, digits = 1): string => (ok(bytes) ? `${(bytes / 1048576).toFixed(digits)} MB` : DASH);

export const mbps = (bytesPerSec: Maybe, digits = 2): string =>
  ok(bytesPerSec) ? `${(bytesPerSec / 1048576).toFixed(digits)} MB/s` : DASH;

export const fixed = (n: Maybe, digits = 1): string => (ok(n) ? n.toFixed(digits) : DASH);

/** Legacy helper kept for callers that want MB/GB rather than GiB. */
export const gb = (m: Maybe): string =>
  !ok(m) ? DASH : m >= 1024 ? `${(m / 1024).toFixed(1)} GB` : `${m.toFixed(0)} MB`;

export function duration(s: Maybe): string {
  if (!ok(s)) return DASH;
  const t = Math.max(0, Math.floor(s));
  const d = Math.floor(t / 86400), h = Math.floor((t % 86400) / 3600), m = Math.floor((t % 3600) / 60);
  if (d) return `${d}d ${h}h`;
  if (h) return `${h}h ${m}m`;
  return `${m}m ${t % 60}s`;
}

/** Nearest-rank percentile over an unsorted array; null when empty. */
export function percentile(values: number[], p: number): number | null {
  const a = values.filter(ok).sort((x, y) => x - y);
  if (!a.length) return null;
  const i = Math.min(a.length - 1, Math.max(0, Math.round((p / 100) * (a.length - 1))));
  return a[i];
}
