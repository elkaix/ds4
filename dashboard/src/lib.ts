import { DASH, rate } from "./format";
import type { Mtp, MtpCounters, RecentRequest, Stats, Totals } from "./types";
import { WINDOW } from "./useStats";

/* ── Session-baseline machinery: "Session" period subtracts a snapshot taken
   at first successful poll (or the last Clear press) from cumulative stats. ── */

export interface Baseline {
  requests: number; prefill_cancelled: number; prompt_tokens: number;
  cached_tokens: number; generated_tokens: number; hits: number; cold: number;
  totals: Totals | null;
  mtp: MtpCounters | null;
  swap_mb: number | null;
}

export const ZERO_TOTALS: Totals = {
  prefill_fresh_tokens: 0, prefill_compute_ns: 0, prompt_total_ns: 0, decode_tokens: 0, decode_ns: 0,
  first_token_ns: 0, checkpoint_saves: 0, checkpoint_save_ns: 0, checkpoint_save_bytes: 0,
  checkpoint_restores: 0, checkpoint_restore_ns: 0,
  cache_recoveries: 0, replayed_tokens: 0, replay_ns: 0,
};

const ZERO_MTP: MtpCounters = {
  cycles: 0, accepted: 0, committed: 0, rows_cycles: 0, batch_cycles: 0,
  setup_ns: 0, verify_ns: 0, rollback_ns: 0, draft_ns: 0, total_ns: 0,
};
export { ZERO_MTP };

export const ZERO: Baseline = {
  requests: 0, prefill_cancelled: 0, prompt_tokens: 0, cached_tokens: 0, generated_tokens: 0,
  hits: 0, cold: 0, totals: ZERO_TOTALS, mtp: ZERO_MTP, swap_mb: null,
};

export const mtpCounters = (m: Mtp): MtpCounters => ({
  cycles: m.cycles, accepted: m.accepted, committed: m.committed, rows_cycles: m.rows_cycles,
  batch_cycles: m.batch_cycles, setup_ns: m.setup_ns, verify_ns: m.verify_ns,
  rollback_ns: m.rollback_ns, draft_ns: m.draft_ns, total_ns: m.total_ns,
});

export const snapshot = (s: Stats): Baseline => ({
  requests: s.requests, prefill_cancelled: s.prefill_cancelled, prompt_tokens: s.prompt_tokens,
  cached_tokens: s.cached_tokens, generated_tokens: s.generated_tokens,
  hits: s.cache.hits, cold: s.cache.cold,
  totals: s.totals ? { ...s.totals } : null,
  mtp: s.mtp ? mtpCounters(s.mtp) : null,
  swap_mb: s.mem ? s.mem.swap_used_mb : null,
});

export function delta<T extends object>(cur: T | undefined, base: T | null | undefined): T | null {
  if (!cur) return null;
  if (!base) return cur;
  const a = cur as Record<string, number>;
  const b = base as Record<string, number>;
  const out: Record<string, number> = {};
  for (const k of Object.keys(a)) out[k] = Math.max(0, (a[k] ?? 0) - (b[k] ?? 0));
  return out as T;
}

const EMA_ALPHA = 2 / (WINDOW + 1);
export function ema(values: number[]): number | null {
  const nz = values.filter((v) => Number.isFinite(v) && v > 0);
  if (!nz.length) return null;
  return nz.reduce((acc, v, i) => (i === 0 ? v : acc + EMA_ALPHA * (v - acc)), 0);
}

/* ── Recent-request helpers ── */

export const decodeTps = (r: RecentRequest) => rate(r.decode_tokens, r.decode_ns);
export const computeTps = (r: RecentRequest) => rate(r.fresh_tokens, r.prefill_ns);

const SOURCE_SHORT: Record<string, string> = {
  "memory-text": "mem-text", "memory-token": "mem-token", "disk-text": "disk-text", "disk-token": "disk-token", none: "cold",
};
export const shortSource = (s: string): string => SOURCE_SHORT[s] ?? s;

export function clockOf(stats: Stats | null, r: RecentRequest): string {
  if (!stats) return DASH;
  const t = Date.now() - Math.max(0, stats.uptime_s - r.at_s) * 1000;
  return new Date(t).toLocaleTimeString();
}
