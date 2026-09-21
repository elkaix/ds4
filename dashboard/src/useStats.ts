import { useEffect, useRef, useState } from "react";
import type { Stats } from "./types";

const POLL_MS = 1000;
export const WINDOW = 300; // samples kept per series (≈5 min at 1 s)
/** x-axis span for the per-request sparklines — matches the server's recent[] cap. */
export const RECENT_SPAN = 64;

export const PREFILL_LIVE_MIN = 80;
export const DECODE_LIVE_MIN = 8;
// ponytail: 15s/3s EMA on live_tokens Δ; /stats has no in-flight phase. Upgrade if server exposes one.
const PREFILL_ALPHA = 2 / 16;
const DECODE_ALPHA = 2 / 4;

export interface Sample {
  t: number;
  live: number;
  busy: boolean;
  /** last_* from /stats (completed request) */
  prefill: number;
  decode: number;
  /** live_tokens growth EMA */
  prefillEma: number;
  decodeEma: number;
}

/** Instantaneous rates derived from the cumulative `proc` counters. Null until the
 *  second poll (one sample cannot make a delta) or when the server omits `proc`. */
export interface Rates {
  cpuPct: number;
  diskReadBps: number;
  diskWriteBps: number;
  pageinsPerSec: number;
}

export interface StatsState {
  stats: Stats | null;
  error: string | null;
  failures: number;
  samples: Sample[];
  rates: Rates | null;
}

/** Polls /stats once a second; keeps a rolling 5-minute window of throughput
 *  samples plus per-second rates derived from the cumulative counters. */
export function useStats(): StatsState {
  const [state, setState] = useState<StatsState>({ stats: null, error: null, failures: 0, samples: [], rates: null });
  // Deltas are computed *outside* the setState updater: StrictMode double-invokes
  // updaters in dev, which would otherwise diff a sample against itself.
  const prev = useRef<{ stats: Stats; t: number } | null>(null);
  const emaRef = useRef({ prefill: 0, decode: 0 });
  // live_tokens only moves on prefill chunks (~2048) / decode batches. Rate
  // must use time since the last change, not since the last 1s poll.
  const lastLive = useRef<{ t: number; live: number } | null>(null);

  useEffect(() => {
    let timer: number | undefined;
    let cancelled = false;

    const tick = async () => {
      try {
        const r = await fetch("/stats", { cache: "no-store" });
        if (!r.ok) throw new Error(`HTTP ${String(r.status)}`);
        const s = (await r.json()) as Stats;
        if (cancelled) return;

        const now = Date.now();
        const p = prev.current;
        let rates: Rates | null = null;
        if (p && s.proc && p.stats.proc) {
          const dtNs = (now - p.t) * 1e6;
          const a = p.stats.proc, b = s.proc;
          // Counters reset if the server restarted under us — drop that sample.
          const cpuNs = (b.cpu_user_ns - a.cpu_user_ns) + (b.cpu_sys_ns - a.cpu_sys_ns);
          if (dtNs > 0 && cpuNs >= 0 && b.disk_read_bytes >= a.disk_read_bytes && b.pageins >= a.pageins) {
            const dtSec = dtNs / 1e9;
            rates = {
              cpuPct: (cpuNs / dtNs) * 100,
              diskReadBps: (b.disk_read_bytes - a.disk_read_bytes) / dtSec,
              diskWriteBps: Math.max(0, b.disk_write_bytes - a.disk_write_bytes) / dtSec,
              pageinsPerSec: (b.pageins - a.pageins) / dtSec,
            };
          }
        }
        let inst = 0;
        if (p) {
          const dLive = s.live_tokens - p.stats.live_tokens;
          if (dLive < 0) {
            emaRef.current = { prefill: 0, decode: 0 };
            lastLive.current = { t: now, live: s.live_tokens };
          } else if (dLive > 0) {
            const from = lastLive.current ?? p;
            const dt = (now - from.t) / 1000;
            if (dt > 0.2) inst = dLive / dt;
            lastLive.current = { t: now, live: s.live_tokens };
          }
        } else {
          lastLive.current = { t: now, live: s.live_tokens };
        }
        let { prefill: prefillEma, decode: decodeEma } = emaRef.current;
        if (inst >= PREFILL_LIVE_MIN) {
          prefillEma = PREFILL_ALPHA * inst + (1 - PREFILL_ALPHA) * prefillEma;
          decodeEma = (1 - DECODE_ALPHA) * decodeEma;
        } else if (s.busy && inst >= DECODE_LIVE_MIN) {
          decodeEma = DECODE_ALPHA * inst + (1 - DECODE_ALPHA) * decodeEma;
          prefillEma = (1 - PREFILL_ALPHA) * prefillEma;
        } else if (!s.busy) {
          prefillEma = (1 - PREFILL_ALPHA) * prefillEma;
          decodeEma = (1 - DECODE_ALPHA) * decodeEma;
        }
        // busy + no live_tokens Δ this poll: hold (between prefill chunks)
        emaRef.current = { prefill: prefillEma, decode: decodeEma };
        const sample: Sample = {
          t: now,
          live: s.live_tokens,
          busy: s.busy,
          prefill: s.last_prefill_tps,
          decode: s.last_decode_tps,
          prefillEma,
          decodeEma,
        };
        prev.current = { stats: s, t: now };

        setState((st) => ({
          stats: s,
          error: null,
          failures: 0,
          samples: [...st.samples, sample].slice(-WINDOW),
          rates: rates ?? st.rates,
        }));
      } catch (e) {
        if (cancelled) return;
        setState((st) => ({ ...st, error: e instanceof Error ? e.message : String(e), failures: st.failures + 1 }));
      } finally {
        if (!cancelled) timer = window.setTimeout(() => { void tick(); }, POLL_MS);
      }
    };

    void tick();
    return () => { cancelled = true; if (timer) clearTimeout(timer); };
  }, []);

  return state;
}
