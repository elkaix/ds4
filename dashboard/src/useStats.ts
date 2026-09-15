import { useEffect, useRef, useState } from "react";
import type { Stats } from "./types";

const POLL_MS = 1000;
export const WINDOW = 300; // samples kept per series (≈5 min at 1 s)
/** x-axis span for the per-request sparklines — matches the server's recent[] cap. */
export const RECENT_SPAN = 64;

export interface Sample {
  t: number;
  /** last_prefill_tps / last_decode_tps as reported at this poll */
  prefill: number;
  decode: number;
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

  useEffect(() => {
    let timer: number | undefined;
    let cancelled = false;

    const tick = async () => {
      try {
        const r = await fetch("/stats", { cache: "no-store" });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
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
        const sample: Sample = {
          t: now,
          prefill: s.last_prefill_tps ?? 0,
          decode: s.last_decode_tps ?? 0,
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
        if (!cancelled) timer = window.setTimeout(tick, POLL_MS);
      }
    };

    tick();
    return () => { cancelled = true; if (timer) clearTimeout(timer); };
  }, []);

  return state;
}
