import { useEffect, useRef, useState } from "react";
import type { Stats } from "./types";

const POLL_MS = 1000;
export const WINDOW = 300; // samples kept per series (≈5 min at 1 s)

export interface Sample { t: number; prefill: number; decode: number }
export interface RequestEvent {
  at: number; requests: number; prompt: number; cached: number; generated: number;
  prefillTps: number; decodeTps: number;
}

export interface StatsState {
  stats: Stats | null;
  error: string | null;
  failures: number;
  samples: Sample[];
  events: RequestEvent[];
}

/** Polls /stats once a second; keeps a rolling throughput window and a log of
 *  completed requests (derived from counter deltas between polls). */
export function useStats(): StatsState {
  const [state, setState] = useState<StatsState>({ stats: null, error: null, failures: 0, samples: [], events: [] });
  const prev = useRef<Stats | null>(null);

  useEffect(() => {
    let timer: number | undefined;
    let cancelled = false;
    const tick = async () => {
      try {
        const r = await fetch("/stats", { cache: "no-store" });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const s = (await r.json()) as Stats;
        if (cancelled) return;
        setState((st) => {
          const samples = [...st.samples, { t: Date.now(), prefill: s.last_prefill_tps, decode: s.last_decode_tps }].slice(-WINDOW);
          let events = st.events;
          const p = prev.current;
          if (p && s.requests > p.requests) {
            events = [{
              at: Date.now(), requests: s.requests - p.requests,
              prompt: s.prompt_tokens - p.prompt_tokens, cached: s.cached_tokens - p.cached_tokens,
              generated: s.generated_tokens - p.generated_tokens,
              prefillTps: s.last_prefill_tps, decodeTps: s.last_decode_tps,
            }, ...st.events].slice(0, 25);
          }
          prev.current = s;
          return { stats: s, error: null, failures: 0, samples, events };
        });
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
