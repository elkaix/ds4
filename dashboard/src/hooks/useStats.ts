import { useEffect, useRef, useState } from 'react';
import { fetchStats, type Stats } from '../api/stats';

export interface Sample {
  t: number; // Date.now() when received
  decode: number;
  prefill: number;
  requests: number;
}

export type Connection =
  | { state: 'connecting' }
  | { state: 'live'; latencyMs: number }
  | { state: 'error'; message: string; retryInMs: number };

export interface StatsFeed {
  stats: Stats | null;
  samples: readonly Sample[];
  connection: Connection;
  /** All-time peaks since the page was opened. */
  peaks: { decode: number; prefill: number };
}

const MAX_BACKOFF_MS = 10_000;

/**
 * Polls `/stats` on a timer chain (never overlapping requests), pauses while the
 * tab is hidden, backs off exponentially on failure, and keeps a bounded history
 * of samples for the sparklines.
 */
export function useStats(base: string, intervalMs: number, keep: number): StatsFeed {
  const [feed, setFeed] = useState<StatsFeed>({
    stats: null,
    samples: [],
    connection: { state: 'connecting' },
    peaks: { decode: 0, prefill: 0 },
  });
  // Mutable history lives in a ref; state gets a fresh frozen snapshot per tick.
  const samplesRef = useRef<Sample[]>([]);
  const peaksRef = useRef({ decode: 0, prefill: 0 });

  useEffect(() => {
    let timer: ReturnType<typeof setTimeout> | undefined;
    // One controller for the effect's lifetime: aborts the in-flight fetch on
    // unmount and doubles as the "stopped" flag for late callbacks.
    const life = new AbortController();
    const stopped = () => life.signal.aborted;
    let failures = 0;

    const schedule = (ms: number) => {
      clearTimeout(timer);
      timer = setTimeout(() => void tick(), ms);
    };

    const tick = async () => {
      if (stopped()) return;
      if (document.hidden) {
        schedule(intervalMs);
        return;
      }
      const started = performance.now();
      try {
        const stats = await fetchStats(base, life.signal);
        if (stopped()) return;
        const now = Date.now();
        const list = samplesRef.current;
        list.push({ t: now, decode: stats.last_decode_tps, prefill: stats.last_prefill_tps, requests: stats.requests });
        if (list.length > keep) list.splice(0, list.length - keep);
        peaksRef.current = {
          decode: Math.max(peaksRef.current.decode, stats.last_decode_tps),
          prefill: Math.max(peaksRef.current.prefill, stats.last_prefill_tps),
        };
        failures = 0;
        setFeed({
          stats,
          samples: [...list],
          connection: { state: 'live', latencyMs: performance.now() - started },
          peaks: peaksRef.current,
        });
        schedule(intervalMs);
      } catch (e) {
        if (stopped() || (e instanceof DOMException && e.name === 'AbortError')) return;
        failures += 1;
        const retryInMs = Math.min(MAX_BACKOFF_MS, intervalMs * 2 ** Math.min(failures, 5));
        setFeed((prev) => ({
          ...prev,
          connection: { state: 'error', message: e instanceof Error ? e.message : String(e), retryInMs },
        }));
        schedule(retryInMs);
      }
    };

    const onVisible = () => {
      if (!document.hidden) schedule(0);
    };
    document.addEventListener('visibilitychange', onVisible);
    schedule(0);

    return () => {
      life.abort();
      clearTimeout(timer);
      document.removeEventListener('visibilitychange', onVisible);
    };
  }, [base, intervalMs, keep]);

  return feed;
}
