import { useEffect, useRef, useState } from 'react';
import { fetchRequests, fetchStats, type RequestRecord, type Stats } from '../api/stats.ts';

export interface Sample {
  t: number;
  decode: number;
  prefill: number;
  requests: number;
  liveTokens: number;
  metalAllocated?: number;
  metalHeadroom?: number;
  swapBytes?: number;
  footprint?: number;
}

export type Connection =
  | { state: 'connecting' }
  | { state: 'live'; latencyMs: number; staleS: number }
  | { state: 'error'; message: string; retryInMs: number };

export interface StatsFeed {
  stats: Stats | null;
  samples: readonly Sample[];
  requests: readonly RequestRecord[];
  connection: Connection;
  peaks: { decode: number; prefill: number };
}

const MAX_BACKOFF_MS = 10_000;

export function useStats(base: string, intervalMs: number, keep: number): StatsFeed {
  const [feed, setFeed] = useState<StatsFeed>({
    stats: null,
    samples: [],
    requests: [],
    connection: { state: 'connecting' },
    peaks: { decode: 0, prefill: 0 },
  });
  const samplesRef = useRef<Sample[]>([]);
  const peaksRef = useRef({ decode: 0, prefill: 0 });
  const lastOkRef = useRef<number | null>(null);

  useEffect(() => {
    let timer: ReturnType<typeof setTimeout> | undefined;
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
        let requests: RequestRecord[] = [];
        try {
          requests = await fetchRequests(base, life.signal);
        } catch {
          requests = [];
        }
        if (stopped()) return;
        const now = Date.now();
        lastOkRef.current = now;
        const allocated = stats.metal?.allocated_bytes;
        const recommended = stats.metal?.recommended_max_bytes;
        const sample: Sample = {
          t: now,
          decode: stats.last_decode_tps,
          prefill: stats.last_prefill_tps,
          requests: stats.requests,
          liveTokens: stats.live_tokens,
        };
        if (allocated != null) sample.metalAllocated = allocated;
        if (allocated != null && recommended != null) sample.metalHeadroom = recommended - allocated;
        if (stats.host?.swap_bytes != null) sample.swapBytes = stats.host.swap_bytes;
        if (stats.process?.physical_footprint_bytes != null) {
          sample.footprint = stats.process.physical_footprint_bytes;
        }
        const list = samplesRef.current;
        list.push(sample);
        if (list.length > keep) list.splice(0, list.length - keep);
        peaksRef.current = {
          decode: Math.max(peaksRef.current.decode, stats.last_decode_tps),
          prefill: Math.max(peaksRef.current.prefill, stats.last_prefill_tps),
        };
        failures = 0;
        setFeed({
          stats,
          samples: [...list],
          requests,
          connection: { state: 'live', latencyMs: performance.now() - started, staleS: 0 },
          peaks: peaksRef.current,
        });
        schedule(intervalMs);
      } catch (e) {
        if (stopped() || (e instanceof DOMException && e.name === 'AbortError')) return;
        failures += 1;
        const retryInMs = Math.min(MAX_BACKOFF_MS, intervalMs * 2 ** Math.min(failures, 5));
        const staleS = lastOkRef.current != null ? (Date.now() - lastOkRef.current) / 1000 : null;
        setFeed((prev) => ({
          ...prev,
          connection: staleS != null
            ? { state: 'live', latencyMs: 0, staleS }
            : { state: 'error', message: e instanceof Error ? e.message : String(e), retryInMs },
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
