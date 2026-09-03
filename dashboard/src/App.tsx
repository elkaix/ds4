import { useEffect, useMemo, useState } from 'react';
import { fetchModelIds, resolveBase, resolveInterval } from './api/stats';
import { useStats } from './hooks/useStats';
import { Header } from './components/Header';
import { StatTile } from './components/StatTile';
import { Sparkline } from './components/Sparkline';
import { Meter, type Severity } from './components/Meter';
import { Panel } from './components/Panel';
import { KeyValueList } from './components/KeyValueList';
import { CachePaths } from './components/CachePaths';
import { Anomalies } from './components/Anomalies';
import { fmtCompact, fmtFixed, fmtInt, fmtPct } from './lib/format';

const BASE = resolveBase();
const INTERVAL = resolveInterval();
const WINDOW_MS = 5 * 60_000;
const KEEP = Math.ceil(WINDOW_MS / INTERVAL);

const sevFor = (pct: number, warn: number, crit: number): Severity =>
  pct >= crit ? 'critical' : pct >= warn ? 'warn' : 'ok';

export default function App() {
  const { stats, samples, connection, peaks } = useStats(BASE, INTERVAL, KEEP);
  const [modelIds, setModelIds] = useState<string[]>([]);
  const [modelName, setModelName] = useState<string | null>(null);

  // Model identity is static for the server's lifetime: fetch once.
  useEffect(() => {
    const c = new AbortController();
    fetchModelIds(BASE, c.signal).then(setModelIds).catch(() => undefined);
    fetch(`${BASE}/health`, { signal: c.signal })
      .then((r) => r.json())
      .then((j: { model?: unknown }) => { if (typeof j.model === 'string') setModelName(j.model); })
      .catch(() => undefined);
    return () => { c.abort(); };
  }, []);

  const derived = useMemo(() => {
    const decode = samples.map((s) => s.decode);
    const prefill = samples.map((s) => s.prefill);
    const times = samples.map((s) => s.t);
    const avg = (a: number[]) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : undefined);
    const last = samples[samples.length - 1];
    const cutoff = last ? last.t - 60_000 : 0;
    const first60 = samples.find((s) => s.t >= cutoff);
    const rpm =
      last && first60 && last.t > first60.t
        ? ((last.requests - first60.requests) / (last.t - first60.t)) * 60_000
        : undefined;
    return { decode, prefill, times, avgDecode: avg(decode), avgPrefill: avg(prefill), rpm };
  }, [samples]);

  const ctxPct = stats?.ctx_size ? (stats.live_tokens / stats.ctx_size) * 100 : 0;
  const cacheTotal = stats ? stats.cache.hits + stats.cache.cold : 0;
  const hitPct = stats && cacheTotal ? (stats.cache.hits / cacheTotal) * 100 : 0;
  const savedPct = stats?.prompt_tokens ? (stats.cached_tokens / stats.prompt_tokens) * 100 : undefined;

  return (
    <div className="page">
      <Header
        connection={connection}
        busy={stats?.busy ?? false}
        modelIds={modelIds}
        modelName={modelName ?? (modelIds[0] ?? null)}
        uptimeS={stats?.uptime_s}
        base={BASE}
      />

      {connection.state === 'error' && (
        <div className="banner" role="alert">
          <b>Cannot reach {BASE}/stats</b> — {connection.message}. Retrying in {fmtFixed(connection.retryInMs / 1000, 0)}s.
          {!location.protocol.startsWith('http') || BASE !== location.origin ? (
            <span> Open <code>{BASE}/dashboard</code> directly, or start the server with <code>--cors</code>.</span>
          ) : null}
        </div>
      )}

      <main className="grid">
        <StatTile
          label="Decode · last request"
          value={fmtFixed(stats?.last_decode_tps, 1)}
          unit="tok/s"
          className="tile--kpi"
          facts={[
            { k: 'Peak', v: fmtFixed(peaks.decode, 1) },
            { k: '5 min avg', v: fmtFixed(derived.avgDecode, 1) },
          ]}
        >
          <Sparkline values={derived.decode} times={derived.times} unit="tok/s" color="var(--series-decode)" title="Decode throughput" />
        </StatTile>

        <StatTile
          label="Prefill · last request"
          value={fmtFixed(stats?.last_prefill_tps, 0)}
          unit="tok/s"
          className="tile--kpi"
          facts={[
            { k: 'Peak', v: fmtFixed(peaks.prefill, 0) },
            { k: '5 min avg', v: fmtFixed(derived.avgPrefill, 0) },
          ]}
        >
          <Sparkline values={derived.prefill} times={derived.times} unit="tok/s" digits={0} color="var(--series-prefill)" title="Prefill throughput" />
        </StatTile>

        <StatTile
          label="Context in use"
          value={fmtFixed(ctxPct, 1)}
          unit="%"
          className="tile--kpi"
          facts={[
            { k: 'Live tokens', v: fmtInt(stats?.live_tokens) },
            { k: 'Window', v: fmtInt(stats?.ctx_size) },
          ]}
        >
          <Meter pct={ctxPct} severity={sevFor(ctxPct, 70, 90)} label="Context window utilisation" />
        </StatTile>

        <StatTile
          label="Cache hit rate"
          value={cacheTotal ? fmtFixed(hitPct, 1) : '—'}
          unit="%"
          className="tile--kpi"
          facts={[
            { k: 'Hits', v: fmtInt(stats?.cache.hits) },
            { k: 'Cold', v: fmtInt(stats?.cache.cold) },
          ]}
        >
          <Meter pct={hitPct} severity={cacheTotal ? sevFor(100 - hitPct, 30, 60) : 'ok'} label="Prompt cache hit rate" />
        </StatTile>

        <Panel title="Load" aside={derived.rpm != null ? `${fmtFixed(derived.rpm, 1)} req/min` : undefined}>
          <KeyValueList
            rows={[
              { k: 'State', v: stats?.busy ? 'Generating' : 'Idle', tone: stats?.busy ? 'warn' : 'ok' },
              { k: 'Queue depth', v: fmtInt(stats?.queue_depth), tone: stats?.queue_depth ? 'warn' : undefined },
              { k: 'Connected clients', v: fmtInt(stats?.clients) },
              { k: 'Requests served', v: fmtInt(stats?.requests) },
              { k: 'Poll latency', v: connection.state === 'live' ? `${fmtFixed(connection.latencyMs, 0)} ms` : '—', tone: 'muted' },
            ]}
          />
        </Panel>

        <Panel title="Tokens" aside={savedPct != null ? `${fmtPct(savedPct, 1)} prefill saved` : undefined}>
          <KeyValueList
            rows={[
              { k: 'Prompt', v: fmtCompact(stats?.prompt_tokens) },
              { k: 'Served from cache', v: fmtCompact(stats?.cached_tokens) },
              { k: 'Generated', v: fmtCompact(stats?.generated_tokens) },
              { k: 'Prefill saved', v: fmtPct(savedPct, 1), tone: savedPct && savedPct > 50 ? 'ok' : undefined },
            ]}
          />
        </Panel>

        <Panel title="Anomalies" className="tile--anom">
          {stats ? <Anomalies stats={stats} /> : <p className="tone-muted">—</p>}
        </Panel>

        <Panel title="Cache paths" aside={cacheTotal ? `${fmtInt(cacheTotal)} requests` : undefined} className="tile--wide">
          {stats ? <CachePaths cache={stats.cache} /> : <p className="tone-muted">—</p>}
        </Panel>
      </main>

      <footer className="foot">
        Polling <code>{BASE}/stats</code> every {String(INTERVAL)} ms · {String(samples.length)} / {String(KEEP)} samples kept
      </footer>
    </div>
  );
}
