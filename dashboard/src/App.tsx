import { useMemo, useState } from "react";
import { Sparkline } from "./Sparkline";
import { duration, gb, num, pct, tps } from "./format";
import type { Stats } from "./types";
import { useStats } from "./useStats";

type Range = "session" | "all";

interface Baseline {
  requests: number; prefill_cancelled: number; prompt_tokens: number;
  cached_tokens: number; generated_tokens: number; hits: number; cold: number;
}

const ZERO: Baseline = { requests: 0, prefill_cancelled: 0, prompt_tokens: 0, cached_tokens: 0, generated_tokens: 0, hits: 0, cold: 0 };
const snapshot = (s: Stats): Baseline => ({
  requests: s.requests, prefill_cancelled: s.prefill_cancelled, prompt_tokens: s.prompt_tokens,
  cached_tokens: s.cached_tokens, generated_tokens: s.generated_tokens, hits: s.cache.hits, cold: s.cache.cold,
});

function windowStats(values: number[]) {
  const nz = values.filter((v) => v > 0);
  return { peak: nz.length ? Math.max(...nz) : 0, avg: nz.length ? nz.reduce((a, b) => a + b, 0) / nz.length : 0 };
}

export function App() {
  const { stats, error, failures, samples, events } = useStats();
  const [range, setRange] = useState<Range>("session");
  // "Session" counts from the moment the page loaded (or Clear was pressed);
  // "All-Time" shows the server's counters since it started.
  const [baseline, setBaseline] = useState<Baseline | null>(null);
  if (stats && baseline === null) setBaseline(snapshot(stats));

  const base = range === "session" && baseline ? baseline : ZERO;
  const totals = stats ? {
    requests: stats.requests - base.requests,
    cancelled: stats.prefill_cancelled - base.prefill_cancelled,
    prompt: stats.prompt_tokens - base.prompt_tokens,
    cached: stats.cached_tokens - base.cached_tokens,
    generated: stats.generated_tokens - base.generated_tokens,
    hits: stats.cache.hits - base.hits,
    cold: stats.cache.cold - base.cold,
  } : null;

  const prefill = useMemo(() => windowStats(samples.map((s) => s.prefill)), [samples]);
  const decode = useMemo(() => windowStats(samples.map((s) => s.decode)), [samples]);
  const prefillSeries = useMemo(() => samples.map((s) => s.prefill), [samples]);
  const decodeSeries = useMemo(() => samples.map((s) => s.decode), [samples]);

  const health: "ok" | "warn" | "bad" = error ? (failures > 3 ? "bad" : "warn") : stats ? "ok" : "warn";
  const ctxPct = stats ? pct(stats.live_tokens, stats.ctx_size) : 0;
  const cacheEff = totals ? pct(totals.cached, totals.prompt) : 0;
  const kv = stats?.kv_disk;
  const kvPct = kv ? pct(kv.used_mb, kv.budget_mb) : 0;

  return (
    <>
      <nav className="topbar">
        <div className="brand">
          <span className={`dot ${health}`} />
          <span className="brand-name">ds4-server</span>
          <span className="brand-sub">{stats?.model ?? "connecting…"}</span>
        </div>
        <div className="topbar-meta">
          <span>up <b>{stats ? duration(stats.uptime_s) : "—"}</b></span>
          <span>rss <b>{stats?.rss_mb ? gb(stats.rss_mb) : "—"}</b></span>
          <span className={`pill ${stats?.busy ? "busy" : "idle"}`}>{error ? "unreachable" : stats?.busy ? "busy" : "idle"}</span>
        </div>
      </nav>

      <main>
        <header className="page-head">
          <div>
            <div className="eyebrow">Dashboard</div>
            <h1>Serving Stats</h1>
          </div>
          <div className="controls">
            <div className="segmented" role="tablist">
              <button role="tab" aria-selected={range === "session"} onClick={() => setRange("session")}>Session</button>
              <button role="tab" aria-selected={range === "all"} onClick={() => setRange("all")}>All-Time</button>
            </div>
            <button className="ghost" onClick={() => stats && setBaseline(snapshot(stats))} disabled={!stats}>↺ Clear</button>
          </div>
        </header>

        {error && <div className="banner error">Cannot reach /stats: {error} — retrying every second.</div>}

        <section className="stat-grid">
          <Stat label="Total prefill tokens" value={totals ? num(totals.prompt) : "—"} />
          <Stat label="Cached tokens" value={totals ? num(totals.cached) : "—"} />
          <Stat label="Cache efficiency" value={totals ? `${cacheEff.toFixed(1)}%` : "—"} hint={totals ? `${num(totals.hits)} hits · ${num(totals.cold)} cold` : undefined} />
          <Stat label="Generated tokens" value={totals ? num(totals.generated) : "—"} hint={totals ? `${num(totals.requests)} requests · ${num(totals.cancelled)} cancelled` : undefined} />
        </section>

        <Panel title="Speed" icon="◔" aside="last request · 5-minute window">
          <div className="split">
            <div className="speed">
              <div className="label">Prompt processing (excl. cached)</div>
              <div className="value">{stats ? tps(stats.last_prefill_tps) : "—"}</div>
              <Sparkline data={prefillSeries} color="#7c5cff" />
              <div className="hint">peak {tps(prefill.peak)} · avg {tps(prefill.avg)}</div>
            </div>
            <div className="speed">
              <div className="label">Token generation</div>
              <div className="value">{stats ? tps(stats.last_decode_tps) : "—"}</div>
              <Sparkline data={decodeSeries} color="#0a84ff" />
              <div className="hint">peak {tps(decode.peak)} · avg {tps(decode.avg)}</div>
            </div>
          </div>
        </Panel>

        <Panel
          title="Active model"
          icon="∿"
          aside={stats ? <Meter value={ctxPct} label={`${num(stats.live_tokens)} / ${num(stats.ctx_size)} ctx tokens live`} hot={ctxPct > 85} /> : undefined}
        >
          {stats ? (
            <>
              <div className="model-row">
                <div>
                  <div className="model-name">{stats.model}</div>
                  <code className="path">{stats.model_path || "—"}</code>
                </div>
                <dl className="facts">
                  <Fact k="clients" v={num(stats.clients)} />
                  <Fact k="queue" v={num(stats.queue_depth)} />
                  <Fact k="slots" v={`${stats.slots.filter((s) => s.busy).length} / ${stats.slot_count} busy`} />
                  <Fact k="context" v={num(stats.ctx_size)} />
                </dl>
              </div>
              <div className="slots">
                {stats.slots.map((s) => {
                  const p = pct(s.live_tokens, s.ctx);
                  return (
                    <div key={s.id} className={`slot ${s.busy ? "busy" : "idle"}`}>
                      <div className="slot-head"><span>slot {s.id}</span><span className="state">{s.busy ? "busy" : "idle"}</span></div>
                      <div className="slot-tokens">{num(s.live_tokens)} / {num(s.ctx)}</div>
                      <div className={`bar ${p > 85 ? "hot" : ""}`}><i style={{ width: `${p}%` }} /></div>
                    </div>
                  );
                })}
              </div>
            </>
          ) : <div className="empty">No model loaded</div>}
        </Panel>

        <Panel
          title="Runtime cache"
          icon="⛁"
          aside={kv?.enabled ? <Meter value={kvPct} label={`SSD ${gb(kv.used_mb)} / ${gb(kv.budget_mb)} · ${num(kv.files)} files`} hot={kvPct > 90} /> : <span className="muted">disk cache off</span>}
        >
          <div className="kv-grid">
            <div>
              <div className="label">KV disk cache directory</div>
              <code className="path">{kv?.enabled ? kv.dir : "—"}</code>
            </div>
            <div>
              <div className="label">Prefix reuse</div>
              <dl className="facts inline">
                <Fact k="hits" v={totals ? num(totals.hits) : "—"} />
                <Fact k="cold" v={totals ? num(totals.cold) : "—"} />
                <Fact k="hit rate" v={totals ? `${pct(totals.hits, totals.hits + totals.cold).toFixed(0)}%` : "—"} />
              </dl>
            </div>
          </div>
        </Panel>

        <Panel title="Recent requests" icon="≡" aside={<span className="muted">derived from counter deltas</span>}>
          {events.length === 0 ? (
            <div className="empty">No completed requests since the page opened</div>
          ) : (
            <table className="log">
              <thead><tr><th>time</th><th>req</th><th>prompt</th><th>cached</th><th>generated</th><th>prefill</th><th>decode</th></tr></thead>
              <tbody>
                {events.map((e) => (
                  <tr key={e.at}>
                    <td>{new Date(e.at).toLocaleTimeString()}</td>
                    <td>+{e.requests}</td>
                    <td>{num(e.prompt)}</td>
                    <td>{num(e.cached)}</td>
                    <td>{num(e.generated)}</td>
                    <td>{tps(e.prefillTps)}</td>
                    <td>{tps(e.decodeTps)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </Panel>

        <footer>
          <span>polling /stats every 1 s</span>
          <a href="/stats">/stats</a><a href="/health">/health</a><a href="/v1/models">/v1/models</a>
        </footer>
      </main>
    </>
  );
}

function Stat({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="stat">
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
      {hint && <div className="hint">{hint}</div>}
    </div>
  );
}

function Panel({ title, icon, aside, children }: { title: string; icon: string; aside?: React.ReactNode; children: React.ReactNode }) {
  return (
    <section className="panel">
      <div className="panel-head">
        <span className="panel-title"><span className="icon">{icon}</span>{title}</span>
        <span className="panel-aside">{aside}</span>
      </div>
      <div className="panel-body">{children}</div>
    </section>
  );
}

function Meter({ value, label, hot }: { value: number; label: string; hot?: boolean }) {
  return (
    <span className="meter">
      <span className={`bar ${hot ? "hot" : ""}`}><i style={{ width: `${Math.min(100, value)}%` }} /></span>
      <span>{label}</span>
    </span>
  );
}

function Fact({ k, v }: { k: string; v: string }) {
  return (<><dt>{k}</dt><dd>{v}</dd></>);
}
