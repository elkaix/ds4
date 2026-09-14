import { useMemo, useState } from "react";
import { Sparkline } from "./Sparkline";
import {
  DASH, duration, fixed, gib, gibDelta, mb, mbps, ms, msNum, num, pct, pctStr,
  percentile, rate, ratioStr, safeDiv, sec, signedPct, tps,
} from "./format";
import type { Mtp, MtpCounters, RecentRequest, Stats, Totals } from "./types";
import { RECENT_SPAN, WINDOW, useStats } from "./useStats";

type Range = "session" | "all";

/* ------------------------------------------------------------------ baseline */

interface Baseline {
  requests: number; prefill_cancelled: number; prompt_tokens: number;
  cached_tokens: number; generated_tokens: number; hits: number; cold: number;
  totals: Totals | null;
  mtp: MtpCounters | null;
  /** captured on the first poll that actually carries `mem`, so an older server
   *  (or one late field) can't pin the delta to the full swap value forever. */
  swap_mb: number | null;
}

const ZERO_TOTALS: Totals = {
  prefill_fresh_tokens: 0, prefill_compute_ns: 0, prompt_total_ns: 0, decode_tokens: 0, decode_ns: 0,
  first_token_ns: 0, checkpoint_saves: 0, checkpoint_save_ns: 0, checkpoint_save_bytes: 0,
  checkpoint_restores: 0, checkpoint_restore_ns: 0,
};
const ZERO_MTP: MtpCounters = {
  cycles: 0, accepted: 0, committed: 0, rows_cycles: 0, batch_cycles: 0,
  setup_ns: 0, verify_ns: 0, rollback_ns: 0, draft_ns: 0, total_ns: 0,
};
const ZERO: Baseline = {
  requests: 0, prefill_cancelled: 0, prompt_tokens: 0, cached_tokens: 0, generated_tokens: 0,
  hits: 0, cold: 0, totals: ZERO_TOTALS, mtp: ZERO_MTP, swap_mb: null,
};

const mtpCounters = (m: Mtp): MtpCounters => ({
  cycles: m.cycles, accepted: m.accepted, committed: m.committed, rows_cycles: m.rows_cycles,
  batch_cycles: m.batch_cycles, setup_ns: m.setup_ns, verify_ns: m.verify_ns,
  rollback_ns: m.rollback_ns, draft_ns: m.draft_ns, total_ns: m.total_ns,
});

const snapshot = (s: Stats): Baseline => ({
  requests: s.requests, prefill_cancelled: s.prefill_cancelled, prompt_tokens: s.prompt_tokens,
  cached_tokens: s.cached_tokens, generated_tokens: s.generated_tokens,
  hits: s.cache?.hits ?? 0, cold: s.cache?.cold ?? 0,
  totals: s.totals ? { ...s.totals } : null,
  mtp: s.mtp ? mtpCounters(s.mtp) : null,
  swap_mb: s.mem ? s.mem.swap_used_mb : null,
});

/** Subtract a baseline from a uniformly cumulative record, clamping counter resets. */
function delta<T extends object>(cur: T | undefined, base: T | null | undefined): T | null {
  if (!cur) return null;
  if (!base) return cur;
  // Both arguments are flat all-number records (Totals / MtpCounters); the views
  // below are the only place that fact is asserted.
  const a = cur as { [k: string]: number };
  const b = base as { [k: string]: number };
  const out: { [k: string]: number } = {};
  for (const k of Object.keys(a)) out[k] = Math.max(0, a[k] - (b[k] ?? 0));
  return out as T;
}

/* --------------------------------------------------------------------- misc */

const EMA_ALPHA = 2 / (WINDOW + 1); // smoothed over the full 5-minute sample window
function ema(values: number[]): number | null {
  const nz = values.filter((v) => Number.isFinite(v) && v > 0);
  if (!nz.length) return null;
  return nz.reduce((acc, v, i) => (i === 0 ? v : acc + EMA_ALPHA * (v - acc)), 0);
}

const decodeTps = (r: RecentRequest) => rate(r.decode_tokens, r.decode_ns);
const computeTps = (r: RecentRequest) => rate(r.fresh_tokens, r.prefill_ns);

/** prompt_ns minus the legs we account for; first_token_ns and decode_ns sit outside it. */
const bookkeepingNs = (r: RecentRequest) =>
  Math.max(0, r.prompt_ns - (r.lookup_ns + r.restore_ns + r.cold_ns + r.prefill_ns + r.store_ns));

interface Segment { key: string; label: string; ns: number; color: string }
const SEGMENT_DEFS: { key: string; label: string; color: string; pick: (r: RecentRequest) => number }[] = [
  { key: "lookup", label: "lookup", color: "#8b93a1", pick: (r) => r.lookup_ns },
  { key: "restore", label: "restore", color: "#0ea5e9", pick: (r) => r.restore_ns },
  { key: "cold", label: "cold prefill", color: "#6366f1", pick: (r) => r.cold_ns },
  { key: "prefill", label: "fresh prefill", color: "#7c5cff", pick: (r) => r.prefill_ns },
  { key: "store", label: "checkpoint save", color: "#f59e0b", pick: (r) => r.store_ns },
  { key: "other", label: "other bookkeeping", color: "#a3a3a3", pick: bookkeepingNs },
  { key: "ttft", label: "first token", color: "#ec4899", pick: (r) => r.first_token_ns },
  { key: "decode", label: "decode", color: "#0a84ff", pick: (r) => r.decode_ns },
];

/* ---------------------------------------------------------------------- app */

export function App() {
  const { stats, error, failures, samples, rates } = useStats();
  const [range, setRange] = useState<Range>("session");
  // "Session" counts from the moment the page loaded (or Clear was pressed);
  // "All-Time" shows the server's counters since it started.
  const [baseline, setBaseline] = useState<Baseline | null>(null);
  if (stats && baseline === null) setBaseline(snapshot(stats));
  // Backfill the pieces the very first poll may not have carried.
  if (stats && baseline) {
    if (baseline.swap_mb === null && stats.mem) setBaseline({ ...baseline, swap_mb: stats.mem.swap_used_mb });
    else if (baseline.totals === null && stats.totals) setBaseline({ ...baseline, totals: { ...stats.totals } });
    else if (baseline.mtp === null && stats.mtp) setBaseline({ ...baseline, mtp: mtpCounters(stats.mtp) });
  }

  const sessionMode = range === "session";
  const base = sessionMode && baseline ? baseline : ZERO;

  const counters = stats ? {
    requests: Math.max(0, stats.requests - base.requests),
    cancelled: Math.max(0, stats.prefill_cancelled - base.prefill_cancelled),
    prompt: Math.max(0, stats.prompt_tokens - base.prompt_tokens),
    cached: Math.max(0, stats.cached_tokens - base.cached_tokens),
    generated: Math.max(0, stats.generated_tokens - base.generated_tokens),
    hits: Math.max(0, (stats.cache?.hits ?? 0) - base.hits),
    cold: Math.max(0, (stats.cache?.cold ?? 0) - base.cold),
  } : null;

  const T = delta(stats?.totals, sessionMode ? baseline?.totals : ZERO_TOTALS);
  const M = delta(stats?.mtp ? mtpCounters(stats.mtp) : undefined, sessionMode ? baseline?.mtp : ZERO_MTP);
  const mtp = stats?.mtp;
  const mem = stats?.mem;
  const kv = stats?.kv_disk;

  const recent = useMemo<RecentRequest[]>(() => (stats?.recent ?? []).slice(0, 64), [stats?.recent]);
  const last = recent[0];

  const footprintSeries = useMemo(() => samples.map((s) => s.footprint), [samples]);
  const decodeSeries = useMemo(() => recent.map(decodeTps).filter((v): v is number => v !== null).reverse(), [recent]);
  const computeSeries = useMemo(() => recent.map(computeTps).filter((v): v is number => v !== null).reverse(), [recent]);
  const decodeEma = useMemo(() => ema(samples.map((s) => s.decode)), [samples]);
  const decodeP50 = useMemo(() => percentile(decodeSeries, 50), [decodeSeries]);
  const decodeP95Low = useMemo(() => percentile(decodeSeries, 5), [decodeSeries]);

  /** Latest request that ran without speculation — the plain-decode reference. */
  const plainStepNs = useMemo(() => {
    const r = recent.find((x) => x.mtp_cycles === 0 && x.decode_tokens > 0 && x.decode_ns > 0);
    return r ? r.decode_ns / r.decode_tokens : null;
  }, [recent]);

  const segments = useMemo<Segment[]>(
    () => (last ? SEGMENT_DEFS.map((d) => ({ key: d.key, label: d.label, color: d.color, ns: Math.max(0, d.pick(last)) })) : []),
    [last],
  );
  const segTotal = segments.reduce((a, s) => a + s.ns, 0);

  const segStats = useMemo(
    () => SEGMENT_DEFS.map((d) => {
      const vals = recent.map(d.pick).filter((v) => Number.isFinite(v));
      return { key: d.key, label: d.label, p50: percentile(vals, 50), p95: percentile(vals, 95) };
    }),
    [recent],
  );

  const health: "ok" | "warn" | "bad" = error ? (failures > 3 ? "bad" : "warn") : stats ? "ok" : "warn";
  const ctxPct = pct(stats?.live_tokens, stats?.ctx_size);
  const kvPct = pct(kv?.used_mb, kv?.budget_mb);
  const metalPct = pct(mem?.metal_allocated_mb, mem?.metal_working_set_mb);
  const swapDelta = mem && baseline?.swap_mb !== null && baseline?.swap_mb !== undefined
    ? mem.swap_used_mb - baseline.swap_mb : null;

  const mtpState = !mtp ? DASH : !mtp.enabled ? "OFF" : mtp.active ? "ON" : "gated off past max_ctx";
  const mtpEffective = rate(M?.committed, M?.total_ns);
  const plainTps = plainStepNs ? 1e9 / plainStepNs : null;
  const netVsPlain = mtpEffective !== null && plainTps !== null && plainTps > 0 ? mtpEffective / plainTps - 1 : null;
  const perCycle = (ns: number | undefined) => safeDiv(ns, M?.cycles);
  const otherNs = M ? Math.max(0, M.total_ns - (M.setup_ns + M.verify_ns + M.rollback_ns + M.draft_ns)) : undefined;

  const storeAlarm = !!last && last.store_ns > 1e9;

  return (
    <>
      <nav className="topbar">
        <div className="brand">
          <span className={`dot ${health}`} />
          <span className="brand-name">ds4-server</span>
          <span className="brand-sub">{stats?.model ?? "connecting…"}</span>
        </div>
        <div className="topbar-meta">
          <span>up <b>{duration(stats?.uptime_s)}</b></span>
          <span>footprint <b>{gib(mem?.footprint_mb ?? stats?.footprint_mb)}</b></span>
          <span>Metal <b>{gib(mem?.metal_allocated_mb)}</b></span>
          <span>swap <b>{gib(mem?.swap_used_mb ?? stats?.swap_used_mb)}</b> <i className="delta">({gibDelta(swapDelta)})</i></span>
          <Badge kind={pressureKind(mem?.pressure)} text={mem ? `pressure ${mem.pressure}` : "pressure —"} />
          <Badge kind={thermalKind(mem?.thermal)} text={mem ? `thermal ${mem.thermal}` : "thermal —"} />
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
              <button role="tab" aria-selected={sessionMode} onClick={() => setRange("session")}>Session</button>
              <button role="tab" aria-selected={!sessionMode} onClick={() => setRange("all")}>All-Time</button>
            </div>
            <button className="ghost" onClick={() => stats && setBaseline(snapshot(stats))} disabled={!stats}>↺ Clear</button>
          </div>
        </header>

        {error && <div className="banner error">Cannot reach /stats: {error} — retrying every second.</div>}

        <section className="stat-grid">
          <Stat label="Requests" value={num(counters?.requests)} hint={counters ? `${num(counters.cancelled)} cancelled` : undefined} />
          <Stat label="Fresh tokens computed" value={num(T?.prefill_fresh_tokens)} hint={counters ? `${num(counters.prompt)} prompt · ${num(counters.cached)} cached` : undefined} />
          <Stat label="Generated tokens" value={num(T?.decode_tokens ?? counters?.generated)} hint={`avg ${tps(rate(T?.decode_tokens, T?.decode_ns))}`} />
          <Stat label="Cache reuse" value={pctStr(counters?.cached, counters?.prompt)} hint={counters ? `${num(counters.hits)} hits · ${num(counters.cold)} cold` : undefined} />
        </section>

        {/* ------------------------------------------------------------ memory */}
        <Panel title="Memory" icon="▦" aside={mem ? <Meter value={metalPct} label={`Metal ${gib(mem.metal_allocated_mb)} / ${gib(mem.metal_working_set_mb)} · ${fixed(metalPct, 0)}%`} hot={metalPct > 90} /> : <span className="muted">no mem telemetry</span>}>
          <div className="metrics">
            <Metric label="Footprint" value={gib(mem?.footprint_mb ?? stats?.footprint_mb)} />
            <Metric label="Peak footprint" value={gib(mem?.peak_footprint_mb)} />
            <Metric label="Metal allocated" value={gib(mem?.metal_allocated_mb)} hint={`limit ${gib(mem?.metal_working_set_mb)}`} />
            <Metric label="Swap used" value={gib(mem?.swap_used_mb ?? stats?.swap_used_mb)} hint={`${gibDelta(swapDelta)} since page open`} />
            <Metric label="Pressure" value={mem?.pressure ?? DASH} tone={pressureKind(mem?.pressure)} />
            <Metric label="Thermal" value={mem?.thermal ?? DASH} tone={thermalKind(mem?.thermal)} />
          </div>
          <div className="sub">RSS {gib(mem?.rss_mb ?? stats?.rss_mb)}</div>
          <Sparkline data={footprintSeries} color="#22c55e" />
          <div className="hint">process footprint · 5-minute window</div>
        </Panel>

        {/* ----------------------------------------------------------- prefill */}
        <Panel title="Prefill" icon="⇥" aside={<span className="muted">last request · session</span>}>
          <div className="metrics">
            <Metric label="Prompt tokens (logical)" value={num(last?.prompt_tokens)} />
            <Metric label="Cached tokens" value={num(last?.cached_tokens)} hint={last ? `source ${last.source}` : undefined} />
            <Metric label="Fresh tokens computed" value={num(last?.fresh_tokens)} />
            <Metric label="GPU prefill compute tok/s" value={tps(last ? computeTps(last) : null)} hint={ms(last?.prefill_ns)} />
            <Metric label="Prompt processing wall" value={sec(last?.prompt_ns)} />
            <Metric label="Effective prompt tok/s" value={tps(rate(last?.fresh_tokens, last?.prompt_ns))} />
            <Metric label="Cache reuse" value={pctStr(last?.cached_tokens, last?.prompt_tokens)} />
            <Metric label="Session compute tok/s" value={tps(rate(T?.prefill_fresh_tokens, T?.prefill_compute_ns))} hint={`wall ${tps(rate(T?.prefill_fresh_tokens, T?.prompt_total_ns))}`} />
          </div>
          <Sparkline data={computeSeries} color="#7c5cff" span={RECENT_SPAN} />
          <div className="hint">GPU prefill compute tok/s per request · oldest → newest</div>
        </Panel>

        {/* ------------------------------------------------------------ decode */}
        <Panel title="Decode" icon="⇢" aside={<span className="muted">{recent.length} recent requests</span>}>
          <div className="metrics">
            <Metric label="Last request" value={tps(last ? decodeTps(last) : null)} hint={`${msNum(safeDiv(last?.decode_ns, last?.decode_tokens))} ms/token`} />
            <Metric label="5-min EMA" value={tps(decodeEma)} />
            <Metric label="Session p50" value={tps(decodeP50)} />
            <Metric label="Session p95-low (slow tail)" value={tps(decodeP95Low)} tone={decodeP95Low !== null && decodeP50 !== null && decodeP95Low < decodeP50 * 0.6 ? "warn" : undefined} />
            <Metric label="Session average" value={tps(rate(T?.decode_tokens, T?.decode_ns))} />
            <Metric label="Session ms/token" value={msNum(safeDiv(T?.decode_ns, T?.decode_tokens))} />
          </div>
          <Sparkline data={decodeSeries} color="#0a84ff" span={RECENT_SPAN} />
          <div className="hint">decode tok/s per request · oldest → newest</div>
        </Panel>

        {/* ------------------------------------------------------------- cache */}
        <Panel title="Cache" icon="⛁" aside={kv?.enabled ? <Meter value={kvPct} label={`SSD ${gib(kv.used_mb)} / ${gib(kv.budget_mb)} · ${num(kv.files)} files`} hot={kvPct > 90} /> : <span className="muted">disk cache off</span>}>
          <div className="metrics">
            <Metric label="Hit rate" value={pctStr(counters?.hits, (counters ? counters.hits + counters.cold : undefined))} hint={counters ? `${num(counters.hits)} hits · ${num(counters.cold)} cold` : undefined} />
            <Metric label="Fresh tokens computed" value={num(T?.prefill_fresh_tokens)} hint="session, cache misses only" />
            <Metric label="KV live tokens" value={`${num(stats?.live_tokens)} / ${num(stats?.ctx_size)}`} />
            <Metric label="KV disk bytes" value={kv?.enabled ? `${gib(kv.used_mb)} / ${gib(kv.budget_mb)}` : DASH} hint={kv?.enabled ? `${num(kv.files)} files` : undefined} />
          </div>
          <Meter value={ctxPct} label={`${fixed(ctxPct, 1)}% of context window live`} hot={ctxPct > 85} wide />
          <div className="sub"><code className="path">{kv?.enabled ? kv.dir : "disk cache off"}</code></div>
        </Panel>

        {/* -------------------------------------------------------- checkpoint */}
        <Panel title="Checkpoint" icon="⤓" aside={storeAlarm ? <span className="badge bad">last save &gt; 1 s</span> : <span className="muted">KV persistence</span>}>
          <div className="metrics">
            <Metric label="Last save" value={ms(last?.store_ns)} tone={storeAlarm ? "bad" : undefined} />
            <Metric label="Session avg save" value={ms(safeDiv(T?.checkpoint_save_ns, T?.checkpoint_saves))} />
            <Metric label="Saves" value={num(T?.checkpoint_saves)} />
            <Metric label="Bytes written" value={mb(T?.checkpoint_save_bytes)} />
            <Metric label="Avg restore" value={ms(safeDiv(T?.checkpoint_restore_ns, T?.checkpoint_restores))} />
            <Metric label="Restores" value={num(T?.checkpoint_restores)} />
          </div>
        </Panel>

        {/* --------------------------------------------------------------- mtp */}
        <Panel
          title="Speculative decode (MTP)"
          icon="⚡"
          aside={<span className={`badge ${!mtp ? "" : !mtp.enabled ? "" : mtp.active ? "ok" : "warn"}`}>{mtpState}</span>}
        >
          <div className="metrics">
            <Metric label="Position" value={num(mtp?.pos)} hint={mtp ? (mtp.max_ctx > 0 ? `max_ctx ${num(mtp.max_ctx)}` : "no ceiling") : undefined} />
            <Metric label="Acceptance" value={ratioStr(safeDiv(M?.accepted, M?.cycles))} hint={M ? `${num(M.accepted)} / ${num(M.cycles)} cycles` : undefined} />
            <Metric label="Tokens / cycle" value={fixed(safeDiv(M?.committed, M?.cycles), 2)} hint={M ? `${num(M.committed)} committed` : undefined} />
            <Metric label="Effective decode" value={tps(mtpEffective)} />
            <Metric label="Plain step" value={plainStepNs !== null ? ms(plainStepNs) : DASH} hint={plainTps !== null ? tps(plainTps) : "no unspeculated request"} />
            <Metric label="Net vs plain" value={signedPct(netVsPlain)} tone={netVsPlain === null ? undefined : netVsPlain >= 0 ? "ok" : "bad"} />
            <Metric label="Cycle" value={ms(safeDiv(M?.total_ns, M?.cycles))} />
            <Metric label="Draft" value={ms(perCycle(M?.draft_ns))} />
            <Metric label="Verify" value={ms(perCycle(M?.verify_ns))} />
            <Metric label="Rollback" value={ms(perCycle(M?.rollback_ns))} />
            <Metric label="Setup" value={ms(perCycle(M?.setup_ns))} />
            <Metric label="Other" value={ms(perCycle(otherNs))} />
            <Metric label="Verify path: rows" value={num(M?.rows_cycles)} hint={M ? pctStr(M.rows_cycles, M.cycles, 0) : undefined} />
            <Metric label="Verify path: batch" value={num(M?.batch_cycles)} hint={M ? pctStr(M.batch_cycles, M.cycles, 0) : undefined} />
          </div>
          <div className="hint">counters are {sessionMode ? "deltas since the page opened" : "since server start"}; per-cycle times</div>
        </Panel>

        {/* --------------------------------------------------------- waterfall */}
        <Panel title="Latency waterfall" icon="⟼" aside={<span className="muted">{last ? `request #${num(last.seq)} · ${ms(segTotal)} end-to-end` : "no completed request"}</span>}>
          {!last ? (
            <div className="empty">No completed requests reported yet</div>
          ) : (
            <>
              <div className="wf-bar">
                {segments.map((s) => (
                  s.ns > 0 && segTotal > 0 ? (
                    <i key={s.key} style={{ width: `${(100 * s.ns) / segTotal}%`, background: s.color }} title={`${s.label} · ${ms(s.ns)}`} />
                  ) : null
                ))}
              </div>
              <div className="wf-legend">
                {segments.map((s) => (
                  <div key={s.key} className="wf-item">
                    <span className="swatch" style={{ background: s.color }} />
                    <span className="wf-label">{s.label}</span>
                    <span className="wf-val">{ms(s.ns)}</span>
                    <span className="wf-pct">{segTotal > 0 ? `${((100 * s.ns) / segTotal).toFixed(1)}%` : DASH}</span>
                  </div>
                ))}
              </div>
              <div className="table-wrap">
                <table className="log">
                  <thead><tr><th>component</th><th>p50</th><th>p95</th></tr></thead>
                  <tbody>
                    {segStats.map((s) => (
                      <tr key={s.key}><td>{s.label}</td><td>{ms(s.p50)}</td><td>{ms(s.p95)}</td></tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="hint">percentiles over the last {recent.length} requests</div>
            </>
          )}
        </Panel>

        {/* ------------------------------------------------------------ system */}
        <Panel title="System" icon="◈" aside={<span className="muted">{stats?.tensor_route ?? "route —"}</span>}>
          <div className="metrics">
            <Metric label="CPU (of one core)" value={rates ? `${fixed(rates.cpuPct, 1)}%` : DASH} />
            <Metric label="Disk read" value={mbps(rates?.diskReadBps)} />
            <Metric label="Disk write" value={mbps(rates?.diskWriteBps)} />
            <Metric label="Page-ins" value={rates ? `${fixed(rates.pageinsPerSec, 1)} /s` : DASH} />
            <Metric label="Tensor route" value={stats?.tensor_route ?? DASH} />
            <Metric label="Clients" value={num(stats?.clients)} />
            <Metric label="Queue depth" value={num(stats?.queue_depth)} />
            <Metric label="Slots busy" value={stats ? `${stats.slots?.filter((s) => s.busy).length ?? 0} / ${stats.slot_count}` : DASH} />
          </div>
          <div className="slots">
            {(stats?.slots ?? []).map((s) => {
              const p = pct(s.live_tokens, s.ctx);
              return (
                <div key={s.id} className={`slot ${s.busy ? "busy" : "idle"}`}>
                  <div className="slot-head"><span>slot {s.id}</span><span className="state">{s.busy ? "busy" : "idle"}</span></div>
                  <div className="slot-tokens">{num(s.live_tokens)} / {num(s.ctx)}</div>
                  <div className={`bar ${p > 85 ? "hot" : ""}`}><i style={{ width: `${Math.min(100, p)}%` }} /></div>
                </div>
              );
            })}
          </div>
          <div className="sub"><code className="path">{stats?.model_path || DASH}</code></div>
        </Panel>

        {/* ------------------------------------------------------------ recent */}
        <Panel title="Recent requests" icon="≡" aside={<span className="muted">reported by the server · newest first</span>}>
          {recent.length === 0 ? (
            <div className="empty">{stats?.recent ? "No completed requests yet" : "This server does not report per-request history"}</div>
          ) : (
            <div className="table-wrap">
              <table className="log">
                <thead>
                  <tr>
                    <th>time</th><th>kind</th><th>pos</th><th>fresh</th><th>cached</th><th>prompt s</th>
                    <th>compute tok/s</th><th>TTFT ms</th><th>gen</th><th>decode tok/s</th><th>mtp</th>
                    <th>source</th><th>finish</th>
                  </tr>
                </thead>
                <tbody>
                  {recent.map((r) => (
                    <tr key={r.seq}>
                      <td>{clockOf(stats, r)}</td>
                      <td>{r.kind}</td>
                      <td>{num(r.pos)}</td>
                      <td>{num(r.fresh_tokens)}</td>
                      <td>{num(r.cached_tokens)}</td>
                      <td>{fixed(r.prompt_ns / 1e9, 2)}</td>
                      <td>{fixed(computeTps(r), 0)}</td>
                      <td>{msNum(r.first_token_ns)}</td>
                      <td>{num(r.decode_tokens)}</td>
                      <td>{fixed(decodeTps(r), 1)}</td>
                      <td>{num(r.mtp_cycles)}</td>
                      <td>{r.source}</td>
                      <td>{r.finish}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
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

/* -------------------------------------------------------------- components */

type Tone = "ok" | "warn" | "bad" | undefined;

const pressureKind = (p: string | undefined): Tone =>
  p === "normal" ? "ok" : p === "warn" ? "warn" : p === "critical" ? "bad" : undefined;
const thermalKind = (t: string | undefined): Tone =>
  t === "nominal" ? "ok" : t === "fair" || t === "serious" ? "warn" : t === "critical" ? "bad" : undefined;

/** Wall-clock time of a completed request, from the server uptime it carries. */
function clockOf(stats: Stats | null, r: RecentRequest): string {
  if (!stats) return DASH;
  const t = Date.now() - Math.max(0, stats.uptime_s - r.at_s) * 1000;
  return new Date(t).toLocaleTimeString();
}

function Badge({ kind, text }: { kind: Tone; text: string }) {
  return <span className={`badge ${kind ?? ""}`}>{text}</span>;
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

function Metric({ label, value, hint, tone }: { label: string; value: string; hint?: string; tone?: Tone }) {
  return (
    <div className="metric">
      <div className="label">{label}</div>
      <div className={`metric-value ${tone ?? ""}`}>{value}</div>
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

function Meter({ value, label, hot, wide }: { value: number; label: string; hot?: boolean; wide?: boolean }) {
  return (
    <span className={`meter ${wide ? "wide" : ""}`}>
      <span className={`bar ${hot ? "hot" : ""}`}><i style={{ width: `${Math.min(100, Math.max(0, value))}%` }} /></span>
      <span>{label}</span>
    </span>
  );
}
