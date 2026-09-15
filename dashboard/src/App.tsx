import { useMemo, useState } from "react";
import { Sparkline } from "./Sparkline";
import {
  DASH, compact, duration, fixed, gib, gibDelta, mb, mbps, ms, msNum, num, pct, pctStr,
  percentile, rate, ratioStr, safeDiv, sec, tps,
} from "./format";
import type { Mtp, MtpCounters, RecentRequest, Stats, Totals } from "./types";
import { RECENT_SPAN, WINDOW, useStats } from "./useStats";

type Range = "session" | "all";
type Tone = "ok" | "warn" | "bad" | undefined;

/** Below this many completed requests, percentiles are noise — show min/med/max. */
const DISTRIBUTION_MIN_N = 20;
/** Below this many, a session average is not worth its own tile. */
const AVERAGE_MIN_N = 10;

/* ----------------------------------------------------------- persisted state */

const readBool = (key: string, fallback: boolean): boolean => {
  try {
    const v = localStorage.getItem(key);
    return v === "1" ? true : v === "0" ? false : fallback;
  } catch { return fallback; } // private mode / blocked storage
};

/** A boolean that survives reloads. Reads lazily so the first paint already has
 *  the saved value, and never throws where localStorage is unavailable. */
function usePersisted(key: string, fallback: boolean): [boolean, (v: boolean) => void] {
  const [v, setV] = useState(() => readBool(key, fallback));
  const set = (next: boolean) => {
    setV(next);
    try { localStorage.setItem(key, next ? "1" : "0"); } catch { /* ignore */ }
  };
  return [v, set];
}

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

const SOURCE_SHORT: Record<string, string> = {
  "memory-text": "mem", "memory-token": "mem-tok",
  "disk-text": "disk", "disk-token": "disk-tok",
  none: "cold", cold: "cold",
};
const shortSource = (s: string): string => SOURCE_SHORT[s] ?? s;

const worst = (...tones: Tone[]): Tone =>
  tones.includes("bad") ? "bad" : tones.includes("warn") ? "warn" : tones.includes("ok") ? "ok" : undefined;

const pressureKind = (p: string | undefined): Tone =>
  p === "normal" ? "ok" : p === "warn" ? "warn" : p === "critical" ? "bad" : undefined;
const thermalKind = (t: string | undefined): Tone =>
  t === "nominal" ? "ok" : t === "fair" || t === "serious" ? "warn" : t === "critical" ? "bad" : undefined;

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
  const [ckptOpen, setCkptOpen] = usePersisted("ds4.dash.panel.checkpoint", false);
  // null = follow mtp.active; a click pins an explicit choice until reload, so a
  // poll that flips `active` can never fight the user.
  const [mtpOverride, setMtpOverride] = useState<boolean | null>(null);
  const [openRow, setOpenRow] = useState<number | null>(null);

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
  const n = recent.length;

  const decodeSeries = useMemo(() => recent.map(decodeTps).filter((v): v is number => v !== null).reverse(), [recent]);
  const computeSeries = useMemo(() => recent.map(computeTps).filter((v): v is number => v !== null).reverse(), [recent]);
  const decodeEma = useMemo(() => ema(samples.map((s) => s.decode)), [samples]);
  const decodeMedian = useMemo(() => percentile(decodeSeries, 50), [decodeSeries]);
  // The slow tail of the tok/s distribution is its 5th percentile; inverted, that
  // is the *high* per-token latency — always ≥ 1000 / median tok/s.
  const p95LatencyMs = useMemo(() => {
    const slow = percentile(decodeSeries, 5);
    const r = safeDiv(1, slow);
    return r === null ? null : r * 1000;
  }, [decodeSeries]);

  const segments = useMemo<Segment[]>(
    () => (last ? SEGMENT_DEFS.map((d) => ({ key: d.key, label: d.label, color: d.color, ns: Math.max(0, d.pick(last)) })) : []),
    [last],
  );
  const segTotal = segments.reduce((a, s) => a + s.ns, 0);

  // Under DISTRIBUTION_MIN_N samples, percentiles are noise: show the plain range.
  // Headers and accessors move together so they can never disagree.
  const dist = useMemo(() => {
    const wide = n >= DISTRIBUTION_MIN_N;
    const ps: [number, number, number] = wide ? [50, 95, 95] : [0, 50, 100];
    const headers = wide ? ["p50", "p95"] : ["min", "median", "max"];
    const rows = SEGMENT_DEFS.map((d) => {
      const vals = recent.map(d.pick).filter((v) => Number.isFinite(v));
      return {
        key: d.key,
        label: d.label,
        cells: wide
          ? [percentile(vals, ps[0]), percentile(vals, ps[1])]
          : [percentile(vals, 0), percentile(vals, 50), percentile(vals, 100)],
      };
    });
    return { headers, rows };
  }, [recent, n]);

  const health: Tone = error ? (failures > 3 ? "bad" : "warn") : stats ? "ok" : "warn";
  const ctxPct = pct(stats?.live_tokens, stats?.ctx_size);
  const kvPct = pct(kv?.used_mb, kv?.budget_mb);
  const metalPct = pct(mem?.metal_allocated_mb, mem?.metal_working_set_mb);
  const headroomMb = mem ? Math.max(0, mem.metal_working_set_mb - mem.metal_allocated_mb) : undefined;
  const swapDelta = mem && baseline?.swap_mb !== null && baseline?.swap_mb !== undefined
    ? mem.swap_used_mb - baseline.swap_mb : null;
  const dotTone = worst(health, pressureKind(mem?.pressure), thermalKind(mem?.thermal));

  /* Qwen MTP has no per-cycle timing counters; effective decode is the observed
   * decode tok/s while speculation is enabled (last request, else session). */
  const mtpTimingInstrumented = !!(M && M.total_ns > 0);
  const mtpEffective = mtpTimingInstrumented
    ? rate(M?.committed, M?.total_ns)
    : (last && last.mtp_cycles > 0 ? decodeTps(last) : (stats?.last_decode_tps || null));
  const mtpActive = !!(mtp?.enabled && mtp.active);
  const mtpOpen = mtpOverride ?? mtpActive;

  const storeAlarm = !!last && last.store_ns > 1e9;

  // Prefill: the effective rate only earns a tile when it diverges from the raw
  // GPU compute rate — otherwise the two numbers just repeat each other.
  const lastCompute = last ? computeTps(last) : null;
  const lastEffective = rate(last?.fresh_tokens, last?.prompt_ns);
  const showEffective = lastCompute !== null && lastEffective !== null
    && Math.abs(lastEffective - lastCompute) / lastCompute > 0.05;

  const showKind = n > 0 && !recent.every((r) => r.kind === "chat");
  const columns = [
    "Time", ...(showKind ? ["Kind"] : []), "Context", "Fresh", "Reuse %",
    "Prompt s", "TTFT ms", "Output", "Decode tok/s", "MTP", "Finish",
  ];

  return (
    <>
      <nav className="topbar">
        <div className="brand">
          <span className="brand-name">ds4-server</span>
          <span className="brand-sub">{stats?.model ?? "connecting…"}</span>
        </div>
        <div className="health-strip">
          <span className={`dot ${dotTone ?? ""}`} />
          <b>{error ? "Unreachable" : !stats ? "Connecting" : stats.busy ? "Busy" : "Healthy"}</b>
          <Sep />
          <span>up {duration(stats?.uptime_s)}</span>
          <Sep />
          <span>{tps(last ? decodeTps(last) : stats?.last_decode_tps)}</span>
          <Sep />
          <span>{compact(stats?.live_tokens)}/{compact(stats?.ctx_size)} ctx</span>
          <Sep />
          <span>Metal {mem ? `${fixed(metalPct, 0)}%` : DASH}</span>
          <Sep />
          <span>{gibDelta(swapDelta)} swap</span>
          <Sep />
          <span className={`tone-${pressureKind(mem?.pressure) ?? "none"}`}>pressure {mem?.pressure ?? DASH}</span>
          <Sep />
          <span className={`tone-${thermalKind(mem?.thermal) ?? "none"}`}>thermal {mem?.thermal ?? DASH}</span>
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
              <button role="tab" aria-selected={sessionMode} onClick={() => setRange("session")}>Since page open</button>
              <button role="tab" aria-selected={!sessionMode} onClick={() => setRange("all")}>Since server start</button>
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

        {/* ================================================== performance */}
        <Section id="performance" title="Performance">
          <Panel title="Prefill" icon="⇥" tone="secondary" aside={<span className="muted">last request</span>}>
            <div className="metrics">
              <Metric label="Prompt" value={num(last?.prompt_tokens)} />
              <Metric label="Cached" value={num(last?.cached_tokens)} hint={`${pctStr(last?.cached_tokens, last?.prompt_tokens)} reused · ${last ? shortSource(last.source) : DASH}`} />
              <Metric label="Fresh" value={num(last?.fresh_tokens)} />
              <Metric label="Compute" value={ms(last?.prefill_ns)} />
              <Metric label="Compute rate" value={tps(lastCompute)} hint={`session ${tps(rate(T?.prefill_fresh_tokens, T?.prefill_compute_ns))}`} />
              <Metric label="Prompt wall" value={sec(last?.prompt_ns)} />
              {showEffective && (
                <Metric label="Effective prompt tok/s" value={tps(lastEffective)} hint="incl. lookup, restore and save" />
              )}
            </div>
            <Chart label="GPU prefill compute tok/s per request" data={computeSeries} color="#7c5cff" unit="tok/s" />
          </Panel>

          <Panel title="Decode" icon="⇢" tone="primary" aside={<span className="muted">{n} recent requests</span>}>
            <div className="metrics">
              <Metric label="Last request" value={tps(last ? decodeTps(last) : null)} hint={`${msNum(safeDiv(last?.decode_ns, last?.decode_tokens))} ms/token`} big />
              <Metric label="5-min EMA" value={tps(decodeEma)} big />
              <Metric label="Median" value={tps(decodeMedian)} big />
              <Metric label="P95 token latency" value={p95LatencyMs === null ? DASH : `${fixed(p95LatencyMs, 1)} ms/token`} big
                tone={p95LatencyMs !== null && decodeMedian !== null && p95LatencyMs > (1000 / decodeMedian) * 1.6 ? "warn" : undefined} />
              {n >= AVERAGE_MIN_N && (
                <Metric label="Session average" value={tps(rate(T?.decode_tokens, T?.decode_ns))} big />
              )}
            </div>
            <Chart label="decode tok/s per request" data={decodeSeries} color="#0a84ff" unit="tok/s" />
          </Panel>

          <Panel title="Latency" icon="⟼" tone="primary" aside={<span className="muted">{last ? `last request #${num(last.seq)}` : "no completed request"}</span>}>
            {!last ? (
              <div className="empty">No completed requests reported yet</div>
            ) : (
              <>
                <div className="block-head">Last request only · {ms(segTotal)} end-to-end</div>
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
                <div className="block-head">session distribution (n={n})</div>
                <div className="table-wrap">
                  <table className="log">
                    <thead><tr><th>component</th>{dist.headers.map((h) => <th key={h}>{h}</th>)}</tr></thead>
                    <tbody>
                      {dist.rows.map((r) => (
                        <tr key={r.key}>
                          <td>{r.label}</td>
                          {r.cells.map((c, i) => <td key={i}>{ms(c)}</td>)}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </>
            )}
          </Panel>
        </Section>

        {/* ================================================== memory & kv */}
        <Section id="memory" title="Memory & KV">
          <Panel
            title="Memory"
            icon="▦"
            tone="primary"
            aside={mem
              ? <Meter value={metalPct} label={`Metal ${fixed(metalPct, 0)}% · headroom ${gib(headroomMb)}`} hot={metalPct > 90} />
              : <span className="muted">no mem telemetry</span>}
          >
            <div className="metrics">
              <Metric label="Footprint" value={gib(mem?.footprint_mb ?? stats?.footprint_mb)} big />
              <Metric label="Peak footprint" value={gib(mem?.peak_footprint_mb)} big />
              <Metric label="Metal allocated" value={gib(mem?.metal_allocated_mb)} hint={`of ${gib(mem?.metal_working_set_mb)}`} big />
              <Metric label="Headroom" value={gib(headroomMb)} hint="working set − allocated" big />
              <Metric label="Swap used" value={gib(mem?.swap_used_mb ?? stats?.swap_used_mb)} hint={`${gibDelta(swapDelta)} since page open`} big />
              <Metric label="Pressure" value={mem?.pressure ?? DASH} tone={pressureKind(mem?.pressure)} big />
              <Metric label="Thermal" value={mem?.thermal ?? DASH} tone={thermalKind(mem?.thermal)} big />
            </div>
            <details className="advanced">
              <summary>Advanced process metrics</summary>
              <div className="metrics">
                <Metric label="RSS" value={gib(mem?.rss_mb ?? stats?.rss_mb)} />
                <Metric label="Metal working set" value={gib(mem?.metal_working_set_mb)} />
                <Metric
                  label="PLE sidecar"
                  value={stats?.ple?.mode ?? "not instrumented"}
                  hint={stats?.ple?.mode === "demand-paged"
                    ? "CPU-side Q4_1 n-gram, demand-paged"
                    : stats?.ple?.mode === "off" ? "--ple not set" : undefined}
                />
              </div>
            </details>
          </Panel>

          <Panel
            title="Cache"
            icon="⛁"
            tone="secondary"
            aside={kv?.enabled
              ? <Meter value={kvPct} label={`SSD ${gib(kv.used_mb)} / ${gib(kv.budget_mb)} · ${num(kv.files)} files`} hot={kvPct > 90} />
              : <span className="muted">disk cache off</span>}
          >
            <div className="metrics">
              <Metric label="Hit rate" value={pctStr(counters?.hits, (counters ? counters.hits + counters.cold : undefined))} />
              <Metric label="Hits / cold" value={counters ? `${num(counters.hits)} / ${num(counters.cold)}` : DASH} />
              <Metric label="Last source" value={last ? shortSource(last.source) : DASH} />
              <Metric label="KV disk" value={kv?.enabled ? `${gib(kv.used_mb)} / ${gib(kv.budget_mb)}` : DASH} hint={kv?.enabled ? `${num(kv.files)} files` : undefined} />
            </div>
            <Meter value={ctxPct} label={`${num(stats?.live_tokens)} / ${num(stats?.ctx_size)} KV live · ${fixed(ctxPct, 1)}%`} hot={ctxPct > 85} wide />
            <div className="sub"><code className="path">{kv?.enabled ? kv.dir : "disk cache off"}</code></div>
          </Panel>

          {ckptOpen ? (
            <Panel
              title="Checkpoint"
              icon="⤓"
              tone="secondary"
              onToggle={() => setCkptOpen(false)}
              open
              aside={storeAlarm ? <span className="badge bad">last save &gt; 1 s</span> : <span className="muted">KV persistence</span>}
            >
              <div className="metrics">
                <Metric label="Last save" value={ms(last?.store_ns)} tone={storeAlarm ? "bad" : undefined} />
                <Metric label="Session avg save" value={ms(safeDiv(T?.checkpoint_save_ns, T?.checkpoint_saves))} />
                <Metric label="Saves" value={num(T?.checkpoint_saves)} />
                <Metric label="Bytes written" value={mb(T?.checkpoint_save_bytes)} />
                <Metric label="Avg restore" value={ms(safeDiv(T?.checkpoint_restore_ns, T?.checkpoint_restores))} />
                <Metric label="Restores" value={num(T?.checkpoint_restores)} />
              </div>
            </Panel>
          ) : (
            <CollapsedPanel
              icon="⤓"
              onClick={() => setCkptOpen(true)}
              badge={storeAlarm ? <span className="badge bad">last save &gt; 1 s</span> : undefined}
              text={T
                ? `Checkpoint · ${num(T.checkpoint_saves)} saves · avg ${ms(safeDiv(T.checkpoint_save_ns, T.checkpoint_saves))} · ${num(T.checkpoint_restores)} restores`
                : "Checkpoint · no data"}
            />
          )}
        </Section>

        {/* ==================================================== speculation */}
        <Section id="speculation" title="Speculation">
          {!mtp?.enabled ? (
            <CollapsedPanel icon="⚡" onClick={() => setMtpOverride(true)} text="MTP off (--mtp not set)" />
          ) : mtpOpen ? (
            <Panel
              title="Speculative decode (MTP)"
              icon="⚡"
              tone="secondary"
              open
              onToggle={() => setMtpOverride(false)}
              aside={<span className={`badge ${mtpActive ? "ok" : ""}`}>{mtpActive ? "ON" : "OFF"}</span>}
            >
              <div className="metrics">
                <Metric label="Position" value={num(mtp?.pos)} hint="not context-gated on Qwen" />
                <Metric label="Acceptance" value={ratioStr(safeDiv(M?.accepted, M?.cycles))} hint={M ? `${num(M.accepted)} / ${num(M.cycles)} cycles` : undefined} />
                <Metric label="Tokens / cycle" value={fixed(safeDiv(M?.committed, M?.cycles), 2)} hint={M ? `${num(M.committed)} draft tokens accepted` : undefined} />
                <Metric label="Effective decode" value={tps(mtpEffective)} hint="last request decode tok/s while MTP on" />
              </div>
              {mtpTimingInstrumented ? null : (
                <div className="hint">break-even C/S · not instrumented on Qwen</div>
              )}
              <div className="hint">counters are {sessionMode ? "deltas since the page opened" : "since server start"}; draft width 1 (width flag inert)</div>
            </Panel>
          ) : (
            <CollapsedPanel icon="⚡" onClick={() => setMtpOverride(true)} text={mtpSummary(mtp)} />
          )}
        </Section>

        {/* ========================================================= system */}
        <Section id="system" title="System">
          <Panel title="System" icon="◈" tone="secondary" aside={<span className="muted">{stats?.tensor_route ?? "route —"}</span>}>
            <div className="metrics">
              <Metric label="CPU (of one core)" value={rates ? `${fixed(rates.cpuPct, 1)}%` : DASH} />
              <Metric label="Disk read" value={mbps(rates?.diskReadBps)} />
              <Metric label="Disk write" value={mbps(rates?.diskWriteBps)} />
              <Metric label="Page-ins" value={rates ? `${fixed(rates.pageinsPerSec, 1)} /s` : DASH} />
              <Metric
                label="Tensor route"
                value={stats?.tensor_route ?? DASH}
                tone={stats?.tensor_route && stats.tensor_route !== "auto" ? "warn" : undefined}
                hint={stats?.tensor_route && stats.tensor_route !== "auto" ? "expected auto" : undefined}
              />
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
        </Section>

        {/* ======================================================= requests */}
        <Section id="requests" title="Requests">
          <Panel title="Recent requests" icon="≡" tone="secondary" aside={<span className="muted">newest first · click a row for detail</span>}>
            {n === 0 ? (
              <div className="empty">{stats?.recent ? "No completed requests yet" : "This server does not report per-request history"}</div>
            ) : (
              <div className="table-wrap">
                <table className="log rows">
                  <thead><tr>{columns.map((c) => <th key={c}>{c}</th>)}</tr></thead>
                  <tbody>
                    {recent.map((r) => (
                      <RowGroup
                        key={r.seq}
                        r={r}
                        stats={stats}
                        showKind={showKind}
                        colSpan={columns.length}
                        open={openRow === r.seq}
                        onClick={() => setOpenRow(openRow === r.seq ? null : r.seq)}
                      />
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Panel>
        </Section>

        <footer>
          <span>polling /stats every 1 s</span>
          <a href="/stats">/stats</a><a href="/health">/health</a><a href="/v1/models">/v1/models</a>
        </footer>
      </main>
    </>
  );
}

/* -------------------------------------------------------------- components */

const Sep = () => <span className="sep">·</span>;

/** The one-line form of the MTP panel, per state. Qwen is never context-gated. */
function mtpSummary(mtp: Mtp | undefined): string {
  if (!mtp) return `MTP ${DASH}`;
  if (!mtp.enabled) return "MTP off (--mtp not set)";
  return "MTP on";
}

/** Wall-clock time of a completed request, from the server uptime it carries. */
function clockOf(stats: Stats | null, r: RecentRequest): string {
  if (!stats) return DASH;
  const t = Date.now() - Math.max(0, stats.uptime_s - r.at_s) * 1000;
  return new Date(t).toLocaleTimeString();
}

function RowGroup({ r, stats, showKind, colSpan, open, onClick }: {
  r: RecentRequest; stats: Stats | null; showKind: boolean; colSpan: number; open: boolean; onClick: () => void;
}) {
  return (
    <>
      <tr className={`row ${open ? "open" : ""}`} onClick={onClick}>
        <td>{clockOf(stats, r)}</td>
        {showKind && <td>{r.kind}</td>}
        <td>{num(r.pos)}</td>
        <td>{num(r.fresh_tokens)}</td>
        <td>{pctStr(r.cached_tokens, r.prompt_tokens, 0)}</td>
        <td>{fixed(r.prompt_ns / 1e9, 2)}</td>
        <td>{msNum(r.first_token_ns)}</td>
        <td>{num(r.decode_tokens)}</td>
        <td>{fixed(decodeTps(r), 1)}</td>
        <td>{r.mtp_cycles > 0 ? `${num(r.mtp_committed)}/${num(r.mtp_cycles)}` : "—"}</td>
        <td>{r.finish}</td>
      </tr>
      {open && (
        <tr className="row-detail">
          <td colSpan={colSpan}>
            <dl className="detail">
              <dt>seq</dt><dd>{num(r.seq)}</dd>
              <dt>prompt</dt><dd>{num(r.prompt_tokens)}</dd>
              <dt>cached</dt><dd>{num(r.cached_tokens)}</dd>
              <dt>compute tok/s</dt><dd>{fixed(computeTps(r), 0)}</dd>
              <dt>mtp cycles</dt><dd>{num(r.mtp_cycles)}</dd>
              <dt>mtp committed</dt><dd>{num(r.mtp_committed)}</dd>
              <dt>source</dt><dd>{shortSource(r.source)}</dd>
            </dl>
          </td>
        </tr>
      )}
    </>
  );
}

function Section({ id, title, children }: { id: string; title: string; children: React.ReactNode }) {
  const [open, setOpen] = usePersisted(`ds4.dash.section.${id}`, true);
  return (
    <section className="group" id={id}>
      <button type="button" className="group-head" onClick={() => setOpen(!open)} aria-expanded={open}>
        <span className={`chev ${open ? "open" : ""}`}>▸</span>
        <span className="group-title">{title}</span>
      </button>
      {open && <div className="group-body">{children}</div>}
    </section>
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

function Metric({ label, value, hint, tone, big }: { label: string; value: string; hint?: string; tone?: Tone; big?: boolean }) {
  return (
    <div className="metric">
      <div className="label">{label}</div>
      <div className={`metric-value ${big ? "big" : ""} ${tone ?? ""}`}>{value}</div>
      {hint && <div className="hint">{hint}</div>}
    </div>
  );
}

function Panel({ title, icon, aside, tone, open, onToggle, children }: {
  title: string; icon: string; aside?: React.ReactNode; tone?: "primary" | "secondary";
  open?: boolean; onToggle?: () => void; children: React.ReactNode;
}) {
  const head = (
    <>
      <span className="panel-title">
        {onToggle && <span className={`chev ${open ? "open" : ""}`}>▸</span>}
        <span className="icon">{icon}</span>{title}
      </span>
      <span className="panel-aside">{aside}</span>
    </>
  );
  return (
    <section className={`panel ${tone ?? "secondary"}`}>
      {onToggle
        ? <button type="button" className="panel-head as-button" onClick={onToggle} aria-expanded={open}>{head}</button>
        : <div className="panel-head">{head}</div>}
      <div className="panel-body">{children}</div>
    </section>
  );
}

/** Tertiary, one-line form of a panel. The muted styling lives here, not on the
 *  panel body, so expanding it yields a normal-weight panel. */
function CollapsedPanel({ icon, text, badge, onClick }: { icon: string; text: string; badge?: React.ReactNode; onClick: () => void }) {
  return (
    <button type="button" className="panel collapsed" onClick={onClick} aria-expanded={false}>
      <span className="chev">▸</span>
      <span className="icon">{icon}</span>
      <span className="collapsed-text">{text}</span>
      {badge}
    </button>
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

/** A full-width sparkline with its own caption and min/max scale. */
function Chart({ label, data, color, unit }: { label: string; data: number[]; color: string; unit: string }) {
  const lo = data.length ? Math.min(...data) : null;
  const hi = data.length ? Math.max(...data) : null;
  return (
    <div className="chart">
      <div className="chart-head">
        <span className="label">{label}</span>
        <span className="hint">{lo === null ? "no data yet" : `min ${fixed(lo, 1)} · max ${fixed(hi, 1)} ${unit}`}</span>
      </div>
      <Sparkline data={data} color={color} span={RECENT_SPAN} height={90} />
      <div className="chart-foot"><span>oldest</span><span>newest</span></div>
    </div>
  );
}
