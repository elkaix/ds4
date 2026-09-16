import { useMemo, useState } from "react";
import type { ChangeEvent, ReactNode } from "react";
import {
  DASH, compact, duration, fixed, gib, gibDelta, mb, mbps, ms, msNum, num, pct, pctStr,
  percentile, rate, safeDiv, sec, signedPct, tps,
} from "./format";
import type { Mtp, MtpCounters, RecentRequest, Stats, Totals } from "./types";
import { WINDOW, useStats } from "./useStats";

type Period = "Session" | "All-Time";
type BottomTab = "cache" | "requests";

interface Baseline {
  requests: number; prefill_cancelled: number; prompt_tokens: number;
  cached_tokens: number; generated_tokens: number; hits: number; cold: number;
  totals: Totals | null;
  mtp: MtpCounters | null;
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
  hits: s.cache.hits, cold: s.cache.cold,
  totals: s.totals ? { ...s.totals } : null,
  mtp: s.mtp ? mtpCounters(s.mtp) : null,
  swap_mb: s.mem ? s.mem.swap_used_mb : null,
});

function delta<T extends object>(cur: T | undefined, base: T | null | undefined): T | null {
  if (!cur) return null;
  if (!base) return cur;
  const a = cur as Record<string, number>;
  const b = base as Record<string, number>;
  const out: Record<string, number> = {};
  for (const k of Object.keys(a)) out[k] = Math.max(0, (a[k] ?? 0) - (b[k] ?? 0));
  return out as T;
}

const EMA_ALPHA = 2 / (WINDOW + 1);
function ema(values: number[]): number | null {
  const nz = values.filter((v) => Number.isFinite(v) && v > 0);
  if (!nz.length) return null;
  return nz.reduce((acc, v, i) => (i === 0 ? v : acc + EMA_ALPHA * (v - acc)), 0);
}

const decodeTps = (r: RecentRequest) => rate(r.decode_tokens, r.decode_ns);
const computeTps = (r: RecentRequest) => rate(r.fresh_tokens, r.prefill_ns);

const SOURCE_SHORT: Record<string, string> = {
  "memory-text": "mem-text", "memory-token": "mem-token", "disk-text": "disk-text", "disk-token": "disk-token", none: "cold",
};
const shortSource = (s: string): string => SOURCE_SHORT[s] ?? s;

function clockOf(stats: Stats | null, r: RecentRequest): string {
  if (!stats) return DASH;
  const t = Date.now() - Math.max(0, stats.uptime_s - r.at_s) * 1000;
  return new Date(t).toLocaleTimeString();
}

type NavIcon = "grid" | "bars" | "doc" | "cube" | "terminal" | "database" | "chart" | "bell" | "users" | "settings" | "help";

const navItems: {
  name: NavIcon;
  label: string;
  targetId: string;
  tab?: BottomTab | undefined;
  openAdvanced?: boolean | undefined;
}[] = [
  { name: "grid", label: "Overview", targetId: "sec-hero" },
  { name: "bars", label: "Performance", targetId: "sec-performance" },
  { name: "doc", label: "Recent Requests", targetId: "sec-bottom", tab: "requests" },
  { name: "cube", label: "Tokens & KV", targetId: "sec-tokens" },
  { name: "terminal", label: "Speculative MTP", targetId: "sec-mtp" },
  { name: "database", label: "Cache Storage", targetId: "sec-bottom", tab: "cache" },
  { name: "chart", label: "System Telemetry", targetId: "sec-system" },
  { name: "bell", label: "Health & Anomalies", targetId: "sec-anomalies" },
  { name: "users", label: "Serving & Clients", targetId: "sec-serving" },
  { name: "settings", label: "Advanced Diagnostics", targetId: "sec-advanced", openAdvanced: true },
  { name: "help", label: "API Endpoints", targetId: "sec-footer" },
];

export function App() {
  const { stats, error, failures, samples, rates } = useStats();
  const [period, setPeriod] = useState<Period>("Session");
  const [baseline, setBaseline] = useState<Baseline | null>(null);
  const [bottomTab, setBottomTab] = useState<BottomTab>("cache");
  const [activeNav, setActiveNav] = useState<number>(0);
  const [advOpen, setAdvOpen] = useState<boolean>(false);
  const [query, setQuery] = useState("");
  const [openRow, setOpenRow] = useState<number | null>(null);

  const handleNavClick = (item: (typeof navItems)[number], index: number) => {
    setActiveNav(index);
    if (item.tab) {
      setBottomTab(item.tab);
    }
    if (item.openAdvanced) {
      setAdvOpen(true);
    }
    const el = document.getElementById(item.targetId);
    if (el) {
      el.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  };

  if (stats && baseline === null) setBaseline(snapshot(stats));
  if (stats && baseline) {
    if (baseline.swap_mb === null && stats.mem) setBaseline({ ...baseline, swap_mb: stats.mem.swap_used_mb });
    else if (baseline.totals === null && stats.totals) setBaseline({ ...baseline, totals: { ...stats.totals } });
    else if (baseline.mtp === null && stats.mtp) setBaseline({ ...baseline, mtp: mtpCounters(stats.mtp) });
  }

  const sessionMode = period === "Session";
  const base = sessionMode && baseline ? baseline : ZERO;

  const counters = useMemo(() => {
    if (!stats) return null;
    return {
      requests: Math.max(0, stats.requests - base.requests),
      cancelled: Math.max(0, stats.prefill_cancelled - base.prefill_cancelled),
      prompt: Math.max(0, stats.prompt_tokens - base.prompt_tokens),
      cached: Math.max(0, stats.cached_tokens - base.cached_tokens),
      generated: Math.max(0, stats.generated_tokens - base.generated_tokens),
      hits: Math.max(0, stats.cache.hits - base.hits),
      cold: Math.max(0, stats.cache.cold - base.cold),
    };
  }, [stats, base]);

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

  const plainStepNs = useMemo(() => {
    const r = recent.find((x) => x.mtp_cycles === 0 && x.decode_tokens > 0 && x.decode_ns > 0);
    return r ? r.decode_ns / r.decode_tokens : null;
  }, [recent]);

  const ctxPct = pct(stats?.live_tokens, stats?.ctx_size);
  const metalPct = pct(mem?.metal_allocated_mb, mem?.metal_working_set_mb);
  const headroomMb = mem ? Math.max(0, mem.metal_working_set_mb - mem.metal_allocated_mb) : undefined;
  const swapDelta = mem && baseline?.swap_mb !== null && baseline?.swap_mb !== undefined
    ? mem.swap_used_mb - baseline.swap_mb : null;

  const mtpEffective = rate(M?.committed, M?.total_ns);
  const plainTps = plainStepNs ? 1e9 / plainStepNs : null;
  const netVsPlain = mtpEffective !== null && plainTps !== null && plainTps > 0 ? mtpEffective / plainTps - 1 : null;
  const perCycle = (ns: number | undefined) => safeDiv(ns, M?.cycles);
  const mtpActive = !!(mtp?.enabled && mtp.active);
  const mtpGated = !!(mtp?.enabled && !mtp.active);

  const lastDecodeTps = last ? decodeTps(last) : null;
  const lastCompute = last ? computeTps(last) : null;
  const msPerTok = safeDiv(last?.decode_ns, last?.decode_tokens);
  const remaining = stats ? Math.max(0, stats.ctx_size - stats.live_tokens) : undefined;
  const turnNs = last ? last.prompt_ns + last.first_token_ns + last.decode_ns : undefined;
  const storeAlarm = !!last && last.store_ns > 1e9;

  const isHealthy = !error && failures === 0;
  const statusText = error ? "Unreachable" : !stats ? "Connecting" : stats.busy ? "Generating" : "Healthy";

  // Build Cache Paths rows from live stats
  const cacheRows = useMemo(() => {
    const defaultPaths = [
      { path: "Memory · token", requests: counters ? counters.hits : 6, tone: "primary" as const },
      { path: "Memory · text", requests: 0, tone: undefined },
      { path: "Thinking visible", requests: 0, tone: undefined },
      { path: "Tool visible", requests: 0, tone: undefined },
      { path: "Responses visible", requests: 0, tone: undefined },
      { path: "Responses tool output", requests: 0, tone: undefined },
      { path: "Anthropic tool output", requests: 0, tone: undefined },
      { path: "Disk · text", requests: 0, tone: undefined },
      { path: "Cold (no cache)", requests: counters ? counters.cold : 1, tone: "muted" as const },
    ];
    const total = defaultPaths.reduce((a, b) => a + b.requests, 0) || 1;
    return defaultPaths.map((r) => ({
      ...r,
      share: Math.round((r.requests * 100) / total),
    }));
  }, [counters]);

  const filteredCacheRows = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return cacheRows;
    return cacheRows.filter((row) => row.path.toLowerCase().includes(q));
  }, [cacheRows, query]);

  const filteredRecent = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return recent;
    return recent.filter((r) =>
      r.kind.toLowerCase().includes(q) ||
      r.finish.toLowerCase().includes(q) ||
      r.source.toLowerCase().includes(q)
    );
  }, [recent, query]);

  const endpointUrl = typeof window !== "undefined"
    ? `${window.location.protocol}//${window.location.hostname}:${window.location.port || "8000"}`
    : "http://127.0.0.1:8000";

  return (
    <main className="canvas">
      <section className="shell">
        {/* ── Topbar ── */}
        <header className="topbar">
          <div className="brandCluster">
            <div className="logo">ds4</div>
            <div className="brand">ds4-server</div>
            <Pill icon="dot" text={statusText} success={isHealthy && !!stats} />
          </div>

          <div className="runtimeMeta">
            <Pill icon="cube" text={stats?.model ?? "DeepSeek V4 Flash Vision Experimental"} />
            <Pill icon="dot" text={`up ${duration(stats?.uptime_s)}`} success />
            <a href="/stats" className="pill" target="_blank" rel="noreferrer">
              <Icon name="link" />
              <span>{endpointUrl}</span>
            </a>
          </div>
        </header>

        <div className="body">
          {/* ── Sidebar ── */}
          <aside className="sidebar" aria-label="Dashboard navigation">
            {navItems.map((item, i) => (
              <button
                key={item.name}
                className={`navButton ${activeNav === i ? "navButtonActive" : ""}`}
                aria-label={item.label}
                title={item.label}
                type="button"
                onClick={() => { handleNavClick(item, i); }}
              >
                <Icon name={item.name} />
                {item.name === "bell" && (failures > 0 || (counters?.cancelled ?? 0) > 0 || storeAlarm) && (
                  <span className="notificationDot" />
                )}
                <span className="navTooltip">{item.label}</span>
              </button>
            ))}
          </aside>

          {/* ── Main Content ── */}
          <div className="content">
            {/* Hero Section */}
            <section className="hero" id="sec-hero">
              <div>
                <div className="eyebrow">Monitor and diagnose local inference</div>
                <h1>Server Dashboard</h1>
              </div>

              <div className="heroControls">
                <div className="periodTabs" role="tablist" aria-label="Time range">
                  {(["Session", "All-Time"] as Period[]).map((item) => (
                    <button
                      key={item}
                      type="button"
                      className={`periodButton ${period === item ? "periodButtonActive" : ""}`}
                      onClick={() => { setPeriod(item); }}
                      role="tab"
                      aria-selected={period === item}
                    >
                      {item}
                    </button>
                  ))}
                  <button
                    type="button"
                    className="periodButton"
                    onClick={() => { if (stats) setBaseline(snapshot(stats)); }}
                    disabled={!stats}
                    title="Reset session counters"
                  >
                    ↺ Clear
                  </button>
                </div>

                <label className="search">
                  <Icon name="search" />
                  <input
                    value={query}
                    onChange={(e: ChangeEvent<HTMLInputElement>) => { setQuery(e.target.value); }}
                    placeholder="Search metrics or requests"
                    aria-label="Search metrics or requests"
                  />
                </label>
              </div>
            </section>

            {error && <div className="banner error">Cannot reach /stats: {error} — retrying every second.</div>}

            {/* ── Top Grid (4 Primary Cards) ── */}
            <section className="topGrid" id="sec-performance">
              <MetricCard
                title="Decode"
                value={tps(lastDecodeTps).replace(" tok/s", "")}
                unit="tok/s"
                subValue={msPerTok != null ? `${fixed(msPerTok / 1e6, 1)} ms/token` : undefined}
                accent="blue"
                sparkKind="decode"
                sparkData={decodeSeries}
                footerLeft={["5m EMA", tps(decodeEma).replace(" tok/s", "")]}
                footerRight={["Median", tps(decodeMedian).replace(" tok/s", "")]}
              />
              <MetricCard
                title="Prefill"
                value={tps(lastCompute).replace(" tok/s", "")}
                unit="tok/s"
                accent="green"
                sparkKind="prefill"
                sparkData={computeSeries}
                footerLeft={["Fresh", num(last?.fresh_tokens ?? T?.prefill_fresh_tokens)]}
                footerRight={["Reuse", pctStr(last?.cached_tokens ?? counters?.cached, last?.prompt_tokens ?? counters?.prompt, 1)]}
              />
              <ProgressCard
                title="Context in use"
                value={`${fixed(ctxPct, 1)}%`}
                progress={ctxPct}
                icon="layers"
                left={["Live tokens", compact(stats?.live_tokens)]}
                right={["Window", compact(stats?.ctx_size)]}
                note={`Remaining ${compact(remaining)}`}
              />
              <ProgressCard
                title="Metal memory"
                value={mem ? `${fixed(metalPct, 0)}%` : "0%"}
                progress={metalPct}
                icon="cpu"
                left={["Allocated", gib(mem?.metal_allocated_mb)]}
                right={["Limit", gib(mem?.metal_working_set_mb)]}
                note={`Headroom ${gib(headroomMb)} · Swap Δ ${gibDelta(swapDelta)}`}
                hot={metalPct > 90}
              />
            </section>

            {/* ── Middle Grid (3 Info Cards) ── */}
            <section className="middleGrid">
              <InfoCard id="sec-serving" title="Serving" actionIcon="play">
                <StatRow icon="play" label="State">
                  <span className={stats?.busy ? "badge" : "successChip"}>{statusText}</span>
                </StatRow>
                <StatRow icon="queue" label="Queue depth" value={num(stats?.queue_depth)} />
                <StatRow icon="users" label="Connected clients" value={num(stats?.clients)} />
                <StatRow icon="clock" label="Requests served" value={num(counters?.requests)} />
                <StatRow icon="database" label="Last TTFT" value={ms(last?.first_token_ns)} />
                <StatRow icon="chart" label="Last turn" value={turnNs != null ? sec(turnNs) : DASH} />
              </InfoCard>

              <InfoCard
                id="sec-tokens"
                title="Tokens / KV"
                actionIcon="doc"
                badge={`${pctStr(counters?.cached, counters?.prompt, 1)} prefill saved`}
              >
                <StatRow icon="doc" label="Prompt" value={num(counters?.prompt)} />
                <StatRow icon="cube" label="Served from cache" value={num(counters?.cached)} />
                <StatRow icon="play" label="Fresh computed" value={num(T?.prefill_fresh_tokens)} />
                <StatRow icon="clock" label="Generated" value={num(T?.decode_tokens ?? counters?.generated)} />
                <StatRow icon="chart" label="Prefill saved" value={pctStr(counters?.cached, counters?.prompt, 1)} />
                <StatRow icon="database" label="Live KV" value={`${compact(stats?.live_tokens)} / ${compact(stats?.ctx_size)}`} />
              </InfoCard>

              <InfoCard id="sec-anomalies" title="Anomalies" actionIcon="shield">
                <HealthRow label="Queue rejected" value={num(counters?.cancelled)} warn={(counters?.cancelled ?? 0) > 0} />
                <HealthRow label="Dropped on disconnect" value="0" />
                <HealthRow label="Prefill cancelled" value={num(counters?.cancelled)} warn={(counters?.cancelled ?? 0) > 0} />
                <HealthRow label="Swap growth" value={gibDelta(swapDelta)} warn={swapDelta != null && swapDelta > 50} />
                <HealthRow label="Checkpoint stalls" value={storeAlarm ? "1" : "0"} warn={storeAlarm} />
                <div className="cardFooterNote">
                  {(counters?.cancelled ?? 0) === 0 && !storeAlarm ? "No anomalies since start" : "Anomalies reported"}
                </div>
              </InfoCard>
            </section>

            {/* ── System, Speculative MTP & Checkpoint Grid ── */}
            <section className="middleGrid">
              <InfoCard id="sec-system" title="System Telemetry" actionIcon="cpu">
                <StatRow icon="chart" label="Thermal" value={mem?.thermal ?? DASH} />
                <StatRow icon="shield" label="Memory pressure" value={mem?.pressure ?? DASH} />
                <StatRow icon="database" label="Swap used" value={gib(mem?.swap_used_mb ?? stats?.swap_used_mb)} />
                <StatRow icon="cpu" label="CPU (single core)" value={rates ? `${fixed(rates.cpuPct, 1)}%` : DASH} />
                <StatRow icon="clock" label="Page-ins" value={rates ? `${fixed(rates.pageinsPerSec, 1)}/s` : DASH} />
                <StatRow icon="terminal" label="Tensor route" value={stats?.tensor_route ?? DASH} />
                <StatRow icon="users" label="Slots busy" value={stats ? `${String(stats.slots.filter((s) => s.busy).length)} / ${String(stats.slot_count)}` : DASH} />
              </InfoCard>

              <InfoCard
                id="sec-mtp"
                title="Speculative / MTP"
                actionIcon="terminal"
                badge={mtpActive ? "Active" : mtpGated ? "Gated Off" : "Disabled"}
              >
                {mtpActive ? (
                  <>
                    <StatRow icon="chart" label="Net vs plain" value={signedPct(netVsPlain)} />
                    <StatRow icon="play" label="Acceptance" value={(() => {
                      const a = safeDiv(M?.accepted, M?.cycles);
                      return a != null ? `${fixed(a * 100, 0)}%` : DASH;
                    })()} />
                    <StatRow icon="clock" label="Committed / cycle" value={fixed(safeDiv(M?.committed, M?.cycles), 2)} />
                    <StatRow icon="database" label="Cycle time" value={ms(safeDiv(M?.total_ns, M?.cycles))} />
                    <StatRow icon="cube" label="Draft / Verify" value={`${ms(perCycle(M?.draft_ns))} / ${ms(perCycle(M?.verify_ns))}`} />
                  </>
                ) : (
                  <>
                    <StatRow icon="terminal" label="Status" value={mtpGated ? `ctx ${compact(mtp.pos)} > max ${compact(mtp.max_ctx)}` : "Inactive"} />
                    <StatRow icon="chart" label="Plain decode" value={tps(plainTps)} />
                    <StatRow icon="clock" label="Context pos" value={num(mtp?.pos)} />
                    <StatRow icon="doc" label="Ceiling" value={mtp && mtp.max_ctx > 0 ? num(mtp.max_ctx) : "none"} />
                  </>
                )}
              </InfoCard>

              <InfoCard
                id="sec-checkpoint"
                title="Checkpoint KV"
                actionIcon="database"
                badge={T && T.checkpoint_saves > 0 ? `${num(T.checkpoint_saves)} saves` : "Idle"}
              >
                <StatRow icon="database" label="Saves" value={num(T?.checkpoint_saves ?? 0)} />
                <StatRow icon="clock" label="Avg save duration" value={ms(safeDiv(T?.checkpoint_save_ns, T?.checkpoint_saves))} />
                <StatRow icon="cube" label="Restores" value={num(T?.checkpoint_restores ?? 0)} />
                <StatRow icon="doc" label="Data written" value={mb(T?.checkpoint_save_bytes)} />
                <StatRow icon="database" label="KV SSD" value={kv?.enabled ? `${gib(kv.used_mb)} / ${gib(kv.budget_mb)}` : "off"} />
              </InfoCard>
            </section>

            {/* ── Bottom Section (Cache paths / Recent requests tabs) ── */}
            <section className="cacheCard" id="sec-bottom">
              <div className="cacheHeader">
                <div className="cardTitleWithIcon">
                  <span className="titleIcon"><Icon name={bottomTab === "cache" ? "database" : "doc"} /></span>
                  <h2>{bottomTab === "cache" ? "Cache paths" : "Recent requests"}</h2>
                </div>

                <div className="infoHeaderRight">
                  <div className="viewTabs">
                    <button
                      type="button"
                      className={`viewTabBtn ${bottomTab === "cache" ? "viewTabBtnActive" : ""}`}
                      onClick={() => { setBottomTab("cache"); }}
                    >
                      Cache paths
                    </button>
                    <button
                      type="button"
                      className={`viewTabBtn ${bottomTab === "requests" ? "viewTabBtnActive" : ""}`}
                      onClick={() => { setBottomTab("requests"); }}
                    >
                      Recent requests ({n})
                    </button>
                  </div>
                  <button className="roundAction" type="button" aria-label="Toggle details">
                    <Icon name="arrow" />
                  </button>
                </div>
              </div>

              {bottomTab === "cache" ? (
                <div className="cacheTable">
                  <div className="cacheRow cacheHead">
                    <span>Path</span>
                    <span>Requests</span>
                    <span>Share</span>
                    <span />
                  </div>

                  {filteredCacheRows.map((row) => (
                    <div className="cacheRow" key={row.path}>
                      <span className="cachePath">{row.path}</span>
                      <span>{row.requests}</span>
                      <span>{row.share}%</span>
                      <div className="barTrack">
                        <div
                          className={`barFill ${
                            row.tone === "primary" ? "barPrimary" :
                            row.tone === "muted" ? "barMuted" : ""
                          }`}
                          style={{ width: `${String(row.share)}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="tableWrap">
                  <div className="reqRow reqHead">
                    <span>Time</span>
                    <span>Context</span>
                    <span>Fresh</span>
                    <span>Reuse</span>
                    <span>Prompt</span>
                    <span>TTFT</span>
                    <span>Out</span>
                    <span>Decode</span>
                    <span>MTP</span>
                    <span>Finish</span>
                  </div>

                  {filteredRecent.length === 0 ? (
                    <div style={{ textAlign: "center", padding: "18px 0", color: "#69747e" }}>
                      No requests match filter
                    </div>
                  ) : (
                    filteredRecent.map((r) => (
                      <div key={r.seq}>
                        <div
                          className={`reqRow ${openRow === r.seq ? "reqRowOpen" : ""}`}
                          onClick={() => { setOpenRow(openRow === r.seq ? null : r.seq); }}
                        >
                          <span>{clockOf(stats, r)}</span>
                          <span>{compact(r.pos)}</span>
                          <span>{num(r.fresh_tokens)}</span>
                          <span>{pctStr(r.cached_tokens, r.prompt_tokens, 0)}</span>
                          <span>{fixed(r.prompt_ns / 1e9, 2)}s</span>
                          <span>{msNum(r.first_token_ns)}</span>
                          <span>{num(r.decode_tokens)}</span>
                          <span>{fixed(decodeTps(r), 1)}</span>
                          <span>{r.mtp_cycles > 0 ? fixed(safeDiv(r.mtp_committed, r.mtp_cycles), 1) : "off"}</span>
                          <span>{r.finish}</span>
                        </div>

                        {openRow === r.seq && (
                          <div className="reqDetail">
                            <dl className="detailGrid">
                              <div><dt>Seq</dt><dd>#{num(r.seq)}</dd></div>
                              <div><dt>Prompt tokens</dt><dd>{num(r.prompt_tokens)}</dd></div>
                              <div><dt>Cached tokens</dt><dd>{num(r.cached_tokens)}</dd></div>
                              <div><dt>Compute rate</dt><dd>{tps(computeTps(r))}</dd></div>
                              <div><dt>Lookup time</dt><dd>{ms(r.lookup_ns)}</dd></div>
                              <div><dt>Restore time</dt><dd>{ms(r.restore_ns)}</dd></div>
                              <div><dt>Cold time</dt><dd>{ms(r.cold_ns)}</dd></div>
                              <div><dt>Fresh prefill</dt><dd>{ms(r.prefill_ns)}</dd></div>
                              <div><dt>KV save time</dt><dd>{ms(r.store_ns)}</dd></div>
                              <div><dt>Source</dt><dd>{shortSource(r.source)}</dd></div>
                              <div><dt>MTP cycles</dt><dd>{num(r.mtp_cycles)}</dd></div>
                              <div><dt>MTP committed</dt><dd>{num(r.mtp_committed)}</dd></div>
                            </dl>
                          </div>
                        )}
                      </div>
                    ))
                  )}
                </div>
              )}
            </section>

            {/* ── Advanced Diagnostics & Footer ── */}
            <details
              className="advDetails"
              id="sec-advanced"
              open={advOpen}
              onToggle={(e) => { setAdvOpen(e.currentTarget.open); }}
            >
              <summary>Advanced Runtime & Model Cache Diagnostics ▸</summary>
              <div className="advBody">
                <dl className="detailGrid">
                  <div><dt>Physical RSS</dt><dd>{gib(mem?.rss_mb ?? stats?.rss_mb)}</dd></div>
                  <div><dt>Peak footprint</dt><dd>{gib(mem?.peak_footprint_mb)}</dd></div>
                  <div><dt>Metal working set</dt><dd>{gib(mem?.metal_working_set_mb)}</dd></div>
                  <div><dt>Disk read throughput</dt><dd>{mbps(rates?.diskReadBps)}</dd></div>
                  <div><dt>Disk write throughput</dt><dd>{mbps(rates?.diskWriteBps)}</dd></div>
                  <div><dt>KV cache directory</dt><dd style={{ fontSize: "11px" }}>{kv?.enabled ? kv.dir : "none"}</dd></div>
                  <div><dt>Model weights path</dt><dd style={{ fontSize: "11px" }}>{stats?.model_path ?? DASH}</dd></div>
                </dl>
              </div>
            </details>

            <footer className="bottomSection" id="sec-footer">
              <span>polling /stats every 1s</span>
              <div>
                <a href="/stats" target="_blank" rel="noreferrer">/stats</a>
                <a href="/health" target="_blank" rel="noreferrer">/health</a>
                <a href="/v1/models" target="_blank" rel="noreferrer">/v1/models</a>
              </div>
            </footer>
          </div>
        </div>
      </section>
    </main>
  );
}

/* ── Sub-components matching ds4-dashboard-design ── */

function Pill({
  icon,
  text,
  success = false,
}: {
  icon: IconName;
  text: string;
  success?: boolean | undefined;
}) {
  return (
    <div className="pill">
      <Icon name={icon} success={success} />
      <span>{text}</span>
    </div>
  );
}

function MetricCard({
  title,
  value,
  unit,
  subValue,
  accent,
  sparkKind,
  sparkData,
  footerLeft,
  footerRight,
}: {
  title: string;
  value: string;
  unit: string;
  subValue?: string | undefined;
  accent: "blue" | "green";
  sparkKind: "decode" | "prefill";
  sparkData: number[];
  footerLeft: [string, string];
  footerRight: [string, string];
}) {
  return (
    <article className="metricCard">
      <div className="metricHeader">
        <h2>{title}</h2>
        <button className="roundAction" type="button" aria-label={`Open ${title} details`}>
          <Icon name="arrow" />
        </button>
      </div>
      <div className="metricValue">
        <strong>{value}</strong>
        <span>{unit}</span>
        {subValue && <span style={{ fontSize: "11px", color: "#69747e", marginLeft: "auto", alignSelf: "center" }}>{subValue}</span>}
      </div>
      <DynamicSparkline kind={sparkKind} accent={accent} data={sparkData} />
      <div className="metricFooter">
        <MiniStat label={footerLeft[0]} value={footerLeft[1]} />
        <MiniStat label={footerRight[0]} value={footerRight[1]} align="right" />
      </div>
    </article>
  );
}

function ProgressCard({
  title,
  value,
  progress,
  icon,
  left,
  right,
  note,
  hot = false,
}: {
  title: string;
  value: string;
  progress: number;
  icon: IconName;
  left: [string, string];
  right: [string, string];
  note?: string | undefined;
  hot?: boolean | undefined;
}) {
  return (
    <article className="metricCard">
      <div className="metricHeader">
        <h2>{title}</h2>
        <button className="roundAction" type="button" aria-label={`Open ${title} details`}>
          <Icon name={icon} />
        </button>
      </div>
      <div className="metricValue">
        <strong>{value}</strong>
      </div>
      <div className="progressTrack">
        <div
          className={`progressFill ${hot ? "progressFillHot" : ""}`}
          style={{ width: `${String(Math.min(100, Math.max(0, progress)))}%` }}
        />
      </div>
      <div className="metricFooter">
        <MiniStat label={left[0]} value={left[1]} />
        <MiniStat label={right[0]} value={right[1]} align="right" />
      </div>
      {note && <div className="progressNote">{note}</div>}
    </article>
  );
}

function InfoCard({
  id,
  title,
  actionIcon,
  badge,
  children,
}: {
  id?: string | undefined;
  title: string;
  actionIcon: IconName;
  badge?: string | undefined;
  children: ReactNode;
}) {
  return (
    <article className="infoCard" id={id}>
      <div className="infoHeader">
        <h2>{title}</h2>
        <div className="infoHeaderRight">
          {badge && <span className="badge">{badge}</span>}
          <button className="roundAction" type="button" aria-label={`Open ${title}`}>
            <Icon name={actionIcon} />
          </button>
        </div>
      </div>
      <div className="infoRows">{children}</div>
    </article>
  );
}

function StatRow({
  icon,
  label,
  value,
  children,
}: {
  icon: IconName;
  label: string;
  value?: string | undefined;
  children?: ReactNode | undefined;
}) {
  return (
    <div className="statRow">
      <span className="statLabel">
        <Icon name={icon} />
        {label}
      </span>
      <span className="statValue">{children ?? value}</span>
    </div>
  );
}

function HealthRow({ label, value, warn = false }: { label: string; value: string; warn?: boolean | undefined }) {
  return (
    <div className="healthRow">
      <span className="healthLabel">
        <span className={warn ? "healthWarn" : "healthCheck"}>{warn ? "!" : "✓"}</span>
        {label}
      </span>
      <span>{value}</span>
    </div>
  );
}

function MiniStat({
  label,
  value,
  align = "left",
}: {
  label: string;
  value: string;
  align?: "left" | "right" | undefined;
}) {
  return (
    <div className={`miniStat ${align === "right" ? "alignRight" : ""}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function DynamicSparkline({
  kind,
  accent,
  data,
}: {
  kind: "decode" | "prefill";
  accent: "blue" | "green";
  data: number[];
}) {
  const color = accent === "blue" ? "#2589ff" : "#2ab46c";

  // If we have actual telemetry points, build smooth curve
  const path = useMemo(() => {
    if (data.length < 2) {
      return kind === "decode"
        ? "M3 50 C12 47,12 34,20 38 C26 43,29 22,40 27 C52 31,58 42,70 42 C84 42,87 29,101 32 C114 34,119 43,132 42 C144 42,150 34,162 35 C176 37,181 24,194 26 C208 26,214 38,225 34"
        : "M3 48 C10 48,11 35,20 38 C28 43,32 27,41 34 C50 39,54 29,65 31 C76 33,79 18,89 23 C99 30,102 39,113 35 C124 30,128 41,138 35 C148 29,150 22,158 29 C166 39,170 18,179 24 C188 30,191 39,202 35 C211 32,215 41,225 20";
    }
    const max = Math.max(1, ...data);
    const min = Math.min(...data);
    const range = max - min || 1;
    const w = 222;
    const h = 48;
    const step = w / Math.max(1, data.length - 1);

    const pts = data.map((v, i) => ({
      x: 3 + i * step,
      y: 56 - ((v - min) / range) * h,
    }));

    const first = pts[0];
    if (!first) return "";
    let d = `M ${first.x.toFixed(1)} ${first.y.toFixed(1)}`;
    for (let i = 0; i < pts.length - 1; i++) {
      const p0 = pts[i];
      const p1 = pts[i + 1];
      if (!p0 || !p1) continue;
      const mx = (p0.x + p1.x) / 2;
      d += ` C ${mx.toFixed(1)} ${p0.y.toFixed(1)}, ${mx.toFixed(1)} ${p1.y.toFixed(1)}, ${p1.x.toFixed(1)} ${p1.y.toFixed(1)}`;
    }
    return d;
  }, [data, kind]);

  return (
    <svg className="sparkline" viewBox="0 0 228 62" role="img" aria-label={`${kind} trend`}>
      <defs>
        <linearGradient id={`${kind}-fill`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity=".18" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>
      <path d={`${path} L225 60 L3 60 Z`} fill={`url(#${kind}-fill)`} />
      <path
        d={path}
        fill="none"
        stroke={color}
        strokeWidth="1.8"
        strokeLinecap="round"
      />
    </svg>
  );
}

type IconName =
  | NavIcon
  | "dot"
  | "link"
  | "search"
  | "layers"
  | "cpu"
  | "play"
  | "queue"
  | "clock"
  | "shield"
  | "arrow";

function Icon({ name, success = false }: { name: IconName; success?: boolean | undefined }) {
  if (name === "dot") {
    return <span className={`dot ${success ? "dotSuccess" : ""}`} />;
  }

  const common = {
    width: 18,
    height: 18,
    viewBox: "0 0 24 24",
    fill: "none",
    stroke: "currentColor",
    strokeWidth: 1.7,
    strokeLinecap: "round" as const,
    strokeLinejoin: "round" as const,
    "aria-hidden": true,
  };

  const paths: Record<string, ReactNode> = {
    grid: <><rect x="4" y="4" width="6" height="6" rx="1.4"/><rect x="14" y="4" width="6" height="6" rx="1.4"/><rect x="4" y="14" width="6" height="6" rx="1.4"/><rect x="14" y="14" width="6" height="6" rx="1.4"/></>,
    bars: <><path d="M5 19V10"/><path d="M9.5 19V5"/><path d="M14.5 19v-7"/><path d="M19 19V8"/></>,
    doc: <><path d="M7 3h7l4 4v14H7z"/><path d="M14 3v5h5"/><path d="M10 12h5"/><path d="M10 16h5"/></>,
    cube: <><path d="m12 3 8 4.5v9L12 21l-8-4.5v-9z"/><path d="m4.5 7.7 7.5 4.2 7.5-4.2"/><path d="M12 12v9"/></>,
    terminal: <><path d="m5 8 4 4-4 4"/><path d="M12 17h7"/></>,
    database: <><ellipse cx="12" cy="5" rx="7" ry="3"/><path d="M5 5v6c0 1.7 3.1 3 7 3s7-1.3 7-3V5"/><path d="M5 11v6c0 1.7 3.1 3 7 3s7-1.3 7-3v-6"/></>,
    chart: <><path d="M4 18 9 12l4 3 7-9"/><path d="M4 4v16h16"/></>,
    bell: <><path d="M6 17h12l-1.2-2.3V10a4.8 4.8 0 0 0-9.6 0v4.7z"/><path d="M10 20h4"/></>,
    users: <><circle cx="9" cy="8" r="3"/><path d="M3.5 19c.7-3.4 2.6-5 5.5-5s4.8 1.6 5.5 5"/><path d="M16 7.5a2.5 2.5 0 0 1 0 5"/><path d="M16 14.5c2.5.2 4 1.7 4.5 4.5"/></>,
    settings: <><circle cx="12" cy="12" r="3"/><path d="M19 13.5v-3l-2-.7-.8-1.8.9-1.9L15 4l-1.9.9-1.8-.8L10.5 2h-3l-.7 2-1.8.8L3.1 4 1 6.1 1.9 8 1.1 9.8 0 10.5v3l2 .7.8 1.8-.9 1.9L4 20l1.9-.9 1.8.8.8 2.1h3l.7-2 1.8-.8 1.9.9L18 18l-.9-1.9.8-1.8z" transform="translate(2 0) scale(.83)"/></>,
    help: <><circle cx="12" cy="12" r="9"/><path d="M9.8 9a2.4 2.4 0 1 1 3.7 2c-1 .7-1.5 1.1-1.5 2.5"/><path d="M12 17h.01"/></>,
    link: <><path d="M10 13a4 4 0 0 0 5.7 0l2.2-2.2a4 4 0 0 0-5.7-5.7L11 6.3"/><path d="M14 11a4 4 0 0 0-5.7 0l-2.2 2.2a4 4 0 0 0 5.7 5.7l1.2-1.2"/></>,
    search: <><circle cx="10.5" cy="10.5" r="6.5"/><path d="m15.5 15.5 5 5"/></>,
    layers: <><path d="m12 3 8 4-8 4-8-4z"/><path d="m4 12 8 4 8-4"/><path d="m4 17 8 4 8-4"/></>,
    cpu: <><rect x="7" y="7" width="10" height="10" rx="2"/><path d="M9 1v3M15 1v3M9 20v3M15 20v3M1 9h3M1 15h3M20 9h3M20 15h3"/><rect x="10" y="10" width="4" height="4" rx=".5"/></>,
    play: <path d="m9 7 8 5-8 5z"/>,
    queue: <><path d="M7 7h10M7 12h10M7 17h10"/><circle cx="4" cy="7" r=".7" fill="currentColor" stroke="none"/><circle cx="4" cy="12" r=".7" fill="currentColor" stroke="none"/><circle cx="4" cy="17" r=".7" fill="currentColor" stroke="none"/></>,
    clock: <><circle cx="12" cy="12" r="8"/><path d="M12 7v5l3 2"/></>,
    shield: <><path d="M12 3 19 6v5c0 4.6-2.5 7.8-7 10-4.5-2.2-7-5.4-7-10V6z"/></>,
    arrow: <><path d="M7 17 17 7"/><path d="M9 7h8v8"/></>,
  };

  return <svg {...common}>{paths[name]}</svg>;
}

export default App;
