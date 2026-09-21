import { useMemo, useState } from "react";
import type { ChangeEvent } from "react";
import {
  DASH, compact, duration, fixed, gib, gibDelta, mb, mbps, ms, msNum, num, pct, pctStr,
  percentile, rate, safeDiv, sec, signedPct, tps,
} from "./format";
import type { RecentRequest } from "./types";
import { DECODE_LIVE_MIN, PREFILL_LIVE_MIN, useStats } from "./useStats";
import {
  ZERO, ZERO_MTP, ZERO_TOTALS, clockOf, computeTps, decodeTps, delta, ema, holdSmooth, mtpCounters,
  shortSource, snapshot, weightedPrefillTps,
} from "./lib";
import type { Baseline } from "./lib";
import { Icon } from "./components/Icon";
import type { NavIcon } from "./components/Icon";
import { HealthRow, InfoCard, MetricCard, ProgressCard, StatRow } from "./components/Cards";
import { Pill } from "./components/Icon";

type Period = "Session" | "All-Time";
type BottomTab = "cache" | "requests";

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

  const decodeSeries = useMemo(
    () => holdSmooth(samples.map((s) => (s.busy && s.decodeEma >= DECODE_LIVE_MIN && s.prefillEma < PREFILL_LIVE_MIN ? s.decodeEma : s.decode))),
    [samples],
  );
  const computeSeries = useMemo(
    () => holdSmooth(samples.map((s) => (s.prefillEma >= PREFILL_LIVE_MIN ? s.prefillEma : s.prefill))),
    [samples],
  );
  const decodeEma = useMemo(() => ema(decodeSeries), [decodeSeries]);
  const decodeMedian = useMemo(
    () => percentile(decodeSeries.filter((v) => v > 0), 50)
      ?? percentile(recent.map(decodeTps).filter((v): v is number => v !== null), 50),
    [decodeSeries, recent],
  );

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

  const lastS = samples.at(-1);
  const livePrefill = lastS && lastS.busy && lastS.prefillEma >= PREFILL_LIVE_MIN ? lastS.prefillEma : null;
  const liveDecode = lastS && lastS.busy && livePrefill == null && lastS.decodeEma >= DECODE_LIVE_MIN ? lastS.decodeEma : null;
  const lastDecodeTps = last ? decodeTps(last) : null;
  const decodeHeadline = liveDecode ?? lastDecodeTps ?? stats?.last_decode_tps ?? null;
  const prefillHeadline = livePrefill ?? weightedPrefillTps(T) ?? (last ? computeTps(last) : null);
  const msPerTok = decodeHeadline != null && decodeHeadline > 0 ? 1000 / decodeHeadline : null;
  const remaining = stats ? Math.max(0, stats.ctx_size - stats.live_tokens) : undefined;
  const turnNs = last ? last.prompt_ns + last.first_token_ns + last.decode_ns : undefined;
  const storeAlarm = !!last && last.store_ns > 1e9;

  const isHealthy = !error && failures === 0;
  const statusText = error ? "Unreachable" : !stats ? "Connecting" : livePrefill ? "Prefilling" : stats.busy ? "Generating" : "Healthy";

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
                value={tps(decodeHeadline).replace(" tok/s", "")}
                unit="tok/s"
                subValue={msPerTok != null ? `${fixed(msPerTok, 1)} ms/token` : undefined}
                accent="blue"
                sparkKind="decode"
                sparkData={decodeSeries}
                footerLeft={["5m EMA", tps(decodeEma).replace(" tok/s", "")]}
                footerRight={["Median", tps(decodeMedian).replace(" tok/s", "")]}
              />
              <MetricCard
                title="Prefill"
                value={tps(prefillHeadline).replace(" tok/s", "")}
                unit="tok/s"
                subValue={livePrefill ? "live" : "session"}
                accent="green"
                sparkKind="prefill"
                sparkData={computeSeries}
                footerLeft={["Fresh", num(T?.prefill_fresh_tokens)]}
                footerRight={["Reuse", pctStr(counters?.cached, counters?.prompt, 1)]}
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
                <StatRow icon="clock" label="Avg restore duration" value={ms(safeDiv(T?.checkpoint_restore_ns, T?.checkpoint_restores))} />
                <StatRow icon="warn" label="Cache recoveries" value={T && T.cache_recoveries > 0 ? `${num(T.cache_recoveries)} (live KV lost)` : "0"} />
                <StatRow icon="clock" label="Replay time" value={T && T.replay_ns > 0 ? fixed(T.replay_ns / 1e9, 1) + "s" : "0s"} />
                <StatRow icon="doc" label="Replayed tokens" value={num(T?.replayed_tokens ?? 0)} />
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
                              {r.frontier_loss_tokens > 0 && (
                                <>
                                  <div><dt>Frontier loss</dt><dd>{num(r.frontier_loss_tokens)} tok</dd></div>
                                  <div><dt>Replayed</dt><dd>{num(r.replayed_tokens)} tok / {fixed(r.replay_ns / 1e9, 1)}s</dd></div>
                                </>
                              )}
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

export default App;
