import { useMemo, useState } from "react";
import type { ChangeEvent, ReactNode } from "react";
import styles from "./dashboard.module.css";
import { useStats } from "./useStats";
import type { Sample } from "./useStats";
import type { Stats, RecentRequest } from "./types";
import { compact, duration, fixed, gib, num, ratioStr, DASH } from "./format";

/** Sparkline horizon selector — trims the shared 5-minute sample buffer. */
type Horizon = "1 min" | "5 min";
const HORIZON_SAMPLES: Record<Horizon, number> = { "1 min": 60, "5 min": 300 };

/** This deploy's Metal wired-memory ceiling (iogpu.wired_limit_mb=118000 →
 *  115.2 GiB), set by run-glm-ds4.sh's preflight. The server does not report
 *  the sysctl in /stats, so the dashboard carries the deploy constant. */
const WIRED_LIMIT_MB = 118000;

const ENDPOINT = "http://127.0.0.1:8000";

/** Cache-path rows derived from the server's recent[] ring (last 64 requests),
 *  so requests and shares always come from one consistent window. */
function cacheRowsFrom(recent: RecentRequest[] | undefined, s: Stats | null) {
  const rows: { path: string; requests: number; share: number; tone?: "primary" | "muted" | undefined }[] = [];
  if (recent && recent.length > 0) {
    const counts = new Map<string, number>();
    for (const r of recent) {
      const key = r.source && r.source.length > 0 ? r.source : "unknown";
      counts.set(key, (counts.get(key) ?? 0) + 1);
    }
    const total = recent.length;
    const entries = [...counts.entries()].sort((a, b) => b[1] - a[1]);
    entries.forEach(([path, requests], i) => {
      rows.push({ path, requests, share: (requests / total) * 100, tone: i === 0 ? "primary" : undefined });
    });
    return rows;
  }
  // Older server without recent[]: degrade to the cumulative hit/cold counters.
  const hits = s?.cache?.hits ?? 0;
  const cold = s?.cache?.cold ?? 0;
  const total = hits + cold;
  if (total > 0) {
    rows.push({ path: "Cache hits", requests: hits, share: (hits / total) * 100, tone: "primary" });
    rows.push({ path: "Cold (no cache)", requests: cold, share: (cold / total) * 100, tone: "muted" });
  }
  return rows;
}

/** Normalises a numeric series into an SVG polyline path inside a w×h box,
 *  padded so a flat line still has room. Returns null when there is no data. */
function seriesPath(series: number[], w: number, h: number): string | null {
  if (series.length < 2) return null;
  const max = Math.max(...series);
  const min = Math.min(...series);
  const span = max - min || 1;
  const pad = 4;
  const usable = h - pad * 2;
  const x = (i: number) => (i / (series.length - 1)) * w;
  const y = (v: number) => h - pad - ((v - min) / span) * usable;
  return series.map((v, i) => `${i === 0 ? "M" : "L"}${x(i).toFixed(1)} ${y(v).toFixed(1)}`).join(" ");
}

export default function App() {
  const { stats, error, failures, samples, rates } = useStats();
  const [horizon, setHorizon] = useState<Horizon>("5 min");
  const [query, setQuery] = useState("");

  const win = useMemo(
    () => samples.slice(-HORIZON_SAMPLES[horizon]),
    [samples, horizon],
  );

  const decodeSeries = useMemo(() => win.map((p: Sample) => p.decode), [win]);
  const prefillSeries = useMemo(() => win.map((p: Sample) => p.prefill), [win]);

  const decodePeak = decodeSeries.length ? Math.max(...decodeSeries) : null;
  const decodeAvg = decodeSeries.length ? decodeSeries.reduce((a, b) => a + b, 0) / decodeSeries.length : null;
  const prefillPeak = prefillSeries.length ? Math.max(...prefillSeries) : null;
  const prefillAvg = prefillSeries.length ? prefillSeries.reduce((a, b) => a + b, 0) / prefillSeries.length : null;

  const cacheRows = useMemo(
    () => cacheRowsFrom(stats?.recent, stats ?? null),
    [stats],
  );

  const filteredCacheRows = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return cacheRows;
    return cacheRows.filter((row) => row.path.toLowerCase().includes(q));
  }, [cacheRows, query]);

  const memMb = stats?.mem?.metal_allocated_mb ?? stats?.mem?.footprint_mb ?? stats?.rss_mb ?? null;
  const memPct = memMb === null ? null : Math.min(100, (memMb / WIRED_LIMIT_MB) * 100);
  const headroomMb = memMb === null ? null : WIRED_LIMIT_MB - memMb;

  const ctxPct = stats ? Math.min(100, (stats.live_tokens / Math.max(1, stats.ctx_size)) * 100) : null;

  const busy = stats?.busy === true;
  const alive = stats !== null;
  const stateText = !alive ? "Connecting" : busy ? "Generating" : "Idle";
  const anomalies = stats?.prefill_cancelled ?? 0;

  return (
    <main className={styles.canvas}>
      <section className={styles.shell}>
        <header className={styles.topbar}>
          <div className={styles.brandCluster}>
            <div className={styles.logo}>ds4</div>
            <div className={styles.brand}>ds4-server</div>
            <Pill icon="dot" text={stateText} success={alive} />
          </div>

          <div className={styles.runtimeMeta}>
            <Pill icon="cube" text={stats?.model ?? "loading model…"} />
            {stats?.mtp?.enabled === true && <Pill icon="chart" text={`MTP · ${ratioStr(stats.mtp.max_ctx > 0 ? stats.mtp.pos / stats.mtp.max_ctx : null) ?? ""}`} />}
            <Pill icon="dot" text={stats ? `up ${duration(stats.uptime_s)}` : "up —"} success={alive} />
            <Pill icon="link" text={ENDPOINT} />
          </div>
        </header>

        <div className={styles.body}>
          <aside className={styles.sidebar} aria-label="Dashboard navigation">
            <NavRail failures={failures} error={error} />
          </aside>

          <div className={styles.content}>
            <section className={styles.hero}>
              <div>
                <div className={styles.eyebrow}>Monitor and diagnose local inference</div>
                <h1>Server Dashboard</h1>
              </div>

              <div className={styles.heroControls}>
                <div className={styles.periodTabs} role="tablist" aria-label="Trend window">
                  {(Object.keys(HORIZON_SAMPLES) as Horizon[]).map((item) => (
                    <button
                      key={item}
                      type="button"
                      className={`${styles.periodButton} ${horizon === item ? styles.periodButtonActive : ""}`}
                      onClick={() => setHorizon(item)}
                      role="tab"
                      aria-selected={horizon === item}
                    >
                      {item}
                    </button>
                  ))}
                </div>

                <label className={styles.search}>
                  <Icon name="search" />
                  <input
                    value={query}
                    onChange={(e: ChangeEvent<HTMLInputElement>) => setQuery(e.target.value)}
                    placeholder="Search cache paths"
                    aria-label="Search cache paths"
                  />
                </label>
              </div>
            </section>

            <section className={styles.topGrid}>
              <MetricCard
                title="Decode"
                value={stats ? fixed(stats.last_decode_tps, 1) : DASH}
                unit="tok/s"
                accent="blue"
                series={decodeSeries}
                footerLeft={["Peak", decodePeak === null ? DASH : fixed(decodePeak, 1)]}
                footerRight={[`${horizon} avg`, decodeAvg === null ? DASH : fixed(decodeAvg, 1)]}
              />
              <MetricCard
                title="Prefill"
                value={stats ? fixed(stats.last_prefill_tps, 0) : DASH}
                unit="tok/s"
                accent="green"
                series={prefillSeries}
                footerLeft={["Peak", prefillPeak === null ? DASH : fixed(prefillPeak, 0)]}
                footerRight={[`${horizon} avg`, prefillAvg === null ? DASH : fixed(prefillAvg, 0)]}
              />
              <ProgressCard
                title="Context in use"
                value={ctxPct === null ? DASH : `${fixed(ctxPct, 1)}%`}
                progress={ctxPct ?? 0}
                icon="layers"
                left={["Live tokens", stats ? num(stats.live_tokens) : DASH]}
                right={["Window", stats ? num(stats.ctx_size) : DASH]}
              />
              <ProgressCard
                title="Metal memory"
                value={memPct === null ? DASH : `${fixed(memPct, 1)}%`}
                progress={memPct ?? 0}
                icon="cpu"
                left={["Allocated", gib(memMb)]}
                right={["Limit", gib(WIRED_LIMIT_MB)]}
                note={headroomMb === null ? undefined : `Headroom ${gib(headroomMb)}`}
              />
            </section>

            <section className={styles.middleGrid}>
              <InfoCard title="Serving" actionIcon="play">
                <StatRow icon="play" label="State">
                  <span className={styles.successChip}>{stateText}</span>
                </StatRow>
                <StatRow icon="queue" label="Queue depth" value={stats ? num(stats.queue_depth) : DASH} />
                <StatRow icon="users" label="Connected clients" value={stats ? num(stats.clients) : DASH} />
                <StatRow icon="clock" label="Requests served" value={stats ? num(stats.requests) : DASH} />
                <StatRow icon="cpu" label="Process CPU" value={rates ? `${fixed(rates.cpuPct, 1)}%` : DASH} />
              </InfoCard>

              <InfoCard
                title="Tokens"
                actionIcon="doc"
                badge={stats ? `${ratioStr(safeRatio(stats.cached_tokens, stats.prompt_tokens))} prefill saved` : undefined}
              >
                <StatRow icon="doc" label="Prompt" value={stats ? compact(stats.prompt_tokens) : DASH} />
                <StatRow icon="cube" label="Served from cache" value={stats ? compact(stats.cached_tokens) : DASH} />
                <StatRow icon="clock" label="Generated" value={stats ? compact(stats.generated_tokens) : DASH} />
                <StatRow
                  icon="chart"
                  label="Prefill saved"
                  value={stats ? ratioStr(safeRatio(stats.cached_tokens, stats.prompt_tokens)) : DASH}
                />
              </InfoCard>

              <InfoCard title="Anomalies" actionIcon="shield">
                <HealthRow label="Prefill cancelled" value={stats ? num(stats.prefill_cancelled) : DASH} />
                <HealthRow label="Checkpoint saves" value={stats?.totals ? num(stats.totals.checkpoint_saves) : DASH} />
                <HealthRow label="Swap in use" value={stats?.mem ? gib(stats.mem.swap_used_mb) : DASH} />
                <div className={styles.cardFooterNote}>
                  {anomalies === 0 ? "No anomalies since start" : `${num(anomalies)} prefill cancellation${anomalies === 1 ? "" : "s"}`}
                </div>
              </InfoCard>
            </section>

            <section className={styles.cacheCard}>
              <div className={styles.cacheHeader}>
                <div className={styles.cardTitleWithIcon}>
                  <span className={styles.titleIcon}><Icon name="database" /></span>
                  <h2>Cache paths</h2>
                </div>
                <span className={styles.badge}>
                  {stats?.recent?.length ? `last ${stats.recent.length} requests` : "cumulative"}
                </span>
              </div>

              <div className={styles.cacheTable}>
                <div className={`${styles.cacheRow} ${styles.cacheHead}`}>
                  <span>Path</span>
                  <span>Requests</span>
                  <span>Share</span>
                  <span />
                </div>

                {filteredCacheRows.map((row) => (
                  <div className={styles.cacheRow} key={row.path}>
                    <span className={styles.cachePath}>{row.path}</span>
                    <span>{num(row.requests)}</span>
                    <span>{fixed(row.share, 0)}%</span>
                    <div className={styles.barTrack}>
                      <div
                        className={`${styles.barFill} ${
                          row.tone === "primary" ? styles.barPrimary :
                          row.tone === "muted" ? styles.barMuted : ""
                        }`}
                        style={{ width: `${Math.min(100, row.share)}%` }}
                      />
                    </div>
                  </div>
                ))}
                {filteredCacheRows.length === 0 && (
                  <div className={styles.cacheRow}>
                    <span className={styles.cachePath}>{error ? `stats unreachable: ${error}` : "no requests yet"}</span>
                    <span>{DASH}</span>
                    <span>{DASH}</span>
                    <div className={styles.barTrack} />
                  </div>
                )}
              </div>
            </section>
          </div>
        </div>
      </section>
    </main>
  );
}

function safeRatio(a: number | null | undefined, b: number | null | undefined): number | null {
  if (typeof a !== "number" || typeof b !== "number" || !Number.isFinite(a) || !Number.isFinite(b) || b <= 0) return null;
  return a / b;
}

function NavRail({ failures, error }: { failures: number; error: string | null }) {
  const icons = ["grid", "bars", "doc", "cube", "terminal", "database", "chart", "bell", "users", "settings", "help"] as const;
  return (
    <>
      {icons.map((name, i) => (
        <button
          key={name}
          className={`${styles.navButton} ${i === 0 ? styles.navButtonActive : ""}`}
          aria-label={name}
          type="button"
        >
          <Icon name={name} />
          {name === "bell" && failures > 0 && error !== null && <span className={styles.notificationDot} />}
        </button>
      ))}
    </>
  );
}

function Pill({ icon, text, success = false }: { icon: IconName; text: string; success?: boolean }) {
  return (
    <div className={styles.pill}>
      <Icon name={icon} success={success} />
      <span>{text}</span>
    </div>
  );
}

function MetricCard({
  title,
  value,
  unit,
  accent,
  series,
  footerLeft,
  footerRight,
}: {
  title: string;
  value: string;
  unit: string;
  accent: "blue" | "green";
  series: number[];
  footerLeft: [string, string];
  footerRight: [string, string];
}) {
  return (
    <article className={styles.metricCard}>
      <div className={styles.metricHeader}>
        <h2>{title}</h2>
      </div>
      <div className={styles.metricValue}>
        <strong>{value}</strong>
        <span>{unit}</span>
      </div>
      <Sparkline series={series} accent={accent} label={title.toLowerCase()} />
      <div className={styles.metricFooter}>
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
}: {
  title: string;
  value: string;
  progress: number;
  icon: IconName;
  left: [string, string];
  right: [string, string];
  note?: string | undefined;
}) {
  return (
    <article className={styles.metricCard}>
      <div className={styles.metricHeader}>
        <h2>{title}</h2>
        <span className={styles.titleIcon}><Icon name={icon} /></span>
      </div>
      <div className={styles.metricValue}>
        <strong>{value}</strong>
      </div>
      <div className={styles.progressTrack}>
        <div className={styles.progressFill} style={{ width: `${Math.min(100, Math.max(0, progress))}%` }} />
      </div>
      <div className={styles.metricFooter}>
        <MiniStat label={left[0]} value={left[1]} />
        <MiniStat label={right[0]} value={right[1]} align="right" />
      </div>
      {note && <div className={styles.progressNote}>{note}</div>}
    </article>
  );
}

function InfoCard({
  title,
  actionIcon,
  badge,
  children,
}: {
  title: string;
  actionIcon: IconName;
  badge?: string | undefined;
  children: ReactNode;
}) {
  return (
    <article className={styles.infoCard}>
      <div className={styles.infoHeader}>
        <h2>{title}</h2>
        <div className={styles.infoHeaderRight}>
          {badge && <span className={styles.badge}>{badge}</span>}
          <span className={styles.titleIcon}><Icon name={actionIcon} /></span>
        </div>
      </div>
      <div className={styles.infoRows}>{children}</div>
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
  value?: string;
  children?: ReactNode;
}) {
  return (
    <div className={styles.statRow}>
      <span className={styles.statLabel}>
        <Icon name={icon} />
        {label}
      </span>
      <span className={styles.statValue}>{children ?? value}</span>
    </div>
  );
}

function HealthRow({ label, value }: { label: string; value: string }) {
  return (
    <div className={styles.healthRow}>
      <span className={styles.healthLabel}>
        <span className={styles.healthCheck}>✓</span>
        {label}
      </span>
      <span>{value}</span>
    </div>
  );
}

function MiniStat({ label, value, align = "left" }: { label: string; value: string; align?: "left" | "right" }) {
  return (
    <div className={`${styles.miniStat} ${align === "right" ? styles.alignRight : ""}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function Sparkline({
  series,
  accent,
  label,
}: {
  series: number[];
  accent: "blue" | "green";
  label: string;
}) {
  const W = 228;
  const H = 62;
  const path = seriesPath(series, W, H);
  const color = accent === "blue" ? "#2589ff" : "#2ab46c";
  const gid = `${label}-fill`;
  return (
    <svg className={styles.sparkline} viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${label} trend`}>
      <defs>
        <linearGradient id={gid} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity=".16" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>
      {path && (
        <>
          <path d={`${path} L${W} ${H} L0 ${H} Z`} fill={`url(#${gid})`} />
          <path d={path} fill="none" stroke={color} strokeWidth="1.7" strokeLinecap="round" />
        </>
      )}
    </svg>
  );
}

type IconName =
  | "grid" | "bars" | "doc" | "cube" | "terminal" | "database" | "chart" | "bell"
  | "users" | "settings" | "help"
  | "dot" | "link" | "search" | "layers" | "cpu" | "play" | "queue" | "clock" | "shield" | "arrow";

function Icon({ name, success = false }: { name: IconName; success?: boolean }) {
  if (name === "dot") {
    return <span className={`${styles.dot} ${success ? styles.dotSuccess : ""}`} />;
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
