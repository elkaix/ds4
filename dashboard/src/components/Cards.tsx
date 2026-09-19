import { useMemo } from "react";
import type { ReactNode } from "react";
import { Icon } from "./Icon";
import type { IconName } from "./Icon";

export function MetricCard({
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

export function ProgressCard({
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

export function InfoCard({
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

export function StatRow({
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

export function HealthRow({ label, value, warn = false }: { label: string; value: string; warn?: boolean | undefined }) {
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
