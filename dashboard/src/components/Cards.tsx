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
  const N = 32;

  const bars = useMemo(() => {
    if (!data.length) return Array.from({ length: N }, () => 0);
    const out: number[] = [];
    const w = data.length / N;
    for (let i = 0; i < N; i++) {
      const a = Math.floor(i * w);
      const b = Math.max(a + 1, Math.floor((i + 1) * w));
      let s = 0;
      let c = 0;
      for (let j = a; j < b; j++) {
        const v = data[j] ?? 0;
        if (v > 0) {
          s += v;
          c++;
        }
      }
      out.push(c ? s / c : 0);
    }
    return out;
  }, [data]);

  const max = Math.max(1, ...bars);
  const barW = 222 / N;
  const gap = 1.4;

  return (
    <svg className="sparkline" viewBox="0 0 228 62" role="img" aria-label={`${kind} histogram`}>
      {bars.map((v, i) => {
        const h = Math.max(v > 0 ? 2 : 0, (v / max) * 48);
        return (
          <rect
            key={i}
            x={3 + i * barW}
            y={56 - h}
            width={Math.max(0.8, barW - gap)}
            height={h}
            rx={1.2}
            fill={color}
            opacity={v > 0 ? 0.88 : 0.12}
          />
        );
      })}
    </svg>
  );
}
