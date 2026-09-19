import type { ReactNode } from "react";

export type NavIcon =
  | "grid" | "bars" | "doc" | "cube" | "terminal" | "database" | "chart"
  | "bell" | "users" | "settings" | "help";

export type IconName =
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
  | "arrow"
  | "warn";

export function Icon({ name, success = false }: { name: IconName; success?: boolean | undefined }) {
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
    warn: <><path d="M12 4 2.5 20h19z"/><path d="M12 10v4"/><path d="M12 17.5h.01"/></>,
  };

  return <svg {...common}>{paths[name]}</svg>;
}

export function Pill({
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
