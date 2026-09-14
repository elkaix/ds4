import { useEffect, useRef } from "react";
import { WINDOW } from "./useStats";

interface Props {
  data: number[];
  color: string;
  height?: number;
  /** x-axis capacity, so a partially-filled series doesn't rescale every tick. */
  span?: number;
}

export function Sparkline({ data, color, height = 64, span = WINDOW }: Props) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    const draw = () => {
      const dpr = window.devicePixelRatio || 1;
      const w = c.clientWidth, h = c.clientHeight;
      if (w <= 0 || h <= 0) return;
      if (c.width !== w * dpr || c.height !== h * dpr) { c.width = w * dpr; c.height = h * dpr; }
      const g = c.getContext("2d");
      if (!g) return;
      g.setTransform(dpr, 0, 0, dpr, 0, 0);
      g.clearRect(0, 0, w, h);
      const pts = data.filter((v) => Number.isFinite(v));
      if (pts.length < 2) return;
      const max = Math.max(1, ...pts);
      const step = w / Math.max(1, span - 1);
      const off = Math.max(0, w - (pts.length - 1) * step);
      g.beginPath();
      pts.forEach((v, i) => {
        const x = off + i * step, y = h - 3 - (v / max) * (h - 6);
        i ? g.lineTo(x, y) : g.moveTo(x, y);
      });
      g.strokeStyle = color; g.lineWidth = 1.75; g.lineJoin = "round"; g.stroke();
      g.lineTo(off + (pts.length - 1) * step, h); g.lineTo(off, h); g.closePath();
      const grad = g.createLinearGradient(0, 0, 0, h);
      grad.addColorStop(0, color + "33"); grad.addColorStop(1, color + "00");
      g.fillStyle = grad; g.fill();
    };
    draw();
    const ro = new ResizeObserver(draw);
    ro.observe(c);
    return () => ro.disconnect();
  }, [data, color, span]);
  return <canvas ref={ref} className="spark" style={{ height }} />;
}
