import { useEffect, useRef } from "react";
import { WINDOW } from "./useStats";

interface Props { data: number[]; color: string; height?: number }

export function Sparkline({ data, color, height = 64 }: Props) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    const draw = () => {
      const dpr = window.devicePixelRatio || 1;
      const w = c.clientWidth, h = c.clientHeight;
      if (c.width !== w * dpr || c.height !== h * dpr) { c.width = w * dpr; c.height = h * dpr; }
      const g = c.getContext("2d");
      if (!g) return;
      g.setTransform(dpr, 0, 0, dpr, 0, 0);
      g.clearRect(0, 0, w, h);
      if (data.length < 2) return;
      const max = Math.max(1, ...data);
      const step = w / (WINDOW - 1);
      const off = w - (data.length - 1) * step;
      g.beginPath();
      data.forEach((v, i) => {
        const x = off + i * step, y = h - 3 - (v / max) * (h - 6);
        i ? g.lineTo(x, y) : g.moveTo(x, y);
      });
      g.strokeStyle = color; g.lineWidth = 1.75; g.lineJoin = "round"; g.stroke();
      g.lineTo(w, h); g.lineTo(off, h); g.closePath();
      const grad = g.createLinearGradient(0, 0, 0, h);
      grad.addColorStop(0, color + "33"); grad.addColorStop(1, color + "00");
      g.fillStyle = grad; g.fill();
    };
    draw();
    const ro = new ResizeObserver(draw);
    ro.observe(c);
    return () => ro.disconnect();
  }, [data, color]);
  return <canvas ref={ref} className="spark" style={{ height }} />;
}
