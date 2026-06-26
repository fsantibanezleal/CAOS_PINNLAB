import { useEffect, useRef } from "react";

/** The animated double-pendulum linkage. Draws, at the current frame: the RK45 reference linkage (solid) with the
 *  lower bob's traced path, overlaid by the PINN linkage (dashed ghost) — which turns red once the trajectory has
 *  passed the leave-time, making the chaotic divergence literal. Angles are measured from the downward vertical:
 *  a bob at angle θ sits at (ℓ sinθ, ℓ cosθ) with y pointing down. */
export function PendulumCanvas({
  th1Ref, th2Ref, th1Pinn, th2Pinn, frame, l1, l2, leaveIdx, size = 360,
}: {
  th1Ref: number[]; th2Ref: number[]; th1Pinn: number[]; th2Pinn: number[];
  frame: number; l1: number; l2: number; leaveIdx: number; size?: number;
}) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    c.width = size * dpr;
    c.height = size * dpr;
    const ctx = c.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, size, size);

    const cx = size / 2;
    const cy = size / 2;
    const reach = l1 + l2;
    const scale = (size / 2 - 16) / reach;
    const pos = (a: number, b: number) => {
      const x1 = Math.sin(a) * l1;
      const y1 = Math.cos(a) * l1;
      const x2 = x1 + Math.sin(b) * l2;
      const y2 = y1 + Math.cos(b) * l2;
      return {
        p1: [cx + x1 * scale, cy + y1 * scale] as const,
        p2: [cx + x2 * scale, cy + y2 * scale] as const,
      };
    };
    const f = Math.max(0, Math.min(th1Ref.length - 1, frame));

    // RK45 lower-bob trail (fading), up to current frame
    ctx.lineWidth = 1.4;
    const trailStart = Math.max(0, f - 220);
    for (let i = trailStart + 1; i <= f; i++) {
      const a = pos(th1Ref[i - 1], th2Ref[i - 1]).p2;
      const b = pos(th1Ref[i], th2Ref[i]).p2;
      const alpha = 0.05 + 0.5 * ((i - trailStart) / Math.max(1, f - trailStart));
      ctx.strokeStyle = `rgba(120,170,255,${alpha.toFixed(3)})`;
      ctx.beginPath();
      ctx.moveTo(a[0], a[1]);
      ctx.lineTo(b[0], b[1]);
      ctx.stroke();
    }

    const drawLinkage = (a: number, b: number, color: string, dashed: boolean, w: number, r: number) => {
      const { p1, p2 } = pos(a, b);
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.lineWidth = w;
      ctx.setLineDash(dashed ? [5, 4] : []);
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(p1[0], p1[1]);
      ctx.lineTo(p2[0], p2[1]);
      ctx.stroke();
      ctx.setLineDash([]);
      for (const p of [p1, p2]) {
        ctx.beginPath();
        ctx.arc(p[0], p[1], r, 0, 2 * Math.PI);
        ctx.fill();
      }
    };

    // RK45 reference (solid, accent)
    drawLinkage(th1Ref[f], th2Ref[f], "#4ea1ff", false, 3, 6);
    // PINN ghost (dashed) — red once it has left the anchor
    const left = f >= leaveIdx;
    drawLinkage(th1Pinn[f], th2Pinn[f], left ? "#ff5d5d" : "rgba(120,220,170,0.85)", true, 2, 4.5);

    // pivot
    ctx.fillStyle = "var(--muted)";
    ctx.beginPath();
    ctx.arc(cx, cy, 3, 0, 2 * Math.PI);
    ctx.fillStyle = "#888";
    ctx.fill();
  }, [th1Ref, th2Ref, th1Pinn, th2Pinn, frame, l1, l2, leaveIdx, size]);

  return <canvas ref={ref} className="pendulum-canvas" style={{ width: size, height: size, maxWidth: "100%" }} />;
}
