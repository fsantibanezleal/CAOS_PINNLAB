import { useEffect, useMemo, useRef, useState } from "react";

import { viridis } from "../../lib/colormap";
import { markersFor, MarkerLayer } from "./MarkerLayer";
import { useFitBox } from "./useFitBox";
import type { KitProps } from "./types";

// bilinear sample of a [nx][ny] field at data coords (x,y), clamped to the grid
function sample(field: number[][], xs: number[], ys: number[], x: number, y: number): number {
  const nx = field.length;
  const ny = field[0].length;
  const fx = ((x - xs[0]) / (xs[nx - 1] - xs[0])) * (nx - 1);
  const fy = ((y - ys[0]) / (ys[ny - 1] - ys[0])) * (ny - 1);
  const ix = Math.floor(fx);
  const iy = Math.floor(fy);
  if (ix < 0 || iy < 0 || ix >= nx - 1 || iy >= ny - 1) {
    const cx = Math.max(0, Math.min(nx - 1, Math.round(fx)));
    const cy = Math.max(0, Math.min(ny - 1, Math.round(fy)));
    return field[cx][cy];
  }
  const tx = fx - ix;
  const ty = fy - iy;
  return (
    field[ix][iy] * (1 - tx) * (1 - ty) + field[ix + 1][iy] * tx * (1 - ty) +
    field[ix][iy + 1] * (1 - tx) * ty + field[ix + 1][iy + 1] * tx * ty
  );
}

/** Integrate one streamline (RK4) of the (u,v) field from a seed, in `dir` (+1 downstream, -1 upstream). */
function streamline(u: number[][], v: number[][], xs: number[], ys: number[], x0: number, y0: number, step: number, maxSteps: number, dir: number): [number, number][] {
  const pts: [number, number][] = [[x0, y0]];
  let x = x0;
  let y = y0;
  const xN = xs[xs.length - 1];
  const yN = ys[ys.length - 1];
  const f = (px: number, py: number): [number, number] => [sample(u, xs, ys, px, py) * dir, sample(v, xs, ys, px, py) * dir];
  for (let i = 0; i < maxSteps; i++) {
    const [a1, b1] = f(x, y);
    const [a2, b2] = f(x + 0.5 * step * a1, y + 0.5 * step * b1);
    const [a3, b3] = f(x + 0.5 * step * a2, y + 0.5 * step * b2);
    const [a4, b4] = f(x + step * a3, y + step * b3);
    const dx = (step / 6) * (a1 + 2 * a2 + 2 * a3 + a4);
    const dy = (step / 6) * (b1 + 2 * b2 + 2 * b3 + b4);
    if (Math.hypot(dx, dy) < 1e-5) break; // stagnation point
    x += dx;
    y += dy;
    if (x < xs[0] || x > xN || y < ys[0] || y > yN) break;
    pts.push([x, y]);
  }
  return pts;
}

type Bg = "speed" | "p" | "vort";

/** VectorFieldKit: a 2-D VELOCITY field (e.g. the lid-driven cavity u,v,p) shown as RK4 streamlines + quiver over
 *  a scalar background (speed / pressure / vorticity), with a hover read-out of (u,v,|U|,p). Static: streamlines are
 *  computed once from the baked field (no animation, no autoplay). Reveals the recirculating vortex that a per-scalar
 *  heatmap hides. (ADR-0063.) */
export function VectorFieldKit({ manifest, trace, active, lang }: KitProps) {
  const es = lang === "es";
  const [bg, setBg] = useState<Bg>("speed");
  const [showQuiver, setShowQuiver] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hover, setHover] = useState<{ x: number; y: number; u: number; v: number; p: number } | null>(null);

  const W = 460;
  const H = 460;
  const fit = useFitBox<HTMLDivElement>(1, 4);
  const fa = manifest.field_axes;
  const hasUV = manifest.outputs.includes("u") && manifest.outputs.includes("v");

  const d = useMemo(() => {
    if (!trace || !hasUV) return null;
    const u = trace.fields.u as number[][];
    const v = trace.fields.v as number[][];
    // guard a mismatched/stale trace (e.g. mid case-switch): u/v must be 2-D grids or we bail to the loader.
    if (!Array.isArray(u) || !Array.isArray(u[0]) || !Array.isArray(v) || !Array.isArray(v[0])) return null;
    const p = (trace.fields.p as number[][]) ?? u;
    const xs = (trace.axes[fa[0]] ?? []) as number[];
    const ys = (trace.axes[fa[1]] ?? []) as number[];
    const nx = u.length;
    const ny = u[0].length;
    // background scalar
    const scal: number[][] = [];
    let lo = Infinity;
    let hi = -Infinity;
    for (let i = 0; i < nx; i++) {
      const col: number[] = [];
      for (let j = 0; j < ny; j++) {
        let s: number;
        if (bg === "speed") s = Math.hypot(u[i][j], v[i][j]);
        else if (bg === "p") s = p[i][j];
        else {
          // vorticity dv/dx - du/dy (central diff)
          const ip = Math.min(nx - 1, i + 1);
          const im = Math.max(0, i - 1);
          const jp = Math.min(ny - 1, j + 1);
          const jm = Math.max(0, j - 1);
          const dvdx = (v[ip][j] - v[im][j]) / ((xs[ip] - xs[im]) || 1);
          const dudy = (u[i][jp] - u[i][jm]) / ((ys[jp] - ys[jm]) || 1);
          s = dvdx - dudy;
        }
        col.push(s);
        if (Number.isFinite(s)) { if (s < lo) lo = s; if (s > hi) hi = s; }
      }
      scal.push(col);
    }
    // streamlines from a seed grid (both directions)
    const span = (xs[nx - 1] - xs[0]);
    const step = span / 400;
    const maxSteps = 800;
    const lines: [number, number][][] = [];
    const S = 11;
    for (let a = 0; a < S; a++) {
      for (let b = 0; b < S; b++) {
        const x0 = xs[0] + ((a + 0.5) / S) * span;
        const y0 = ys[0] + ((b + 0.5) / S) * (ys[ny - 1] - ys[0]);
        const fwd = streamline(u, v, xs, ys, x0, y0, step, maxSteps, 1);
        const bwd = streamline(u, v, xs, ys, x0, y0, step, maxSteps, -1);
        lines.push(bwd.reverse().concat(fwd.slice(1)));
      }
    }
    return { u, v, p, xs, ys, nx, ny, scal, lo: lo === hi ? lo - 1: lo, hi: hi === lo ? hi + 1: hi, lines };
  }, [trace, bg, fa, hasUV]);

  useEffect(() => {
    const cv = canvasRef.current;
    if (!cv || !d) return;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    cv.width = W * dpr;
    cv.height = H * dpr;
    const ctx = cv.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const { scal, lo, hi, nx, ny, xs, ys, u, v, lines } = d;

    // background heatmap via an offscreen nx*ny image, scaled up smoothly
    const off = document.createElement("canvas");
    off.width = nx;
    off.height = ny;
    const octx = off.getContext("2d")!;
    const img = octx.createImageData(nx, ny);
    const span = hi - lo || 1;
    for (let i = 0; i < nx; i++) {
      for (let j = 0; j < ny; j++) {
        const [r, g, bl] = viridis((scal[i][j] - lo) / span);
        const idx = ((ny - 1 - j) * nx + i) * 4;
        img.data[idx] = r; img.data[idx + 1] = g; img.data[idx + 2] = bl; img.data[idx + 3] = 255;
      }
    }
    octx.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(off, 0, 0, W, H);

    const x0 = xs[0];
    const y0 = ys[0];
    const xspan = xs[nx - 1] - x0 || 1;
    const yspan = ys[ny - 1] - y0 || 1;
    const mapX = (x: number) => ((x - x0) / xspan) * W;
    const mapY = (y: number) => (1 - (y - y0) / yspan) * H;

    // streamlines
    ctx.strokeStyle = "rgba(255,255,255,0.78)";
    ctx.lineWidth = 1.1;
    for (const ln of lines) {
      if (ln.length < 2) continue;
      ctx.beginPath();
      ctx.moveTo(mapX(ln[0][0]), mapY(ln[0][1]));
      for (let k = 1; k < ln.length; k++) ctx.lineTo(mapX(ln[k][0]), mapY(ln[k][1]));
      ctx.stroke();
    }

    // quiver
    if (showQuiver) {
      const Q = 16;
      let vmax = 1e-6;
      for (let i = 0; i < nx; i++) for (let j = 0; j < ny; j++) vmax = Math.max(vmax, Math.hypot(u[i][j], v[i][j]));
      const al = (W / Q) * 0.42 / vmax;
      ctx.strokeStyle = "rgba(0,0,0,0.55)";
      ctx.lineWidth = 1;
      for (let a = 0; a < Q; a++) {
        for (let b = 0; b < Q; b++) {
          const x = x0 + ((a + 0.5) / Q) * xspan;
          const y = y0 + ((b + 0.5) / Q) * yspan;
          const uu = sample(u, xs, ys, x, y);
          const vv = sample(v, xs, ys, x, y);
          const sx = mapX(x);
          const sy = mapY(y);
          const ex = sx + uu * al;
          const ey = sy - vv * al;
          ctx.beginPath();
          ctx.moveTo(sx, sy);
          ctx.lineTo(ex, ey);
          ctx.stroke();
          const ang = Math.atan2(ey - sy, ex - sx);
          ctx.beginPath();
          ctx.moveTo(ex, ey);
          ctx.lineTo(ex - 4 * Math.cos(ang - 0.4), ey - 4 * Math.sin(ang - 0.4));
          ctx.moveTo(ex, ey);
          ctx.lineTo(ex - 4 * Math.cos(ang + 0.4), ey - 4 * Math.sin(ang + 0.4));
          ctx.stroke();
        }
      }
    }
  }, [d, showQuiver]);

  if (!hasUV) return <div className="banner warn">{es ? "VectorFieldKit requiere salidas u,v.": "VectorFieldKit needs u,v outputs."}</div>;
  if (!d) return <div className="loading">{es ? "Cargando campo…": "Loading field…"}</div>;

  const { xs, ys, nx, ny, u, v, p } = d;
  function onMove(e: React.MouseEvent) {
    const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
    const fx = (e.clientX - rect.left) / rect.width;
    const fy = (e.clientY - rect.top) / rect.height;
    const ix = Math.max(0, Math.min(nx - 1, Math.round(fx * (nx - 1))));
    const iy = Math.max(0, Math.min(ny - 1, Math.round((1 - fy) * (ny - 1))));
    setHover({ x: xs[ix] ?? ix, y: ys[iy] ?? iy, u: u[ix][iy], v: v[ix][iy], p: p[ix][iy] });
  }

  const bgLabel = { speed: es ? "rapidez |U|": "speed |U|", p: es ? "presión p": "pressure p", vort: es ? "vorticidad": "vorticity" };

  return (
    <div className="vf-kit">
      <div className="vf-controls">
        <span className="muted" style={{ fontSize: 13 }}>{es ? "Fondo": "Background"}</span>
        <div className="variant-chips">
          {(["speed", "p", "vort"] as Bg[]).map((k) => (
            <button key={k} type="button" className={"variant-chip" + (bg === k ? " active": "")} onClick={() => setBg(k)}>{bgLabel[k]}</button>
          ))}
        </div>
        <label className="t-ctl"><input type="checkbox" checked={showQuiver} onChange={(e) => setShowQuiver(e.target.checked)} /> {es ? "Flechas": "Quiver"}</label>
      </div>
      <div className="vf-fitarea" ref={fit.areaRef}>
      <div className="vf-map" onMouseMove={onMove} onMouseLeave={() => setHover(null)} style={{ width: fit.w || W, height: fit.h || undefined, position: "relative" }}>
        <canvas ref={canvasRef} style={{ width: "100%", aspectRatio: "1 / 1", display: "block", borderRadius: 8, border: "1px solid var(--border)" }} />
        <MarkerLayer markers={markersFor(manifest, active)} a0={d.xs[0]} a1={d.xs[d.xs.length - 1]} b0={d.ys[0]} b1={d.ys[d.ys.length - 1]} lang={lang} />
      </div>
      </div>
      <div className="vf-readout mono">
        {hover
          ? <>{fa[0]}={hover.x.toFixed(3)} {fa[1]}={hover.y.toFixed(3)} → <strong>u={hover.u.toFixed(3)} v={hover.v.toFixed(3)} |U|={Math.hypot(hover.u, hover.v).toFixed(3)} p={hover.p.toFixed(3)}</strong></>
         : <span className="muted">{es ? "pasa el cursor para leer (u, v, |U|, p) · líneas blancas = streamlines del flujo": "hover to read (u, v, |U|, p) · white lines = flow streamlines"}</span>}
      </div>
      <p className="hint">
        {es
          ? "Streamlines (RK4) del campo de velocidad precalculado: revelan el vórtice recirculante de la cavidad que un heatmap escalar esconde. Estático (sin animación). La pestaña Live re-evalúa el ONNX."
         : "RK4 streamlines of the baked velocity field: they reveal the recirculating cavity vortex that a per-scalar heatmap hides. Static (no animation). The Live tab re-evaluates the ONNX."}
      </p>
    </div>
  );
}
