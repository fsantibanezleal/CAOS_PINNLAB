import { useEffect, useMemo, useRef, useState } from "react";

import { viridis } from "../lib/colormap";

export interface FieldAxis {
  label: string;
  lo: number;
  hi: number;
}

const TIME_NAMES = new Set(["t", "time", "tau", "tt", "τ"]);
const dimName = (l: string) => (TIME_NAMES.has(l) ? `${l} — time →` : l);

/** Interactive 2-D field viewer: a viridis heatmap with a hover crosshair + value read-out, a colorbar, and two
 *  line-cut profiles (u along x at the cursor row, u along y at the cursor column). `field[ix][iy]` with ix on the
 *  horizontal axis and larger iy drawn at the top (math convention). Reacts to the cursor — the value read-out and
 *  both profiles update live as you move the pointer. */
export function FieldView({
  field,
  axisX,
  axisY,
  outputLabel,
}: {
  field: number[][];
  axisX: FieldAxis;
  axisY: FieldAxis;
  outputLabel: string;
}) {
  const nx = field.length;
  const ny = field[0]?.length ?? 0;
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [cur, setCur] = useState<{ ix: number; iy: number } | null>(null);

  const { lo, hi } = useMemo(() => {
    let lo = Infinity;
    let hi = -Infinity;
    for (const row of field) for (const v of row) {
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    return { lo, hi };
  }, [field]);
  const span = hi - lo || 1;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !nx || !ny) return;
    canvas.width = nx;
    canvas.height = ny;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const img = ctx.createImageData(nx, ny);
    for (let ix = 0; ix < nx; ix++) {
      for (let iy = 0; iy < ny; iy++) {
        const [r, g, b] = viridis((field[ix][iy] - lo) / span);
        const idx = ((ny - 1 - iy) * nx + ix) * 4;
        img.data[idx] = r;
        img.data[idx + 1] = g;
        img.data[idx + 2] = b;
        img.data[idx + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
  }, [field, nx, ny, lo, span]);

  function onMove(e: React.MouseEvent) {
    const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
    const fx = (e.clientX - rect.left) / rect.width;
    const fy = (e.clientY - rect.top) / rect.height;
    const ix = Math.max(0, Math.min(nx - 1, Math.round(fx * (nx - 1))));
    const iy = Math.max(0, Math.min(ny - 1, Math.round((1 - fy) * (ny - 1))));
    setCur({ ix, iy });
  }

  const ax = (i: number, a: FieldAxis, n: number) => a.lo + ((a.hi - a.lo) * i) / Math.max(1, n - 1);
  const val = cur ? field[cur.ix][cur.iy] : null;
  const cx = cur ? cur.ix / Math.max(1, nx - 1) : 0;
  const cy = cur ? 1 - cur.iy / Math.max(1, ny - 1) : 0;

  // line-cut profiles at the cursor
  const rowProfile = cur ? field.map((col) => col[cur.iy]) : [];
  const colProfile = cur ? field[cur.ix] : [];
  // initial-state overlay: for a space-time field, show the profile at t=0 (dashed) under the selected-time one,
  // so you compare the INITIAL state vs the SELECTED state. (Index 0 on whichever axis is time.)
  const timeIsX = TIME_NAMES.has(axisX.label);
  const timeIsY = TIME_NAMES.has(axisY.label);
  const ref1 = cur && timeIsY ? field.map((col) => col[0]) : undefined; // u vs x at t=0
  const ref2 = cur && timeIsX ? field[0] : undefined; // u vs y-axis at t=0 (when x is time)

  return (
    <div className="fieldview">
      <div className="fieldview-map">
        <div className="axis-y-label">{dimName(axisY.label)}</div>
        <div className="fw-stack">
          <div className="fw-row">
            <div className="field-wrap" onMouseMove={onMove} onMouseLeave={() => setCur(null)}>
              <canvas ref={canvasRef} className="field-canvas" style={{ aspectRatio: `${nx} / ${ny || 1}` }} />
              {cur && (
                <>
                  <div className="xhair xhair-v" style={{ left: `${cx * 100}%` }} />
                  <div className="xhair xhair-h" style={{ top: `${cy * 100}%` }} />
                  <div className="xhair-dot" style={{ left: `${cx * 100}%`, top: `${cy * 100}%` }} />
                </>
              )}
            </div>
            <Colorbar lo={lo} hi={hi} value={val} label={outputLabel} />
          </div>
          <div className="axis-x-label">{dimName(axisX.label)}</div>
        </div>
      </div>

      <div className="fieldview-side">
        <div className="readout">
          {cur ? (
            <span className="mono">
              {axisX.label}={ax(cur.ix, axisX, nx).toFixed(3)} &nbsp; {axisY.label}={ax(cur.iy, axisY, ny).toFixed(3)}
              &nbsp; → &nbsp; <strong>{outputLabel}={val!.toExponential(3)}</strong>
            </span>
          ) : (
            <span className="muted">hover the field to read the value at any point · field min {lo.toExponential(2)} · max {hi.toExponential(2)}</span>
          )}
        </div>

        <div className="profiles">
          <Profile title={`${outputLabel} vs ${axisX.label}`} n={nx} values={rowProfile} cursorIdx={cur?.ix ?? -1} reference={ref1} />
          <Profile title={`${outputLabel} vs ${axisY.label}`} n={ny} values={colProfile} cursorIdx={cur?.iy ?? -1} reference={ref2} />
        </div>
      </div>
    </div>
  );
}

function Colorbar({ lo, hi, value, label }: { lo: number; hi: number; value: number | null; label: string }) {
  const stops = Array.from({ length: 24 }, (_, i) => i / 23);
  const t = value !== null ? (value - lo) / (hi - lo || 1) : null;
  return (
    <div className="colorbar" title={label}>
      <div className="colorbar-scale">
        {stops.map((s, i) => {
          const [r, g, b] = viridis(1 - s);
          return <div key={i} style={{ background: `rgb(${r},${g},${b})` }} />;
        })}
        {t !== null && <div className="colorbar-tick" style={{ top: `${(1 - t) * 100}%` }} />}
      </div>
      <div className="colorbar-labels">
        <span className="mono">{hi.toExponential(1)}</span>
        <span className="mono">{lo.toExponential(1)}</span>
      </div>
    </div>
  );
}

function Profile({
  title,
  n,
  values,
  cursorIdx,
  reference,
}: {
  title: string;
  n: number;
  values: number[];
  cursorIdx: number;
  reference?: number[];
}) {
  const W = 280;
  const H = 96;
  const pad = 6;
  const hasRef = !!reference && reference.length > 1;
  const all = hasRef ? values.concat(reference!) : values;
  const lo = Math.min(...(all.length ? all : [0]));
  const hi = Math.max(...(all.length ? all : [1]));
  const span = hi - lo || 1;
  const toPts = (arr: number[]) =>
    arr
      .map((v, i) => {
        const x = pad + (i / Math.max(1, n - 1)) * (W - 2 * pad);
        const y = H - pad - ((v - lo) / span) * (H - 2 * pad);
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(" ");
  const cxFrac = cursorIdx >= 0 ? cursorIdx / Math.max(1, n - 1) : null;
  return (
    <div className="profile">
      <div className="profile-title muted">
        {title}
        {hasRef && <span className="profile-legend"> · <span className="pl-sel">selected</span> vs <span className="pl-ref">initial (t=0)</span></span>}
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="profile-svg" preserveAspectRatio="none">
        <line x1={pad} y1={H - pad} x2={W - pad} y2={H - pad} stroke="var(--border)" strokeWidth="0.8" />
        {hasRef && <polyline points={toPts(reference!)} fill="none" stroke="var(--muted)" strokeWidth="1.2" strokeDasharray="4 3" />}
        {values.length > 1 && <polyline points={toPts(values)} fill="none" stroke="var(--accent)" strokeWidth="1.6" />}
        {cxFrac !== null && (
          <line
            x1={pad + cxFrac * (W - 2 * pad)}
            y1={pad}
            x2={pad + cxFrac * (W - 2 * pad)}
            y2={H - pad}
            stroke="var(--accent-2)"
            strokeWidth="1"
            strokeDasharray="3 2"
          />
        )}
      </svg>
    </div>
  );
}
