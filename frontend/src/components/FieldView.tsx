import { useEffect, useMemo, useRef, useState } from "react";

// viridis colormap (10 anchors, linear interp)
const VIRIDIS = [
  [68, 1, 84], [72, 40, 120], [62, 74, 137], [49, 104, 142], [38, 130, 142],
  [31, 158, 137], [53, 183, 121], [110, 206, 88], [181, 222, 43], [253, 231, 37],
];
function viridis(t: number): [number, number, number] {
  t = Math.max(0, Math.min(1, t));
  const x = t * (VIRIDIS.length - 1);
  const i = Math.floor(x);
  const f = x - i;
  const a = VIRIDIS[i];
  const b = VIRIDIS[Math.min(i + 1, VIRIDIS.length - 1)];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

export interface FieldAxis {
  label: string;
  lo: number;
  hi: number;
}

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

  return (
    <div className="fieldview">
      <div className="fieldview-main">
        <div className="field-wrap" onMouseMove={onMove} onMouseLeave={() => setCur(null)}>
          <canvas ref={canvasRef} className="field-canvas" style={{ aspectRatio: `${nx} / ${ny || 1}` }} />
          {cur && (
            <>
              <div className="xhair xhair-v" style={{ left: `${cx * 100}%` }} />
              <div className="xhair xhair-h" style={{ top: `${(1 - cy) * 100}%` }} />
              <div className="xhair-dot" style={{ left: `${cx * 100}%`, top: `${(1 - cy) * 100}%` }} />
            </>
          )}
        </div>
        <Colorbar lo={lo} hi={hi} value={val} label={outputLabel} />
      </div>

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
        <Profile title={`${outputLabel} vs ${axisX.label}`} n={nx} values={rowProfile} cursorIdx={cur?.ix ?? -1} />
        <Profile title={`${outputLabel} vs ${axisY.label}`} n={ny} values={colProfile} cursorIdx={cur?.iy ?? -1} />
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
}: {
  title: string;
  n: number;
  values: number[];
  cursorIdx: number;
}) {
  const W = 280;
  const H = 96;
  const pad = 6;
  const lo = Math.min(...(values.length ? values : [0]));
  const hi = Math.max(...(values.length ? values : [1]));
  const span = hi - lo || 1;
  const pts = values
    .map((v, i) => {
      const x = pad + (i / Math.max(1, n - 1)) * (W - 2 * pad);
      const y = H - pad - ((v - lo) / span) * (H - 2 * pad);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  const cxFrac = cursorIdx >= 0 ? cursorIdx / Math.max(1, n - 1) : null;
  return (
    <div className="profile">
      <div className="profile-title muted">{title}</div>
      <svg viewBox={`0 0 ${W} ${H}`} className="profile-svg" preserveAspectRatio="none">
        <line x1={pad} y1={H - pad} x2={W - pad} y2={H - pad} stroke="var(--border)" strokeWidth="0.8" />
        {values.length > 1 && <polyline points={pts} fill="none" stroke="var(--accent)" strokeWidth="1.6" />}
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
