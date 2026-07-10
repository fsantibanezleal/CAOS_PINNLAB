import { useEffect, useMemo, useState } from "react";

import type { CaseManifest, Diagnostics } from "../../lib/contract";
import { loadDiagnostics } from "../../lib/data";
import { snapshotElement } from "../../lib/snapshot";

function SnapBtn({ name }: { name: string }) {
  return (
    <button type="button" className="snap-btn" title="Save PNG"
      onClick={(e) => snapshotElement((e.currentTarget as HTMLElement).closest(".diag-block") as HTMLElement, name)}>
      ⤓
    </button>
  );
}

/** DiagnosticsKit (issue #25) - the WHY behind the method contrast. For Helmholtz: the wavenumber sweep (the naive
 *  PINN's error climbs as the wavenumber grows while the Fourier lane stays low) and the radial spectral energy (the
 *  naive lane cannot reach the high-|k| band the pattern lives in). All values are the baked, real numbers. */
export function DiagnosticsKit({ manifest, lang }: { manifest: CaseManifest; lang: "en" | "es" }) {
  const es = lang === "es";
  const path = manifest.diagnostics?.path;
  const [d, setD] = useState<Diagnostics | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    if (!path) return;
    let alive = true;
    loadDiagnostics(path).then((x) => alive && setD(x)).catch((e) => alive && setErr(String(e)));
    return () => { alive = false; };
  }, [path]);

  if (!path) return null;
  if (err) return <div className="banner error">⚠ {err}</div>;
  if (!d) return <div className="loading">{es ? "Cargando diagnósticos…" : "Loading diagnostics…"}</div>;

  return (
    <div className="diag-kit">
      {d.wavenumber_sweep && (
        <div className="diag-block">
          <h4>{es ? "Barrido de número de onda: dónde la PINN ingenua colapsa" : "Wavenumber sweep: where the naive PINN collapses"}<SnapBtn name={`${manifest.case_id}-sweep`} /></h4>
          <LineChart
            xs={d.wavenumber_sweep.n}
            series={[
              { label: es ? "PINN ingenua" : "naive PINN", color: "var(--bad)", ys: d.wavenumber_sweep.naive },
              { label: es ? "PINN Fourier" : "Fourier PINN", color: "var(--good)", ys: d.wavenumber_sweep.adapted },
            ]}
            xLabel={es ? "número de onda n (k0 = 2πn)" : "wavenumber n (k0 = 2πn)"}
            yLabel={es ? "L2 relativa vs exacta" : "relative L2 vs exact"}
            logY

          />
          <p className="hint">{es ? "La PINN ingenua (tanh) empeora al subir n (sesgo espectral); la de Fourier se mantiene baja. Números reales del barrido." : "The naive tanh PINN gets worse as n grows (spectral bias); the Fourier PINN stays low. Real numbers from the sweep."}</p>
        </div>
      )}
      {d.radial_spectrum && (
        <div className="diag-block">
          <h4>{es ? "Energía espectral radial: la banda de alta frecuencia que la ingenua no alcanza" : "Radial spectral energy: the high-frequency band the naive lane misses"}<SnapBtn name={`${manifest.case_id}-spectrum`} /></h4>
          <LineChart
            xs={d.radial_spectrum.k}
            series={[
              { label: es ? "estándar (FDM)" : "standard (FDM)", color: "var(--muted)", ys: d.radial_spectrum.standard },
              { label: es ? "PINN ingenua" : "naive PINN", color: "var(--bad)", ys: d.radial_spectrum.naive },
              { label: es ? "PINN Fourier" : "Fourier PINN", color: "var(--good)", ys: d.radial_spectrum.adapted },
            ]}
            xLabel={es ? "|k| radial" : "radial |k|"}
            yLabel={es ? "energía |FFT|" : "|FFT| energy"}

          />
        </div>
      )}
      {d.line_comparisons?.map((lc, i) => (
        <div key={i} className="diag-block">
          <h4>{es ? lc.title_es : lc.title_en}<SnapBtn name={`${manifest.case_id}-diag-${i}`} /></h4>
          <XYChart series={lc.series} xLabel={lc.xLabel} yLabel={lc.yLabel} />
        </div>
      ))}
    </div>
  );
}

/** A chart where each series carries its OWN x,y (a benchmark scatter of points vs a model curve). */
function XYChart({ series, xLabel, yLabel }: { series: { label: string; color: string; scatter?: boolean; x: number[]; y: number[] }[]; xLabel: string; yLabel: string }) {
  const W = 640;
  const H = 300;
  const pad = { l: 52, r: 12, t: 12, b: 40 };
  const all = series.flatMap((s) => s.x.map((xv, i) => ({ x: xv, y: s.y[i] }))).filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y));
  const xMin = Math.min(...all.map((p) => p.x)), xMax = Math.max(...all.map((p) => p.x));
  const yMin = Math.min(...all.map((p) => p.y)), yMax = Math.max(...all.map((p) => p.y));
  const px = (x: number) => pad.l + ((x - xMin) / (xMax - xMin || 1)) * (W - pad.l - pad.r);
  const py = (y: number) => H - pad.b - ((y - yMin) / (yMax - yMin || 1)) * (H - pad.t - pad.b);
  return (
    <div className="diag-chart">
      <svg viewBox={`0 0 ${W} ${H}`} className="diag-svg">
        <text className="diag-ylab" x={14} y={H / 2} transform={`rotate(-90 14 ${H / 2})`} textAnchor="middle">{yLabel}</text>
        <text className="diag-xlab" x={(pad.l + W - pad.r) / 2} y={H - 6} textAnchor="middle">{xLabel}</text>
        {Array.from({ length: 5 }, (_, i) => {
          const yv = yMin + (i / 4) * (yMax - yMin);
          const y = H - pad.b - (i / 4) * (H - pad.t - pad.b);
          return (
            <g key={i}>
              <line x1={pad.l} y1={y} x2={W - pad.r} y2={y} stroke="var(--border)" strokeWidth="0.6" />
              <text className="diag-tick" x={pad.l - 6} y={y + 3} textAnchor="end">{yv.toFixed(2)}</text>
            </g>
          );
        })}
        {[0, 0.25, 0.5, 0.75, 1].map((f) => {
          const xv = xMin + f * (xMax - xMin);
          return <text key={f} className="diag-tick" x={px(xv)} y={H - pad.b + 16} textAnchor="middle">{xv.toFixed(2)}</text>;
        })}
        {series.map((s) =>
          s.scatter ? (
            s.x.map((xv, i) => <circle key={s.label + i} cx={px(xv)} cy={py(s.y[i])} r={3} fill={s.color} />)
          ) : (
            <polyline key={s.label} fill="none" stroke={s.color} strokeWidth="2" points={s.x.map((xv, i) => `${px(xv)},${py(s.y[i])}`).join(" ")} />
          ),
        )}
      </svg>
      <div className="diag-legend">
        {series.map((s) => (
          <span key={s.label} className="diag-leg"><span className="diag-swatch" style={{ background: s.color, height: s.scatter ? "8px" : "3px", width: s.scatter ? "8px" : "12px", borderRadius: s.scatter ? "50%" : "2px" }} />{s.label}</span>
        ))}
      </div>
    </div>
  );
}

type Series = { label: string; color: string; ys: number[] };

function LineChart({ xs, series, xLabel, yLabel, logY }: { xs: number[]; series: Series[]; xLabel: string; yLabel: string; logY?: boolean }) {
  const W = 640;
  const H = 300;
  const pad = { l: 56, r: 12, t: 12, b: 40 };
  const [hi, setHi] = useState<number | null>(null);

  const { xMin, xMax, yMin, yMax, tf } = useMemo(() => {
    const all = series.flatMap((s) => s.ys).filter((v) => Number.isFinite(v));
    const tf = (v: number) => (logY ? Math.log10(Math.max(v, 1e-6)) : v);
    const ys = all.map(tf);
    return { xMin: Math.min(...xs), xMax: Math.max(...xs), yMin: Math.min(...ys), yMax: Math.max(...ys), tf };
  }, [xs, series, logY]);

  const px = (x: number) => pad.l + ((x - xMin) / (xMax - xMin || 1)) * (W - pad.l - pad.r);
  const py = (v: number) => H - pad.b - ((tf(v) - yMin) / (yMax - yMin || 1)) * (H - pad.t - pad.b);
  const yTicks = 4;

  return (
    <div className="diag-chart">
      <svg viewBox={`0 0 ${W} ${H}`} className="diag-svg" onMouseLeave={() => setHi(null)}
        onMouseMove={(e) => {
          const r = (e.currentTarget as SVGElement).getBoundingClientRect();
          const fx = ((e.clientX - r.left) / r.width) * W;
          const xi = Math.round(xMin + ((fx - pad.l) / (W - pad.l - pad.r)) * (xMax - xMin));
          const idx = xs.findIndex((x) => x === xi);
          setHi(idx >= 0 ? idx : null);
        }}>
        <text className="diag-ylab" x={14} y={H / 2} transform={`rotate(-90 14 ${H / 2})`} textAnchor="middle">{yLabel}</text>
        <text className="diag-xlab" x={(pad.l + W - pad.r) / 2} y={H - 6} textAnchor="middle">{xLabel}</text>
        {/* y grid + ticks */}
        {Array.from({ length: yTicks + 1 }, (_, i) => {
          const yv = yMin + (i / yTicks) * (yMax - yMin);
          const y = H - pad.b - (i / yTicks) * (H - pad.t - pad.b);
          const lbl = logY ? (Math.pow(10, yv)).toExponential(0) : yv.toFixed(2);
          return (
            <g key={i}>
              <line x1={pad.l} y1={y} x2={W - pad.r} y2={y} stroke="var(--border)" strokeWidth="0.6" />
              <text className="diag-tick" x={pad.l - 6} y={y + 3} textAnchor="end">{lbl}</text>
            </g>
          );
        })}
        {/* x ticks */}
        {xs.map((x) => (
          <text key={x} className="diag-tick" x={px(x)} y={H - pad.b + 16} textAnchor="middle">{x}</text>
        ))}
        {series.map((sr) => (
          <polyline key={sr.label} fill="none" stroke={sr.color} strokeWidth="2"
            points={sr.ys.map((v, i) => `${px(xs[i])},${py(v)}`).join(" ")} />
        ))}
        {series.map((sr) => sr.ys.map((v, i) => <circle key={sr.label + i} cx={px(xs[i])} cy={py(v)} r={hi === i ? 4 : 2.5} fill={sr.color} />))}
        {hi !== null && <line x1={px(xs[hi])} y1={pad.t} x2={px(xs[hi])} y2={H - pad.b} stroke="var(--accent-2)" strokeDasharray="3 2" strokeWidth="1" />}
      </svg>
      <div className="diag-legend">
        {series.map((sr) => (
          <span key={sr.label} className="diag-leg"><span className="diag-swatch" style={{ background: sr.color }} />{sr.label}
            {hi !== null && <span className="mono muted"> · {Number.isFinite(sr.ys[hi]) ? sr.ys[hi].toExponential(2) : "-"}</span>}
          </span>
        ))}
      </div>
    </div>
  );
}
