import { useMemo, useState } from "react";

import { TracePlot } from "./TracePlot";
import type { KitProps } from "./types";

const TIME_NAMES = new Set(["t", "time", "tau"]);

/** UQBandKit: a Bayesian/ensemble PINN field with epistemic uncertainty (mean `c` + std `c_std`). At a chosen
 *  time (a paused slider: NO autoplay) it draws the mean with a filled ±1σ / ±2σ band, plus the σ(x) magnitude
 *  curve and the calibration coverage. Honest: the band is at true scale (this ensemble is well-calibrated, so it
 *  is thin) and the σ curve shows WHERE uncertainty concentrates. (ADR-0063.) */
export function UQBandKit({ manifest, trace, lang }: KitProps) {
  const es = lang === "es";
  const fa = manifest.field_axes;
  const tIdx = TIME_NAMES.has(fa[1]) ? 1: TIME_NAMES.has(fa[0]) ? 0: 1;
  const tAxis = fa[tIdx];
  const spaceAxis = fa[1 - tIdx];

  const d = useMemo(() => {
    if (!trace) return null;
    const mean = (trace.fields.c ?? trace.fields[manifest.outputs[0]]) as number[][];
    const std = (trace.fields.c_std ?? trace.fields[manifest.outputs[1]]) as number[][];
    if (!mean || !std) return null;
    const tArr = (trace.axes[tAxis] ?? []) as number[];
    const sArr = (trace.axes[spaceAxis] ?? []) as number[];
    const at = (f: number[][], it: number) => (tIdx === 1 ? f.map((col) => col[it]): f[it]);
    const sm = (trace.summary ?? {}) as Record<string, number>;
    return { mean, std, tArr, sArr, at, nt: tArr.length, cov: sm.uq_calibration_2sigma, K: sm.ensemble_K };
  }, [trace, tAxis, spaceAxis, tIdx, manifest.outputs]);

  const nt = d?.nt ?? 1;
  const [it, setIt] = useState(0);
  const itc = Math.min(it, nt - 1);

  if (!d) return <div className="loading">{es ? "Cargando…": "Loading…"}</div>;
  const { mean, std, tArr, sArr, at } = d;
  const mx = at(mean, itc);
  const sx = at(std, itc);
  const tVal = tArr[itc] ?? 0;

  // band polygons (±2σ outer, ±1σ inner) + mean line, in data coords for an inline SVG
  const n = mx.length;
  const x0 = sArr[0] ?? 0;
  const x1 = sArr[n - 1] ?? 1;
  let lo = Infinity;
  let hi = -Infinity;
  for (let i = 0; i < n; i++) { lo = Math.min(lo, mx[i] - 2 * sx[i]); hi = Math.max(hi, mx[i] + 2 * sx[i]); }
  if (!(hi > lo)) { hi = lo + 1; }
  const W = 580;
  const H = 240;
  const padL = 52;
  const padR = 14;
  const padT = 12;
  const padB = 30;
  const sX = (i: number) => padL + (n > 1 ? (i / (n - 1)) * (W - padL - padR): 0);
  const sY = (v: number) => H - padB - ((v - lo) / (hi - lo)) * (H - padT - padB);
  const bandPath = (k: number) => {
    const up = mx.map((m, i) => `${sX(i).toFixed(1)},${sY(m + k * sx[i]).toFixed(1)}`);
    const dn = mx.map((m, i) => `${sX(i).toFixed(1)},${sY(m - k * sx[i]).toFixed(1)}`).reverse();
    return "M" + up.join(" L") + " L" + dn.join(" L") + " Z";
  };
  const meanLine = mx.map((m, i) => `${sX(i).toFixed(1)},${sY(m).toFixed(1)}`).join(" ");

  const { cov, K } = d;

  return (
    <div className="uq-kit">
      <label className="ctl uq-slider">
        <span>{tAxis} = <strong className="mono">{tVal.toFixed(3)}</strong> &nbsp;<span className="muted">({itc + 1}/{nt})</span></span>
        <input className="scrub" type="range" min={0} max={Math.max(0, nt - 1)} step={1} value={itc} onChange={(e) => setIt(Number(e.target.value))} />
      </label>

      <svg viewBox={`0 0 ${W} ${H}`} className="uq-svg">
        <line x1={padL} y1={padT} x2={padL} y2={H - padB} stroke="var(--border)" />
        <line x1={padL} y1={H - padB} x2={W - padR} y2={H - padB} stroke="var(--border)" />
        <text x={padL - 6} y={sY(hi) + 4} textAnchor="end" className="lp-tick">{hi.toExponential(1)}</text>
        <text x={padL - 6} y={sY(lo) + 4} textAnchor="end" className="lp-tick">{lo.toExponential(1)}</text>
        <text x={padL} y={H - padB + 16} textAnchor="start" className="lp-tick">{x0.toFixed(2)}</text>
        <text x={W - padR} y={H - padB + 16} textAnchor="end" className="lp-tick">{x1.toFixed(2)}</text>
        <text x={(padL + W - padR) / 2} y={H - 4} textAnchor="middle" className="lp-axislabel">{spaceAxis}</text>
        <path d={bandPath(2)} fill="var(--accent)" opacity={0.16} />
        <path d={bandPath(1)} fill="var(--accent)" opacity={0.28} />
        <polyline points={meanLine} fill="none" stroke="var(--accent)" strokeWidth="2" />
      </svg>
      <div className="uq-legend muted">
        <span><i className="sw" style={{ background: "var(--accent)" }} /> {es ? "media": "mean"} c</span>
        <span><i className="sw" style={{ background: "var(--accent)", opacity: 0.28 }} /> ±1σ</span>
        <span><i className="sw" style={{ background: "var(--accent)", opacity: 0.16 }} /> ±2σ</span>
      </div>

      <div className="uq-sigma">
        <TracePlot
          title={es ? `Incertidumbre σ(${spaceAxis}) en ${tAxis}=${tVal.toFixed(2)}`: `Uncertainty σ(${spaceAxis}) at ${tAxis}=${tVal.toFixed(2)}`}
          series={[{ points: sx.map((vv, i) => [sArr[i] ?? i, vv]), color: "#ff8a3d", width: 1.6 }]}
          xRange={[x0, x1]} yRange={[0, Math.max(1e-6, ...sx)]} xLabel={spaceAxis} yLabel="σ" height={150}
        />
      </div>

      <p className="hint">
        {es
          ? `PINN bayesiana (deep ensemble, K=${K ?? "?"}): la banda es la incertidumbre epistémica a escala real: fina porque está bien calibrada (cobertura @2σ = ${cov != null ? (cov * 100).toFixed(1) + "%": "n/d"}). La curva naranja muestra DÓNDE crece σ. Mueve el deslizador de ${tAxis} (en pausa, sin animación).`
         : `Bayesian PINN (deep ensemble, K=${K ?? "?"}): the band is epistemic uncertainty at true scale: thin because it is well-calibrated (coverage @2σ = ${cov != null ? (cov * 100).toFixed(1) + "%": "n/a"}). The orange curve shows WHERE σ grows. Drag the ${tAxis} slider (paused, no animation).`}
      </p>
    </div>
  );
}
