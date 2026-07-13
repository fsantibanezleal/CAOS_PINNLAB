import { useMemo } from "react";

import type { CompareLane, CompareTrace } from "../../lib/contract";
import { fmtTick, niceTicks } from "../../lib/plot";
import { Transport } from "./Transport";
import { useAnimator } from "./useAnimator";

const TIME_NAMES = new Set(["t", "time", "tau", "tt", "τ"]);

/** CompareEvolution (issue #36): ANIMATE the method ladder. For a comparison whose trace has a time dimension,
 *  play the standard / naive / adapted profiles evolving TOGETHER over t on one locked y-scale - you literally
 *  watch the naive lane peel away from the standard while the fix tracks it. Paused by default (no autoplay);
 *  a separate component so its hooks never race the parent's loading returns. */
export function CompareEvolution({ trace, lanes, vRange, lang }: { trace: CompareTrace; lanes: CompareLane[]; vRange: [number, number]; lang: "en" | "es" }) {
  const es = lang === "es";
  const tDim = trace.dims.findIndex((d) => TIME_NAMES.has(d.toLowerCase()));
  const spaceDim = tDim === 0 ? 1 : 0;
  const tArr = trace.axes[trace.dims[tDim]] ?? [];
  const sArr = trace.axes[trace.dims[spaceDim]] ?? [];
  const nT = tArr.length;
  const anim = useAnimator(nT, { fps: 10 });
  const it = Math.min(anim.frame, Math.max(0, nT - 1));

  const series = useMemo(() => {
    const roleColor: Record<string, string> = { reference: "var(--muted)", exact: "var(--accent)", baseline: "var(--bad)", fix: "var(--good)", data: "var(--accent-2)" };
    return lanes
      .filter((l) => trace.fields[l.key])
      .map((l) => {
        const f = trace.fields[l.key];
        const at = (ti: number) => (tDim === 1 ? f.map((col) => col[ti]) : f[ti]);
        return { key: l.key, label_en: l.label_en, label_es: l.label_es, color: roleColor[l.role] ?? "var(--text)", at };
      });
  }, [trace, lanes, tDim]);

  if (tDim < 0 || nT < 2 || !series.length) return null;

  const W = 880;
  const H = 260;
  const pad = { l: 56, r: 14, t: 12, b: 34 };
  const [lo, hi] = vRange;
  const span = hi - lo || 1;
  const n = sArr.length;
  const sx = (i: number) => pad.l + (n > 1 ? (i / (n - 1)) * (W - pad.l - pad.r) : 0);
  const sy = (v: number) => H - pad.b - ((v - lo) / span) * (H - pad.t - pad.b);
  const poly = (arr: number[]) => arr.map((v, i) => `${sx(i).toFixed(1)},${sy(v).toFixed(1)}`).join(" ");

  return (
    <div className="cmpev">
      <h4 className="cmpev-title">
        {es ? "VER la escalera evolucionar: cada carril en el mismo instante" : "WATCH the ladder evolve: every lane at the same instant"}
      </h4>
      <Transport anim={anim} lang={lang} axisLabel={trace.dims[tDim]} axisValue={tArr[it] ?? 0} />
      <svg viewBox={`0 0 ${W} ${H}`} className="cmpev-svg">
        {niceTicks(lo, hi, 5).map((v) => (
          <g key={"y" + v}>
            <line x1={pad.l} y1={sy(v)} x2={W - pad.r} y2={sy(v)} stroke="var(--border)" strokeWidth="0.6" opacity="0.7" />
            <text x={pad.l - 6} y={sy(v) + 3} textAnchor="end" className="lp-tick">{fmtTick(v)}</text>
          </g>
        ))}
        {niceTicks(sArr[0] ?? 0, sArr[n - 1] ?? 1, 6).map((v) => {
          const x = pad.l + (((v - (sArr[0] ?? 0)) / (((sArr[n - 1] ?? 1) - (sArr[0] ?? 0)) || 1)) * (W - pad.l - pad.r));
          return (
            <g key={"x" + v}>
              <line x1={x} y1={pad.t} x2={x} y2={H - pad.b} stroke="var(--border)" strokeWidth="0.5" opacity="0.45" />
              <text x={x} y={H - pad.b + 14} textAnchor="middle" className="lp-tick">{fmtTick(v)}</text>
            </g>
          );
        })}
        <line x1={pad.l} y1={pad.t} x2={pad.l} y2={H - pad.b} stroke="var(--border)" strokeWidth="1" />
        <line x1={pad.l} y1={H - pad.b} x2={W - pad.r} y2={H - pad.b} stroke="var(--border)" strokeWidth="1" />
        <text x={(pad.l + W - pad.r) / 2} y={H - 4} textAnchor="middle" className="lp-axislabel">{trace.dims[spaceDim]}</text>
        {series.map((s) => (
          <polyline key={s.key} points={poly(s.at(it))} fill="none" stroke={s.color} strokeWidth={s.key === "standard" ? 2.4 : 1.8} strokeDasharray={s.key === "standard" ? undefined : undefined} opacity={s.key === "analytic" ? 0.65 : 1} />
        ))}
      </svg>
      <div className="diag-legend">
        {series.map((s) => (
          <span key={s.key} className="diag-leg"><span className="diag-swatch" style={{ background: s.color }} />{es ? s.label_es : s.label_en}</span>
        ))}
      </div>
      <p className="hint">
        {es
          ? "Pulsa ▶: los perfiles de cada carril evolucionan JUNTOS en el tiempo (escala y fija). Se ve al carril ingenuo separarse del estándar mientras la corrección lo sigue."
          : "Press ▶: each lane's profile evolves TOGETHER in time (fixed y-scale). You see the naive lane peel away from the standard while the fix tracks it."}
      </p>
    </div>
  );
}
