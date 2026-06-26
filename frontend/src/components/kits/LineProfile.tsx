import { useState } from "react";

/** An SVG line plot of u over a spatial axis, with a faint GHOST of the initial frame, a hover crosshair +
 *  value read-out, and a y-scale LOCKED to the whole field's range so the curve doesn't rescale every frame
 *  (the motion you see is the real solution evolving, not an autoscale artifact). Used as the animated hero
 *  of TimeEvolutionKit and the probe of SpatioTemporalKit. */
export function LineProfile({
  values,
  ghost,
  spaceArr,
  yRange,
  spaceLabel,
  outLabel,
}: {
  values: number[];
  ghost?: number[];
  spaceArr: number[];
  yRange: [number, number];
  spaceLabel: string;
  outLabel: string;
}) {
  const W = 580;
  const H = 248;
  const padL = 52;
  const padR = 14;
  const padT = 14;
  const padB = 32;
  const n = values.length;
  const [lo, hi] = yRange;
  const span = hi - lo || 1;
  const x0 = spaceArr[0] ?? 0;
  const x1 = spaceArr[spaceArr.length - 1] ?? 1;

  const sx = (i: number) => padL + (n > 1 ? (i / (n - 1)) * (W - padL - padR) : 0);
  const sy = (v: number) => H - padB - ((v - lo) / span) * (H - padT - padB);
  const poly = (arr: number[]) => arr.map((v, i) => `${sx(i).toFixed(1)},${sy(v).toFixed(1)}`).join(" ");

  const [hoverI, setHoverI] = useState<number | null>(null);
  function onMove(e: React.MouseEvent<SVGSVGElement>) {
    const rect = e.currentTarget.getBoundingClientRect();
    const fx = (e.clientX - rect.left) / rect.width;
    const i = Math.max(0, Math.min(n - 1, Math.round(fx * (n - 1))));
    setHoverI(i);
  }

  const hv = hoverI != null ? values[hoverI] : null;
  const hx = hoverI != null ? (spaceArr[hoverI] ?? x0 + ((x1 - x0) * hoverI) / Math.max(1, n - 1)) : null;

  return (
    <div className="lineprofile">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="lineprofile-svg"
        onMouseMove={onMove}
        onMouseLeave={() => setHoverI(null)}
      >
        {/* frame */}
        <line x1={padL} y1={padT} x2={padL} y2={H - padB} stroke="var(--border)" strokeWidth="1" />
        <line x1={padL} y1={H - padB} x2={W - padR} y2={H - padB} stroke="var(--border)" strokeWidth="1" />
        {/* y ticks */}
        <text x={padL - 6} y={sy(hi) + 4} textAnchor="end" className="lp-tick">{hi.toExponential(1)}</text>
        <text x={padL - 6} y={sy(lo) + 4} textAnchor="end" className="lp-tick">{lo.toExponential(1)}</text>
        {/* x ticks */}
        <text x={padL} y={H - padB + 16} textAnchor="start" className="lp-tick">{x0.toFixed(2)}</text>
        <text x={W - padR} y={H - padB + 16} textAnchor="end" className="lp-tick">{x1.toFixed(2)}</text>
        <text x={(padL + W - padR) / 2} y={H - 4} textAnchor="middle" className="lp-axislabel">{spaceLabel}</text>
        {/* ghost (initial frame) */}
        {ghost && ghost.length === n && (
          <polyline points={poly(ghost)} fill="none" stroke="var(--accent-2)" strokeWidth="1.2" strokeDasharray="4 3" opacity={0.5} />
        )}
        {/* current frame */}
        {n > 1 && <polyline points={poly(values)} fill="none" stroke="var(--accent)" strokeWidth="2" />}
        {/* hover crosshair + dot */}
        {hoverI != null && hv != null && (
          <>
            <line x1={sx(hoverI)} y1={padT} x2={sx(hoverI)} y2={H - padB} stroke="var(--accent-2)" strokeWidth="0.8" strokeDasharray="3 2" />
            <circle cx={sx(hoverI)} cy={sy(hv)} r="3.2" fill="var(--accent)" />
          </>
        )}
      </svg>
      <div className="lp-readout mono">
        {hoverI != null && hv != null
          ? <>{spaceLabel}={hx!.toFixed(3)} &nbsp;→&nbsp; <strong>{outLabel}={hv.toExponential(3)}</strong></>
          : <span className="muted">{`hover the curve to read ${outLabel}(${spaceLabel}) · dashed = initial frame`}</span>}
      </div>
    </div>
  );
}
