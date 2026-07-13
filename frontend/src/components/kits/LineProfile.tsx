import { useState } from "react";

import { fmtTick, fmtVal, niceTicks } from "../../lib/plot";

/** An SVG line plot of u over a spatial axis with REAL axes (nice ticks + a recessive grid), a faint GHOST of
 *  the initial frame, a hover crosshair + value read-out, and a y-scale LOCKED to the whole field's range so the
 *  curve doesn't rescale every frame (the motion you see is the real solution evolving, not an autoscale
 *  artifact). The animated hero of the time cases and of TimeEvolutionKit. */
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
  const W = 640;
  const H = 220;
  const padL = 50;
  const padR = 12;
  const padT = 10;
  const padB = 34;
  const n = values.length;
  const [lo, hi] = yRange;
  const span = hi - lo || 1;
  const x0 = spaceArr[0] ?? 0;
  const x1 = spaceArr[spaceArr.length - 1] ?? 1;

  const sx = (i: number) => padL + (n > 1 ? (i / (n - 1)) * (W - padL - padR) : 0);
  const sxv = (x: number) => padL + ((x - x0) / (x1 - x0 || 1)) * (W - padL - padR);
  const sy = (v: number) => H - padB - ((v - lo) / span) * (H - padT - padB);
  const poly = (arr: number[]) => arr.map((v, i) => `${sx(i).toFixed(1)},${sy(v).toFixed(1)}`).join(" ");

  const yTicks = niceTicks(lo, hi, 5);
  const xTicks = niceTicks(x0, x1, 6);

  const [hoverI, setHoverI] = useState<number | null>(null);
  function onMove(e: React.MouseEvent<SVGSVGElement>) {
    const rect = e.currentTarget.getBoundingClientRect();
    const fx = (e.clientX - rect.left) / rect.width;
    const px = fx * W;
    const i = Math.max(0, Math.min(n - 1, Math.round(((px - padL) / (W - padL - padR)) * (n - 1))));
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
        {/* recessive grid + real ticks */}
        {yTicks.map((v) => (
          <g key={"y" + v}>
            <line x1={padL} y1={sy(v)} x2={W - padR} y2={sy(v)} stroke="var(--border)" strokeWidth="0.6" opacity="0.7" />
            <text x={padL - 6} y={sy(v) + 3} textAnchor="end" className="lp-tick">{fmtTick(v)}</text>
          </g>
        ))}
        {xTicks.map((v) => (
          <g key={"x" + v}>
            <line x1={sxv(v)} y1={padT} x2={sxv(v)} y2={H - padB} stroke="var(--border)" strokeWidth="0.5" opacity="0.45" />
            <text x={sxv(v)} y={H - padB + 14} textAnchor="middle" className="lp-tick">{fmtTick(v)}</text>
          </g>
        ))}
        {/* frame */}
        <line x1={padL} y1={padT} x2={padL} y2={H - padB} stroke="var(--border)" strokeWidth="1" />
        <line x1={padL} y1={H - padB} x2={W - padR} y2={H - padB} stroke="var(--border)" strokeWidth="1" />
        <text x={(padL + W - padR) / 2} y={H - 6} textAnchor="middle" className="lp-axislabel">{spaceLabel}</text>
        {/* ghost (initial frame) */}
        {ghost && ghost.length === n && (
          <polyline points={poly(ghost)} fill="none" stroke="var(--accent-2)" strokeWidth="1.2" strokeDasharray="4 3" opacity={0.55} />
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
      <div className="lp-readout">
        {hoverI != null && hv != null
          ? <span className="mono">{spaceLabel}={fmtVal(hx!)} → {outLabel}={fmtVal(hv)}</span>
          : <span className="muted">{outLabel}({spaceLabel})</span>}
      </div>
    </div>
  );
}
