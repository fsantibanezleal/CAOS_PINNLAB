import { fmtTick, niceTicks } from "../../lib/plot";

export interface Series {
  points: [number, number][];
  color: string;
  dashed?: boolean;
  width?: number;
}
export interface Marker {
  x: number;
  label?: string;
  color?: string;
}
export interface Dot {
  x: number;
  y: number;
  color: string;
  r?: number;
}

/** A small, flexible SVG line/scatter plotter shared by the trajectory kit: phase portraits (θ1 vs θ2),
 *  the butterfly separation curve (semilog), and the angle time-series. Multiple series, an optional vertical
 *  cursor + labelled markers (e.g. the leave-time), current-point dots, and an optional log-y axis. */
export function TracePlot({
  series,
  xRange,
  yRange,
  xLabel,
  yLabel,
  yLog = false,
  cursorX,
  markers = [],
  dots = [],
  height = 200,
  title,
}: {
  series: Series[];
  xRange: [number, number];
  yRange: [number, number];
  xLabel: string;
  yLabel: string;
  yLog?: boolean;
  cursorX?: number;
  markers?: Marker[];
  dots?: Dot[];
  height?: number;
  title?: string;
}) {
  const W = 360;
  const H = height;
  const padL = 46;
  const padR = 10;
  const padT = title ? 20 : 10;
  const padB = 28;
  const [x0, x1] = xRange;
  const yl = yLog ? Math.log10(Math.max(yRange[0], 1e-9)) : yRange[0];
  const yh = yLog ? Math.log10(Math.max(yRange[1], 1e-9)) : yRange[1];
  const xspan = x1 - x0 || 1;
  const yspan = yh - yl || 1;

  const sx = (x: number) => padL + ((x - x0) / xspan) * (W - padL - padR);
  const sy = (y: number) => {
    const yy = yLog ? Math.log10(Math.max(y, 1e-9)) : y;
    return H - padB - ((yy - yl) / yspan) * (H - padT - padB);
  };
  const path = (pts: [number, number][]) =>
    pts.map((p, i) => `${i === 0 ? "M" : "L"}${sx(p[0]).toFixed(1)},${sy(p[1]).toFixed(1)}`).join(" ");

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="trace-svg" preserveAspectRatio="xMidYMid meet">
      {title && <text x={W / 2} y={13} textAnchor="middle" className="tp-title">{title}</text>}
      {/* recessive grid + nice ticks (log-y keeps decade labels via the range ends) */}
      {!yLog &&
        niceTicks(yRange[0], yRange[1], 4).map((v) => (
          <g key={"y" + v}>
            <line x1={padL} y1={sy(v)} x2={W - padR} y2={sy(v)} stroke="var(--border)" strokeWidth="0.5" opacity="0.65" />
            <text x={padL - 5} y={sy(v) + 3} textAnchor="end" className="tp-tick">{fmtTick(v)}</text>
          </g>
        ))}
      {yLog && (
        <>
          <text x={padL - 5} y={sy(yRange[1]) + 3} textAnchor="end" className="tp-tick">{fmt(yRange[1], true)}</text>
          <text x={padL - 5} y={sy(yRange[0]) + 3} textAnchor="end" className="tp-tick">{fmt(yRange[0], true)}</text>
        </>
      )}
      {niceTicks(x0, x1, 5).map((v) => (
        <g key={"x" + v}>
          <line x1={sx(v)} y1={padT} x2={sx(v)} y2={H - padB} stroke="var(--border)" strokeWidth="0.4" opacity="0.4" />
          <text x={sx(v)} y={H - padB + 14} textAnchor="middle" className="tp-tick">{fmtTick(v)}</text>
        </g>
      ))}
      {/* frame */}
      <line x1={padL} y1={padT} x2={padL} y2={H - padB} stroke="var(--border)" strokeWidth="1" />
      <line x1={padL} y1={H - padB} x2={W - padR} y2={H - padB} stroke="var(--border)" strokeWidth="1" />
      <text x={6} y={(padT + H - padB) / 2} textAnchor="middle" className="tp-axis" transform={`rotate(-90 8 ${(padT + H - padB) / 2})`}>{yLabel}</text>
      <text x={(padL + W - padR) / 2} y={H - 3} textAnchor="middle" className="tp-axis">{xLabel}</text>
      {/* markers (vertical) */}
      {markers.map((m, i) => (
        <g key={i}>
          <line x1={sx(m.x)} y1={padT} x2={sx(m.x)} y2={H - padB} stroke={m.color ?? "#ff8a3d"} strokeWidth="1" strokeDasharray="3 2" />
          {m.label && <text x={sx(m.x) + 3} y={padT + 9} className="tp-tick" fill={m.color ?? "#ff8a3d"}>{m.label}</text>}
        </g>
      ))}
      {/* series */}
      {series.map((s, i) => (
        <path key={i} d={path(s.points)} fill="none" stroke={s.color} strokeWidth={s.width ?? 1.6}
          strokeDasharray={s.dashed ? "4 3" : undefined} strokeLinejoin="round" />
      ))}
      {/* cursor */}
      {cursorX != null && (
        <line x1={sx(cursorX)} y1={padT} x2={sx(cursorX)} y2={H - padB} stroke="var(--accent-2)" strokeWidth="0.9" strokeDasharray="2 2" />
      )}
      {/* dots */}
      {dots.map((d, i) => <circle key={i} cx={sx(d.x)} cy={sy(d.y)} r={d.r ?? 3.4} fill={d.color} stroke="var(--panel)" strokeWidth="0.8" />)}
    </svg>
  );
}

function fmt(v: number, log: boolean): string {
  if (log) return v >= 0.01 && v < 1000 ? String(+v.toPrecision(2)) : v.toExponential(0);
  return Math.abs(v) >= 100 || (v !== 0 && Math.abs(v) < 0.01) ? v.toExponential(1) : v.toFixed(2);
}
