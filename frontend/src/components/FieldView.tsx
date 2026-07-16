import { useEffect, useMemo, useRef, useState } from "react";

import { viridis } from "../lib/colormap";
import { fmtTick, fmtVal, niceTicks } from "../lib/plot";
import { MarkerLayer } from "./kits/MarkerLayer";
import { Transport } from "./kits/Transport";
import { useAnimator } from "./kits/useAnimator";
import { useFitBox } from "./kits/useFitBox";

export interface FieldAxis {
  label: string;
  lo: number;
  hi: number;
}

const TIME_NAMES = new Set(["t", "time", "tau", "tt", "τ"]);
const isTime = (l: string) => TIME_NAMES.has(l.toLowerCase());
/** Axis caption: tag a time axis so the reader knows which direction is time. */
const dimTag = (l: string) => (isTime(l) ? `${l} (time)`: l);

/** Interactive field viewer: the single Field/Live visualization for every scalar-field case (ADR-0063).
 *  A viridis heatmap of `field[ix][iy]` (ix on the horizontal axis, larger iy drawn at the top) with:
 *   - a hover read-out (value under the pointer at any time),
 *   - CLICK-TO-PIN: click the map to lock a location; the two side graphs + crosshair follow the pin,
 *   - a PLAY button (only when one axis is time): animate the evolution forward; the pinned time index is the
 *     single shared time-cursor (play advances it, clicking/scrubbing sets it), so both modes coexist,
 *   - two line-cut graphs with real axis labels: the SPATIAL profile at the selected time (dashed = initial
 *     state t=0, solid = selected time, so you compare initial vs selected) and the TEMPORAL trace at the
 *     pinned location. For a steady (no-time) field both graphs are spatial cross-sections. */
export function FieldView({
  field,
  axisX,
  axisY,
  outputLabel,
  lang = "en",
  markers = [],
}: {
  field: number[][];
  axisX: FieldAxis;
  axisY: FieldAxis;
  outputLabel: string;
  lang?: "en" | "es";
  /** computed answers drawn ON the field (issue #49 S3), in (axisX, axisY) units */
  markers?: import("../lib/contract").EstimateMarker[];
}) {
  const es = lang === "es";
  const nx = field.length;
  const ny = field[0]?.length ?? 0;
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const timeIsX = isTime(axisX.label);
  const timeIsY = isTime(axisY.label);
  const hasTime = timeIsX || timeIsY;
  const nT = timeIsX ? nx: timeIsY ? ny: 1;

  // ONE shared time-cursor: the animator drives it (play), scrubbing/clicking seeks it. Default paused.
  const anim = useAnimator(nT, { fps: 12 });
  const it = hasTime ? Math.min(anim.frame, nT - 1): 0;

  // sel = the pinned/last location. pinned=false -> the two graphs FOLLOW the cursor (the released mode); a single
  // click PINS them here; a double click RELEASES back to follow. Both modes in one.
  const [sel, setSel] = useState<{ ix: number; iy: number }>(() => ({
    ix: timeIsX ? 0: Math.floor(nx / 2),
    iy: timeIsY ? 0: Math.floor(ny / 2),
  }));
  const [pinned, setPinned] = useState(false);
  const [hov, setHov] = useState<{ ix: number; iy: number } | null>(null);

  // released + hovering -> the graphs follow the cursor fully; pinned (or off-map) -> space from sel, time from the
  // animator (so Play still drives the evolution at the pinned location).
  const following = !pinned && !!hov;
  const base = following ? hov! : sel;
  const curTime = hasTime ? (following ? (timeIsY ? hov!.iy : hov!.ix) : it) : 0;
  const cur = timeIsY ? { ix: base.ix, iy: curTime }: timeIsX ? { ix: curTime, iy: base.iy }: base;

  const { lo, hi } = useMemo(() => {
    let lo = Infinity;
    let hi = -Infinity;
    for (const row of field) for (const v of row) {
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    return { lo: Number.isFinite(lo) ? lo: 0, hi: Number.isFinite(hi) ? hi: 1 };
  }, [field]);
  const span = hi - lo || 1;

  // WHEEL ZOOM into the map (issue #49 S4): a zoom WINDOW in field indices; the canvas renders the slice, and
  // every overlay (crosshair, markers, ticks, read-out) maps through the same window, so nothing lies.
  const [win, setWin] = useState<{ x0: number; x1: number; y0: number; y1: number } | null>(null);
  const wx0 = win?.x0 ?? 0, wx1 = win?.x1 ?? nx - 1, wy0 = win?.y0 ?? 0, wy1 = win?.y1 ?? ny - 1;
  const wnx = wx1 - wx0 + 1, wny = wy1 - wy0 + 1;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !wnx || !wny) return;
    canvas.width = wnx;
    canvas.height = wny;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const img = ctx.createImageData(wnx, wny);
    for (let ix = 0; ix < wnx; ix++) {
      for (let iy = 0; iy < wny; iy++) {
        const [r, g, b] = viridis((field[wx0 + ix][wy0 + iy] - lo) / span);
        const idx = ((wny - 1 - iy) * wnx + ix) * 4;
        img.data[idx] = r;
        img.data[idx + 1] = g;
        img.data[idx + 2] = b;
        img.data[idx + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
  }, [field, wnx, wny, wx0, wy0, lo, span]);

  function idxFromEvent(e: React.MouseEvent) {
    const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
    const fx = (e.clientX - rect.left) / rect.width;
    const fy = (e.clientY - rect.top) / rect.height;
    const ix = Math.max(0, Math.min(nx - 1, wx0 + Math.round(fx * (wnx - 1))));
    const iy = Math.max(0, Math.min(ny - 1, wy0 + Math.round((1 - fy) * (wny - 1))));
    return { ix, iy };
  }
  function onWheel(e: React.WheelEvent) {
    if (nx < 8 || ny < 8) return;
    const { ix, iy } = idxFromEvent(e as unknown as React.MouseEvent);
    const factor = e.deltaY < 0 ? 0.75 : 1 / 0.75; // in : out
    const spanX = Math.min(nx - 1, Math.max(6, Math.round((wx1 - wx0) * factor)));
    const spanY = Math.min(ny - 1, Math.max(6, Math.round((wy1 - wy0) * factor)));
    if (spanX >= nx - 1 && spanY >= ny - 1) { setWin(null); return; }
    const fx = (ix - wx0) / Math.max(1, wnx - 1);
    const fy = (iy - wy0) / Math.max(1, wny - 1);
    let x0 = Math.round(ix - fx * spanX), y0 = Math.round(iy - fy * spanY);
    x0 = Math.max(0, Math.min(nx - 1 - spanX, x0));
    y0 = Math.max(0, Math.min(ny - 1 - spanY, y0));
    setWin({ x0, x1: x0 + spanX, y0, y1: y0 + spanY });
  }
  function onMove(e: React.MouseEvent) {
    setHov(idxFromEvent(e));
  }
  function onClick(e: React.MouseEvent) {
    const { ix, iy } = idxFromEvent(e);
    setPinned(true); // single click -> pin/fix the graphs here
    anim.setPlaying(false);
    if (timeIsY) {
      setSel((s) => ({ ...s, ix }));
      anim.setFrame(iy);
    } else if (timeIsX) {
      setSel((s) => ({ ...s, iy }));
      anim.setFrame(ix);
    } else {
      setSel({ ix, iy });
    }
  }
  function onDblClick() {
    setPinned(false); // double click -> release: the graphs follow the cursor again
  }

  const axVal = (i: number, a: FieldAxis, n: number) => a.lo + ((a.hi - a.lo) * i) / Math.max(1, n - 1);
  // read-out follows the hover if present, else the pin
  const readIx = hov ? hov.ix: cur.ix;
  const readIy = hov ? hov.iy: cur.iy;
  const readVal = field[readIx]?.[readIy] ?? null;
  // crosshair mapped through the zoom window (clamped to its edges when the pin is outside the view)
  const cx = Math.max(0, Math.min(1, (cur.ix - wx0) / Math.max(1, wnx - 1)));
  const cy = 1 - Math.max(0, Math.min(1, (cur.iy - wy0) / Math.max(1, wny - 1)));
  // the map box sizes to the zoom window's aspect within the available area (plan E2): overlays stay truthful.
  // The ref is on the WHOLE body row (measures the real space); cols=1.9 reserves ~47% of the width for the
  // profile column, so the map never collapses when the profiles are present (the shrink bug). It is capped by
  // the body height too, so a square field fills the height and takes just its share of the width.
  const fit = useFitBox<HTMLDivElement>(wnx / Math.max(1, wny), 8, 1.9, 6);
  const tVal = hasTime ? (timeIsY ? axVal(curTime, axisY, ny): axVal(curTime, axisX, nx)): 0;
  const timeLabel = timeIsY ? axisY.label: axisX.label;

  // ---- the two line-cut graphs -------------------------------------------------------------------
  // Graph 1 = SPATIAL profile at the selected time (animated); dashed reference = the same profile at t=0.
  // Graph 2 = the OTHER cross-section (a temporal trace at the pinned location for a time case; the second
  // spatial cut for a steady case).
  let g1: { xLabel: string; n: number; values: number[]; init?: number[]; cursor: number };
  let g2: { xLabel: string; n: number; values: number[]; cursor: number };
  if (timeIsY) {
    g1 = { xLabel: axisX.label, n: nx, values: field.map((c) => c[curTime]), init: field.map((c) => c[0]), cursor: cur.ix };
    g2 = { xLabel: axisY.label, n: ny, values: field[base.ix] ?? [], cursor: curTime };
  } else if (timeIsX) {
    g1 = { xLabel: axisY.label, n: ny, values: field[curTime] ?? [], init: field[0] ?? [], cursor: cur.iy };
    g2 = { xLabel: axisX.label, n: nx, values: field.map((c) => c[base.iy]), cursor: curTime };
  } else {
    g1 = { xLabel: axisX.label, n: nx, values: field.map((c) => c[cur.iy]), cursor: cur.ix };
    g2 = { xLabel: axisY.label, n: ny, values: field[cur.ix] ?? [], cursor: cur.iy };
  }

  // the animated hero cut runs along the SPACE axis (the non-time field axis)
  const heroAxis = timeIsY ? axisX : axisY;

  return (
    <div className="fieldview2">
      {hasTime && (
        <div className="fv2-transport">
          <Transport anim={anim} lang={lang} axisLabel={timeLabel} axisValue={tVal} />
        </div>
      )}
      {/* the fit ref is on the STABLE body row (its size does not depend on the map), so setting the map size
          never resizes the measured element: no ResizeObserver feedback loop ("good for a moment then shrinks"). */}
      <div className="fv2-body" ref={fit.areaRef}>
        {/* LEFT: the map, sized square from the available space; colorbar beside it.
            The column must NOT be pinned to fit.w: the row inside it is the map (fit.w) PLUS the gap and the
            colorbar, so pinning the column to the map's width alone overflowed it by ~52px which, being
            centered, spilled 26px past each edge. The left spill fell outside .pl-res-viz (overflow:hidden)
            and clipped the map's left edge. Sizing the column to its content keeps the map whole. */}
        <div className="fv2-mapcol">
          <div className="fv2-maprow" style={fit.h ? { height: fit.h } : undefined}>
            <div className="field-wrap" style={fit.w ? { width: fit.w, height: fit.h } : undefined}
              onMouseMove={onMove} onMouseLeave={() => setHov(null)} onClick={onClick} onWheel={onWheel}
              onDoubleClick={() => { onDblClick(); setWin(null); }}
              title={es ? (pinned ? "Fijado. Doble clic para soltar y restablecer el zoom. Rueda = zoom." : "Clic para fijar; rueda = zoom; doble clic restablece") : (pinned ? "Pinned. Double-click releases + resets zoom. Wheel = zoom." : "Click to pin; wheel = zoom; double-click resets")}>
              <canvas ref={canvasRef} className="field-canvas" style={{ width: "100%", height: "100%" }} />
              {win && <span className="fw-zoombadge mono">zoom {Math.round(((nx - 1) / Math.max(1, wnx - 1)) * 10) / 10}x · {es ? "doble clic restablece" : "dbl-click resets"}</span>}
              <div className="xhair xhair-v" style={{ left: `${cx * 100}%` }} />
              <div className="xhair xhair-h" style={{ top: `${cy * 100}%` }} />
              <div className="xhair-dot" style={{ left: `${cx * 100}%`, top: `${cy * 100}%` }} />
              <MarkerLayer markers={markers}
                a0={axVal(wx0, axisX, nx)} a1={axVal(wx1, axisX, nx)}
                b0={axVal(wy0, axisY, ny)} b1={axVal(wy1, axisY, ny)} lang={lang} />
            </div>
            <Colorbar lo={lo} hi={hi} value={readVal} label={outputLabel} />
          </div>
          <div className="fv2-axhint muted mono">{dimTag(axisX.label)} × {dimTag(axisY.label)}</div>
        </div>

        {/* RIGHT: the read-out + the profile cuts, filling the remaining width and height */}
        <div className="fv2-side">
          <div className="readout mono">
            {axisX.label}={axVal(readIx, axisX, nx).toFixed(3)} &nbsp; {axisY.label}={axVal(readIy, axisY, ny).toFixed(3)}
            &nbsp; → &nbsp; <strong>{outputLabel}={readVal != null ? fmtVal(readVal): "-"}</strong>
          </div>
          {hasTime && (
            <Profile
              title={`${outputLabel}(${g1.xLabel}) ${es ? "en" : "at"} ${timeLabel}=${fmtVal(tVal)}`}
              legend={g1.init ? { tVal, timeLabel }: undefined}
              xLabel={dimTag(g1.xLabel)} yLabel={outputLabel} n={g1.n} values={g1.values} reference={g1.init}
              cursorIdx={g1.cursor} yLo={lo} yHi={hi} xLo={heroAxis.lo} xHi={heroAxis.hi} es={es} grow
            />
          )}
          {!hasTime && (
            <Profile
              title={`${outputLabel} vs ${dimTag(g1.xLabel)}`}
              xLabel={dimTag(g1.xLabel)} yLabel={outputLabel} n={g1.n} values={g1.values} reference={g1.init}
              cursorIdx={g1.cursor} yLo={lo} yHi={hi} xLo={axisX.lo} xHi={axisX.hi} es={es} grow
            />
          )}
          <Profile
            title={`${outputLabel} vs ${dimTag(g2.xLabel)}${hasTime ? (es ? " (en el punto fijado)" : " (at the pinned spot)") : ""}`}
            xLabel={dimTag(g2.xLabel)} yLabel={outputLabel} n={g2.n} values={g2.values}
            cursorIdx={g2.cursor} yLo={lo} yHi={hi} xLo={timeIsY ? axisY.lo : axisX.lo} xHi={timeIsY ? axisY.hi : axisX.hi} es={es} grow
          />
          <p className="fv2-hint muted">
            {hasTime
              ? (es ? "Los gráficos siguen el cursor; clic FIJA, doble clic suelta. ▶ anima el tiempo (discontinua = t=0). Rueda sobre el mapa = zoom." : "Graphs follow the cursor; click PINS, double-click releases. ▶ animates time (dashed = t=0). Wheel over the map = zoom.")
              : (es ? "Clic en el mapa para fijar; los cortes son fila y columna del campo. Rueda = zoom." : "Click the map to pin; the cuts are the field's row and column. Wheel = zoom.")}
          </p>
        </div>
      </div>
    </div>
  );
}

function Colorbar({ lo, hi, value, label }: { lo: number; hi: number; value: number | null; label: string }) {
  const stops = Array.from({ length: 24 }, (_, i) => i / 23);
  const t = value !== null ? (value - lo) / (hi - lo || 1): null;
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
        <span className="mono">{fmtTick(hi)}</span>
        <span className="mono">{fmtTick(lo)}</span>
      </div>
    </div>
  );
}

function Profile({
  title,
  legend,
  xLabel,
  yLabel,
  n,
  values,
  cursorIdx,
  reference,
  yLo,
  yHi,
  xLo,
  xHi,
  es,
  grow,
}: {
  title: string;
  legend?: { tVal: number; timeLabel: string };
  xLabel: string;
  yLabel: string;
  n: number;
  values: number[];
  cursorIdx: number;
  reference?: number[];
  yLo: number;
  yHi: number;
  xLo?: number;
  xHi?: number;
  es: boolean;
  grow?: boolean;
}) {
  const W = 280;
  const H = 108;
  const padL = 40;
  const padR = 8;
  const padT = 6;
  const padB = 8;
  const hasRef = !!reference && reference.length > 1;
  // y-scale LOCKED to the whole field [yLo,yHi] so the animated profile does not jump frame-to-frame and the
  // initial-vs-selected comparison is on one honest scale.
  const lo = yLo;
  const hi = yHi;
  const span = hi - lo || 1;
  const yTicks = niceTicks(lo, hi, 3);
  const toPts = (arr: number[]) =>
    arr
      .map((v, i) => {
        const x = padL + (i / Math.max(1, n - 1)) * (W - padL - padR);
        const y = H - padB - ((v - lo) / span) * (H - padT - padB);
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(" ");
  const sy = (v: number) => H - padB - ((v - lo) / span) * (H - padT - padB);
  const cxFrac = cursorIdx >= 0 ? cursorIdx / Math.max(1, n - 1): null;
  return (
    <div className={"profile" + (grow ? " profile-grow" : "")}>
      <div className="profile-title muted">
        {title}
        {legend && (
          <span className="profile-legend">
            {" "}
            · <span className="pl-sel">{es ? "seleccionado": "selected"} ({legend.timeLabel}={legend.tVal.toFixed(2)})</span>{" "}
            {es ? "vs": "vs"} <span className="pl-ref">{es ? "inicial": "initial"} ({legend.timeLabel}=0)</span>
          </span>
        )}
      </div>
      <div className="profile-plot">
        <span className="profile-ylabel">{yLabel}</span>
        <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" className="profile-svg">
          {yTicks.map((v) => (
            <g key={v}>
              <line x1={padL} y1={sy(v)} x2={W - padR} y2={sy(v)} stroke="var(--border)" strokeWidth="0.5" opacity="0.7" />
              <text x={padL - 4} y={sy(v) + 3} textAnchor="end" className="lp-tick">{fmtTick(v)}</text>
            </g>
          ))}
          <line x1={padL} y1={H - padB} x2={W - padR} y2={H - padB} stroke="var(--border)" strokeWidth="0.8" />
          <line x1={padL} y1={padT} x2={padL} y2={H - padB} stroke="var(--border)" strokeWidth="0.8" />
          {xLo !== undefined && xHi !== undefined && (
            <>
              <text x={padL + 2} y={H - 1} textAnchor="start" className="lp-tick">{fmtTick(xLo)}</text>
              <text x={W - padR} y={H - 1} textAnchor="end" className="lp-tick">{fmtTick(xHi)}</text>
            </>
          )}
          {hasRef && <polyline points={toPts(reference!)} fill="none" stroke="var(--muted)" strokeWidth="1.2" strokeDasharray="4 3" />}
          {values.length > 1 && <polyline points={toPts(values)} fill="none" stroke="var(--accent)" strokeWidth="1.6" />}
          {cxFrac !== null && (
            <line
              x1={padL + cxFrac * (W - padL - padR)}
              y1={padT}
              x2={padL + cxFrac * (W - padL - padR)}
              y2={H - padB}
              stroke="var(--accent-2)"
              strokeWidth="1"
              strokeDasharray="3 2"
            />
          )}
        </svg>
      </div>
      <div className="profile-xlabel">{xLabel}</div>
    </div>
  );
}
