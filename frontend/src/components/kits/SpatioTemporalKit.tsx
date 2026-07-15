import { useEffect, useMemo, useState } from "react";

import type { EvolutionFrames, FieldTrace } from "../../lib/contract";
import { fieldRange, viridis } from "../../lib/colormap";
import { loadEvolution, loadTrace } from "../../lib/data";
import { HeatCanvas } from "./HeatCanvas";
import { markersFor, MarkerLayer } from "./MarkerLayer";
import { useFitBox } from "./useFitBox";
import { Transport } from "./Transport";
import type { KitProps } from "./types";
import { useAnimator } from "./useAnimator";

/** SpatioTemporalKit: a 2-D spatial field that EVOLVES in time, where time is the swept parameter and each
 *  variant is a time snapshot (e.g. ocean-transport, heap-leach). It loads ALL variant traces and animates the
 *  2-D heatmap forward over t, with a color scale fixed across every frame (so the motion is real, not a
 *  per-frame autoscale) and a hover value read-out. No retraining: every snapshot is already baked. */
export function SpatioTemporalKit({ manifest, active, lang }: KitProps) {
  const es = lang === "es";
  const [outIdx, setOutIdx] = useState(0);
  const outName = manifest.outputs[outIdx] ?? manifest.outputs[0];
  const fa = manifest.field_axes;
  const tKey = manifest.param_specs[0]?.key ?? "t";

  const [traces, setTraces] = useState<FieldTrace[] | null>(null);
  const [ev, setEv] = useState<EvolutionFrames | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    setTraces(null);
    setEv(null);
    // Prefer a baked evolution-frames file (a SMOOTH offline-ONNX sequence, issue #36); fall back to the
    // per-variant snapshot traces when a case has no dedicated frames bake.
    if (manifest.evolution?.path) {
      loadEvolution(manifest.evolution.path).then((e) => alive && setEv(e)).catch((e) => alive && setErr(String(e)));
    } else {
      Promise.all(manifest.variants.map((v) => loadTrace(v.trace.path)))
        .then((ts) => alive && setTraces(ts))
        .catch((e) => alive && setErr(String(e)));
    }
    return () => { alive = false; };
  }, [manifest.case_id]);

  const data = useMemo(() => {
    let frames: number[][][] | null = null;
    let ax0: number[] = [0, 1];
    let ax1: number[] = [0, 1];
    let tVals: number[] = [];
    if (ev && ev.frames[outName]) {
      frames = ev.frames[outName];
      ax0 = ev.axes[fa[0]] ?? ax0;
      ax1 = ev.axes[fa[1]] ?? ax1;
      tVals = ev.t;
    } else if (traces && traces.length) {
      frames = traces.map((t) => t.fields[outName] as number[][]);
      ax0 = (traces[0]?.axes[fa[0]] ?? ax0) as number[];
      ax1 = (traces[0]?.axes[fa[1]] ?? ax1) as number[];
      tVals = manifest.variants.map((v) => v.params[tKey] ?? 0);
    }
    if (!frames || !frames.length) return null;
    let lo = Infinity;
    let hi = -Infinity;
    for (const fr of frames) {
      const [a, b] = fieldRange(fr);
      if (a < lo) lo = a;
      if (b > hi) hi = b;
    }
    const range: [number, number] = [lo, hi > lo ? hi: lo + 1];
    return { frames, range, ax0, ax1, tVals };
  }, [traces, ev, outName, fa, tKey, manifest.variants]);

  const nF = data?.frames.length ?? manifest.variants.length;
  // real field aspect (nH/nV) so the map is contained, not forced square; ref goes on a map-ONLY wrapper below
  // (excluding the colorbar) so fit.w never overflows into the colorbar (the "map runs off the edge" bug).
  const aspect0 = data && data.frames[0] ? data.frames[0].length / Math.max(1, data.frames[0][0]?.length ?? 1) : 1;
  const fit = useFitBox<HTMLDivElement>(aspect0, 6);
  const anim = useAnimator(nF, { fps: 6 });
  const f = Math.min(anim.frame, nF - 1);
  const [hover, setHover] = useState<{ x: number; y: number; v: number } | null>(null);

  if (err) return <div className="banner error">{err}</div>;
  if (!data) return <div className="loading">{es ? "Cargando fotogramas…": "Loading frames…"}</div>;

  const { frames, range, ax0, ax1, tVals } = data;
  const frame = frames[f];
  const nH = frame.length;
  const nV = frame[0]?.length ?? 0;

  function onMove(e: React.MouseEvent) {
    const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
    const fx = (e.clientX - rect.left) / rect.width;
    const fy = (e.clientY - rect.top) / rect.height;
    const iH = Math.max(0, Math.min(nH - 1, Math.round(fx * (nH - 1))));
    const iV = Math.max(0, Math.min(nV - 1, Math.round((1 - fy) * (nV - 1))));
    setHover({ x: ax0[iH] ?? iH, y: ax1[iV] ?? iV, v: frame[iH][iV] });
  }

  return (
    <div className="st-kit">
      {manifest.outputs.length > 1 && (
        <div className="row" style={{ gap: 8, marginBottom: 8 }}>
          <span className="muted" style={{ fontSize: 13 }}>{es ? "Campo": "Field"}</span>
          <div className="variant-chips">
            {manifest.outputs.map((o, i) => (
              <button key={o} type="button" className={"variant-chip" + (i === outIdx ? " active": "")} onClick={() => setOutIdx(i)}>{o}</button>
            ))}
          </div>
        </div>
      )}
      <Transport anim={anim} lang={lang} axisLabel={tKey} axisValue={tVals[f] ?? 0} />
      <div className="st-grid">
        {/* the fit ref measures the map area ONLY (colorbar is a separate flex child), so fit.w can never spill
            into the colorbar and overflow the stage. */}
        <div className="st-mapwrap" ref={fit.areaRef}>
          <div className="st-map fitted" onMouseMove={onMove} onMouseLeave={() => setHover(null)} style={{ position: "relative", width: fit.w || undefined, height: fit.h || undefined }}>
            <HeatCanvas field={frame} range={range} ariaLabel={`${outName} at ${tKey}=${(tVals[f] ?? 0).toFixed(2)}`} />
            <MarkerLayer markers={markersFor(manifest, active)} a0={ax0[0] ?? 0} a1={ax0[ax0.length - 1] ?? 1} b0={ax1[0] ?? 0} b1={ax1[ax1.length - 1] ?? 1} lang={lang} />
          </div>
        </div>
        <div className="st-cbar" aria-hidden>
          <div className="st-cbar-scale">
            {Array.from({ length: 24 }, (_, i) => {
              const [r, g, b] = viridis(1 - i / 23);
              return <div key={i} style={{ background: `rgb(${r},${g},${b})` }} />;
            })}
          </div>
          <div className="st-cbar-labels mono">
            <span>{range[1].toExponential(1)}</span>
            <span>{range[0].toExponential(1)}</span>
          </div>
        </div>
      </div>
      <div className="st-readout mono">
        {hover
          ? <>{fa[0]}={hover.x.toFixed(3)} &nbsp; {fa[1]}={hover.y.toFixed(3)} &nbsp;→&nbsp; <strong>{outName}={hover.v.toExponential(3)}</strong></>
         : <span className="muted">{`hover the field · ${outName} scale fixed across frames [${range[0].toExponential(1)}, ${range[1].toExponential(1)}]`}</span>}
      </div>
      <p className="hint">
        {es
          ? `Pulsa ▶ para ver el campo 2-D evolucionar en ${tKey} (o arrastra la barra). Cada fotograma es un instante. (La pestaña Live re-evalúa el ONNX en cualquier ${tKey}.)`
         : `Press ▶ to watch the 2-D field evolve in ${tKey} (or drag the bar). Each frame is one instant. (The Live tab re-evaluates the ONNX at any ${tKey}.)`}
      </p>
    </div>
  );
}
