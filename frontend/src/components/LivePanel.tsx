import { useCallback, useEffect, useRef, useState } from "react";

import type { CaseManifest, FieldTrace } from "../lib/contract";
import { onnxUrl } from "../lib/data";
import { evalNet } from "../lib/onnx";
import { FieldView } from "./FieldView";

/** A zoomed/panned sub-window of an axis range. zoom>=1 (1 = full domain); pan in [-1,1] shifts the window center
 *  within the domain (no effect at zoom 1). The window always stays inside the trained domain, so every probe is a
 *  valid in-domain evaluation: zooming/panning genuinely re-runs the net on a different sub-grid (it visibly moves,
 *  which is the point: it proves the browser is computing, not replaying). */
function windowRange([lo, hi]: [number, number], zoom: number, pan: number): [number, number] {
  const span = hi - lo;
  const half = span / (2 * Math.max(1, zoom));
  const c = (lo + hi) / 2 + pan * (span / 2 - half);
  return [c - half, c + half];
}

/** The `Live` tab: re-evaluates the field IN THE BROWSER via onnxruntime-web. For a parametric case the param
 *  sliders drive a live re-evaluation of the shared ONNX (move k -> the field recomputes); EVERY live case also gets
 *  an Explore (zoom + pan) control that re-runs the net on a sub-window (visibly moves -> proves it is computing) and
 *  a resolution slider; a precompute (field-IO operator) case explains the baked result is replayed. */
export function LivePanel({ manifest, lang }: { manifest: CaseManifest; lang: "en" | "es" }) {
  const es = lang === "es";
  const fieldAxes = manifest.field_axes;
  const isParam = manifest.param_specs.length > 0;
  const livable = manifest.lane === "live";

  const [ranges, setRanges] = useState<Record<string, [number, number]> | null>(null);
  const [params, setParams] = useState<Record<string, number>>(
    Object.fromEntries(manifest.param_specs.map((p) => [p.key, p.default])),
  );
  const [res, setRes] = useState(81);
  const [oIdx, setOIdx] = useState(0);
  const [view, setView] = useState({ zoom: 1, panX: 0, panY: 0 });
  const [field, setField] = useState<number[][] | null>(null);
  const [ms, setMs] = useState<number | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const reqId = useRef(0);

  // field-axis ranges from the default variant's baked trace (its coordinate arrays)
  useEffect(() => {
    const v = manifest.variants[0];
    if (!v) return;
    fetch(`${import.meta.env.BASE_URL}data/derived/${v.trace.path}`)
      .then((r) => r.json() as Promise<FieldTrace>)
      .then((tr) => {
        const rg: Record<string, [number, number]> = {};
        for (const a of fieldAxes) {
          const arr = tr.axes[a] ?? [0, 1];
          rg[a] = [arr[0], arr[arr.length - 1]];
        }
        setRanges(rg);
      })
      .catch((e) => setErr(String(e)));
  }, [manifest, fieldAxes.join(",")]);

  const recompute = useCallback(async () => {
    if (!ranges || !livable) return;
    const id = ++reqId.current;
    setErr(null);
    try {
      const url = onnxUrl(manifest.onnx.path);
      const nx = res;
      const ny = res;
      const n = nx * ny;
      const d = manifest.inputs.length;
      const coords = new Float32Array(n * d);
      const a0 = fieldAxes[0];
      const a1 = fieldAxes[1];
      const [x0, x1] = windowRange(ranges[a0], view.zoom, view.panX);
      const [y0, y1] = windowRange(ranges[a1], view.zoom, view.panY);
      let k = 0;
      for (let ix = 0; ix < nx; ix++) {
        const xv = x0 + ((x1 - x0) * ix) / (nx - 1);
        for (let iy = 0; iy < ny; iy++) {
          const yv = y0 + ((y1 - y0) * iy) / (ny - 1);
          for (let j = 0; j < d; j++) {
            const ax = manifest.inputs[j];
            coords[k * d + j] = ax === a0 ? xv: ax === a1 ? yv: (params[ax] ?? 0);
          }
          k++;
        }
      }
      const t0 = performance.now();
      const out = await evalNet(url, coords, n, d);
      const dt = performance.now() - t0;
      if (id !== reqId.current) return;
      // The ONNX exports a SINGLE output tensor [n, nOut] (the export names it "u" even for multi-output nets), so the
      // browser gets one flat array with the nOut columns interleaved. Pick column oIdx for the selected output field.
      const raw = Object.values(out)[0];
      const nOut = manifest.outputs.length;
      const f: number[][] = [];
      for (let ix = 0; ix < nx; ix++) {
        const col: number[] = [];
        for (let iy = 0; iy < ny; iy++) col.push(raw[(ix * ny + iy) * nOut + oIdx]);
        f.push(col);
      }
      setField(f);
      setMs(dt);
    } catch (e) {
      if (id === reqId.current) setErr(String(e));
    }
  }, [ranges, livable, manifest, res, params, oIdx, view, fieldAxes.join(",")]);

  useEffect(() => {
    void recompute();
  }, [recompute]);

  if (!livable) {
    return (
      <div className="live-panel">
        <p className="banner warn">
          {es
            ? "Caso precompute (operador campo→campo): el motor corre OFFLINE; el navegador reproduce un resultado representativo horneado. No hay inferencia en vivo por coordenadas para este lane."
           : "Precompute case (field→field operator): the engine runs OFFLINE; the browser replays a representative baked result. There is no coordinate-driven live inference for this lane."}
        </p>
      </div>
    );
  }

  const wx: [number, number] = ranges ? windowRange(ranges[fieldAxes[0]], view.zoom, view.panX): [0, 1];
  const wy: [number, number] = ranges ? windowRange(ranges[fieldAxes[1]], view.zoom, view.panY): [0, 1];
  const zoomed = view.zoom > 1.0001;

  return (
    <div className="live-panel">
      <div className="live-controls" title={es ? "Inferencia EN VIVO en tu navegador (onnxruntime-web) del .onnx exportado: la misma red entrenada offline." : "LIVE inference in your browser (onnxruntime-web) of the exported .onnx: the same network trained offline."}>
        {manifest.outputs.length > 1 && (
          <label className="ctl">
            <span>{es ? "Campo": "Field"}</span>
            <div className="variant-chips">
              {manifest.outputs.map((o, i) => (
                <button key={o} type="button" className={"variant-chip" + (i === oIdx ? " active": "")} onClick={() => setOIdx(i)}>{o}</button>
              ))}
            </div>
          </label>
        )}
        {manifest.param_specs.map((p) => (
          <label key={p.key} className="ctl">
            <span>
              {(es ? p.label_es: p.label_en)}
              {p.unit ? ` (${p.unit})`: ""}: <strong className="mono">{(params[p.key] ?? p.default).toFixed(2)}</strong>
            </span>
            <input
              className="scrub"
              type="range"
              min={p.min}
              max={p.max}
              step={p.step}
              value={params[p.key] ?? p.default}
              onChange={(e) => setParams((q) => ({ ...q, [p.key]: Number(e.target.value) }))}
            />
          </label>
        ))}
        <label className="ctl">
          <span>{es ? "Zoom (explorar)": "Zoom (explore)"}: <strong className="mono">{view.zoom.toFixed(2)}×</strong></span>
          <input className="scrub" type="range" min={1} max={5} step={0.25} value={view.zoom}
            onChange={(e) => setView((v) => ({ ...v, zoom: Number(e.target.value) }))} />
        </label>
        {zoomed && (
          <>
            <label className="ctl">
              <span>{es ? "Desplazar X": "Pan X"}: <strong className="mono">{view.panX.toFixed(2)}</strong></span>
              <input className="scrub" type="range" min={-1} max={1} step={0.05} value={view.panX}
                onChange={(e) => setView((v) => ({ ...v, panX: Number(e.target.value) }))} />
            </label>
            <label className="ctl">
              <span>{es ? "Desplazar Y": "Pan Y"}: <strong className="mono">{view.panY.toFixed(2)}</strong></span>
              <input className="scrub" type="range" min={-1} max={1} step={0.05} value={view.panY}
                onChange={(e) => setView((v) => ({ ...v, panY: Number(e.target.value) }))} />
            </label>
          </>
        )}
        <label className="ctl">
          <span>{es ? "Resolución": "Resolution"}: <strong className="mono">{res}×{res}</strong></span>
          <input className="scrub" type="range" min={33} max={161} step={4} value={res} onChange={(e) => setRes(Number(e.target.value))} />
        </label>
      </div>

      {err && <div className="banner error">{err}</div>}
      {!field && !err && <div className="loading">{es ? "Cargando runtime…": "Loading runtime…"}</div>}
      {field && (
        <div className="live-fieldarea">
          <FieldView
            field={field}
            axisX={{ label: fieldAxes[0], lo: wx[0], hi: wx[1] }}
            axisY={{ label: fieldAxes[1], lo: wy[0], hi: wy[1] }}
            outputLabel={manifest.outputs[oIdx]}
            lang={lang}
          />
        </div>
      )}
      {field && ms !== null && (
        <p className="live-hint muted">
          {isParam
            ? (es ? `Re-evaluado en vivo en ${ms.toFixed(1)} ms para el régimen: un ONNX cubre toda la familia.` : `Re-evaluated live in ${ms.toFixed(1)} ms at the regime: one ONNX covers the whole family.`)
            : (es ? `Re-evaluado en vivo en ${ms.toFixed(1)} ms a ${res}×${res}.` : `Re-evaluated live in ${ms.toFixed(1)} ms at ${res}×${res}.`)}
        </p>
      )}
    </div>
  );
}
