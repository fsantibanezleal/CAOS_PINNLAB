import { useCallback, useEffect, useRef, useState } from "react";

import type { CaseManifest, FieldTrace } from "../lib/contract";
import { onnxUrl } from "../lib/data";
import { evalNet } from "../lib/onnx";
import { FieldView } from "./FieldView";

/** The `Live` tab: re-evaluates the field IN THE BROWSER via onnxruntime-web. For a parametric case the param
 *  sliders drive a live re-evaluation of the shared ONNX (move k -> the field recomputes); for a non-parametric live
 *  case a resolution slider re-evaluates the trained net at any grid; a precompute (field-IO operator) case explains
 *  that the operator ran offline and the baked result is replayed. This is the live counterpart of the pipeline. */
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
      const [x0, x1] = ranges[a0];
      const [y0, y1] = ranges[a1];
      let k = 0;
      for (let ix = 0; ix < nx; ix++) {
        const xv = x0 + ((x1 - x0) * ix) / (nx - 1);
        for (let iy = 0; iy < ny; iy++) {
          const yv = y0 + ((y1 - y0) * iy) / (ny - 1);
          for (let j = 0; j < d; j++) {
            const ax = manifest.inputs[j];
            coords[k * d + j] = ax === a0 ? xv : ax === a1 ? yv : (params[ax] ?? 0);
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
  }, [ranges, livable, manifest, res, params, oIdx, fieldAxes.join(",")]);

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

  return (
    <div className="live-panel">
      <p className="muted" style={{ marginTop: 0 }}>
        {es
          ? "Inferencia EN VIVO en tu navegador (onnxruntime-web) del .onnx exportado — la misma red entrenada offline. Mueve los controles y el campo se re-evalúa."
          : "LIVE inference in your browser (onnxruntime-web) of the exported .onnx — the same network trained offline. Move the controls and the field re-evaluates."}
      </p>

      <div className="live-controls">
        {manifest.outputs.length > 1 && (
          <label className="ctl">
            <span>{es ? "Campo" : "Field"}</span>
            <div className="variant-chips">
              {manifest.outputs.map((o, i) => (
                <button key={o} type="button" className={"variant-chip" + (i === oIdx ? " active" : "")} onClick={() => setOIdx(i)}>{o}</button>
              ))}
            </div>
          </label>
        )}
        {manifest.param_specs.map((p) => (
          <label key={p.key} className="ctl">
            <span>
              {(es ? p.label_es : p.label_en)}
              {p.unit ? ` (${p.unit})` : ""}: <strong className="mono">{(params[p.key] ?? p.default).toFixed(2)}</strong>
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
          <span>{es ? "Resolución" : "Resolution"}: <strong className="mono">{res}×{res}</strong></span>
          <input className="scrub" type="range" min={33} max={161} step={4} value={res} onChange={(e) => setRes(Number(e.target.value))} />
        </label>
      </div>

      {err && <div className="banner error">{err}</div>}
      {!field && !err && <div className="loading">{es ? "Cargando runtime…" : "Loading runtime…"}</div>}
      {field && (
        <>
          <FieldView
            field={field}
            axisX={{ label: fieldAxes[0], lo: ranges![fieldAxes[0]][0], hi: ranges![fieldAxes[0]][1] }}
            axisY={{ label: fieldAxes[1], lo: ranges![fieldAxes[1]][0], hi: ranges![fieldAxes[1]][1] }}
            outputLabel={manifest.outputs[oIdx]}
          />
          {ms !== null && (
            <p className="hint">
              {isParam
                ? es
                  ? `Re-evaluado en vivo en ${ms.toFixed(1)} ms para el régimen seleccionado — un solo ONNX cubre toda la familia de parámetros.`
                  : `Re-evaluated live in ${ms.toFixed(1)} ms at the selected regime — one ONNX covers the whole parameter family.`
                : es
                  ? `Re-evaluado en vivo en ${ms.toFixed(1)} ms a ${res}×${res}.`
                  : `Re-evaluated live in ${ms.toFixed(1)} ms at ${res}×${res}.`}
            </p>
          )}
        </>
      )}
    </div>
  );
}
