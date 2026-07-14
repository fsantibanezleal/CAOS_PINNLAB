import { useMemo, useState } from "react";

import type { CaseManifest, FieldTrace, VariantEntry } from "../lib/contract";
import { fieldRange, viridis } from "../lib/colormap";
import { fmtTick } from "../lib/plot";
import { HeatCanvas } from "./kits/HeatCanvas";
import { markersFor, MarkerLayer } from "./kits/MarkerLayer";
import { useFitBox } from "./kits/useFitBox";

/** THE RESULTS KEY GRAPH (plan §2.2.1): ONE fitted visualization of the case's primary field, with its answer
 *  markers and a hover read-out: clean and motivating, guaranteed to fit the Results box (no scroll). The full
 *  multi-panel / animated / interactive kit lives in the Field and evidence tabs; Results shows the headline. */
export function ResultKeyViz({ manifest, trace, active, lang }: { manifest: CaseManifest; trace: FieldTrace | null; active: VariantEntry; lang: "en" | "es" }) {
  const es = lang === "es";
  const fa = manifest.field_axes;
  const [hov, setHov] = useState<{ x: number; y: number; v: number } | null>(null);

  // pick the primary 2-D field to show: the first output that is a 2-D grid on the field axes. For a
  // vector case, prefer the speed magnitude; for an inverse case, the recovered field is outputs[0].
  const picked = useMemo(() => {
    if (!trace) return null;
    const f = trace.fields as Record<string, unknown>;
    const is2D = (k: string) => Array.isArray(f[k]) && Array.isArray((f[k] as unknown[])[0]);
    if (manifest.outputs.includes("u") && manifest.outputs.includes("v") && is2D("u") && is2D("v")) {
      const u = f.u as number[][];
      const v = f.v as number[][];
      const speed = u.map((col, i) => col.map((uv, j) => Math.hypot(uv, v[i][j])));
      return { field: speed, label: es ? "rapidez |u|" : "speed |u|" };
    }
    const primary = manifest.outputs.find((o) => is2D(o));
    if (!primary) return null;
    return { field: f[primary] as number[][], label: primary };
  }, [trace, manifest.outputs, es]);

  const xs = (trace?.axes[fa[0]] ?? [0, 1]) as number[];
  const ys = (trace?.axes[fa[1]] ?? [0, 1]) as number[];
  const nH = picked?.field.length ?? 1;
  const nV = picked?.field[0]?.length ?? 1;
  const fit = useFitBox<HTMLDivElement>(nH / Math.max(1, nV), 22);
  const range = useMemo(() => (picked ? fieldRange(picked.field) : ([0, 1] as [number, number])), [picked]);

  if (!trace) return <div className="loading">{es ? "Cargando…" : "Loading…"}</div>;
  if (!picked) return <div className="loading">{es ? "Sin campo 2-D" : "No 2-D field"}</div>;

  function onMove(e: React.MouseEvent) {
    if (!picked) return;
    const r = (e.currentTarget as HTMLElement).getBoundingClientRect();
    const fx = (e.clientX - r.left) / r.width;
    const fy = (e.clientY - r.top) / r.height;
    const iH = Math.max(0, Math.min(nH - 1, Math.round(fx * (nH - 1))));
    const iV = Math.max(0, Math.min(nV - 1, Math.round((1 - fy) * (nV - 1))));
    setHov({ x: xs[iH] ?? iH, y: ys[iV] ?? iV, v: picked.field[iH][iV] });
  }

  return (
    <div className="rkv">
      <div className="rkv-caption muted">{picked.label} · {fa[0]} × {fa[1]}</div>
      <div className="rkv-area" ref={fit.areaRef}>
        <div className="rkv-map fitted" onMouseMove={onMove} onMouseLeave={() => setHov(null)}
          style={{ width: fit.w || undefined, height: fit.h || undefined, position: "relative" }}>
          <HeatCanvas field={picked.field} range={range} ariaLabel={picked.label} />
          <MarkerLayer markers={markersFor(manifest, active)} a0={xs[0]} a1={xs[xs.length - 1]} b0={ys[0]} b1={ys[ys.length - 1]} lang={lang} />
        </div>
        <div className="rkv-cbar" aria-hidden style={fit.h ? { height: fit.h } : undefined}>
          <div className="rkv-cbar-scale">
            {Array.from({ length: 20 }, (_, i) => {
              const [r, g, b] = viridis(1 - i / 19);
              return <div key={i} style={{ background: `rgb(${r},${g},${b})` }} />;
            })}
          </div>
          <div className="rkv-cbar-labels mono"><span>{fmtTick(range[1])}</span><span>{fmtTick(range[0])}</span></div>
        </div>
      </div>
      <div className="rkv-readout mono">
        {hov
          ? <>{fa[0]}={hov.x.toFixed(3)} &nbsp; {fa[1]}={hov.y.toFixed(3)} &nbsp;→&nbsp; <strong>{picked.label}={hov.v.toExponential(3)}</strong></>
          : <span className="muted">{es ? "El gráfico principal del resultado. Pasa el cursor para leer valores; abre Campo para el visualizador completo." : "The result's key graph. Hover to read values; open Field for the full interactive viewer."}</span>}
      </div>
    </div>
  );
}
