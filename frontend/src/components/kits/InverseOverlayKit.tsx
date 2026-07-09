import { useMemo, useState } from "react";

import { fieldRange } from "../../lib/colormap";
import { HeatCanvas } from "./HeatCanvas";
import type { KitProps } from "./types";

/** InverseOverlayKit: an INVERSE case: a hidden field (here the conductivity `k(x,y)`) is recovered from
 *  sparse noisy sensor data. Instead of one heatmap it shows the comparison that MAKES it an inverse result:
 *  the recovered field with the sparse OBSERVATION markers overlaid (the evidence the PDE prior interpolates
 *  between), the true field `k*` on the SAME color scale, the pointwise error `|k-k*|`, and the measured `T`,
 *  all side by side. Hover any map for the value read-out. Static. (ADR-0063.) */
export function InverseOverlayKit({ manifest, trace, lang }: KitProps) {
  const es = lang === "es";
  const [hover, setHover] = useState<{ which: string; x: number; y: number; v: number } | null>(null);
  const fa = manifest.field_axes;

  const d = useMemo(() => {
    if (!trace) return null;
    const f = trace.fields as Record<string, number[][]>;
    const k = f.k;
    const kt = f.k_true;
    const T = f.T;
    if (!k) return null;
    const nx = k.length;
    const ny = k[0].length;
    const err: number[][] | undefined = kt ? k.map((col, i) => col.map((v, j) => Math.abs(v - kt[i][j]))): undefined;
    const xs = (trace.axes[fa[0]] ?? [0, 1]) as number[];
    const ys = (trace.axes[fa[1]] ?? [0, 1]) as number[];
    const obs = ((trace.inverse?.observations as number[][]) ?? []);
    const s = (trace.summary ?? {}) as Record<string, number>;
    // shared k / k* scale for an honest visual comparison; err + T get their own.
    const kRange = fieldRange(kt ? k.concat(kt): k);
    return { k, kt, T, err, nx, ny, xs, ys, obs, kRange, kL2: s.l2_relative, tL2: s.T_l2_relative };
  }, [trace, fa]);

  if (!d) return <div className="loading">{es ? "Cargando…": "Loading…"}</div>;
  const { k, kt, T, err, nx, ny, xs, ys, obs, kRange, kL2, tL2 } = d;
  const x0 = xs[0];
  const x1 = xs[xs.length - 1];
  const y0 = ys[0];
  const y1 = ys[ys.length - 1];

  function mkMove(which: string, fld: number[][]) {
    return (e: React.MouseEvent) => {
      const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
      const fx = (e.clientX - rect.left) / rect.width;
      const fy = (e.clientY - rect.top) / rect.height;
      const ix = Math.max(0, Math.min(nx - 1, Math.round(fx * (nx - 1))));
      const iy = Math.max(0, Math.min(ny - 1, Math.round((1 - fy) * (ny - 1))));
      setHover({ which, x: xs[ix] ?? ix, y: ys[iy] ?? iy, v: fld[ix][iy] });
    };
  }

  const panels: { id: string; label: string; field: number[][]; range?: [number, number]; dots?: boolean }[] = [
    { id: "k", label: es ? "k recuperada (con sensores)": "recovered k (with sensors)", field: k, range: kRange, dots: true },
  ];
  if (kt) panels.push({ id: "k_true", label: es ? "k* verdadera": "true k*", field: kt, range: kRange });
  if (err) panels.push({ id: "err", label: "|k − k*|", field: err });
  if (T) panels.push({ id: "T", label: es ? "T medida": "measured T", field: T });

  return (
    <div className="inv-kit">
      <div className="inv-panels">
        {panels.map((p) => (
          <figure key={p.id} className="inv-panel">
            <figcaption className="inv-cap muted">{p.label}</figcaption>
            <div className="inv-map" onMouseMove={mkMove(p.id, p.field)} onMouseLeave={() => setHover(null)}>
              <HeatCanvas field={p.field} range={p.range ?? fieldRange(p.field)} ariaLabel={`${p.id} field`} />
              {p.dots &&
                obs.map((o, i) => (
                  <div
                    key={i}
                    className="inv-obs"
                    title={`${fa[0]}=${o[0]} ${fa[1]}=${o[1]} · T=${o[2]}`}
                    style={{ left: `${((o[0] - x0) / (x1 - x0)) * 100}%`, top: `${(1 - (o[1] - y0) / (y1 - y0)) * 100}%` }}
                  />
                ))}
            </div>
          </figure>
        ))}
      </div>
      <div className="inv-readout mono">
        {hover ? (
          <>
            {hover.which}: {fa[0]}={hover.x.toFixed(3)} {fa[1]}={hover.y.toFixed(3)} → <strong>{hover.v.toFixed(4)}</strong>
          </>
        ): (
          <span className="muted">
            {es
              ? `${obs.length} sensores dispersos (puntos blancos) · k L2 vs k* = ${kL2 != null ? (kL2 * 100).toFixed(2) + "%": "n/d"} · T L2 = ${tL2 != null ? (tL2 * 100).toFixed(2) + "%": "n/d"}`
             : `${obs.length} sparse sensors (white dots) · k L2 vs k* = ${kL2 != null ? (kL2 * 100).toFixed(2) + "%": "n/a"} · T L2 = ${tL2 != null ? (tL2 * 100).toFixed(2) + "%": "n/a"}`}
          </span>
        )}
      </div>
      <p className="hint">
        {es
          ? "Problema INVERSO: el campo de conductividad k(x,y) se recupera desde ~100 sensores T dispersos y ruidosos (los puntos blancos sobre el primer mapa). La física (la EDP) rellena los huecos entre sensores. Compara la k recuperada con la k* verdadera (misma escala de color) y su error |k−k*|."
         : "INVERSE problem: the conductivity field k(x,y) is recovered from ~100 sparse noisy T sensors (the white dots on the first map). The physics (the PDE) fills the gaps between sensors. Compare the recovered k against the true k* (same color scale) and their error |k−k*|."}
      </p>
    </div>
  );
}
