import { useMemo, useState } from "react";

import { fieldRange } from "../../lib/colormap";
import { HeatCanvas } from "./HeatCanvas";
import type { KitProps } from "./types";

/** HiddenFlowKit (issue #48): the HFM mechanism made visible. The case recovers a hidden VELOCITY field from
 *  sparse dye observations + the transport physics, so the kit shows exactly that comparison: the recovered
 *  current speed with the dye-sample dots overlaid (the only evidence the net ever saw), the true speed on the
 *  SAME color scale, the pointwise velocity error (read it against the dye-swept region: the current is
 *  identifiable only where dye gradients passed), and the reconstructed dye at the variant's t. Static; reuses
 *  the proven InverseOverlayKit layout (.inv-*). */
export function HiddenFlowKit({ manifest, trace, active, lang }: KitProps) {
  const es = lang === "es";
  const [hover, setHover] = useState<{ which: string; x: number; y: number; v: number } | null>(null);
  const fa = manifest.field_axes;

  const d = useMemo(() => {
    if (!trace) return null;
    const f = trace.fields as Record<string, number[][]>;
    const u = f.u;
    const v = f.v;
    const c = f.c;
    const ut = f.u_true;
    const vt = f.v_true;
    const swept = f.swept;
    if (!u || !v) return null;
    const speed = u.map((col, i) => col.map((uv, j) => Math.hypot(uv, v[i][j])));
    const speedTrue = ut && vt ? ut.map((col, i) => col.map((uv, j) => Math.hypot(uv, vt[i][j]))) : undefined;
    const err = ut && vt ? u.map((col, i) => col.map((uv, j) => Math.hypot(uv - ut[i][j], v[i][j] - vt[i][j]))) : undefined;
    const xs = (trace.axes[fa[0]] ?? [0, 1]) as number[];
    const ys = (trace.axes[fa[1]] ?? [0, 1]) as number[];
    const obs = ((trace.inverse?.observations as number[][]) ?? []);
    const s = (trace.summary ?? {}) as Record<string, number>;
    const range = fieldRange(speedTrue ? speed.concat(speedTrue) : speed);
    return {
      speed, speedTrue, err, c, swept, xs, ys, obs, range,
      uL2: s.l2_relative, swRmse: s.speed_rel_rmse_swept, deadRmse: s.speed_rel_rmse_dead, sweptFrac: s.swept_area_frac,
    };
  }, [trace, fa]);

  if (!d) return <div className="loading">{es ? "Cargando…" : "Loading…"}</div>;
  const { speed, speedTrue, err, c, swept, xs, ys, obs, range, uL2, swRmse, deadRmse, sweptFrac } = d;
  const nx = speed.length;
  const ny = speed[0].length;
  const x0 = xs[0];
  const x1 = xs[xs.length - 1];
  const y0 = ys[0];
  const y1 = ys[ys.length - 1];
  const tNow = Number(active?.params?.t ?? NaN);

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
    { id: "speed", label: es ? "corriente recuperada |u| (con muestras de tinte)" : "recovered current |u| (with dye samples)", field: speed, range, dots: true },
  ];
  if (speedTrue) panels.push({ id: "speed_true", label: es ? "corriente verdadera |u*|" : "true current |u*|", field: speedTrue, range });
  if (err) panels.push({ id: "err", label: "|u − u*|", field: err });
  if (c) panels.push({ id: "c", label: es ? `tinte reconstruido c${Number.isFinite(tNow) ? ` (t=${tNow})` : ""}` : `reconstructed dye c${Number.isFinite(tNow) ? ` (t=${tNow})` : ""}`, field: c });
  if (swept) panels.push({ id: "swept", label: es ? "región barrida por el tinte (identificable)" : "dye-swept region (identifiable)", field: swept });

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
                    title={`${fa[0]}=${o[0]} ${fa[1]}=${o[1]} t=${o[2]} · c=${o[3]}`}
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
        ) : (
          <span className="muted">
            {es
              ? `${obs.length} muestras de tinte (puntos blancos; t en el tooltip) · u L2 global = ${uL2 != null ? (uL2 * 100).toFixed(1) + "%" : "n/d"} · velocidad RMSE rel: barrida ${swRmse != null ? (swRmse * 100).toFixed(1) + "%" : "n/d"} vs zona muerta ${deadRmse != null ? (deadRmse * 100).toFixed(1) + "%" : "n/d"} (${sweptFrac != null ? (sweptFrac * 100).toFixed(0) + "% barrido" : ""})`
              : `${obs.length} dye samples (white dots; t in the tooltip) · global u L2 = ${uL2 != null ? (uL2 * 100).toFixed(1) + "%" : "n/a"} · rel speed RMSE: swept ${swRmse != null ? (swRmse * 100).toFixed(1) + "%" : "n/a"} vs dead zone ${deadRmse != null ? (deadRmse * 100).toFixed(1) + "%" : "n/a"} (${sweptFrac != null ? (sweptFrac * 100).toFixed(0) + "% swept" : ""})`}
          </span>
        )}
      </div>
      <p className="hint">
        {es
          ? "El mecanismo de Hidden Fluid Mechanics (Science 2020): la red solo vio muestras dispersas y ruidosas del TINTE + la física del transporte (sin datos de velocidad, sin CI/CB). La corriente recuperada es precisa DENTRO de la región barrida por el tinte y no identificable donde nunca pasó tinte: compara el mapa de error con la máscara barrida; esa limitación es física, no del método."
          : "The Hidden Fluid Mechanics mechanism (Science 2020): the net only ever saw sparse noisy DYE samples + the transport physics (no velocity data, no IC/BC). The recovered current is accurate INSIDE the dye-swept region and unidentifiable where dye never passed: compare the error map against the swept mask; that limitation is physics, not the method."}
      </p>
    </div>
  );
}
