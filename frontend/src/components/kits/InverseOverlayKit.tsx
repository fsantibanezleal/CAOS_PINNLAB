import { useMemo, useState } from "react";

import { fieldRange } from "../../lib/colormap";
import { HeatCanvas } from "./HeatCanvas";
import type { KitProps } from "./types";

type Which = "k" | "k_true" | "err" | "T";

/** InverseOverlayKit — an INVERSE case: a hidden field (here the conductivity `k(x,y)`) is recovered from sparse
 *  noisy sensor data. It shows the recovered field with the sparse OBSERVATION markers overlaid (the evidence the
 *  PDE prior interpolates between), a toggle to compare recovered `k` vs the true `k*` and their error, and the
 *  measured field `T`. Static. (ADR-0063.) */
export function InverseOverlayKit({ manifest, trace, lang }: KitProps) {
  const es = lang === "es";
  const [which, setWhich] = useState<Which>("k");
  const [hover, setHover] = useState<{ x: number; y: number; v: number } | null>(null);
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
    const err: number[][] = kt ? k.map((col, i) => col.map((v, j) => Math.abs(v - kt[i][j]))) : k;
    const xs = (trace.axes[fa[0]] ?? [0, 1]) as number[];
    const ys = (trace.axes[fa[1]] ?? [0, 1]) as number[];
    const obs = ((trace.inverse?.observations as number[][]) ?? []);
    const s = (trace.summary ?? {}) as Record<string, number>;
    return { k, kt, T, err, nx, ny, xs, ys, obs, kL2: s.l2_relative, tL2: s.T_l2_relative };
  }, [trace, fa]);

  if (!d) return <div className="loading">{es ? "Cargando…" : "Loading…"}</div>;
  const { k, kt, T, err, nx, ny, xs, ys, obs, kL2, tL2 } = d;

  const fields: Record<Which, number[][] | undefined> = { k, k_true: kt, err, T };
  const field = fields[which] ?? k;
  const range = fieldRange(field);
  const x0 = xs[0];
  const x1 = xs[xs.length - 1];
  const y0 = ys[0];
  const y1 = ys[ys.length - 1];

  function onMove(e: React.MouseEvent) {
    const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
    const fx = (e.clientX - rect.left) / rect.width;
    const fy = (e.clientY - rect.top) / rect.height;
    const ix = Math.max(0, Math.min(nx - 1, Math.round(fx * (nx - 1))));
    const iy = Math.max(0, Math.min(ny - 1, Math.round((1 - fy) * (ny - 1))));
    setHover({ x: xs[ix] ?? ix, y: ys[iy] ?? iy, v: field[ix][iy] });
  }

  const tabs: { id: Which; label: string }[] = [
    { id: "k", label: es ? "k recuperada" : "recovered k" },
    { id: "k_true", label: es ? "k* verdadera" : "true k*" },
    { id: "err", label: es ? "|k−k*|" : "|k−k*|" },
    { id: "T", label: "T" },
  ];

  return (
    <div className="inv-kit">
      <div className="inv-controls">
        <span className="muted" style={{ fontSize: 13 }}>{es ? "Campo" : "Field"}</span>
        <div className="variant-chips">
          {tabs.filter((t) => t.id !== "k_true" || kt).map((t) => (
            <button key={t.id} type="button" className={"variant-chip" + (which === t.id ? " active" : "")} onClick={() => setWhich(t.id)}>{t.label}</button>
          ))}
        </div>
      </div>
      <div className="inv-map" onMouseMove={onMove} onMouseLeave={() => setHover(null)} style={{ width: 440, maxWidth: "100%" }}>
        <HeatCanvas field={field} range={range} ariaLabel={`${which} field`} />
        {/* sparse observation markers (where T was measured) */}
        {obs.map((o, i) => (
          <div
            key={i}
            className="inv-obs"
            title={`${fa[0]}=${o[0]} ${fa[1]}=${o[1]} · T=${o[2]}`}
            style={{ left: `${((o[0] - x0) / (x1 - x0)) * 100}%`, top: `${(1 - (o[1] - y0) / (y1 - y0)) * 100}%` }}
          />
        ))}
      </div>
      <div className="inv-readout mono">
        {hover
          ? <>{fa[0]}={hover.x.toFixed(3)} {fa[1]}={hover.y.toFixed(3)} → <strong>{which}={hover.v.toFixed(4)}</strong></>
          : <span className="muted">{es ? `${obs.length} sensores dispersos (puntos) · k L2 vs k* = ${kL2 != null ? (kL2 * 100).toFixed(2) + "%" : "n/d"} · T L2 = ${tL2 != null ? (tL2 * 100).toFixed(2) + "%" : "n/d"}` : `${obs.length} sparse sensors (dots) · k L2 vs k* = ${kL2 != null ? (kL2 * 100).toFixed(2) + "%" : "n/a"} · T L2 = ${tL2 != null ? (tL2 * 100).toFixed(2) + "%" : "n/a"}`}</span>}
      </div>
      <p className="hint">
        {es
          ? "Problema INVERSO: el campo de conductividad k(x,y) se recupera desde ~100 sensores T dispersos y ruidosos (los puntos). La física (la EDP) rellena los huecos entre sensores — por eso una PINN gana aquí. Compara la k recuperada con la k* verdadera y su error."
          : "INVERSE problem: the conductivity field k(x,y) is recovered from ~100 sparse noisy T sensors (the dots). The physics (the PDE) fills the gaps between sensors — that is why a PINN wins here. Compare the recovered k against the true k* and their error."}
      </p>
    </div>
  );
}
