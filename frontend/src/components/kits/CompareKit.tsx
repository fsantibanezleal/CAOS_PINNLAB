import { useEffect, useMemo, useState } from "react";

import type { CaseManifest, CompareTrace } from "../../lib/contract";
import { fieldRange, viridis } from "../../lib/colormap";
import { fmtTick } from "../../lib/plot";
import { loadComparison } from "../../lib/data";
import { snapshotElement } from "../../lib/snapshot";
import { CompareEvolution } from "./CompareEvolution";
import { HeatCanvas } from "./HeatCanvas";

function SnapBtn({ target, name }: { target: string; name: string }) {
  return (
    <button
      type="button"
      className="snap-btn"
      title="Save PNG"
      onClick={(e) => snapshotElement((e.currentTarget as HTMLElement).closest(target) as HTMLElement, name)}
    >
      ⤓
    </button>
  );
}

/** CompareKit (issue #25) - the method-ladder demonstration: the STANDARD PDE solution vs the naive / adapted /
 *  data-driven PINN on ONE grid, plus the error maps, with a shared hover probe that reads EVERY lane at a point.
 *  Driven entirely by `manifest.comparison` (lanes + fields) so it is not case-specific - every case whose pipeline
 *  computes a comparison gets the same view. Nothing here is invented: it shows the baked fields as they were computed. */
export function CompareKit({ manifest, lang }: { manifest: CaseManifest; lang: "en" | "es" }) {
  const es = lang === "es";
  const cmp = manifest.comparison;
  const [trace, setTrace] = useState<CompareTrace | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [hov, setHov] = useState<{ ix: number; iy: number } | null>(null);
  const [detail, setDetail] = useState<string | null>(null); // a lane key to show enlarged in a modal

  useEffect(() => {
    if (!cmp) return;
    let alive = true;
    setTrace(null);
    loadComparison(cmp.trace).then((t) => alive && setTrace(t)).catch((e) => alive && setErr(String(e)));
    return () => { alive = false; };
  }, [cmp?.trace]);

  const grids = useMemo(() => {
    if (!trace || !cmp) return null;
    const valueLanes = cmp.lanes.filter((l) => trace.fields[l.key]);
    const errKeys = cmp.lanes.map((l) => l.err).filter((k): k is string => !!k && !!trace.fields[k]);
    // shared color scale across the value lanes (same physical field u), so the panels are visually comparable
    const allVals: number[] = [];
    for (const l of valueLanes) for (const col of trace.fields[l.key]) for (const v of col) allVals.push(v);
    const vRange: [number, number] = allVals.length ? [Math.min(...allVals), Math.max(...allVals)] : [0, 1];
    return { valueLanes, errKeys, vRange };
  }, [trace, cmp]);

  if (!cmp) return null;
  if (err) return <div className="banner error">⚠ {err}</div>;
  if (!trace || !grids) return <div className="loading">{es ? "Cargando comparación…" : "Loading comparison…"}</div>;

  const { valueLanes, errKeys, vRange } = grids;
  const xs = trace.axes[trace.dims[0]] ?? [0, 1];
  const ys = trace.axes[trace.dims[1]] ?? [0, 1];
  const nx = xs.length;
  const ny = ys.length;
  const roleClass: Record<string, string> = { reference: "cmp-ref", exact: "cmp-exact", baseline: "cmp-bad", fix: "cmp-good", data: "cmp-data" };

  function onMove(e: React.MouseEvent) {
    const r = (e.currentTarget as HTMLElement).getBoundingClientRect();
    const fx = (e.clientX - r.left) / r.width;
    const fy = (e.clientY - r.top) / r.height;
    setHov({ ix: Math.max(0, Math.min(nx - 1, Math.round(fx * (nx - 1)))), iy: Math.max(0, Math.min(ny - 1, Math.round((1 - fy) * (ny - 1)))) });
  }

  const at = (key: string) => (hov && trace.fields[key] ? trace.fields[key][hov.ix]?.[hov.iy] : undefined);
  const s = trace.summary ?? {};

  return (
    <div className="cmp-kit">
      {/* headline contrast numbers (the real baked L2 vs the standard) */}
      <div className="cmp-headline">
        {"naive_vs_std" in s && <span className="cmp-num cmp-bad"><b>{es ? "PINN ingenua" : "naive PINN"}</b> L2 vs {es ? "estándar" : "standard"} = {fmt(s.naive_vs_std)}</span>}
        {"adapted_vs_std" in s && <span className="cmp-num cmp-good"><b>{es ? "PINN adaptada" : "adapted PINN"}</b> L2 vs {es ? "estándar" : "standard"} = {fmt(s.adapted_vs_std)}</span>}
        {"std_vs_analytic" in s && <span className="cmp-num muted">{es ? "estándar (FDM) vs analítica" : "standard (FDM) vs analytic"} = {fmt(s.std_vs_analytic)}</span>}
      </div>

      {/* the value lanes, shared color scale */}
      <div className="cmp-panels">
        {valueLanes.map((l) => (
          <figure key={l.key} className="cmp-panel">
            <figcaption className={"cmp-cap " + (roleClass[l.role] ?? "")}>{es ? l.label_es : l.label_en}<SnapBtn target=".cmp-panel" name={`${manifest.case_id}-${l.key}`} /></figcaption>
            <div className="cmp-map cmp-clickable" onMouseMove={onMove} onMouseLeave={() => setHov(null)} onClick={() => setDetail(l.key)} title={es ? "Clic para ampliar" : "Click to enlarge"}>
              <HeatCanvas field={trace.fields[l.key]} range={vRange} ariaLabel={l.key} />
            </div>
            <div className="cmp-val mono">{hov ? fmtv(at(l.key)) : " "}</div>
          </figure>
        ))}
        <div className="cmp-cbar" aria-hidden>
          <div className="cmp-cbar-scale">
            {Array.from({ length: 24 }, (_, i) => {
              const [r, g, b] = viridis(1 - i / 23);
              return <div key={i} style={{ background: `rgb(${r},${g},${b})` }} />;
            })}
          </div>
          <div className="cmp-cbar-labels mono">
            <span>{fmtTick(vRange[1])}</span>
            <span>{fmtTick((vRange[0] + vRange[1]) / 2)}</span>
            <span>{fmtTick(vRange[0])}</span>
          </div>
        </div>
      </div>

      {/* the error maps (|lane - standard|), own scales */}
      {errKeys.length > 0 && (
        <div className="cmp-panels cmp-errrow">
          {cmp.lanes.filter((l) => l.err && trace.fields[l.err]).map((l) => (
            <figure key={l.err} className="cmp-panel">
              <figcaption className="cmp-cap muted">|{es ? (l.label_es) : l.label_en} − {es ? "estándar" : "standard"}|</figcaption>
              <div className="cmp-map" onMouseMove={onMove} onMouseLeave={() => setHov(null)}>
                <HeatCanvas field={trace.fields[l.err!]} range={fieldRange(trace.fields[l.err!])} ariaLabel={l.err!} />
              </div>
              <div className="cmp-val mono">{hov ? fmtv(at(l.err!)) : " "}</div>
            </figure>
          ))}
        </div>
      )}

      {/* DYNAMICS (issue #36): animate the lanes together over time when the trace has a time axis */}
      <CompareEvolution trace={trace} lanes={valueLanes} vRange={vRange} lang={lang} />

      <div className="cmp-readout mono">
        {hov ? (
          <>x={fmtc(xs[hov.ix])} y={fmtc(ys[hov.iy])} → {valueLanes.map((l) => <span key={l.key}> {es ? l.label_es : l.label_en}={fmtv(at(l.key))} </span>)}</>
        ) : (
          <span className="muted">{es ? "pasa el cursor por cualquier panel para leer todas las soluciones en ese punto" : "hover any panel to read every solution at that point"}</span>
        )}
      </div>
      <p className="hint">{es ? cmp.note_es : cmp.note_en}</p>

      {detail && trace.fields[detail] && (
        <div className="modal-backdrop" onClick={() => setDetail(null)}>
          <div className="modal-card cmp-detail" onClick={(e) => e.stopPropagation()}>
            <div className="modal-head">
              <strong>{(() => { const l = cmp.lanes.find((x) => x.key === detail); return l ? (es ? l.label_es : l.label_en) : detail; })()}</strong>
              <span className="row" style={{ gap: 6 }}>
                <SnapBtn target=".cmp-detail" name={`${manifest.case_id}-${detail}-detail`} />
                <button type="button" className="iconbtn" onClick={() => setDetail(null)} aria-label="Close">✕</button>
              </span>
            </div>
            <div className="modal-body">
              <div className="cmp-detail-map"><HeatCanvas field={trace.fields[detail]} range={vRange} ariaLabel={detail} /></div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

const fmt = (v: number | undefined) => (typeof v === "number" ? (v * 100).toFixed(1) + "%" : "n/a");
const fmtv = (v: number | undefined) => (typeof v === "number" ? v.toExponential(2) : "-");
const fmtc = (v: number | undefined) => (typeof v === "number" ? v.toFixed(3) : "-");
