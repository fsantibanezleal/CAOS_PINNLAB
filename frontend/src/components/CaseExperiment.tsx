import { useEffect, useState, type ReactNode } from "react";

import type { CaseManifest, FieldTrace } from "../lib/contract";
import { loadManifest, loadTrace } from "../lib/data";
import { Equation } from "./Equation";
import { CompareKit } from "./kits/CompareKit";
import { DiagnosticsKit } from "./kits/DiagnosticsKit";
import { resolveKit } from "./kits/registry";
import { LivePanel } from "./LivePanel";
import { VariantCharts } from "./VariantCharts";

type TabId = "compare" | "field" | "live" | "charts" | "context" | "diagnostics";

/** One case = a full-width workbench (ADR-0016 §9 + ADR-0063), mirroring CAOS_RES_Lidar3D's `.work` shell:
 *  a LEFT control rail (the case selector, passed in as `selector`, then the regime chips + the view switch),
 *  a CENTER stage that fills the width with the active view (Field / Live / Charts / Context), and a RIGHT
 *  stats rail (governing equation + method/engine/L2/ONNX + expected band). No control is lost from the old
 *  stacked layout; they are re-slotted into the three columns so the content finally uses the full width. */
export function CaseExperiment({
  manifestId,
  selector,
  description,
  lang,
}: {
  manifestId: string;
  selector: ReactNode;
  description: ReactNode;
  lang: "en" | "es";
}) {
  const es = lang === "es";
  const [manifest, setManifest] = useState<CaseManifest | null>(null);
  const [activeId, setActiveId] = useState<string>("");
  const [trace, setTrace] = useState<FieldTrace | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [tab, setTab] = useState<TabId>("field");

  useEffect(() => {
    // When a case ships the method-ladder comparison (issue #25), land on it - it is the richest view.
    if (manifest) setTab(manifest.comparison ? "compare" : "field");
  }, [manifest?.case_id]);

  useEffect(() => {
    let alive = true;
    setManifest(null);
    // Reset the trace too: otherwise a render can pair the NEW manifest (e.g. a VectorField case) with the
    // PREVIOUS case's trace before the trace effect runs, and a kit crashes reading fields that trace lacks.
    setTrace(null);
    setErr(null);
    loadManifest(manifestId)
      .then((m) => {
        if (!alive) return;
        setManifest(m);
        setActiveId(m.variants[0]?.id ?? "");
      })
      .catch((e) => alive && setErr(String(e)));
    return () => {
      alive = false;
    };
  }, [manifestId]);

  const active = manifest?.variants.find((v) => v.id === activeId) ?? manifest?.variants[0];

  useEffect(() => {
    if (!active) return;
    let alive = true;
    setTrace(null);
    loadTrace(active.trace.path).then((t) => alive && setTrace(t)).catch((e) => alive && setErr(String(e)));
    return () => {
      alive = false;
    };
  }, [active?.trace.path]);

  const VIEWS: { id: TabId; label: string; sub: string }[] = [
    ...(manifest?.comparison ? [{ id: "compare" as TabId, label: es ? "Comparar" : "Compare", sub: es ? "estándar vs PINN" : "standard vs PINN" }] : []),
    { id: "field", label: es ? "Campo" : "Field", sub: es ? "campo interactivo" : "interactive field" },
    { id: "live", label: "Live", sub: es ? "inferencia en el navegador" : "in-browser inference" },
    { id: "charts", label: "Charts", sub: es ? "comparación de regímenes" : "regime comparison" },
    ...(manifest?.diagnostics ? [{ id: "diagnostics" as TabId, label: es ? "Diagnóstico" : "Diagnostics", sub: es ? "por qué falla" : "why it fails" }] : []),
    { id: "context", label: "Context", sub: es ? "teoría y ecuaciones" : "theory + equations" },
  ];

  if (err) {
    return (
      <div className="pl-work">
        <aside className="pl-rail">{selector}</aside>
        <section className="pl-stage"><div className="banner error">⚠ {err}</div></section>
        <aside className="pl-rail" />
      </div>
    );
  }
  if (!manifest || !active) {
    return (
      <div className="pl-work">
        <aside className="pl-rail">{selector}</aside>
        <section className="pl-stage"><div className="loading">{es ? "Cargando…" : "Loading…"}</div></section>
        <aside className="pl-rail" />
      </div>
    );
  }

  const Kit = resolveKit(manifest.view_kit);
  const l2 = active.metrics.l2_relative;

  const stage =
    tab === "compare" ? (
      <CompareKit manifest={manifest} lang={lang} />
    ) : tab === "diagnostics" ? (
      <DiagnosticsKit manifest={manifest} lang={lang} />
    ) : tab === "field" ? (
      <Kit key={manifest.case_id} manifest={manifest} trace={trace} active={active} lang={lang} />
    ) : tab === "live" ? (
      <LivePanel manifest={manifest} lang={lang} />
    ) : tab === "charts" ? (
      <VariantCharts variants={manifest.variants} activeId={active.id} onSelect={setActiveId} lang={lang} />
    ) : (
      <div className="prose">{description}</div>
    );

  return (
    <div className="pl-work">
      {/* LEFT rail: case selector (from AppPage) + regime + view switch */}
      <aside className="pl-rail">
        {selector}
        <hr className="pl-rail-sep" />
        <h4>
          {es ? "Régimen" : "Regime"} ({manifest.variants.length}){" "}
          <span className={"badge " + (manifest.lane === "live" ? "live" : "precomputed")}>
            {manifest.lane === "live" ? "live" : "precompute"}
          </span>
        </h4>
        <div className="variant-chips">
          {manifest.variants.map((v) => (
            <button key={v.id} className={"variant-chip" + (v.id === active.id ? " active" : "")} onClick={() => setActiveId(v.id)}>
              {es ? v.label_es : v.label_en}
            </button>
          ))}
        </div>
        {active && <p className="variant-note">{es ? active.note_es : active.note_en}</p>}
        <h4>{es ? "Vista" : "View"}</h4>
        <div className="pl-viewbtns">
          {VIEWS.map((v) => (
            <button key={v.id} type="button" className={"pl-viewbtn" + (v.id === tab ? " on" : "")} onClick={() => setTab(v.id)} aria-pressed={v.id === tab}>
              {v.label}
              <small>{v.sub}</small>
            </button>
          ))}
        </div>
      </aside>

      {/* CENTER stage: the active view, full width */}
      <section className="pl-stage" aria-label={manifest.title}>
        {stage}
      </section>

      {/* RIGHT rail: governing equation + metrics + expected band */}
      <aside className="pl-rail">
        <h4>{es ? "Ecuación gobernante" : "Governing equation"}</h4>
        <div className="pl-right-eq"><Equation tex={manifest.governing_equations} /></div>
        <h4>{es ? "Métricas" : "Metrics"}</h4>
        <div className="pl-metrics">
          <span className="k">{es ? "Método" : "Method"}</span><span className="v mono">{manifest.method}</span>
          <span className="k">{es ? "Motor" : "Engine"}</span><span className="v mono">{manifest.engine.framework}</span>
          <span className="k">L2</span><span className="v mono">{typeof l2 === "number" ? l2.toExponential(2) : String(l2)}</span>
          <span className="k">{es ? "Paridad ONNX" : "ONNX parity"}</span><span className="v mono">{manifest.onnx.parity_max_abs.toExponential(1)}</span>
        </div>
        <h4>{es ? "Banda esperada" : "Expected band"}</h4>
        <p className="muted" style={{ fontSize: 12.5, margin: 0, lineHeight: 1.5 }}>{manifest.expected_band}</p>
      </aside>
    </div>
  );
}
