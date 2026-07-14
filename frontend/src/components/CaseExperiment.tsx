import { useEffect, useState, type ReactNode } from "react";

import type { CaseManifest, FieldTrace } from "../lib/contract";
import { CONSTRAINTS, PIN_LABELS } from "../content/constraints";
import { RESULTS } from "../content/results";
import { loadManifest, loadTrace } from "../lib/data";
import { CaseResults } from "./CaseResults";
import { Equation } from "./Equation";
import { CompareKit } from "./kits/CompareKit";
import { DiagnosticsKit } from "./kits/DiagnosticsKit";
import { TrainingKit } from "./kits/TrainingKit";
import { resolveKit } from "./kits/registry";
import { LivePanel } from "./LivePanel";
import { VariantCharts } from "./VariantCharts";

type TabId = "results" | "context" | "field" | "live" | "compare" | "training" | "diagnostics" | "charts";

/** One case = an INVESTIGATION workbench (unified remediation plan, issue #49). Owner-fixed structure:
 *  the tabs start with RESULTS (the landing view: situation -> what the model calculates -> what we know ->
 *  how the PINN helps -> the answer + verdict), then CONTEXT (the theory), then the evidence views. The tabs
 *  sit ON the stage they control; the regime chips sit directly ABOVE the stage (control next to effect);
 *  the left rail carries ONLY case selection; the method/engine/error meta is a one-line stage footer. The
 *  instrument fits the screen: no page scroll to operate it. */
export function CaseExperiment({
  manifestId,
  selector,
  description,
  lang,
  requestedView,
  onViewChange,
}: {
  manifestId: string;
  selector: ReactNode;
  description: ReactNode;
  lang: "en" | "es";
  /** A view to land on (deep link / story chapter), applied when it exists for this case (issue #38). */
  requestedView?: string;
  /** Notifies the host when the user switches views, so the URL can follow (shareable deep links). */
  onViewChange?: (view: string) => void;
}) {
  const es = lang === "es";
  const [manifest, setManifest] = useState<CaseManifest | null>(null);
  const [activeId, setActiveId] = useState<string>("");
  const [trace, setTrace] = useState<FieldTrace | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [tab, setTab] = useState<TabId>("results");

  useEffect(() => {
    // Land on the requested view when it exists (deep link / story chapter); otherwise on RESULTS: every case
    // opens with what was asked and what was found (owner decision D1), never with raw solver output.
    if (!manifest) return;
    const avail = new Set<TabId>(["results", "context", "field", "live", "charts"]);
    if (manifest.comparison) avail.add("compare");
    if (manifest.training) avail.add("training");
    if (manifest.diagnostics) avail.add("diagnostics");
    const req = requestedView as TabId | undefined;
    setTab(req && avail.has(req) ? req : "results");
  }, [manifest?.case_id, requestedView]);

  function switchTab(id: TabId) {
    setTab(id);
    onViewChange?.(id);
  }

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

  // Owner-fixed order (D1): Results, Context, then the rest.
  const VIEWS: { id: TabId; label: string; sub: string }[] = [
    { id: "results", label: es ? "Resultados" : "Results", sub: es ? "la pregunta y la respuesta" : "the question and the answer" },
    { id: "context", label: es ? "Contexto" : "Context", sub: es ? "teoría y ecuaciones" : "theory + equations" },
    { id: "field", label: es ? "Campo" : "Field", sub: es ? "campo interactivo" : "interactive field" },
    { id: "live", label: "Live", sub: es ? "inferencia en el navegador" : "in-browser inference" },
    ...(manifest?.comparison ? [{ id: "compare" as TabId, label: es ? "Comparar" : "Compare", sub: es ? "estándar vs PINN" : "standard vs PINN" }] : []),
    ...(manifest?.training ? [{ id: "training" as TabId, label: es ? "Entrenamiento" : "Training", sub: es ? "míralo aprender" : "watch it learn" }] : []),
    ...(manifest?.diagnostics ? [{ id: "diagnostics" as TabId, label: es ? "Diagnóstico" : "Diagnostics", sub: es ? "validación independiente" : "independent validation" }] : []),
    { id: "charts", label: "Charts", sub: es ? "errores por régimen" : "regime errors" },
  ];

  if (err) {
    return (
      <div className="pl-work">
        <aside className="pl-rail">{selector}</aside>
        <section className="pl-stage"><div className="banner error">⚠ {err}</div></section>
      </div>
    );
  }
  if (!manifest || !active) {
    return (
      <div className="pl-work">
        <aside className="pl-rail">{selector}</aside>
        <section className="pl-stage"><div className="loading">{es ? "Cargando…" : "Loading…"}</div></section>
      </div>
    );
  }

  const Kit = resolveKit(manifest.view_kit);
  const l2 = active.metrics.l2_relative;
  const r = RESULTS[manifest.case_id];

  const stage =
    tab === "results" ? (
      <CaseResults manifest={manifest} trace={trace} active={active} lang={lang} kit={Kit} onGoto={(v) => switchTab(v as TabId)} onSelectVariant={setActiveId} />
    ) : tab === "context" ? (
      <div className="pl-ctxview">
        <div className="pl-ctx-eq"><Equation tex={manifest.governing_equations} /></div>
        {CONSTRAINTS[manifest.case_id] && (
          <div className="pl-pins" title={es ? "Qué fija la solución: una EDP sola tiene infinitas soluciones" : "What pins the solution down: a PDE alone has infinitely many solutions"}>
            <span className="pl-pins-label">{es ? "Qué fija la solución" : "What pins the solution"}</span>
            {CONSTRAINTS[manifest.case_id].map((pin, i) => (
              <span key={i} className={"pin pin-" + pin.kind}>
                <b>{PIN_LABELS[pin.kind][lang]}</b> {es ? pin.es : pin.en}
              </span>
            ))}
          </div>
        )}
        <div className="prose">{description}</div>
      </div>
    ) : tab === "compare" ? (
      <CompareKit manifest={manifest} lang={lang} />
    ) : tab === "training" ? (
      <TrainingKit manifest={manifest} lang={lang} />
    ) : tab === "diagnostics" ? (
      <DiagnosticsKit manifest={manifest} lang={lang} />
    ) : tab === "field" ? (
      <Kit key={manifest.case_id} manifest={manifest} trace={trace} active={active} lang={lang} />
    ) : tab === "live" ? (
      <LivePanel manifest={manifest} lang={lang} />
    ) : (
      <VariantCharts variants={manifest.variants} activeId={active.id} onSelect={setActiveId} lang={lang} />
    );

  return (
    <div className="pl-work">
      {/* LEFT rail: case selection ONLY (search + domain + list, from AppPage) */}
      <aside className="pl-rail">{selector}</aside>

      {/* MAIN: regime strip -> tabs ON the stage -> the stage filling the rest -> a one-line meta footer */}
      <div className="pl-main">
        {manifest.variants.length > 1 && (
          <div className="pl-regime-strip" role="group" aria-label={es ? "Régimen" : "Regime"}>
            <span className="pl-regime-label">{es ? "RÉGIMEN" : "REGIME"}</span>
            {manifest.variants.map((v) => (
              <button key={v.id} className={"variant-chip" + (v.id === active.id ? " active" : "")} title={es ? v.note_es : v.note_en} onClick={() => setActiveId(v.id)}>
                {es ? v.label_es : v.label_en}
              </button>
            ))}
            <span className="pl-regime-note muted">{es ? active.note_es : active.note_en}</span>
          </div>
        )}
        <div className="pl-stagetabs" role="tablist" aria-label={es ? "Vistas" : "Views"}>
          {VIEWS.map((v) => (
            <button key={v.id} type="button" role="tab" className={"pl-viewbtn" + (v.id === tab ? " on" : "")} onClick={() => switchTab(v.id)} aria-selected={v.id === tab} title={v.sub}>
              {v.label}
            </button>
          ))}
        </div>
        <section className="pl-stage" aria-label={manifest.title}>
          {stage}
        </section>
        <div className="pl-meta-footer mono">
          <span><b className="muted">{es ? "Método" : "Method"}</b> {manifest.method}</span>
          <span><b className="muted">{es ? "Motor" : "Engine"}</b> {manifest.engine.framework}</span>
          <span><b className="muted">L2</b> {typeof l2 === "number" ? l2.toExponential(2) : String(l2)}</span>
          <span><b className="muted">{es ? "Paridad ONNX" : "ONNX parity"}</b> {manifest.onnx.parity_max_abs.toExponential(1)}</span>
          {r && <span className="pl-meta-verdict" title={es ? r.verdict_es : r.verdict_en}><b className="muted">{es ? "Veredicto" : "Verdict"}</b> {(es ? r.verdict_es : r.verdict_en).split(":")[0]}</span>}
        </div>
      </div>
    </div>
  );
}
