import { useEffect, useState, type ReactNode } from "react";

import type { CaseManifest, FieldTrace } from "../lib/contract";
import { CONSTRAINTS, PIN_LABELS } from "../content/constraints";
import { RESULTS } from "../content/results";
import { SCENARIOS } from "../content/scenarios";
import { loadManifest, loadTrace } from "../lib/data";
import { CaseResults } from "./CaseResults";
import { Equation } from "./Equation";
import { SectionPager } from "./SectionPager";
import { CompareKit } from "./kits/CompareKit";
import { DiagnosticsKit } from "./kits/DiagnosticsKit";
import { TrainingKit } from "./kits/TrainingKit";
import { resolveKit } from "./kits/registry";
import { LivePanel } from "./LivePanel";
import { VariantCharts } from "./VariantCharts";

type TabId = "results" | "context" | "field" | "live" | "compare" | "training" | "diagnostics" | "charts";

/** One case = the investigation workbench (real-review plan 2026-07-15, issue #49). Owner-fixed structure:
 *  LEFT RAIL (one purpose set, stable for every case): choose the case (passed in as `selector`) + the case's
 *  context chips (what pins the solution) + the REGIME parameter control, once. RIGHT AREA: the governing
 *  EQUATION as a full-width, always-visible strip (the case's identity, invariant across views) above the
 *  tabs; tabs in the owner-fixed order (Results, Context, then the evidence); the STAGE filling all remaining
 *  height, with every tab designed to fit it at full view: no scrolling. */
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
  /** A view to land on (deep link), applied when it exists for this case (issue #38). */
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
    if (!manifest) return;
    const avail = new Set<TabId>(["results", "context", "field", "charts"]);
    if (manifest.lane === "live") avail.add("live");
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

  // Owner-fixed order: Results, Context, then the evidence views.
  const VIEWS: { id: TabId; label: string; sub: string }[] = [
    { id: "results", label: es ? "Resultados" : "Results", sub: es ? "la respuesta" : "the answer" },
    { id: "context", label: es ? "Contexto" : "Context", sub: es ? "resumen, teoría y método" : "summary, theory + method" },
    { id: "field", label: es ? "Campo" : "Field", sub: es ? "campo interactivo" : "interactive field" },
    // Live only for the live lane: a precompute (field->field operator / chaotic ODE) case has no coordinate-driven
    // in-browser inference, so it does not get an empty Live tab (issue #49 E5 audit).
    ...(manifest?.lane === "live" ? [{ id: "live" as TabId, label: "Live", sub: es ? "inferencia en el navegador" : "in-browser inference" }] : []),
    ...(manifest?.comparison ? [{ id: "compare" as TabId, label: es ? "Comparar" : "Compare", sub: es ? "estándar vs PINN" : "standard vs PINN" }] : []),
    ...(manifest?.training ? [{ id: "training" as TabId, label: es ? "Entrenamiento" : "Training", sub: es ? "míralo aprender" : "watch it learn" }] : []),
    ...(manifest?.diagnostics ? [{ id: "diagnostics" as TabId, label: es ? "Diagnóstico" : "Diagnostics", sub: es ? "validación independiente" : "independent validation" }] : []),
    { id: "charts", label: es ? "Regímenes" : "Regimes", sub: es ? "los números por régimen" : "the per-regime numbers" },
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
  const sc = SCENARIOS[manifest.case_id];
  const pins = CONSTRAINTS[manifest.case_id];
  const e = manifest.estimate;

  // CONTEXT (plan §2.2.2): a sectioned, full-width, no-scroll view. First section = the case SUMMARY (the
  // narrative: situation / what the model calculates / know vs seek / how the PINN helps + assumptions);
  // the deep prose then auto-splits by its own headings.
  const summary = (
    <div className="pl-sumgrid">
      {sc && (
        <section className="pl-res-card">
          <h4>{es ? "La situación" : "The situation"}</h4>
          <p>{es ? sc.situation_es : sc.situation_en}</p>
        </section>
      )}
      {r && (
        <section className="pl-res-card">
          <h4>{es ? "Qué calcula el modelo" : "What the model calculates"}</h4>
          <p>{es ? r.calculates_es : r.calculates_en}</p>
        </section>
      )}
      {sc && (
        <section className="pl-res-card">
          <h4>{es ? "Qué sabemos y qué buscamos" : "What we know and what we seek"}</h4>
          <p>{es ? sc.measured_es : sc.measured_en}</p>
        </section>
      )}
      {(e || r) && (
        <section className="pl-res-card">
          <h4>{es ? "Cómo ayuda la PINN, y bajo qué supuestos" : "How the PINN helps, and under what assumptions"}</h4>
          {e && <p>{es ? e.why_es : e.why_en}</p>}
          {r && (
            <ul className="pl-res-assumptions">
              {(es ? r.assumptions_es : r.assumptions_en).map((a, i) => <li key={i}>{a}</li>)}
            </ul>
          )}
        </section>
      )}
    </div>
  );

  const stage =
    tab === "results" ? (
      <CaseResults manifest={manifest} trace={trace} active={active} lang={lang} kit={Kit} onGoto={(v) => switchTab(v as TabId)} />
    ) : tab === "context" ? (
      <SectionPager key={manifest.case_id + lang} lang={lang} extra={[{ title: es ? "Resumen" : "Summary", body: summary }]}>
        {description}
      </SectionPager>
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
      {/* LEFT RAIL: choose the case (selector) + context chips + the REGIME control, once (owner-defined purpose) */}
      <aside className="pl-rail">
        {selector}
        {pins && (
          <div className="pl-railpins">
            <h4>{es ? "Qué fija la solución" : "What pins the solution"}</h4>
            <div className="pl-pins">
              {pins.map((pin, i) => (
                <span key={i} className={"pin pin-" + pin.kind}>
                  <b>{PIN_LABELS[pin.kind][lang]}</b> {es ? pin.es : pin.en}
                </span>
              ))}
            </div>
          </div>
        )}
        {manifest.variants.length > 1 && (
          <div className="pl-railregime">
            <h4>{es ? "Régimen" : "Regime"} ({manifest.variants.length})</h4>
            <div className="variant-chips">
              {manifest.variants.map((v) => (
                <button key={v.id} className={"variant-chip" + (v.id === active.id ? " active" : "")} title={es ? v.note_es : v.note_en} onClick={() => setActiveId(v.id)}>
                  {es ? v.label_es : v.label_en}
                </button>
              ))}
            </div>
            <p className="variant-note">{es ? active.note_es : active.note_en}</p>
          </div>
        )}
      </aside>

      {/* RIGHT AREA: the equation (full width, ALWAYS visible) -> tabs -> the stage that every tab must fit */}
      <div className="pl-main">
        <div className="pl-eqrow">
          <Equation tex={manifest.governing_equations} />
        </div>
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
