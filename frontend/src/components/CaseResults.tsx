import type { ReactNode } from "react";

import type { CaseManifest, FieldTrace, VariantEntry } from "../lib/contract";
import { RESULTS } from "../content/results";
import { SCENARIOS } from "../content/scenarios";
import { CONSTRAINTS, PIN_LABELS } from "../content/constraints";
import { Equation } from "./Equation";

/** THE RESULTS TAB (unified remediation plan §3.1, issue #49): the landing view of every case, owner-ordered
 *  first. A designed screen, not a scroll: LEFT = the investigation narrative in plain language (the situation,
 *  what the model calculates, what we know vs seek, how the PINN helps + declared assumptions, THE ANSWER as a
 *  sentence with its values, and THE VERDICT derived from the baked numbers). RIGHT = the case's key
 *  visualization (the view kit) + the evidence index: one line per remaining tab stating what it proves. */
export function CaseResults({
  manifest,
  trace,
  active,
  lang,
  kit: Kit,
  onGoto,
  onSelectVariant,
}: {
  manifest: CaseManifest;
  trace: FieldTrace | null;
  active: VariantEntry;
  lang: "en" | "es";
  kit: (p: { manifest: CaseManifest; trace: FieldTrace | null; active: VariantEntry; lang: "en" | "es" }) => ReactNode;
  onGoto: (view: string) => void;
  onSelectVariant: (id: string) => void;
}) {
  const es = lang === "es";
  const r = RESULTS[manifest.case_id];
  const sc = SCENARIOS[manifest.case_id];
  const e = manifest.estimate;
  const pins = CONSTRAINTS[manifest.case_id];

  const variantLabel = (vid: string) =>
    manifest.variants.find((v) => v.id === vid)?.[es ? "label_es" : "label_en"] ?? vid;

  const EVIDENCE: { view: string; label: string; proves: string }[] = [
    ...(manifest.comparison
      ? [{ view: "compare", label: es ? "Comparar" : "Compare", proves: es ? "el mismo problema resuelto por la referencia clásica y por cada variante de PINN, campo contra campo, con sus mapas de error" : "the same problem solved by the classical reference and by each PINN variant, field against field, with the error maps" }]
      : []),
    { view: "field", label: es ? "Campo" : "Field", proves: es ? "el campo completo, interactivo: pasa el cursor para leer valores; clic fija las curvas de corte" : "the full field, interactive: hover to read values; click pins the profile cuts" },
    { view: "live", label: "Live", proves: es ? "la red entrenada corriendo AHORA en tu navegador: mueve los controles y re-evalúa la física" : "the trained network running NOW in your browser: move the controls and it re-evaluates the physics" },
    ...(manifest.training
      ? [{ view: "training", label: es ? "Entrenamiento" : "Training", proves: es ? "el campo en instantes reales del entrenamiento: mira a la variante ingenua NO aprender" : "the field at real training checkpoints: watch the naive variant FAIL to learn" }]
      : []),
    ...(manifest.diagnostics
      ? [{ view: "diagnostics", label: es ? "Diagnóstico" : "Diagnostics", proves: es ? "la validación independiente: perfiles contra la referencia publicada o los datos retenidos" : "the independent validation: profiles against the published reference or held-out data" }]
      : []),
    { view: "charts", label: "Charts", proves: es ? "los números de error de cada régimen, comparados" : "each regime's error numbers, compared" },
  ];

  return (
    <div className="pl-results">
      <div className="pl-res-narrative">
        {sc && (
          <section className="pl-res-block">
            <h4>{es ? "La situación" : "The situation"}</h4>
            <p>{es ? sc.situation_es : sc.situation_en}</p>
          </section>
        )}
        {r && (
          <section className="pl-res-block">
            <h4>{es ? "Qué calcula el modelo" : "What the model calculates"}</h4>
            <p>{es ? r.calculates_es : r.calculates_en}</p>
            <div className="pl-res-eq"><Equation tex={manifest.governing_equations} /></div>
          </section>
        )}
        <section className="pl-res-block">
          <h4>{es ? "Qué sabemos y qué buscamos" : "What we know and what we seek"}</h4>
          {sc && <p>{es ? sc.measured_es : sc.measured_en}</p>}
          {pins && (
            <div className="pl-pins">
              {pins.map((pin, i) => (
                <span key={i} className={"pin pin-" + pin.kind}>
                  <b>{PIN_LABELS[pin.kind][lang]}</b> {es ? pin.es : pin.en}
                </span>
              ))}
            </div>
          )}
        </section>
        {(e || r) && (
          <section className="pl-res-block">
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

      <div className="pl-res-answercol">
        <section className="pl-res-answer">
          <h4>{es ? "La respuesta" : "The answer"}</h4>
          {r && <p className="pl-res-answertext">{es ? r.answer_es : r.answer_en}</p>}
          {e && (
            <div className="pl-ans-grid">
              {e.items.map((it, i) => (
                <div key={i} className="pl-ans-item">
                  <span className="pl-ans-label">{es ? it.label_es : it.label_en}</span>
                  {it.value !== undefined ? (
                    <span className="pl-ans-value mono">{it.value}</span>
                  ) : (
                    <>
                      <span className="pl-ans-value mono">{it.values?.[active.id] ?? Object.values(it.values ?? {})[0]}</span>
                      {it.values && Object.keys(it.values).length > 1 && (
                        <span className="pl-ans-variants">
                          {Object.entries(it.values).map(([vid, val]) => (
                            <button key={vid} type="button" className={"pl-ans-vchip" + (vid === active.id ? " on" : "")} title={val} onClick={() => onSelectVariant(vid)} aria-pressed={vid === active.id}>
                              {variantLabel(vid)}
                            </button>
                          ))}
                        </span>
                      )}
                    </>
                  )}
                </div>
              ))}
            </div>
          )}
          {r && (
            <p className="pl-res-verdict">
              <b>{es ? "Veredicto" : "Verdict"}</b> {es ? r.verdict_es : r.verdict_en}
            </p>
          )}
        </section>
        <section className="pl-res-viz">
          <Kit manifest={manifest} trace={trace} active={active} lang={lang} />
        </section>
        <section className="pl-res-evidence">
          <h4>{es ? "La evidencia (las demás pestañas)" : "The evidence (the other tabs)"}</h4>
          <ul>
            {EVIDENCE.map((ev) => (
              <li key={ev.view}>
                <button type="button" className="pl-res-evlink" onClick={() => onGoto(ev.view)}>{ev.label}</button>
                <span> {ev.proves}</span>
              </li>
            ))}
          </ul>
        </section>
      </div>
    </div>
  );
}
