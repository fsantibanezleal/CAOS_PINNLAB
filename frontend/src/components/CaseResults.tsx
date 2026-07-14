import type { ReactNode } from "react";

import type { CaseManifest, FieldTrace, VariantEntry } from "../lib/contract";
import { RESULTS } from "../content/results";

/** THE RESULTS TAB (real-review plan §2.2.1): the result itself, nothing else, at full view with no scroll.
 *  LEFT: THE ANSWER as a plain sentence, its computed values, THE VERDICT (from the baked numbers), and the
 *  evidence links. RIGHT: the case's key visualization filling the stage height, with its answer markers.
 *  The question lives on the rail's case card; the narrative lives in Context > Summary; the regime control
 *  lives on the rail: nothing here competes with the result. */
export function CaseResults({
  manifest,
  trace,
  active,
  lang,
  kit: Kit,
  onGoto,
}: {
  manifest: CaseManifest;
  trace: FieldTrace | null;
  active: VariantEntry;
  lang: "en" | "es";
  kit: (p: { manifest: CaseManifest; trace: FieldTrace | null; active: VariantEntry; lang: "en" | "es" }) => ReactNode;
  onGoto: (view: string) => void;
}) {
  const es = lang === "es";
  const r = RESULTS[manifest.case_id];
  const e = manifest.estimate;

  const EVIDENCE: { view: string; label: string; proves: string }[] = [
    ...(manifest.comparison
      ? [{ view: "compare", label: es ? "Comparar" : "Compare", proves: es ? "la referencia clásica vs cada PINN, con mapas de error" : "the classical reference vs each PINN, with error maps" }]
      : []),
    { view: "field", label: es ? "Campo" : "Field", proves: es ? "el campo completo interactivo" : "the full interactive field" },
    { view: "live", label: "Live", proves: es ? "la red corriendo en tu navegador" : "the network running in your browser" },
    ...(manifest.training
      ? [{ view: "training", label: es ? "Entrenamiento" : "Training", proves: es ? "mira a la variante ingenua NO aprender" : "watch the naive variant FAIL to learn" }]
      : []),
    ...(manifest.diagnostics
      ? [{ view: "diagnostics", label: es ? "Diagnóstico" : "Diagnostics", proves: es ? "validación independiente" : "independent validation" }]
      : []),
  ];

  return (
    <div className="pl-results">
      <section className="pl-res-answer">
        <span className="pl-ans-eyebrow">{es ? "LA RESPUESTA (computada offline)" : "THE ANSWER (computed offline)"}</span>
        {r && <p className="pl-res-answertext">{es ? r.answer_es : r.answer_en}</p>}
        {e && (
          <div className="pl-ans-grid">
            {e.items.map((it, i) => (
              <div key={i} className="pl-ans-item">
                <span className="pl-ans-label">{es ? it.label_es : it.label_en}</span>
                <span className="pl-ans-value mono">
                  {it.value !== undefined ? it.value : (it.values?.[active.id] ?? Object.values(it.values ?? {})[0])}
                </span>
              </div>
            ))}
          </div>
        )}
        {r && (
          <p className="pl-res-verdict">
            <b>{es ? "Veredicto" : "Verdict"}</b> {es ? r.verdict_es : r.verdict_en}
          </p>
        )}
        <div className="pl-res-evidence">
          <span className="pl-ans-eyebrow">{es ? "LA EVIDENCIA" : "THE EVIDENCE"}</span>
          <ul>
            {EVIDENCE.map((ev) => (
              <li key={ev.view}>
                <button type="button" className="pl-res-evlink" onClick={() => onGoto(ev.view)}>{ev.label}</button>
                <span> {ev.proves}</span>
              </li>
            ))}
          </ul>
        </div>
      </section>
      <section className="pl-res-viz">
        <Kit manifest={manifest} trace={trace} active={active} lang={lang} />
      </section>
    </div>
  );
}
