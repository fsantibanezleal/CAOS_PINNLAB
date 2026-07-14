import type { ReactNode } from "react";

import type { CaseManifest, FieldTrace, VariantEntry } from "../lib/contract";
import { RESULTS } from "../content/results";

/** THE RESULTS TAB (real-review plan §2.2.1, revised 2026-07-15 after the "single image = regression" note):
 *  the RICH landing view: a substantial answer column (question, the computed answer in prominent stat tiles,
 *  the verdict, the evidence links) beside the case's FULL interactive visualization (the real fitted kit, not
 *  a dumbed-down single image). Both fit the stage: no scroll. */
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
    { view: "field", label: es ? "Campo" : "Field", proves: es ? "el campo completo, más grande, con cortes y zoom" : "the full field, larger, with cuts and zoom" },
    { view: "live", label: "Live", proves: es ? "la red corriendo en tu navegador" : "the network running in your browser" },
    ...(manifest.training
      ? [{ view: "training", label: es ? "Entrenamiento" : "Training", proves: es ? "mira a la variante ingenua NO aprender" : "watch the naive variant FAIL to learn" }]
      : []),
    ...(manifest.diagnostics
      ? [{ view: "diagnostics", label: es ? "Diagnóstico" : "Diagnostics", proves: es ? "validación independiente" : "independent validation" }]
      : []),
    { view: "context", label: es ? "Contexto" : "Context", proves: es ? "la teoría, el método y los supuestos" : "the theory, the method and the assumptions" },
  ];

  const answerVal = (it: NonNullable<typeof e>["items"][number]) =>
    it.value !== undefined ? it.value : (it.values?.[active.id] ?? Object.values(it.values ?? {})[0] ?? "");

  return (
    <div className="pl-results">
      <section className="pl-res-answer">
        {e && (
          <div className="pl-res-qrow">
            <span className="pl-ans-eyebrow">{es ? "LA PREGUNTA" : "THE QUESTION"}</span>
            <p className="pl-res-question">{es ? e.question_es : e.question_en}</p>
          </div>
        )}
        <span className="pl-ans-eyebrow">{es ? "LA RESPUESTA (computada offline)" : "THE ANSWER (computed offline)"}</span>
        {r && <p className="pl-res-answertext">{es ? r.answer_es : r.answer_en}</p>}
        {e && (
          <div className="pl-res-tiles">
            {e.items.map((it, i) => (
              <div key={i} className="pl-res-tile">
                <span className="pl-res-tileval mono">{answerVal(it)}</span>
                <span className="pl-res-tilelabel">{es ? it.label_es : it.label_en}</span>
              </div>
            ))}
          </div>
        )}
        {r && (
          <div className="pl-res-verdictbox">
            <span className="pl-ans-eyebrow good">{es ? "EL VEREDICTO" : "THE VERDICT"}</span>
            <p className="pl-res-verdict">{es ? r.verdict_es : r.verdict_en}</p>
          </div>
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
