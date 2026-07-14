import type { CaseManifest } from "../lib/contract";

/** THE QUESTION then THE ESTIMATE (issue #46): the first thing a visitor reads on a case is the engineering
 *  question it answers, and the answer, computed offline by build_estimates.py from the baked artifacts (never
 *  asserted). Per-variant items react to the active regime; clicking another regime's value switches to it, so
 *  the card doubles as a fast what-if over the parameter family. */
export function AnswerCard({
  manifest,
  activeId,
  onSelect,
  lang,
}: {
  manifest: CaseManifest;
  activeId: string;
  onSelect: (variantId: string) => void;
  lang: "en" | "es";
}) {
  const es = lang === "es";
  const e = manifest.estimate;
  if (!e) return null;

  const variantLabel = (vid: string) =>
    manifest.variants.find((v) => v.id === vid)?.[es ? "label_es" : "label_en"] ?? vid;

  return (
    <div className="pl-answer" role="region" aria-label={es ? "La pregunta y la estimación" : "The question and the estimate"}>
      <div className="pl-ans-q">
        <span className="pl-ans-eyebrow">{es ? "LA PREGUNTA" : "THE QUESTION"}</span>
        <p className="pl-ans-question">{es ? e.question_es : e.question_en}</p>
        <p className="pl-ans-why">
          <span className="pl-ans-eyebrow2">{es ? "Por qué una PINN" : "Why a PINN"}</span> {es ? e.why_es : e.why_en}
        </p>
      </div>
      <div className="pl-ans-items">
        <span className="pl-ans-eyebrow">{es ? "LA ESTIMACIÓN (computada offline)" : "THE ESTIMATE (computed offline)"}</span>
        <div className="pl-ans-grid">
          {e.items.map((it, i) => (
            <div key={i} className="pl-ans-item">
              <span className="pl-ans-label">{es ? it.label_es : it.label_en}</span>
              {it.value !== undefined ? (
                <span className="pl-ans-value mono">{it.value}</span>
              ) : (
                <>
                  <span className="pl-ans-value mono">{it.values?.[activeId] ?? Object.values(it.values ?? {})[0]}</span>
                  {it.values && Object.keys(it.values).length > 1 && (
                    <span className="pl-ans-variants">
                      {Object.entries(it.values).map(([vid, val]) => (
                        <button
                          key={vid}
                          type="button"
                          className={"pl-ans-vchip" + (vid === activeId ? " on" : "")}
                          title={val}
                          onClick={() => onSelect(vid)}
                          aria-pressed={vid === activeId}
                        >
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
      </div>
    </div>
  );
}
