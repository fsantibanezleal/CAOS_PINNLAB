import type { VariantEntry } from "../lib/contract";

/** The `Charts` sub-tab: a clickable side-by-side comparison of every baked regime: the relative-L2 (vs the anchor)
 *  per variant as a bar (lower is better). Click a row to load that regime everywhere (mirrors SimLab's
 *  VariantComparison: point -> load regime). */
export function VariantCharts({
  variants,
  activeId,
  onSelect,
  lang,
}: {
  variants: VariantEntry[];
  activeId: string;
  onSelect: (id: string) => void;
  lang: "en" | "es";
}) {
  const es = lang === "es";
  const vals = variants.map((v) => {
    const m = v.metrics.l2_relative;
    return typeof m === "number" ? m: NaN;
  });
  const finite = vals.filter((x) => Number.isFinite(x));
  const max = finite.length ? Math.max(...finite): 1;

  return (
    <div className="charts">
      <h3>{es ? "Comparación de regímenes: error relativo-L2 vs la referencia": "Regime comparison: relative-L2 vs the reference"}</h3>
      <p className="muted" style={{ marginTop: 0 }}>
        {es
          ? "Una barra por variante (menor es mejor). Clic en una fila para cargar ese régimen en todas las vistas."
         : "One bar per variant (lower is better). Click a row to load that regime in every view."}
      </p>
      <div className="cmp-bars">
        {variants.map((v, i) => {
          const l2 = vals[i];
          const pct = Number.isFinite(l2) ? Math.max(2, (l2 / max) * 100): 0;
          const active = v.id === activeId;
          return (
            <button
              key={v.id}
              className={"cmp-row" + (active ? " active": "")}
              onClick={() => onSelect(v.id)}
              title={es ? v.label_es: v.label_en}
            >
              <span className="cmp-label">{es ? v.label_es: v.label_en}</span>
              <span className="cmp-track">
                <span className="cmp-fill" style={{ width: `${pct}%` }} />
              </span>
              <span className="cmp-val mono">{Number.isFinite(l2) ? l2.toExponential(2): ": "}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
