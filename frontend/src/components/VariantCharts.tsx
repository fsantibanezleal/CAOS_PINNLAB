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

  // A one-variant "comparison" chart is meaningless (issue #49 review): for single-regime cases this tab
  // shows every measured number the bake produced, each explained in plain language, instead of a single bar.
  if (variants.length < 2) {
    const v = variants[0];
    const GLOSS: Record<string, { en: string; es: string }> = {
      l2_relative: { en: "overall error vs the reference (relative L2; lower is better)", es: "error global vs la referencia (L2 relativa; menor es mejor)" },
      max_abs_error: { en: "worst single-point error anywhere in the field", es: "peor error puntual en todo el campo" },
      onnx_parity_max_abs: { en: "trained net vs the exported browser model (should be ~0)", es: "red entrenada vs el modelo exportado al navegador (debe ser ~0)" },
      T_l2_relative: { en: "error of the auxiliary temperature field", es: "error del campo auxiliar de temperatura" },
      speed_rel_rmse_swept: { en: "current error INSIDE the dye-swept region", es: "error de corriente DENTRO de la región barrida" },
      speed_rel_rmse_dead: { en: "current error in the never-dyed dead zones", es: "error de corriente en las zonas muertas sin tinte" },
      swept_area_frac: { en: "fraction of the domain the dye swept", es: "fracción del dominio barrida por el tinte" },
      dye_holdout_rmse: { en: "reconstructed dye vs 160 held-out samples (never trained on)", es: "tinte reconstruido vs 160 muestras retenidas (nunca entrenadas)" },
      holdout_rmse_c: { en: "out-of-sample error at depths never seen (degC)", es: "error fuera de muestra en profundidades nunca vistas (°C)" },
      recovered_alpha_mm2_s: { en: "the recovered soil diffusivity (mm²/s)", es: "la difusividad de suelo recuperada (mm²/s)" },
      leave_time: { en: "seconds the surrogate holds the true trajectory", es: "segundos que el sustituto sigue la trayectoria verdadera" },
      uq_calibration_2sigma: { en: "fraction of truth inside the 2-sigma band (honest error bars)", es: "fracción de la verdad dentro de la banda 2 sigma (barras honestas)" },
    };
    const fmt = (x: number) => (Math.abs(x) >= 0.01 && Math.abs(x) < 10000 ? String(Math.round(x * 10000) / 10000) : x.toExponential(2));
    const entries = Object.entries(v.metrics).filter(([, val]) => typeof val === "number") as [string, number][];
    return (
      <div className="charts">
        <h3>{es ? "Los números medidos de este caso" : "This case's measured numbers"}</h3>
        <p className="muted" style={{ marginTop: 0 }}>
          {es
            ? "Este caso tiene un solo régimen, así que no hay barrido que comparar: estos son todos los números que la horneada midió, cada uno explicado. Los mapas de error viven en Comparar / Diagnóstico."
            : "This case has a single regime, so there is no sweep to compare: these are all the numbers the bake measured, each explained. The error maps live in Compare / Diagnostics."}
        </p>
        <div className="pl-ans-grid">
          {entries.map(([k, val]) => (
            <div key={k} className="pl-ans-item">
              <span className="pl-ans-label">{GLOSS[k] ? (es ? GLOSS[k].es : GLOSS[k].en) : k.replace(/_/g, " ")}</span>
              <span className="pl-ans-value mono">{fmt(val)}</span>
            </div>
          ))}
        </div>
      </div>
    );
  }

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
