import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";

import { CATEGORY_LABELS, type CaseManifest } from "../lib/contract";
import { loadIndex, loadManifest } from "../lib/data";
import { useUI } from "../store";

/** Per-category narrative: what the group explores, which SOTA methods it exercises, and the honest caveat. */
const CATEGORY_INTRO: Record<string, { en: string; es: string }> = {
  "canonical-benchmark": {
    en: "The reference layer: textbook PDEs with a known analytic or spectral solution, so the relative-L2 measures the true error, not a modelling choice. They span the elliptic (Poisson), parabolic (heat), hyperbolic (wave) and nonlinear/stiff (Burgers, Allen-Cahn) regimes, and exercise the core methods — hard-constraint output transforms, SIREN, and RAR adaptive sampling for sharp fronts. If a method cannot pass here, it does not graduate to the applied groups.",
    es: "La capa de referencia: EDPs de manual con solución analítica o espectral conocida, así el L2 relativo mide el error verdadero, no una decisión de modelado. Cubren los regímenes elíptico (Poisson), parabólico (calor), hiperbólico (onda) y no lineal/rígido (Burgers, Allen-Cahn), y ejercitan los métodos núcleo — restricciones duras por output-transform, SIREN y muestreo adaptativo RAR para frentes abruptos. Si un método no aprueba aquí, no pasa a los grupos aplicados.",
  },
  "mining-mineral-processing": {
    en: "Process-engineering PDEs and population balances from comminution, flotation, thickening and heap leaching. These are illustrative-synthetic — physically faithful structures with engineering-scale values, validated against analytic or numerical references, NOT calibrated to a specific plant. They show how the same PINN machinery carries to reactive transport, kinetics and settling.",
    es: "EDPs de ingeniería de procesos y balances poblacionales de conminución, flotación, espesamiento y lixiviación en pilas. Son ilustrativo-sintéticos — estructuras físicamente fieles con valores a escala de ingeniería, validados contra referencias analíticas o numéricas, NO calibrados a una planta específica. Muestran cómo la misma maquinaria PINN se traslada a transporte reactivo, cinética y sedimentación.",
  },
  "pollution-environmental": {
    en: "Transport and remediation: an advecting-diffusing pollutant patch (time-scrubber), a low-permeability barrier that slows a plume (domain decomposition), and an inverse case on REAL measured soil-temperature data with uncertainty. The honesty label is explicit per case — exact-solution anchors where one exists, real data where claimed.",
    es: "Transporte y remediación: un parche contaminante que advecta y difunde (scrubber temporal), una barrera de baja permeabilidad que frena un penacho (descomposición de dominio), y un caso inverso sobre datos REALES de temperatura de suelo con incertidumbre. La etiqueta de honestidad es explícita por caso — anclas de solución exacta donde existen, datos reales donde se declaran.",
  },
  "industrial-fluids-heat": {
    en: "Harder forward problems: high-wavenumber Helmholtz (the spectral-bias showcase that needs Fourier features) and lid-driven cavity flow (coupled velocity–pressure with gauge weighting). These push the CPU lane — their relative-L2 is reported honestly, with the note that a GPU and frequency annealing tighten it further.",
    es: "Problemas directos más duros: Helmholtz de alto número de onda (la vitrina del sesgo espectral que necesita Fourier features) y flujo en cavidad con tapa móvil (velocidad–presión acopladas con pesado de gauge). Estos exigen el carril CPU — su L2 relativo se reporta honestamente, con la nota de que una GPU y el recocido de frecuencias lo aprietan más.",
  },
};

export function Experiments() {
  const { t } = useTranslation();
  const lang = useUI((s) => s.lang);
  const [rows, setRows] = useState<CaseManifest[]>([]);

  useEffect(() => {
    loadIndex().then((ix) => Promise.all(ix.cases.map((c) => loadManifest(c.case_id))).then(setRows));
  }, []);

  const byCat: Record<string, CaseManifest[]> = {};
  for (const m of rows) (byCat[m.category] ||= []).push(m);

  return (
    <div className="prose" style={{ maxWidth: 1040 }}>
      <h1>{lang === "es" ? "Experimentos" : "Experiments"}</h1>
      <p className="muted">
        {lang === "es"
          ? "Resumen por categoría de los casos horneados: método, motor, lane y error relativo vs la referencia."
          : "Per-category summary of the baked cases: method, engine, lane, and relative error vs the reference."}
      </p>
      {Object.entries(byCat).map(([cat, ms]) => (
        <div key={cat} className="panel" style={{ marginBottom: 16 }}>
          <h3>{CATEGORY_LABELS[cat]?.[lang] ?? cat}</h3>
          {CATEGORY_INTRO[cat] && (
            <p style={{ fontSize: 14 }}>{CATEGORY_INTRO[cat][lang]}</p>
          )}
          <table className="tbl">
            <thead>
              <tr>
                <th>case</th>
                <th>{t("app.method")}</th>
                <th>{t("app.engine")}</th>
                <th>{t("app.lane")}</th>
                <th>{t("app.relL2")}</th>
              </tr>
            </thead>
            <tbody>
              {ms.map((m) => (
                <tr key={m.case_id}>
                  <td className="mono">{m.case_id}</td>
                  <td>{m.method}</td>
                  <td>{m.engine.framework}</td>
                  <td>
                    <span className={`tag ${m.lane}`}>{m.lane}</span>
                  </td>
                  <td className="mono">
                    {m.variants[0]?.metrics?.l2_relative !== undefined ? Number(m.variants[0].metrics.l2_relative).toExponential(2) : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ))}
    </div>
  );
}
