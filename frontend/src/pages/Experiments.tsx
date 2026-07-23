import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";

import { SubTabs } from "../components/SubTabs";
import { CATEGORY_LABELS, type CaseManifest } from "../lib/contract";
import { loadIndex, loadManifest } from "../lib/data";
import { useUI } from "../store";

/** short tab label per category (the full CATEGORY_LABELS are long sentences unsuited to a tab strip). */
const CAT_TAB: Record<string, { en: string; es: string }> = {
  "canonical-benchmark": { en: "Benchmarks", es: "Benchmarks" },
  "mining-mineral-processing": { en: "Mining", es: "Minería" },
  "pollution-environmental": { en: "Pollution", es: "Polución" },
  "industrial-fluids-heat": { en: "Industrial", es: "Industrial" },
  control: { en: "Control", es: "Control" },
};

/** Per-category narrative: what the group explores, which SOTA methods it exercises, and the honest caveat. */
const CATEGORY_INTRO: Record<string, { en: string; es: string }> = {
  "canonical-benchmark": {
    en: "The reference layer: textbook PDEs with a known analytic or spectral solution, so the relative-L2 measures the true error, not a modelling choice. They span the elliptic (Poisson), parabolic (heat), hyperbolic (wave) and nonlinear/stiff (Burgers, Allen-Cahn) regimes, and exercise the core methods: hard-constraint output transforms, SIREN, and RAR adaptive sampling for sharp fronts. If a method cannot pass here, it does not graduate to the applied groups.",
    es: "La capa de referencia: EDPs de manual con solución analítica o espectral conocida, así el L2 relativo mide el error verdadero, no una decisión de modelado. Cubren los regímenes elíptico (Poisson), parabólico (calor), hiperbólico (onda) y no lineal/rígido (Burgers, Allen-Cahn), y ejercitan los métodos núcleo: restricciones duras por output-transform, SIREN y muestreo adaptativo RAR para frentes abruptos. Si un método no aprueba aquí, no pasa a los grupos aplicados.",
  },
  "mining-mineral-processing": {
    en: "Process-engineering PDEs and population balances from comminution, flotation, thickening and heap leaching. These are illustrative-synthetic: physically faithful structures with engineering-scale values, validated against analytic or numerical references, not calibrated to a specific plant. They show how the same PINN machinery carries to reactive transport, kinetics and settling.",
    es: "EDPs de ingeniería de procesos y balances poblacionales de conminución, flotación, espesamiento y lixiviación en pilas. Son ilustrativo-sintéticos: estructuras físicamente fieles con valores a escala de ingeniería, validados contra referencias analíticas o numéricas, no calibrados a una planta específica. Muestran cómo la misma maquinaria PINN se traslada a transporte reactivo, cinética y sedimentación.",
  },
  "pollution-environmental": {
    en: "Transport and remediation: an advecting-diffusing pollutant patch (time-scrubber), a low-permeability barrier that slows a plume (domain decomposition), and an inverse case on real measured soil-temperature data with uncertainty. The honesty label is explicit per case: exact-solution anchors where one exists, real data where claimed.",
    es: "Transporte y remediación: un parche contaminante que advecta y difunde (scrubber temporal), una barrera de baja permeabilidad que frena un penacho (descomposición de dominio), y un caso inverso sobre datos reales de temperatura de suelo con incertidumbre. La etiqueta de honestidad es explícita por caso: anclas de solución exacta donde existen, datos reales donde se declaran.",
  },
  "industrial-fluids-heat": {
    en: "Harder forward problems: high-wavenumber Helmholtz (the spectral-bias showcase that needs Fourier features) and lid-driven cavity flow (coupled velocity-pressure with gauge weighting). These push the CPU lane: their relative-L2 is reported honestly, with the note that a GPU and frequency annealing tighten it further.",
    es: "Problemas directos más duros: Helmholtz de alto número de onda (la vitrina del sesgo espectral que necesita Fourier features) y flujo en cavidad con tapa móvil (velocidad-presión acopladas con pesado de gauge). Estos exigen el carril CPU: su L2 relativo se reporta honestamente, con la nota de que una GPU y el recocido de frecuencias lo aprietan más.",
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

  const ladder = (
    <>
      <div className="panel" style={{ marginBottom: 0, borderColor: "var(--accent)" }}>
        <h3>{lang === "es" ? "La escalera de métodos: estándar vs PINN ingenua vs la corrección" : "The method ladder: standard vs naive vs adapted PINN"}</h3>
        <p style={{ fontSize: 14 }}>
          {lang === "es"
            ? "La tubería offline computa la comparación que la teoría describe: la solución estándar (forma cerrada o solucionador clásico) frente a la PINN ingenua (sin la corrección) frente a la corrección, más diagnósticos. La vista Comparar / Diagnóstico de cada caso muestra resultados reales precalculados, no afirmaciones."
            : "The offline pipeline computes the comparison the theory describes: the standard solution (closed-form or classical solver) vs the naive PINN (without the fix) vs the fix, plus diagnostics. Each case's Compare / Diagnostics view shows real baked results, not assertions."}
        </p>
        <table className="tbl">
          <thead><tr><th>case</th><th>{lang === "es" ? "estándar" : "standard"}</th><th>{lang === "es" ? "PINN ingenua" : "naive PINN"}</th><th>{lang === "es" ? "la corrección" : "the fix"}</th></tr></thead>
          <tbody>
            <tr><td className="mono">ind-helmholtz</td><td>FDM / spectral</td><td className="mono" style={{ color: "var(--bad)" }}>120.8%</td><td className="mono" style={{ color: "var(--good)" }}>Fourier 9.3%</td></tr>
            <tr><td className="mono">bench-allencahn</td><td>spectral ref</td><td className="mono" style={{ color: "var(--bad)" }}>95.4% (collapse)</td><td className="mono" style={{ color: "var(--good)" }}>hard+RAR 0.4%</td></tr>
            <tr><td className="mono">ind-heat2d-inverse</td><td>MMS k*</td><td className="mono" style={{ color: "var(--bad)" }}>356% (no data)</td><td className="mono" style={{ color: "var(--good)" }}>data+physics 4.0%</td></tr>
            <tr><td className="mono">poll-soil-barrier</td><td>MMS</td><td className="mono">single-domain ~19%</td><td className="mono">FBPINN ~19%</td></tr>
            <tr><td className="mono">bench-darcy-operator</td><td>finite-difference</td><td className="mono">—</td><td className="mono" style={{ color: "var(--good)" }}>FNO 2.5% (one pass)</td></tr>
            <tr><td className="mono">bench-navier-cavity</td><td>Ghia 1982</td><td className="mono">—</td><td className="mono" style={{ color: "var(--good)" }}>centerline RMSE 0.05</td></tr>
            <tr><td className="mono">env-soil-heat-real</td><td>real sensors</td><td className="mono">—</td><td className="mono" style={{ color: "var(--good)" }}>held-out ~1&deg;C</td></tr>
            <tr><td className="mono">bench-darcy-pino</td><td>finite-difference</td><td className="mono">FNO data-only 14.3%</td><td className="mono" style={{ color: "var(--good)" }}>PINO 7.8% (32 labels)</td></tr>
            <tr><td className="mono">dyn-pendulum-hnn</td><td>RK45</td><td className="mono" style={{ color: "var(--bad)" }}>MLP 7.4% E-drift</td><td className="mono" style={{ color: "var(--good)" }}>HNN 0.07%</td></tr>
            <tr><td className="mono">env-aquifer-test</td><td>Theis analytic</td><td className="mono" style={{ color: "var(--bad)" }}>PINN 367%</td><td className="mono" style={{ color: "var(--good)" }}>Cooper-Jacob 0.7%</td></tr>
          </tbody>
        </table>
        <p className="muted" style={{ fontSize: 12.5 }}>
          {lang === "es"
            ? "Honesto: donde una prueba justa no muestra contraste real (wave1d) o un solucionador divergió (una cavidad FDM apresurada), no se muestra ningún resultado inventado. Y donde la PINN pierde, se dice: en env-aquifer-test la corrección es el método clásico (Cooper-Jacob), no la red. Dos casos de operador van por otros ejes (conforme = cobertura, super-resolución = discretización) y aparecen en Benchmark. Abre la vista Comparar de un caso para ver los campos y los mapas de error."
            : "Honest: where a fair test shows no real contrast (wave1d) or a solver diverged (a rushed FDM cavity), no fabricated result is shown. And where the PINN loses, it says so: in env-aquifer-test the fix is the classical method (Cooper-Jacob), not the network. Two operator cases run on other axes (conformal = coverage, super-resolution = discretisation) and appear in Benchmark. Open a case's Compare view for the fields + error maps."}
        </p>
      </div>
    </>
  );

  const whatRun = (
    <>
      <div className="panel" style={{ marginBottom: 0 }}>
        <h3>{lang === "es" ? "Qué se ejecutó de verdad (y cómo se mantiene honesto)" : "What was actually run (and how it stays honest)"}</h3>
        <ul style={{ fontSize: 13.5, lineHeight: 1.6 }}>
          <li>{lang === "es"
            ? "Presupuesto justo. Cuando un caso compara dos carriles (ingenuo vs corrección), ambos se entrenan de cero con el mismo presupuesto (mismos pasos de Adam, misma receta): un contraste solo cuenta si las dos redes tuvieron la misma oportunidad. Un intento temprano que reutilizaba una red preentrenada invirtió el resultado de wave1d; se corrigió."
            : "Fair budget. When a case contrasts two lanes (naive vs fix), both are trained from scratch at the same budget (same Adam steps, same recipe): a contrast only counts if both networks got the same chance. An early attempt that reused a pre-trained net inverted wave1d's result; it was corrected."}</li>
          <li>{lang === "es"
            ? "La curva de identificabilidad (heat2d-inverse). Se entrenó el mismo inverso con n = 0, 10, 25, 50, 100 sensores a un presupuesto fijo: el error del campo k recuperado cae 356%, 17.3%, 16.3%, 13.6%, 12.6%. El acantilado está entre 0 y 10 sensores: cualquier dato de anclaje restaura la identificabilidad de la incógnita de campo; luego los retornos decrecen. Es la respuesta computada a 'cuánta información necesita una PINN'."
            : "The identifiability curve (heat2d-inverse). The same inverse was trained at n = 0, 10, 25, 50, 100 sensors at a fixed budget: the recovered-k field error falls 356%, 17.3%, 16.3%, 13.6%, 12.6%. The cliff is between 0 and 10 sensors: any anchoring data restores identifiability of the field unknown; then returns diminish. It is the computed answer to 'how much information a PINN needs'."}</li>
          <li>{lang === "es"
            ? "Validación fuera de muestra con datos reales (soil-heat-real). Se recuperó una difusividad (0.305 mm²/s) desde dos series reales de sensores NOAA, y se predijeron tres profundidades interiores nunca vistas por el optimizador: RMSE 1.24 / 1.05 / 0.75 °C a 10 / 20 / 50 cm. Es la única validación que cuenta: contra datos retenidos."
            : "Out-of-sample real-data validation (soil-heat-real). One diffusivity (0.305 mm²/s) was recovered from two real NOAA sensor series, and three interior depths the optimizer never saw were predicted: RMSE 1.24 / 1.05 / 0.75 °C at 10 / 20 / 50 cm. It is the only validation that counts: against held-out data."}</li>
          <li>{lang === "es"
            ? "El holdout de tinte (hidden-velocity, el caso insignia). De 800 muestras de tinte, 160 nunca se entrenaron; la ONNX exportada las reconstruye a 0.78% de RMSE. Y el resultado es honesto en su alcance: la corriente se recupera a 16% donde el tinte barrió, 37% en las zonas muertas que nunca vio (no identificable, para cualquier método): el mapa de dónde confiar es parte del resultado."
            : "The dye holdout (hidden-velocity, the flagship). Of 800 dye samples, 160 were never trained on; the exported ONNX reconstructs them to 0.78% RMSE. And the result is honest about its scope: the current is recovered to 16% where the dye swept, 37% in the dead zones it never saw (unidentifiable, for any method): the map of where to trust is part of the result."}</li>
          <li>{lang === "es"
            ? "Dinámica de entrenamiento capturada (helmholtz, allencahn). Se guardó el campo en checkpoints reales: la vista Entrenamiento reproduce al carril ingenuo sin aprender (Helmholtz clavado en ~100% mientras Fourier ya muestra el patrón), no lo afirma."
            : "Training dynamics captured (helmholtz, allencahn). The field was saved at real checkpoints: the Training view replays the naive lane failing to learn (Helmholtz stuck at ~100% while Fourier already shows the pattern), it does not assert it."}</li>
          <li>{lang === "es"
            ? "Resultados negativos publicados. Donde una prueba justa no dio contraste (wave1d, ~11% ambos carriles) o un solucionador divergió (una cavidad FDM apresurada, revertida a favor de las líneas de Ghia), no se inventa un número: el registro honesto es el producto."
            : "Negative results published. Where a fair test gave no contrast (wave1d, ~11% both lanes) or a solver diverged (a rushed FDM cavity, reverted in favor of the Ghia centerlines), no number is invented: the honest record is the product."}</li>
        </ul>
        <p className="muted" style={{ fontSize: 12.5, marginTop: 6 }}>
          {lang === "es"
            ? "Sigue a Benchmark para los números completos y cómo leerlos."
            : "Continue to Benchmark for the full numbers and how to read them."}
        </p>
      </div>
    </>
  );

  const catPanel = (cat: string, ms: CaseManifest[]) => (
    <div className="panel" style={{ marginBottom: 0 }}>
      <h3>{CATEGORY_LABELS[cat]?.[lang] ?? cat}</h3>
      {CATEGORY_INTRO[cat] && <p style={{ fontSize: 14 }}>{CATEGORY_INTRO[cat][lang]}</p>}
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
              <td><span className={`tag ${m.lane}`}>{m.lane}</span></td>
              <td className="mono">
                {m.variants[0]?.metrics?.l2_relative !== undefined ? Number(m.variants[0].metrics.l2_relative).toExponential(2): ": "}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );

  const tabs = [
    { id: "ladder", label: lang === "es" ? "Escalera de métodos" : "Method ladder", content: ladder },
    { id: "whatrun", label: lang === "es" ? "Qué se ejecutó" : "What was run", content: whatRun },
    ...Object.entries(byCat).map(([cat, ms]) => ({
      id: cat,
      label: CAT_TAB[cat]?.[lang] ?? cat,
      content: catPanel(cat, ms),
    })),
  ];

  return (
    <div className="prose" style={{ maxWidth: 1040 }}>
      <h1>{lang === "es" ? "Experimentos": "Experiments"}</h1>
      <p className="muted">
        {lang === "es"
          ? "Resumen por categoría de los casos precalculados: la escalera de métodos, qué se ejecutó de verdad, y método/motor/lane/error por grupo."
         : "Per-category summary of the baked cases: the method ladder, what was actually run, and method/engine/lane/error by group."}
      </p>
      {rows.length === 0 ? (
        <div className="loading">{lang === "es" ? "Cargando…" : "Loading…"}</div>
      ) : (
        <SubTabs ariaLabel={lang === "es" ? "Secciones de experimentos" : "Experiments sections"} tabs={tabs} />
      )}
    </div>
  );
}
