import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";

import { SubTabs } from "../components/SubTabs";
import type { CaseManifest } from "../lib/contract";
import { loadIndex, loadManifest } from "../lib/data";
import { useUI } from "../store";

function honestyClass(flag: string): string {
  return flag.startsWith("validated") ? "validated": "synthetic";
}

export function Benchmark() {
  const { t } = useTranslation();
  const lang = useUI((s) => s.lang);
  const [rows, setRows] = useState<CaseManifest[]>([]);

  useEffect(() => {
    loadIndex().then((ix) => Promise.all(ix.cases.map((c) => loadManifest(c.case_id))).then(setRows));
  }, []);

  const es = lang === "es";
  return (
    <div className="prose" style={{ maxWidth: 1160 }}>
      <h1>Benchmark</h1>
      <p className="muted">
        {es
          ? "Honestidad: qué estima cada caso (la cantidad de interés computada offline), error relativo vs la referencia (analítica/dataset/numérica), paridad ONNX-vs-modelo, y etiqueta de datos sintéticos vs reales. Sin números maquillados."
         : "Honesty: what each case estimates (the quantity of interest, computed offline), relative error vs the reference (analytic/dataset/numerical), ONNX-vs-model parity, and the synthetic-vs-real data label. No dressed-up numbers."}
      </p>
      {rows.length === 0 ? (
        <div className="loading">{es ? "Cargando…" : "Loading…"}</div>
      ) : (
      <SubTabs
        ariaLabel={es ? "Secciones del benchmark" : "Benchmark sections"}
        tabs={[
          { id: "numbers", label: es ? "Los números" : "The numbers", content: (
      <div className="panel" style={{ marginBottom: 0 }}>
        <table className="tbl">
          <thead>
            <tr>
              <th>case</th>
              <th>{es ? "qué estima" : "what it estimates"}</th>
              <th>anchor</th>
              <th>{t("app.relL2")}</th>
              <th>{es ? "ingenua vs estándar" : "naive vs standard"}</th>
              <th>{es ? "corrección vs estándar" : "fix vs standard"}</th>
              <th>{t("app.onnxParity")}</th>
              <th>{t("app.dataLabel")}</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((m) => {
              const s = (m.comparison?.summary ?? {}) as Record<string, number>;
              const pct = (v: number | undefined) => (typeof v === "number" ? (v * 100).toFixed(1) + "%" : "—");
              const it0 = m.estimate?.items?.[0];
              const qoiVal = it0 ? (it0.value ?? Object.values(it0.values ?? {})[0]) : undefined;
              return (
                <tr key={m.case_id}>
                  <td className="mono">
                    <a href={`#/?case=${m.case_id}${m.comparison ? "&view=compare" : ""}`}>{m.case_id}</a>
                  </td>
                  <td style={{ maxWidth: 220 }} title={m.estimate ? (es ? m.estimate.question_es : m.estimate.question_en) : undefined}>
                    {it0 ? (
                      <>
                        <span className="muted" style={{ fontSize: 11, display: "block", lineHeight: 1.3 }}>{es ? it0.label_es : it0.label_en}</span>
                        <span className="mono" style={{ fontSize: 12 }}>{qoiVal}</span>
                      </>
                    ) : "—"}
                  </td>
                  <td>{m.validation_anchor}</td>
                  <td className="mono">
                    {m.variants[0]?.metrics?.l2_relative !== undefined ? Number(m.variants[0].metrics.l2_relative).toExponential(2): ": "}
                  </td>
                  <td className="mono" style={{ color: s.naive_vs_std !== undefined ? "var(--bad)" : undefined }}>{pct(s.naive_vs_std)}</td>
                  <td className="mono" style={{ color: s.adapted_vs_std !== undefined ? "var(--good)" : undefined }}>{pct(s.adapted_vs_std)}</td>
                  <td className="mono">{Number(m.onnx.parity_max_abs).toExponential(2)}</td>
                  <td>
                    <span className={`tag ${honestyClass(m.real_or_synthetic)}`}>
                      {t(`honesty.${m.real_or_synthetic}`, { defaultValue: m.real_or_synthetic })}
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
        <p className="muted" style={{ fontSize: 12.5 }}>
          {lang === "es"
            ? "Las columnas de la escalera vienen de las comparaciones horneadas (la vista Comparar de cada caso); un guion significa que ese carril no existe para el caso (nunca se fabrica un contraste). Clic en un caso abre su comparación."
            : "The ladder columns come from the baked comparisons (each case's Compare view); a dash means that lane does not exist for the case (a contrast is never fabricated). Click a case to open its comparison."}
        </p>
      </div>
          ) },
          { id: "howto", label: es ? "Cómo leerlos" : "How to read them", content: (
      <div className="panel" style={{ marginBottom: 0 }}>
        <h3>{es ? "Cómo leer estos números" : "How to read these numbers"}</h3>
        <ul style={{ fontSize: 13.5, lineHeight: 1.6 }}>
          <li>{es
            ? "El L2 RELATIVO es el error del campo de la PINN contra su ancla de validación, normalizado por la norma de la referencia. La escala del ancla importa: analítica/MMS (forma cerrada exacta), dataset/benchmark (p. ej. las líneas de Ghia), o datos reales retenidos (soil-heat). Un 0.1% contra una forma cerrada y un 5% contra un benchmark de flujo no son comparables sin mirar el ancla."
            : "RELATIVE L2 is the PINN field's error against its validation anchor, normalized by the reference's norm. The anchor class matters: analytic/MMS (exact closed form), dataset/benchmark (e.g. the Ghia centerlines), or held-out real data (soil-heat). A 0.1% against a closed form and a 5% against a flow benchmark are not comparable without reading the anchor."}</li>
          <li>{es
            ? "La PARIDAD ONNX (siempre < 1e-4, típicamente ~1e-6 a 1e-7) mide que la red exportada al navegador reproduce a la red entrenada. NO es un error de física: es la fidelidad del contrato train→web. Que sea diminuta es lo que hace que el tab Live evalúe exactamente la función entrenada."
            : "ONNX PARITY (always < 1e-4, typically ~1e-6 to 1e-7) measures that the network exported to the browser reproduces the trained network. It is NOT a physics error: it is the fidelity of the train→web contract. Its being tiny is what makes the Live tab evaluate exactly the trained function."}</li>
          <li>{es
            ? "Varios casos se sientan HONESTAMENTE en un error alto por diseño, no por defecto: Helmholtz ~9% (los campos oscilatorios son duros; la lección es que la red ingenua da 121%), la cavidad de Navier ~ RMSE 0.05 en las líneas de Ghia, la barrera de suelo ~19% (el salto de material es genuinamente difícil para redes suaves). Se publican tal cual."
            : "Several cases sit HONESTLY at a high error by design, not by defect: Helmholtz ~9% (oscillatory fields are hard; the lesson is the naive net gives 121%), the Navier cavity at ~0.05 RMSE on the Ghia centerlines, the soil barrier ~19% (the material jump is genuinely hard for smooth networks). They are published as-is."}</li>
          <li>{es
            ? "La etiqueta de DATOS es la línea de honestidad: solo 1 de 21 casos usa datos reales (soil-heat-real, temperaturas NOAA); el resto es sintético o sintético-ilustrativo (un cierre de ingeniería estándar, no un gemelo calibrado). Los casos de minería/polución son modelos reducidos etiquetados como tales."
            : "The DATA label is the honesty line: only 1 of 21 cases uses real data (soil-heat-real, NOAA temperatures); the rest is synthetic or synthetic-illustrative (a standard engineering closure, not a calibrated twin). The mining/pollution cases are reduced models labeled as such."}</li>
          <li>{es
            ? "Donde una prueba justa no mostró contraste real (wave1d, ~11% ambos carriles) o un solucionador divergió (un FDM apresurado), NO se muestra ningún resultado inventado: el guion es honesto."
            : "Where a fair test showed no real contrast (wave1d, ~11% both lanes) or a solver diverged (a rushed FDM), NO fabricated result is shown: the dash is honest."}</li>
        </ul>
        <p className="muted" style={{ fontSize: 12.5, marginTop: 6 }}>
          {es
            ? "Cierre del arco: Introducción (por qué las EDPs y las PINN), Metodología (los métodos y cuándo ganan), Implementación (cómo se construyó, con las lecciones), Experimentos (qué se corrió) y Benchmark (los números y su lectura honesta)."
            : "Closing the arc: Introduction (why PDEs and PINNs), Methodology (the methods and when they win), Implementation (how it was built, with the lessons), Experiments (what was run), and Benchmark (the numbers and their honest reading)."}
        </p>
      </div>
          ) },
        ]}
      />
      )}
    </div>
  );
}
