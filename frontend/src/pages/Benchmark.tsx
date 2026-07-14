import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";

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
      <div className="panel">
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
    </div>
  );
}
