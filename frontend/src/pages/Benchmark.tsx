import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";

import type { CaseManifest } from "../lib/contract";
import { loadIndex, loadManifest } from "../lib/data";
import { useUI } from "../store";

function honestyClass(flag: string): string {
  return flag.startsWith("validated") ? "validated" : "synthetic";
}

export function Benchmark() {
  const { t } = useTranslation();
  const lang = useUI((s) => s.lang);
  const [rows, setRows] = useState<CaseManifest[]>([]);

  useEffect(() => {
    loadIndex().then((ix) => Promise.all(ix.cases.map((c) => loadManifest(c.case_id))).then(setRows));
  }, []);

  return (
    <div className="prose" style={{ maxWidth: 1040 }}>
      <h1>Benchmark</h1>
      <p className="muted">
        {lang === "es"
          ? "Honestidad: error relativo vs la referencia (analítica/dataset/FEM), paridad ONNX-vs-modelo, y etiqueta de datos sintéticos vs reales. Sin números maquillados."
          : "Honesty: relative error vs the reference (analytic/dataset/FEM), ONNX-vs-model parity, and the synthetic-vs-real data label. No dressed-up numbers."}
      </p>
      <div className="panel">
        <table className="tbl">
          <thead>
            <tr>
              <th>case</th>
              <th>anchor</th>
              <th>{t("app.relL2")}</th>
              <th>{t("app.onnxParity")}</th>
              <th>{t("app.dataLabel")}</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((m) => (
              <tr key={m.case_id}>
                <td className="mono">{m.case_id}</td>
                <td>{m.validation_anchor}</td>
                <td className="mono">
                  {m.variants[0]?.metrics?.l2_relative !== undefined ? Number(m.variants[0].metrics.l2_relative).toExponential(2) : "—"}
                </td>
                <td className="mono">{Number(m.onnx.parity_max_abs).toExponential(2)}</td>
                <td>
                  <span className={`tag ${honestyClass(m.real_or_synthetic)}`}>
                    {t(`honesty.${m.real_or_synthetic}`, { defaultValue: m.real_or_synthetic })}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
