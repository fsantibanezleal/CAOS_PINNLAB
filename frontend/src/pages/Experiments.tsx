import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";

import { CATEGORY_LABELS, type CaseManifest } from "../lib/contract";
import { loadIndex, loadManifest } from "../lib/data";
import { useUI } from "../store";

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
