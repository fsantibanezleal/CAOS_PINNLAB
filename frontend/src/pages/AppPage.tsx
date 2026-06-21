import { useEffect, useState } from "react";

import { CaseExperiment } from "../components/CaseExperiment";
import { Tabs } from "../components/Tabs";
import { ContextFor, caseLabel } from "../content/cases/registry";
import type { CaseIndex } from "../lib/contract";
import { loadIndex } from "../lib/data";
import { useUI } from "../store";

/** The App page (ADR-0016 §9): lands directly in the tool. A top-level tab per case, each a workbench
 *  (variant bar + Field / Live / Charts / Context). */
export function AppPage() {
  const lang = useUI((s) => s.lang);
  const es = lang === "es";
  const [index, setIndex] = useState<CaseIndex | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    loadIndex().then(setIndex).catch((e) => setErr(String(e)));
  }, []);

  if (err) return <div className="page-body"><div className="banner error">⚠ {err}</div></div>;
  if (!index) return <div className="page-body"><div className="loading">{es ? "Cargando…" : "Loading…"}</div></div>;

  const tabs = index.cases.map((c) => ({
    id: c.case_id,
    label: caseLabel(c.case_id),
    content: <CaseExperiment manifestId={c.case_id} description={ContextFor(c.case_id, lang)} lang={lang} />,
  }));

  return (
    <div className="page-body">
      <div className="page-head">
        <h1>{es ? "Catálogo de PINNs" : "PINN catalogue"}</h1>
        <p className="lede">
          {es
            ? "Cada caso es un banco de trabajo: una familia de regímenes de parámetro, un campo interactivo (lectura de valor al cursor), inferencia en vivo en tu navegador, una comparación de regímenes y el contexto detallado con sus ecuaciones."
            : "Each case is a workbench: a family of parameter regimes, an interactive field (value read-out at the cursor), live in-browser inference, a regime comparison, and the detailed context with its equations."}
        </p>
      </div>
      <Tabs tabs={tabs} ariaLabel={es ? "Casos" : "Cases"} />
    </div>
  );
}
