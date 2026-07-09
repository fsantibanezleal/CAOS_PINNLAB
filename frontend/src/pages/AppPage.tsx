import { useEffect, useMemo, useState } from "react";

import { CaseExperiment } from "../components/CaseExperiment";
import { ContextFor, caseLabel } from "../content/cases/registry";
import {
  CATEGORY_INTRO,
  CATEGORY_LABELS,
  DATA_LABELS,
  VIEW_KIT_LABELS,
  type CaseIndex,
  type CaseIndexEntry,
} from "../lib/contract";
import { loadIndex } from "../lib/data";
import { useUI } from "../store";

const CATEGORY_ORDER = [
  "canonical-benchmark",
  "mining-mineral-processing",
  "pollution-environmental",
  "industrial-fluids-heat",
  "control",
];

/** The App page (ADR-0016 §9 + ADR-0063). Lands directly in the tool, now with a THREE-level structure so the 20
 *  cases read as scenarios + groupings of functionalities: (1) a group nav by physics domain (scenario), (2) a grid
 *  of case cards showing each case's FUNCTIONALITY (its view-kit + method + honesty label), (3) the per-case
 *  four-sub-tab workbench (Field / Live / Charts / Context). */
export function AppPage() {
  const lang = useUI((s) => s.lang);
  const es = lang === "es";
  const [index, setIndex] = useState<CaseIndex | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [group, setGroup] = useState<string>(CATEGORY_ORDER[0]);
  const [caseId, setCaseId] = useState<string>("");

  useEffect(() => {
    loadIndex().then(setIndex).catch((e) => setErr(String(e)));
  }, []);

  const groups = useMemo(() => {
    const g: Record<string, CaseIndexEntry[]> = {};
    if (index) for (const c of index.cases) (g[c.category] = g[c.category] ?? []).push(c);
    return g;
  }, [index]);

  // land on the tool: first case of the default group
  useEffect(() => {
    if (index && !caseId) {
      const first = (groups[group] ?? [])[0] ?? index.cases[0];
      if (first) setCaseId(first.case_id);
    }
  }, [index, group, caseId, groups]);

  if (err) return <div className="page-body"><div className="banner error">⚠ {err}</div></div>;
  if (!index) return <div className="page-body"><div className="loading">{es ? "Cargando…" : "Loading…"}</div></div>;

  const present = CATEGORY_ORDER.filter((c) => groups[c]?.length);
  const activeCases = groups[group] ?? [];
  const active = index.cases.find((c) => c.case_id === caseId);

  function selectGroup(cat: string) {
    setGroup(cat);
    const first = (groups[cat] ?? [])[0];
    if (first) setCaseId(first.case_id);
  }

  return (
    <div className="page-body">
      <div className="page-head">
        <h1>{es ? "Catálogo de PINNs" : "PINN catalogue"}</h1>
        <p className="lede">
          {es
            ? "20 casos en cinco dominios. Elige un dominio, luego un caso: cada uno es un banco de trabajo — un campo interactivo, inferencia en vivo en tu navegador, una comparación de regímenes y el contexto detallado con sus ecuaciones. Las etiquetas de cada tarjeta dicen qué funcionalidad ejercita (la visualización y el método)."
            : "20 cases across five domains. Pick a domain, then a case: each is a workbench — an interactive field, live in-browser inference, a regime comparison, and the detailed context with its equations. Each card's badges say what functionality it exercises (the visualization and the method)."}
        </p>
      </div>

      {/* Level 1 — scenario domains */}
      <div className="group-nav" role="tablist" aria-label={es ? "Dominios" : "Domains"}>
        {present.map((cat) => (
          <button
            key={cat}
            role="tab"
            type="button"
            aria-selected={cat === group}
            className={"group-tab" + (cat === group ? " active" : "")}
            onClick={() => selectGroup(cat)}
          >
            {CATEGORY_LABELS[cat]?.[lang] ?? cat}
            <span className="group-count">{groups[cat].length}</span>
          </button>
        ))}
      </div>
      {CATEGORY_INTRO[group] && <p className="group-intro">{CATEGORY_INTRO[group][lang]}</p>}

      {/* Level 2 — case cards (groupings of functionalities) */}
      <div className="case-grid">
        {activeCases.map((c) => {
          const kit = VIEW_KIT_LABELS[c.view_kit ?? "HeatmapKit"]?.[lang] ?? c.view_kit ?? "";
          const real = c.real_or_synthetic === "validated-real";
          const data = DATA_LABELS[c.real_or_synthetic ?? "synthetic"]?.[lang] ?? c.real_or_synthetic ?? "";
          return (
            <button
              key={c.case_id}
              type="button"
              className={"case-card" + (c.case_id === caseId ? " active" : "")}
              onClick={() => setCaseId(c.case_id)}
              aria-pressed={c.case_id === caseId}
            >
              <span className="case-card-title">{caseLabel(c.case_id)}</span>
              <span className="case-card-badges">
                <span className="cc-badge kit">{kit}</span>
                <span className={"cc-badge data" + (real ? " real" : "")}>{data}</span>
              </span>
              <span className="case-card-method mono">{c.method}</span>
            </button>
          );
        })}
      </div>

      {/* Level 3 — the workbench */}
      {active && (
        <div className="case-workbench">
          <CaseExperiment key={active.case_id} manifestId={active.case_id} description={ContextFor(active.case_id, lang)} lang={lang} />
        </div>
      )}
    </div>
  );
}
