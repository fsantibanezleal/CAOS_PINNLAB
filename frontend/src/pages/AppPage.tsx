import { useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";

import { CaseExperiment } from "../components/CaseExperiment";
import { ContextFor, caseLabel } from "../content/cases/registry";
import {
  CATEGORY_LABELS,
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

/** THE STORY (issue #36): the use-case narrative of when PINNs win and when they lose, each chapter landing on the
 *  case + view that DEMONSTRATES it with real computed content. This is the pedagogical axis the domain grouping
 *  lacks: forward-easy (classical wins) -> forward-hard (needs the fix) -> inverse/data (PINNs win) -> real data ->
 *  operators -> UQ -> the chaotic limit. */
const STORY: { id: string; view: string; star?: boolean; en: string; es: string; subEn: string; subEs: string }[] = [
  { id: "bench-poisson2d", view: "compare", en: "1 · Easy forward: the classical solver already wins", es: "1 · Directo fácil: el solucionador clásico ya gana",
    subEn: "The PINN matches the closed form (<0.1%), but here a classical solver is exact and faster: the PINN is NOT the tool.", subEs: "La PINN iguala la forma cerrada (<0.1%), pero aquí un solucionador clásico es exacto y más rápido: la PINN NO es la herramienta." },
  { id: "ind-helmholtz", view: "training", en: "2 · Hard forward: naive fails (spectral bias), Fourier fixes", es: "2 · Directo difícil: la ingenua falla (sesgo espectral), Fourier corrige",
    subEn: "Watch it (not) learn: the naive lane is a flat blank at 100% while the Fourier lane already shows the pattern by iteration 500.", subEs: "Míralo (no) aprender: el carril ingenuo es un blanco plano al 100% mientras el de Fourier ya muestra el patrón en la iteración 500." },
  { id: "bench-allencahn", view: "training", en: "3 · Stiff dynamics: soft PINN collapses, constraints + RAR track", es: "3 · Dinámica rígida: la PINN suave colapsa, restricciones + RAR siguen",
    subEn: "The metastable collapse as visible training dynamics; the fully-trained fix reaches 0.4% (Compare).", subEs: "El colapso metaestable como dinámica de entrenamiento visible; la corrección totalmente entrenada llega a 0.4% (Comparar)." },
  { id: "ind-heat2d-inverse", view: "compare", en: "4 · WHERE PINNS WIN: inverse + sparse data", es: "4 · DONDE GANAN LAS PINN: inverso + datos dispersos",
    subEn: "Pure physics with no data: 356% (unrecoverable). Physics + 100 sparse sensors: 4%. The data makes it solvable.", subEs: "Física pura sin datos: 356% (irrecuperable). Física + 100 sensores dispersos: 4%. Los datos lo hacen soluble." },
  { id: "env-soil-heat-real", view: "diagnostics", star: true, en: "5 · Real measured data, validated out-of-sample", es: "5 · Datos reales medidos, validado fuera de muestra",
    subEn: "NOAA soil temperatures: held-out depths never seen by the optimizer, reconstructed to ~1 degC RMSE.", subEs: "Temperaturas de suelo NOAA: profundidades nunca vistas por el optimizador, reconstruidas a ~1 gradoC RMSE." },
  { id: "bench-darcy-operator", view: "compare", en: "6 · Operators: amortize a whole PDE family", es: "6 · Operadores: amortizar una familia de EDPs",
    subEn: "One FNO maps a NEW permeability field to its pressure in a single pass (2.5% vs finite differences).", subEs: "Un FNO mapea un campo de permeabilidad NUEVO a su presión en un solo paso (2.5% vs diferencias finitas)." },
  { id: "poll-source-uq-bpinn", view: "field", en: "7 · Uncertainty: error bars where data is sparse", es: "7 · Incertidumbre: barras de error donde faltan datos",
    subEn: "A deep ensemble gives mean +- sigma; sigma grows exactly where the sensors are not.", subEs: "Un ensamble profundo da media +- sigma; sigma crece exactamente donde no hay sensores." },
  { id: "dyn-double-pendulum", view: "field", en: "8 · The honest limit: chaos defeats any surrogate", es: "8 · El límite honesto: el caos vence a cualquier sustituto",
    subEn: "The PINN tracks the integrator ~2 s, then chaos peels it away: no surrogate beats the Lyapunov exponent.", subEs: "La PINN sigue al integrador ~2 s y el caos la despega: ningún sustituto vence al exponente de Lyapunov." },
];
const isFeatured = (id: string) => STORY.some((f) => f.id === id && f.star);

/** The App page (ADR-0016 §9 + ADR-0063), now a FULL-WIDTH workbench mirroring CAOS_RES_Lidar3D: the content
 *  fills the whole viewport (header + footer are already full-width) instead of being boxed in the narrow reading
 *  column. The left rail carries the case selection (highlights + domain + case list); `CaseExperiment` adds the
 *  regime chips + the view switch, the full-width center stage, and the right stats rail. */
export function AppPage() {
  const lang = useUI((s) => s.lang);
  const es = lang === "es";
  const [index, setIndex] = useState<CaseIndex | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [group, setGroup] = useState<string>(CATEGORY_ORDER[0]);
  const [caseId, setCaseId] = useState<string>("");

  // Shareable deep links (issue #38): #/?case=<id>&view=<view> is read on load and kept in sync as you navigate.
  const [searchParams, setSearchParams] = useSearchParams();
  const [view, setView] = useState<string | undefined>(searchParams.get("view") ?? undefined);

  useEffect(() => {
    loadIndex().then(setIndex).catch((e) => setErr(String(e)));
  }, []);

  const groups = useMemo(() => {
    const g: Record<string, CaseIndexEntry[]> = {};
    if (index) for (const c of index.cases) (g[c.category] = g[c.category] ?? []).push(c);
    return g;
  }, [index]);

  function syncUrl(nextCase: string, nextView?: string) {
    const p: Record<string, string> = { case: nextCase };
    if (nextView) p.view = nextView;
    setSearchParams(p, { replace: true });
  }

  // land on the tool: the deep-linked case when valid, else the first case of the default group
  useEffect(() => {
    if (index && !caseId) {
      const linked = searchParams.get("case");
      const found = linked ? index.cases.find((c) => c.case_id === linked) : undefined;
      if (found) {
        setGroup(found.category);
        setCaseId(found.case_id);
        return;
      }
      const first = (groups[group] ?? [])[0] ?? index.cases[0];
      if (first) setCaseId(first.case_id);
    }
  }, [index, group, caseId, groups]);

  // REACT to URL param changes AFTER mount too (external hash navigation / back-forward): keep state in sync.
  useEffect(() => {
    if (!index || !caseId) return;
    const pCase = searchParams.get("case");
    const pView = searchParams.get("view") ?? undefined;
    if (pCase && pCase !== caseId) {
      const found = index.cases.find((c) => c.case_id === pCase);
      if (found) {
        setGroup(found.category);
        setCaseId(found.case_id);
        setView(pView);
      }
    } else if (pCase === caseId && pView !== view) {
      setView(pView);
    }
  }, [searchParams]);

  if (err) return <div className="pl-app"><div className="pl-work"><aside className="pl-rail" /><section className="pl-stage"><div className="banner error">⚠ {err}</div></section><aside className="pl-rail" /></div></div>;
  if (!index || !caseId) return <div className="pl-app"><div className="pl-work"><aside className="pl-rail" /><section className="pl-stage"><div className="loading">{es ? "Cargando…" : "Loading…"}</div></section><aside className="pl-rail" /></div></div>;

  const present = CATEGORY_ORDER.filter((c) => groups[c]?.length);
  const activeCases = groups[group] ?? [];

  function selectGroup(cat: string) {
    setGroup(cat);
    const first = (groups[cat] ?? [])[0];
    if (first) {
      setCaseId(first.case_id);
      setView(undefined);
      syncUrl(first.case_id);
    }
  }
  function selectCase(cat: string, id: string, nextView?: string) {
    setGroup(cat);
    setCaseId(id);
    setView(nextView);
    syncUrl(id, nextView);
  }

  // the left-rail case selector (rendered by CaseExperiment at the top of the left column)
  const selector = (
    <div className="pl-selector">
      <label className="pl-selrow">
        <span className="pl-sellabel">{es ? "La historia: cuándo ganan / pierden las PINN" : "The story: when PINNs win / lose"}</span>
        <select
          className="pl-select"
          value={STORY.some((f) => f.id === caseId) ? caseId : ""}
          onChange={(e) => {
            const id = e.target.value;
            const ch = STORY.find((f) => f.id === id);
            const c = index.cases.find((x) => x.case_id === id);
            if (c) selectCase(c.category, id, ch?.view); // land on the view that DEMONSTRATES the chapter
          }}
        >
          <option value="">{es ? "Recorre la historia en 8 capítulos…" : "Walk the story in 8 chapters…"}</option>
          {STORY.map((f) => {
            const c = index.cases.find((x) => x.case_id === f.id);
            if (!c) return null;
            return <option key={f.id} value={f.id}>{es ? f.es : f.en}</option>;
          })}
        </select>
      </label>
      {(() => {
        const ch = STORY.find((f) => f.id === caseId);
        return ch ? <p className="pl-story-blurb">{es ? ch.subEs : ch.subEn}</p> : null;
      })()}

      <label className="pl-selrow">
        <span className="pl-sellabel">{es ? "Dominio" : "Domain"}</span>
        <select className="pl-select" value={group} onChange={(e) => selectGroup(e.target.value)}>
          {present.map((cat) => (
            <option key={cat} value={cat}>{(CATEGORY_LABELS[cat]?.[lang] ?? cat) + ` (${groups[cat].length})`}</option>
          ))}
        </select>
      </label>

      <h4>{es ? "Caso" : "Case"} ({activeCases.length})</h4>
      <div className="pl-caselist pl-caselist-scroll">
        {activeCases.map((c) => (
          <button
            key={c.case_id}
            type="button"
            data-case={c.case_id}
            className={"pl-caseitem" + (c.case_id === caseId ? " on" : "")}
            onClick={() => { setCaseId(c.case_id); setView(undefined); syncUrl(c.case_id); }}
            aria-pressed={c.case_id === caseId}
          >
            {isFeatured(c.case_id) && <span className="pl-star">★</span>}
            {caseLabel(c.case_id)}
          </button>
        ))}
      </div>
    </div>
  );

  return (
    <div className="pl-app">
      <CaseExperiment
        manifestId={caseId}
        selector={selector}
        description={ContextFor(caseId, lang)}
        lang={lang}
        requestedView={view}
        onViewChange={(v) => { setView(v); syncUrl(caseId, v); }}
      />
    </div>
  );
}
