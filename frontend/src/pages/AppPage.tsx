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
  { id: "bench-poisson2d", view: "results", en: "1 · Easy forward: classical wins", es: "1 · Directo fácil: gana el clásico",
    subEn: "The PINN matches the closed form (<0.1%), but here a classical solver is exact and faster: the PINN is NOT the tool.", subEs: "La PINN iguala la forma cerrada (<0.1%), pero aquí un solucionador clásico es exacto y más rápido: la PINN NO es la herramienta." },
  { id: "ind-helmholtz", view: "results", en: "2 · Hard forward: naive fails, Fourier fixes", es: "2 · Directo difícil: ingenua falla, Fourier corrige",
    subEn: "Watch it (not) learn: the naive lane is a flat blank at 100% while the Fourier lane already shows the pattern by iteration 500.", subEs: "Míralo (no) aprender: el carril ingenuo es un blanco plano al 100% mientras el de Fourier ya muestra el patrón en la iteración 500." },
  { id: "bench-allencahn", view: "results", en: "3 · Stiff: soft collapses, constraints track", es: "3 · Rígida: la suave colapsa, restricciones siguen",
    subEn: "The metastable collapse as visible training dynamics; the fully-trained fix reaches 0.4% (Compare).", subEs: "El colapso metaestable como dinámica de entrenamiento visible; la corrección totalmente entrenada llega a 0.4% (Comparar)." },
  { id: "ind-heat2d-inverse", view: "results", en: "4 · WHERE PINNS WIN: inverse + data", es: "4 · DONDE GANAN: inverso + datos",
    subEn: "Pure physics with no data: 356% (unrecoverable). Physics + 100 sparse sensors: 4%. The data makes it solvable.", subEs: "Física pura sin datos: 356% (irrecuperable). Física + 100 sensores dispersos: 4%. Los datos lo hacen soluble." },
  { id: "env-soil-heat-real", view: "results", star: true, en: "5 · Real data, out-of-sample", es: "5 · Datos reales, fuera de muestra",
    subEn: "NOAA soil temperatures: held-out depths never seen by the optimizer, reconstructed to ~1 degC RMSE.", subEs: "Temperaturas de suelo NOAA: profundidades nunca vistas por el optimizador, reconstruidas a ~1 gradoC RMSE." },
  { id: "bench-darcy-operator", view: "results", en: "6 · Operators: one net, a PDE family", es: "6 · Operadores: una red, una familia",
    subEn: "One FNO maps a NEW permeability field to its pressure in a single pass (2.5% vs finite differences).", subEs: "Un FNO mapea un campo de permeabilidad NUEVO a su presión en un solo paso (2.5% vs diferencias finitas)." },
  { id: "poll-source-uq-bpinn", view: "results", en: "7 · Uncertainty: honest error bars", es: "7 · Incertidumbre: barras honestas",
    subEn: "A deep ensemble gives mean +- sigma; sigma grows exactly where the sensors are not.", subEs: "Un ensamble profundo da media +- sigma; sigma crece exactamente donde no hay sensores." },
  { id: "dyn-double-pendulum", view: "results", en: "8 · The limit: chaos wins", es: "8 · El límite: gana el caos",
    subEn: "The PINN tracks the integrator ~2 s, then chaos peels it away: no surrogate beats the Lyapunov exponent.", subEs: "La PINN sigue al integrador ~2 s y el caos la despega: ningún sustituto vence al exponente de Lyapunov." },
  { id: "ind-hidden-velocity", view: "results", star: true, en: "9 · The flagship: hidden state from dye", es: "9 · El buque insignia: estado oculto desde el tinte",
    subEn: "The HFM mechanism (Science 2020): the whole current recovered from sparse dye samples alone: accurate where the dye swept, honestly unidentifiable where it never went.", subEs: "El mecanismo HFM (Science 2020): toda la corriente recuperada solo desde muestras dispersas de tinte: precisa donde el tinte barrió, honestamente no identificable donde nunca pasó." },
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
  const [q, setQ] = useState<string>("");

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

  // land on the tool: the deep-linked case when valid, else STORY CHAPTER 1 (the narrative spine is the
  // entry, never an arbitrary case), else the first case as a last resort.
  useEffect(() => {
    if (index && !caseId) {
      const linked = searchParams.get("case");
      const found = linked ? index.cases.find((c) => c.case_id === linked) : undefined;
      if (found) {
        setGroup(found.category);
        setCaseId(found.case_id);
        return;
      }
      const ch1 = index.cases.find((c) => c.case_id === STORY[0].id);
      if (ch1) {
        setGroup(ch1.category);
        setCaseId(ch1.case_id);
        setView(STORY[0].view);
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

  // Case search: filters ACROSS all categories by human label + the engineering question (both languages).
  const qn = q.trim().toLowerCase();
  const searched = qn
    ? index.cases.filter((c) =>
        [caseLabel(c.case_id), c.title, c.question_en, c.question_es].some((s) => s?.toLowerCase().includes(qn)))
    : null;
  const listCases = searched ?? activeCases;

  // the left-rail case browser (secondary to the story stepper: "browse all cases")
  const selector = (
    <div className="pl-selector">
      <input
        type="search"
        className="pl-search"
        placeholder={es ? "Buscar un caso o una pregunta…" : "Search a case or a question…"}
        value={q}
        onChange={(e) => setQ(e.target.value)}
        aria-label={es ? "Buscar casos" : "Search cases"}
      />
      {!searched && (
        <label className="pl-selrow">
          <span className="pl-sellabel">{es ? "Dominio" : "Domain"}</span>
          <select className="pl-select" value={group} onChange={(e) => selectGroup(e.target.value)}>
            {present.map((cat) => (
              <option key={cat} value={cat}>{(CATEGORY_LABELS[cat]?.[lang] ?? cat) + ` (${groups[cat].length})`}</option>
            ))}
          </select>
        </label>
      )}
      <h4>{searched ? (es ? `Resultados (${listCases.length})` : `Results (${listCases.length})`) : (es ? `Caso (${listCases.length})` : `Case (${listCases.length})`)}</h4>
      <div className="pl-caselist pl-caselist-scroll">
        {listCases.map((c) => (
          <button
            key={c.case_id}
            type="button"
            data-case={c.case_id}
            className={"pl-caseitem" + (c.case_id === caseId ? " on" : "")}
            onClick={() => { selectCase(c.category, c.case_id); setQ(""); }}
            aria-pressed={c.case_id === caseId}
          >
            {isFeatured(c.case_id) && <span className="pl-star">★</span>}
            {caseLabel(c.case_id)}
            {(es ? c.question_es : c.question_en) && (
              <small className="pl-caseq">{es ? c.question_es : c.question_en}</small>
            )}
          </button>
        ))}
        {searched && !listCases.length && <p className="muted" style={{ fontSize: 12, padding: "4px 8px" }}>{es ? "Sin resultados." : "No matches."}</p>}
      </div>
    </div>
  );

  // THE STORY STEPPER: the narrative spine, visible and first. Each chip is a chapter; the active chapter
  // shows its question + computed evidence line. Browsing outside the story dims the stepper but keeps it.
  const chIdx = STORY.findIndex((f) => f.id === caseId);
  const goChapter = (i: number) => {
    const ch = STORY[i];
    const c = index.cases.find((x) => x.case_id === ch?.id);
    if (c && ch) selectCase(c.category, ch.id, ch.view);
  };
  const activeCh = chIdx >= 0 ? STORY[chIdx] : null;
  const activeEntry = index.cases.find((x) => x.case_id === caseId);
  const stepper = (
    <div className={"pl-stepper" + (chIdx < 0 ? " off-story" : "")} role="navigation" aria-label={es ? "La historia" : "The story"}>
      <span className="pl-stepper-title">{es ? "LA HISTORIA: cuándo ganan y cuándo pierden las PINN" : "THE STORY: when PINNs win and when they lose"}</span>
      <div className="pl-stepper-row">
        <button type="button" className="pl-step-nav" disabled={chIdx <= 0} onClick={() => goChapter(chIdx <= 0 ? 0 : chIdx - 1)} aria-label={es ? "Capítulo anterior" : "Previous chapter"}>‹</button>
        {STORY.map((f, i) => (
          <button
            key={f.id}
            type="button"
            className={"pl-step" + (i === chIdx ? " on" : "")}
            onClick={() => goChapter(i)}
            title={es ? f.es : f.en}
            aria-current={i === chIdx ? "step" : undefined}
          >
            <b>{i + 1}</b>
            <span>{(es ? f.es : f.en).replace(/^\d+ · /, "")}</span>
          </button>
        ))}
        <button type="button" className="pl-step-nav" disabled={chIdx >= STORY.length - 1 && chIdx >= 0} onClick={() => goChapter(chIdx < 0 ? 0 : chIdx + 1)} aria-label={es ? "Siguiente capítulo" : "Next chapter"}>›</button>
      </div>
      {activeCh && (
        <p className="pl-story-blurb">
          {activeEntry && <strong className="pl-story-q">{es ? activeEntry.question_es : activeEntry.question_en}</strong>}{" "}
          {es ? activeCh.subEs : activeCh.subEn}
        </p>
      )}
    </div>
  );

  return (
    <div className="pl-app">
      {stepper}
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
