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

// Owner decision (2026-07-15): a fresh visit lands on a MINING case (mining-first product framing).
// Flotation kinetics is the default: flotation is the emblematic mineral-processing operation and its
// recovery-vs-time curve is the most legible first view. Deep links (?case=…) still win over this.
const DEFAULT_GROUP = "mining-mineral-processing";
const DEFAULT_CASE = "mine-flotation-kinetics";


/** The App page (ADR-0016 §9 + ADR-0063), now a FULL-WIDTH workbench mirroring CAOS_RES_Lidar3D: the content
 *  fills the whole viewport (header + footer are already full-width) instead of being boxed in the narrow reading
 *  column. The left rail carries the case selection (highlights + domain + case list); `CaseExperiment` adds the
 *  regime chips + the view switch, the full-width center stage, and the right stats rail. */
export function AppPage() {
  const lang = useUI((s) => s.lang);
  const es = lang === "es";
  const [index, setIndex] = useState<CaseIndex | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [group, setGroup] = useState<string>(DEFAULT_GROUP);
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

  // land on the tool: the deep-linked case when valid, else the default MINING case (owner decision).
  useEffect(() => {
    if (index && !caseId) {
      const linked = searchParams.get("case");
      const found = linked ? index.cases.find((c) => c.case_id === linked) : undefined;
      if (found) {
        setGroup(found.category);
        setCaseId(found.case_id);
        return;
      }
      // default landing = the mining case; fall back to the mining group's first case, then anything.
      const def = index.cases.find((c) => c.case_id === DEFAULT_CASE);
      const first = def ?? (groups[DEFAULT_GROUP] ?? [])[0] ?? (groups[group] ?? [])[0] ?? index.cases[0];
      if (first) {
        setGroup(first.category);
        setCaseId(first.case_id);
      }
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

  // THE LEFT COLUMN (owner decisions 2026-07-14): Domain dropdown + CASE DROPDOWN (one selection, never the
  // full card list) + the SELECTED case's card, then the case narrative in FIXED positions for EVERY case
  // (stable layout): the question, the situation, what the model calculates, what we know and seek.
  const entry = index.cases.find((c) => c.case_id === caseId);
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
      {searched ? (
        <div className="pl-caselist pl-caselist-scroll">
          {searched.map((c) => (
            <button key={c.case_id} type="button" data-case={c.case_id} className={"pl-caseitem" + (c.case_id === caseId ? " on" : "")}
              onClick={() => { selectCase(c.category, c.case_id); setQ(""); }}>
              {caseLabel(c.case_id)}
              {(es ? c.question_es : c.question_en) && <small className="pl-caseq">{es ? c.question_es : c.question_en}</small>}
            </button>
          ))}
          {!searched.length && <p className="muted" style={{ fontSize: 12, padding: "4px 8px" }}>{es ? "Sin resultados." : "No matches."}</p>}
        </div>
      ) : (
        <>
          <label className="pl-selrow">
            <span className="pl-sellabel">{es ? "Dominio" : "Domain"}</span>
            <select className="pl-select" value={group} onChange={(e) => selectGroup(e.target.value)}>
              {present.map((cat) => (
                <option key={cat} value={cat}>{(CATEGORY_LABELS[cat]?.[lang] ?? cat) + ` (${groups[cat].length})`}</option>
              ))}
            </select>
          </label>
          <label className="pl-selrow">
            <span className="pl-sellabel">{es ? "Caso" : "Case"}</span>
            <select className="pl-select" data-caseselect value={caseId} onChange={(e) => selectCase(group, e.target.value)}>
              {activeCases.map((c) => (
                <option key={c.case_id} value={c.case_id}>{caseLabel(c.case_id)}</option>
              ))}
            </select>
          </label>
          <div className="pl-casecard">
            <b>{caseLabel(caseId)}</b>
            {entry && <p className="pl-casecard-q">{es ? entry.question_es : entry.question_en}</p>}
          </div>
        </>
      )}
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
