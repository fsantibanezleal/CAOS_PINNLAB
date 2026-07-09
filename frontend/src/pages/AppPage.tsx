import { useEffect, useMemo, useState } from "react";

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

/** Cross-cutting HIGHLIGHTS so the standout cases are not lost in the sea of 20: surfaced at the top of the rail.
 *  REAL data first + starred; then hybrid data+physics; then uncertainty; then chaotic dynamics. */
const FEATURED: { id: string; star?: boolean }[] = [
  { id: "env-soil-heat-real", star: true },
  { id: "ind-heat2d-inverse" },
  { id: "poll-source-uq-bpinn" },
  { id: "dyn-double-pendulum" },
];
const isFeatured = (id: string) => FEATURED.some((f) => f.id === id);

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

  if (err) return <div className="pl-app"><div className="pl-work"><aside className="pl-rail" /><section className="pl-stage"><div className="banner error">⚠ {err}</div></section><aside className="pl-rail" /></div></div>;
  if (!index || !caseId) return <div className="pl-app"><div className="pl-work"><aside className="pl-rail" /><section className="pl-stage"><div className="loading">{es ? "Cargando…" : "Loading…"}</div></section><aside className="pl-rail" /></div></div>;

  const present = CATEGORY_ORDER.filter((c) => groups[c]?.length);
  const activeCases = groups[group] ?? [];

  function selectGroup(cat: string) {
    setGroup(cat);
    const first = (groups[cat] ?? [])[0];
    if (first) setCaseId(first.case_id);
  }
  function selectCase(cat: string, id: string) {
    setGroup(cat);
    setCaseId(id);
  }

  // the left-rail case selector (rendered by CaseExperiment at the top of the left column)
  const selector = (
    <div className="pl-selector">
      <h4>{es ? "Destacados" : "Highlights"}</h4>
      <div className="pl-caselist">
        {FEATURED.map((f) => {
          const c = index.cases.find((x) => x.case_id === f.id);
          if (!c) return null;
          return (
            <button
              key={f.id}
              type="button"
              className={"pl-caseitem" + (f.id === caseId ? " on" : "")}
              onClick={() => selectCase(c.category, f.id)}
            >
              <span className="pl-star">{f.star ? "★" : "•"}</span>
              {caseLabel(f.id)}
            </button>
          );
        })}
      </div>

      <h4>{es ? "Dominio" : "Domain"}</h4>
      <div className="pl-domainchips">
        {present.map((cat) => (
          <button
            key={cat}
            type="button"
            className={"pl-domainchip" + (cat === group ? " on" : "")}
            onClick={() => selectGroup(cat)}
          >
            {CATEGORY_LABELS[cat]?.[lang] ?? cat} ({groups[cat].length})
          </button>
        ))}
      </div>

      <h4>{es ? "Caso" : "Case"} ({activeCases.length})</h4>
      <div className="pl-caselist pl-caselist-scroll">
        {activeCases.map((c) => (
          <button
            key={c.case_id}
            type="button"
            data-case={c.case_id}
            className={"pl-caseitem" + (c.case_id === caseId ? " on" : "")}
            onClick={() => setCaseId(c.case_id)}
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
      <CaseExperiment manifestId={caseId} selector={selector} description={ContextFor(caseId, lang)} lang={lang} />
    </div>
  );
}
