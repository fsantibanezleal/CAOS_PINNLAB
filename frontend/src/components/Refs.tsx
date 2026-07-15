import { CITATIONS } from "../data/citations";

/** Inline citation (ADR-0016 §7.5): <Cite id="raissi-2019" /> renders a linked short label in prose. */
export function Cite({ id }: { id: string }) {
  const c = CITATIONS[id];
  if (!c) return <span className="cite-missing">[?]</span>;
  const href = c.doi ? `https://doi.org/${c.doi}` : c.url;
  return (
    <a className="cite" href={href} target="_blank" rel="noreferrer noopener" title={c.citation}>
      [{c.label}]
    </a>
  );
}

/** Per-section reference list (ADR-0016 §7.5): <Refs ids={[...]} /> at the END of a section shows ONLY that
 *  section's references, each with a real DOI/URL link. This REPLACES the banned bottom-of-page bibliography. */
export function Refs({ ids, lang }: { ids: string[]; lang?: "en" | "es" }) {
  const es = lang === "es";
  const items = ids.map((id) => CITATIONS[id]).filter(Boolean);
  if (!items.length) return null;
  return (
    <div className="refs">
      <span className="refs-label">{es ? "Referencias de esta sección" : "References for this section"}</span>
      <ul>
        {items.map((c) => (
          <li key={c.id}>
            {c.citation}{" "}
            <a href={c.doi ? `https://doi.org/${c.doi}` : c.url} target="_blank" rel="noreferrer noopener">
              {c.doi ? `doi:${c.doi}` : c.url}
            </a>
          </li>
        ))}
      </ul>
    </div>
  );
}
