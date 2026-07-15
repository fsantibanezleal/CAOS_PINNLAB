import { useEffect, useRef, useState, type ReactNode } from "react";

/** A deep-prose section heading is a full sentence ("The problem: phase separation with sharp interfaces: the
 *  Allen-Cahn equation"); as a TAB label it must be short and never cut mid-word. Take the lead clause (before
 *  the first colon, when short enough), else clamp at a word boundary with an ellipsis. */
function shortLabel(s: string): string {
  const raw = s.trim();
  const colon = raw.indexOf(":");
  let t = colon > 2 && colon <= 34 ? raw.slice(0, colon) : raw;
  if (t.length > 32) {
    t = t.slice(0, 32);
    const sp = t.lastIndexOf(" ");
    if (sp > 14) t = t.slice(0, sp);
    t = t.replace(/[\s,;:.]+$/, "") + "…";
  }
  return t;
}

/** In-tab section switcher (real-review plan §2.2): long content is NEVER a scroll: it is split into
 *  sections shown one at a time at full width, switched by a pill row. The split is taken from the content's
 *  own h2/h3 headings after render (the deep Context pages already carry exactly these sections). Extra
 *  leading nodes before the first heading join the first section. Re-splits on content change (keyed mount). */
export function SectionPager({ children, extra, lang }: { children: ReactNode; extra?: { title: string; body: ReactNode }[]; lang: "en" | "es" }) {
  const host = useRef<HTMLDivElement>(null);
  const [sections, setSections] = useState<string[]>([]);
  const [active, setActive] = useState(0);
  const es = lang === "es";
  const nExtra = extra?.length ?? 0;

  useEffect(() => {
    const el = host.current;
    if (!el) return;
    const kids = Array.from(el.children);
    const heads: { title: string; start: number }[] = [];
    kids.forEach((k, i) => {
      if (/^H[123]$/.test(k.tagName)) heads.push({ title: shortLabel(k.textContent ?? ""), start: i });
    });
    if (!heads.length) {
      setSections([]);
      return;
    }
    // if content precedes the first heading, fold it into the first section
    if (heads[0].start > 0) heads[0].start = 0;
    // per-heading segment [start,end) with its rendered text length
    const segs = heads.map((h, i) => {
      const end = i + 1 < heads.length ? heads[i + 1].start : kids.length;
      let len = 0;
      for (let k = h.start; k < end; k++) len += (kids[k].textContent ?? "").length;
      return { title: h.title, start: h.start, end, len };
    });
    // GROUP adjacent short sections so each VIEW carries enough text to fill the 2-column stage (no one-paragraph
    // panels floating in an empty stage; 2 columns need ~2x the text to fill the same height). Accumulate until a
    // group passes MIN chars; merge a too-short tail back.
    const MIN = 2200;
    type G = { start: number; end: number; len: number; titles: string[] };
    const groups: G[] = [];
    for (const s of segs) {
      const last = groups[groups.length - 1];
      if (last && last.len < MIN) {
        last.end = s.end;
        last.len += s.len;
        last.titles.push(s.title);
      } else {
        groups.push({ start: s.start, end: s.end, len: s.len, titles: [s.title] });
      }
    }
    // a short tail group (below ~0.8 MIN) would leave the bottom of its 2-column view empty: merge it back so
    // every view fills (a slightly taller last view scrolls within itself, which is fine for a deep reading).
    while (groups.length > 1 && groups[groups.length - 1].len < MIN * 0.8) {
      const tail = groups.pop()!;
      const prev = groups[groups.length - 1];
      prev.end = tail.end;
      prev.len += tail.len;
      prev.titles.push(...tail.titles);
    }
    // when everything collapses to one full-prose view, label it "Deep dive" rather than the first section's name
    setSections(
      groups.length === 1
        ? [es ? "Detalle" : "Deep dive"]
        : groups.map((g) => (g.titles.length > 1 ? `${g.titles[0]} +` : g.titles[0])),
    );
    setActive(0);

    const apply = (idx: number) => {
      kids.forEach((k, i) => {
        const g = groups.findIndex((gr) => i >= gr.start && i < gr.end);
        (k as HTMLElement).style.display = g === idx ? "" : "none";
      });
    };
    apply(0);
    (el as HTMLElement & { __apply?: (i: number) => void }).__apply = apply;
  }, [children]);

  const show = (i: number) => {
    setActive(i);
    const el = host.current as (HTMLDivElement & { __apply?: (i: number) => void }) | null;
    if (i >= nExtra && el?.__apply) el.__apply(i - nExtra);
  };

  const titles = [...(extra?.map((x) => x.title) ?? []), ...sections];
  const showingExtra = active < nExtra;

  return (
    <div className="pl-secpager">
      {titles.length > 1 && (
        <div className="pl-secnav" role="tablist">
          {titles.map((t, i) => (
            <button key={i} type="button" role="tab" className={"pl-secbtn" + (i === active ? " on" : "")} aria-selected={i === active} onClick={() => show(i)}>
              {t}
            </button>
          ))}
        </div>
      )}
      {extra?.map((x, i) => (
        <div key={i} className="pl-secbody" style={{ display: showingExtra && i === active ? "" : "none" }}>{x.body}</div>
      ))}
      <div ref={host} className={"pl-secbody pl-prosewide"} style={{ display: showingExtra ? "none" : "" }} aria-label={es ? "Contenido de la sección" : "Section content"}>
        {children}
      </div>
    </div>
  );
}
