import { useEffect, useRef, useState, type ReactNode } from "react";

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
    const bounds: { title: string; start: number }[] = [];
    kids.forEach((k, i) => {
      if (/^H[123]$/.test(k.tagName)) bounds.push({ title: (k.textContent ?? "").slice(0, 60), start: i });
    });
    if (!bounds.length) {
      setSections([]);
      return;
    }
    // if content precedes the first heading, fold it into the first section
    if (bounds[0].start > 0) bounds[0].start = 0;
    setSections(bounds.map((b) => b.title));
    setActive(0);

    const apply = (idx: number) => {
      kids.forEach((k, i) => {
        const s = bounds.findIndex((b, bi) => i >= b.start && (bi === bounds.length - 1 || i < bounds[bi + 1].start));
        (k as HTMLElement).style.display = s === idx ? "" : "none";
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
