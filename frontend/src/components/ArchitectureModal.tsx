// Architecture / "how it was built" modal — opened by the ⓘ button in the header (Veta/Circuita pattern).
// Tabs, each pairing a hand-authored themed SVG (fetched + INLINED so its CSS-variable tokens resolve against the
// active dark/light palette — an <img> would not inherit them) with a compact bilingual explanation. This panel is
// the evidence that the app is real: it shows the actual modules, stages and data flow, not a demo facade.
import { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";

import { TABS } from "./architecture-tabs";

const svgCache: Record<string, string> = {};

export function ArchitectureModal({ onClose }: { onClose: () => void }) {
  const { t } = useTranslation();
  const [active, setActive] = useState<string>(TABS[0]!.id);
  const [svgMarkup, setSvgMarkup] = useState<string | null>(null);
  const [svgError, setSvgError] = useState<string | null>(null);
  const closeBtnRef = useRef<HTMLButtonElement | null>(null);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
  useEffect(() => closeBtnRef.current?.focus(), []);

  const tab = TABS.find((x) => x.id === active) ?? TABS[0]!;
  useEffect(() => {
    let cancelled = false;
    setSvgError(null);
    const cached = svgCache[tab.svg];
    if (cached) {
      setSvgMarkup(cached);
      return;
    }
    setSvgMarkup(null);
    fetch(`${import.meta.env.BASE_URL}svg/tech/${tab.svg}`)
      .then(async (res) => {
        if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
        const text = await res.text();
        svgCache[tab.svg] = text;
        if (!cancelled) setSvgMarkup(text);
      })
      .catch((err) => !cancelled && setSvgError((err as Error)?.message ?? String(err)));
    return () => {
      cancelled = true;
    };
  }, [tab.svg]);

  return (
    <div className="modal-backdrop" role="dialog" aria-modal="true" aria-labelledby="arch-title" onClick={onClose}>
      <div className="modal-card" onClick={(e) => e.stopPropagation()}>
        <header className="modal-head">
          <strong id="arch-title">{t("arch.title")}</strong>
          <button ref={closeBtnRef} className="iconbtn" onClick={onClose} aria-label={t("arch.close")}>
            ✕
          </button>
        </header>

        <div className="modal-tabs" role="tablist" aria-label={t("arch.title")}>
          {TABS.map((x) => (
            <button
              key={x.id}
              role="tab"
              aria-selected={x.id === active}
              className={`chip${x.id === active ? " on" : ""}`}
              onClick={() => setActive(x.id)}
            >
              {t(x.labelKey)}
            </button>
          ))}
        </div>

        <div className="modal-body" role="tabpanel">
          <p className="muted" style={{ fontSize: 13, lineHeight: 1.6, marginTop: 0 }}>{t(tab.textKey)}</p>
          {svgError && <div style={{ color: "var(--bad)", fontSize: 12 }}>{t("arch.svgFailed", { message: svgError })}</div>}
          {!svgError && svgMarkup === null && <div className="muted" style={{ fontSize: 12 }}>…</div>}
          {svgMarkup !== null && (
            <div className="svg-frame" dangerouslySetInnerHTML={{ __html: svgMarkup }} />
          )}
          {tab.blocks && (
            <div className="modal-blocks">
              {tab.blocks.map((bk, i) => {
                const spanBoth = tab.blocks!.length % 2 === 1 && i === tab.blocks!.length - 1;
                return (
                  <section key={bk.titleKey} className="block" style={spanBoth ? { gridColumn: "1 / -1" } : undefined}>
                    <strong>{t(bk.titleKey)}</strong>
                    <p>{t(bk.textKey)}</p>
                  </section>
                );
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
