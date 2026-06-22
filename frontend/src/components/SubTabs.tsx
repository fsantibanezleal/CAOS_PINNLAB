import { useId, useState, type ReactNode } from "react";

export interface SubTabDef {
  id: string;
  label: ReactNode;
  content: ReactNode;
}

/** Second-level tabs nested inside a Tabs panel (the per-case Field / Live / Charts / Context rail). */
export function SubTabs({ tabs, initial, ariaLabel }: { tabs: SubTabDef[]; initial?: string; ariaLabel?: string }) {
  const baseId = useId();
  const first = tabs[0]?.id ?? "";
  const [active, setActive] = useState<string>(initial ?? first);

  function onKeyDown(e: React.KeyboardEvent<HTMLButtonElement>, idx: number) {
    if (!["ArrowRight", "ArrowLeft", "Home", "End"].includes(e.key)) return;
    e.preventDefault();
    let next = idx;
    if (e.key === "ArrowRight") next = (idx + 1) % tabs.length;
    else if (e.key === "ArrowLeft") next = (idx - 1 + tabs.length) % tabs.length;
    else if (e.key === "Home") next = 0;
    else if (e.key === "End") next = tabs.length - 1;
    const target = tabs[next];
    if (target) {
      setActive(target.id);
      document.getElementById(`${baseId}-subtab-${target.id}`)?.focus();
    }
  }

  return (
    <div className="subtabs">
      <div className="subtablist" role="tablist" aria-label={ariaLabel}>
        {tabs.map((tab, idx) => (
          <button
            key={tab.id}
            id={`${baseId}-subtab-${tab.id}`}
            role="tab"
            type="button"
            aria-selected={tab.id === active}
            aria-controls={`${baseId}-subpanel-${tab.id}`}
            tabIndex={tab.id === active ? 0 : -1}
            className={tab.id === active ? "subtab active" : "subtab"}
            onClick={() => setActive(tab.id)}
            onKeyDown={(e) => onKeyDown(e, idx)}
          >
            {tab.label}
          </button>
        ))}
      </div>
      <div className="subtabpanels">
        {tabs.map((tab) => (
          <div
            key={tab.id}
            id={`${baseId}-subpanel-${tab.id}`}
            role="tabpanel"
            aria-labelledby={`${baseId}-subtab-${tab.id}`}
            hidden={tab.id !== active}
            tabIndex={0}
            className="subtabpanel"
          >
            {tab.id === active ? tab.content : null}
          </div>
        ))}
      </div>
    </div>
  );
}
