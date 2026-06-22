import { useId, useState, type ReactNode } from "react";

export interface TabDef {
  id: string;
  label: ReactNode;
  content: ReactNode;
}

/** Accessible roving-tabindex tab strip (arrow-key nav, aria wiring). Accent-soft pill for active. */
export function Tabs({ tabs, initial, ariaLabel }: { tabs: TabDef[]; initial?: string; ariaLabel?: string }) {
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
      document.getElementById(`${baseId}-tab-${target.id}`)?.focus();
    }
  }

  return (
    <div className="tabs">
      <div className="tablist" role="tablist" aria-label={ariaLabel}>
        {tabs.map((tab, idx) => (
          <button
            key={tab.id}
            id={`${baseId}-tab-${tab.id}`}
            role="tab"
            type="button"
            aria-selected={tab.id === active}
            aria-controls={`${baseId}-panel-${tab.id}`}
            tabIndex={tab.id === active ? 0 : -1}
            className={tab.id === active ? "tab active" : "tab"}
            onClick={() => setActive(tab.id)}
            onKeyDown={(e) => onKeyDown(e, idx)}
          >
            {tab.label}
          </button>
        ))}
      </div>
      {tabs.map((tab) => (
        <div
          key={tab.id}
          id={`${baseId}-panel-${tab.id}`}
          role="tabpanel"
          aria-labelledby={`${baseId}-tab-${tab.id}`}
          hidden={tab.id !== active}
          tabIndex={0}
          className="tabpanel"
        >
          {tab.id === active ? tab.content : null}
        </div>
      ))}
    </div>
  );
}
