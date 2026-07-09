import { useEffect, useState, type ReactNode } from "react";

import type { CaseManifest, FieldTrace } from "../lib/contract";
import { loadManifest, loadTrace } from "../lib/data";
import { Equation } from "./Equation";
import { resolveKit } from "./kits/registry";
import { LivePanel } from "./LivePanel";
import { SubTabs } from "./SubTabs";
import { VariantCharts } from "./VariantCharts";

/** One case = a workbench (ADR-0016 §9): a variant bar (≥6 regime chips + lane badge + a "what this shows" note)
 *  then four sub-tabs: Field (interactive replay) · Live (onnxruntime-web re-eval) · Charts (regime comparison) ·
 *  Context (the deep bilingual write-up). Mirrors CAOS_SIMLAB's ScenarioExperiment. */
export function CaseExperiment({
  manifestId,
  description,
  lang,
}: {
  manifestId: string;
  description: ReactNode;
  lang: "en" | "es";
}) {
  const es = lang === "es";
  const [manifest, setManifest] = useState<CaseManifest | null>(null);
  const [activeId, setActiveId] = useState<string>("");
  const [trace, setTrace] = useState<FieldTrace | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    loadManifest(manifestId)
      .then((m) => {
        if (!alive) return;
        setManifest(m);
        setActiveId(m.variants[0]?.id ?? "");
      })
      .catch((e) => alive && setErr(String(e)));
    return () => {
      alive = false;
    };
  }, [manifestId]);

  const active = manifest?.variants.find((v) => v.id === activeId) ?? manifest?.variants[0];

  useEffect(() => {
    if (!active) return;
    let alive = true;
    setTrace(null);
    loadTrace(active.trace.path).then((t) => alive && setTrace(t)).catch((e) => alive && setErr(String(e)));
    return () => {
      alive = false;
    };
  }, [active?.trace.path]);

  if (err) return <div className="banner error">⚠ {err}</div>;
  if (!manifest || !active) return <div className="loading">{es ? "Cargando…": "Loading…"}</div>;

  const Kit = resolveKit(manifest.view_kit);
  const l2 = active.metrics.l2_relative;

  const fieldTab = (
    <div>
      <Equation tex={manifest.governing_equations} />
      <Kit key={manifest.case_id} manifest={manifest} trace={trace} active={active} lang={lang} />
      <p className="muted" style={{ fontSize: 13 }}>{manifest.expected_band}</p>
      <div className="meta-strip">
        <span><b className="muted">{es ? "Método": "Method"}</b> <span className="mono">{manifest.method}</span></span>
        <span><b className="muted">{es ? "Motor": "Engine"}</b> <span className="mono">{manifest.engine.framework}</span></span>
        <span><b className="muted">L2</b> <span className="mono">{typeof l2 === "number" ? l2.toExponential(2): String(l2)}</span></span>
        <span><b className="muted">{es ? "Paridad ONNX": "ONNX parity"}</b> <span className="mono">{manifest.onnx.parity_max_abs.toExponential(1)}</span></span>
      </div>
    </div>
  );

  const tabs = [
    { id: "field", label: es ? "Campo": "Field", content: fieldTab },
    { id: "live", label: "Live", content: <LivePanel manifest={manifest} lang={lang} /> },
    { id: "charts", label: "Charts", content: <VariantCharts variants={manifest.variants} activeId={active.id} onSelect={setActiveId} lang={lang} /> },
    { id: "context", label: "Context", content: <div className="prose">{description}</div> },
  ];

  return (
    <div>
      <div className="variant-bar">
        <span className="variant-bar-label">
          {es ? "Regímenes": "Variants"} ({manifest.variants.length}) ·{" "}
          <span className={"badge " + (manifest.lane === "live" ? "live": "precomputed")}>
            {manifest.lane === "live" ? (es ? "live": "live"): (es ? "precompute": "precompute")}
          </span>
        </span>
        <div className="variant-chips">
          {manifest.variants.map((v) => (
            <button key={v.id} className={"variant-chip" + (v.id === active.id ? " active": "")} onClick={() => setActiveId(v.id)}>
              {es ? v.label_es: v.label_en}
            </button>
          ))}
        </div>
        {active && <p className="variant-note">{es ? active.note_es: active.note_en}</p>}
      </div>
      <SubTabs tabs={tabs} ariaLabel={manifest.title} />
    </div>
  );
}
