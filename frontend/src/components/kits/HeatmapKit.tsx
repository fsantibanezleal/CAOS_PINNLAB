import { useState } from "react";

import { FieldView } from "../FieldView";
import type { KitProps } from "./types";

/** HeatmapKit — the steady-field render kit and the default fallback. A viridis heatmap of
 *  `trace.fields[out]` over `field_axes`, with a multi-output chip selector. This IS the original
 *  universal Field view (ADR-0016 §9), now one kit among several (ADR-0063). For genuinely STEADY
 *  elliptic fields (Poisson, zero-source) a still heatmap is the honest, correct presentation. */
export function HeatmapKit({ manifest, trace, lang }: KitProps) {
  const es = lang === "es";
  const [outIdx, setOutIdx] = useState(0);
  const outName = manifest.outputs[outIdx] ?? manifest.outputs[0];
  const fa = manifest.field_axes;
  const field2d = trace ? (trace.fields[outName] as number[][]) : null;
  const ax0 = trace?.axes[fa[0]] ?? [0, 1];
  const ax1 = trace?.axes[fa[1]] ?? [0, 1];

  return (
    <div>
      {manifest.outputs.length > 1 && (
        <div className="row" style={{ gap: 8, marginBottom: 8 }}>
          <span className="muted" style={{ fontSize: 13 }}>{es ? "Campo" : "Field"}</span>
          <div className="variant-chips">
            {manifest.outputs.map((o, i) => (
              <button key={o} className={"variant-chip" + (i === outIdx ? " active" : "")} onClick={() => setOutIdx(i)}>{o}</button>
            ))}
          </div>
        </div>
      )}
      {field2d ? (
        <FieldView
          field={field2d}
          axisX={{ label: fa[0], lo: ax0[0], hi: ax0[ax0.length - 1] }}
          axisY={{ label: fa[1], lo: ax1[0], hi: ax1[ax1.length - 1] }}
          outputLabel={outName}
        />
      ) : (
        <div className="loading">{es ? "Cargando campo…" : "Loading field…"}</div>
      )}
    </div>
  );
}
