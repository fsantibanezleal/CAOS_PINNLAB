import { useState } from "react";

import { FieldView } from "../FieldView";
import { markersFor } from "./MarkerLayer";
import type { KitProps } from "./types";

const TIME_NAMES = new Set(["t", "time", "tau", "tt", "τ"]);

/** HeatmapKit: the steady-field render kit and the default fallback. A viridis heatmap of
 *  `trace.fields[out]` over `field_axes`, with a multi-output chip selector. This IS the original
 *  universal Field view (ADR-0016 §9), now one kit among several (ADR-0063). A caption states exactly what
 *  the axes ARE: for a time-dependent PDE the map is the SPACE-TIME field (the whole evolution in one image,
 *  not a static final state); for a steady field it is the spatial domain. */
export function HeatmapKit({ manifest, trace, active, lang }: KitProps) {
  const es = lang === "es";
  const [outIdx, setOutIdx] = useState(0);
  const outName = manifest.outputs[outIdx] ?? manifest.outputs[0];
  const fa = manifest.field_axes;
  const field2d = trace ? (trace.fields[outName] as number[][]): null;
  const ax0 = trace?.axes[fa[0]] ?? [0, 1];
  const ax1 = trace?.axes[fa[1]] ?? [0, 1];

  // what do the axes MEAN? -> a caption so the user knows if they are seeing the evolution or a snapshot.
  const timeAxis = fa.find((a) => TIME_NAMES.has(a));
  const timeParam = manifest.param_specs.find((p) => TIME_NAMES.has(p.key));
  let caption: string;
  if (timeAxis) {
    const spaceAxis = fa.find((a) => a !== timeAxis) ?? fa[0];
    caption = es
      ? `Campo ESPACIO-TIEMPO ${outName}(${spaceAxis}, ${timeAxis}): toda la evolución en una imagen. El eje ${timeAxis} es el TIEMPO: cada línea horizontal es el perfil en ese instante; leer hacia arriba en el tiempo muestra cómo evoluciona. NO es un estado final estático.`
     : `SPACE-TIME field ${outName}(${spaceAxis}, ${timeAxis}): the whole evolution in one image. The ${timeAxis} axis is TIME: each horizontal line is the profile at that instant; reading up the time axis shows how it evolves. It is NOT a static final state.`;
  } else if (timeParam) {
    const tv = active?.params?.[timeParam.key];
    caption = es
      ? `Instantánea espacial en ${timeParam.key}=${tv != null ? tv: "?"} (el tiempo es una entrada de la red: usa los chips de régimen de arriba para avanzar en el tiempo). Ejes ${fa[0]} × ${fa[1]}.`
     : `Spatial snapshot at ${timeParam.key}=${tv != null ? tv: "?"} (time is a network input: use the regime chips above to step through time). Axes ${fa[0]} × ${fa[1]}.`;
  } else {
    caption = es
      ? `Campo ${outName} sobre el dominio espacial 2-D (${fa[0]} × ${fa[1]}): un problema estacionario, sin dimensión temporal.`
     : `${outName} field over the 2-D spatial domain (${fa[0]} × ${fa[1]}): a steady problem, no time dimension.`;
  }

  return (
    <div className="heatmap-kit">
      <p className="dim-caption">{caption}</p>
      {manifest.outputs.length > 1 && (
        <div className="hk-outputs row" style={{ gap: 8 }}>
          <span className="muted" style={{ fontSize: 12 }}>{es ? "Campo": "Field"}</span>
          <div className="variant-chips">
            {manifest.outputs.map((o, i) => (
              <button key={o} className={"variant-chip" + (i === outIdx ? " active": "")} onClick={() => setOutIdx(i)}>{o}</button>
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
          lang={lang}
          markers={markersFor(manifest, active)}
        />
      ): (
        <div className="loading">{es ? "Cargando campo…": "Loading field…"}</div>
      )}
    </div>
  );
}
