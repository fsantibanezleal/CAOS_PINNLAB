import { useMemo, useState } from "react";

import { fieldRange } from "../../lib/colormap";
import { HeatCanvas } from "./HeatCanvas";
import { LineProfile } from "./LineProfile";
import { Transport } from "./Transport";
import type { KitProps } from "./types";
import { useAnimator } from "./useAnimator";

const TIME_NAMES = new Set(["t", "time", "tau", "tt", "τ"]);

/** TimeEvolutionKit — a 1-D field that EVOLVES in time. The baked trace is a 2-D `[space, t]` grid; this kit
 *  animates the profile `u(space)` forward over t (with a faint ghost of the initial frame and a y-scale
 *  locked to the whole run), and shows the `[space, t]` carpet as a click-to-seek bar. No retraining — the
 *  trace already holds every frame; the kit just walks the time columns. The Live tab still re-evals the ONNX. */
export function TimeEvolutionKit({ manifest, trace, lang }: KitProps) {
  const es = lang === "es";
  const [outIdx, setOutIdx] = useState(0);
  const outName = manifest.outputs[outIdx] ?? manifest.outputs[0];
  const fa = manifest.field_axes;
  const tIdx = TIME_NAMES.has(fa[1]) ? 1 : TIME_NAMES.has(fa[0]) ? 0 : 1;
  const tAxis = fa[tIdx];
  const spaceAxis = fa[1 - tIdx];

  const data = useMemo(() => {
    if (!trace) return null;
    const field = trace.fields[outName] as number[][];
    if (!field || !field.length) return null;
    const tArr = trace.axes[tAxis] ?? [];
    const sArr = trace.axes[spaceAxis] ?? [];
    const yRange = fieldRange(field);
    const nT = tArr.length || (tIdx === 1 ? (field[0]?.length ?? 0) : field.length);
    const nS = sArr.length || (tIdx === 1 ? field.length : (field[0]?.length ?? 0));
    // carpet indexed [space][t]
    let carpet: number[][];
    if (tIdx === 1) {
      carpet = field;
    } else {
      carpet = [];
      for (let is = 0; is < nS; is++) {
        const col: number[] = [];
        for (let it = 0; it < nT; it++) col.push(field[it][is]);
        carpet.push(col);
      }
    }
    return { field, tArr, sArr, yRange, carpet, nT };
  }, [trace, outName, tAxis, spaceAxis, tIdx]);

  const nT = data?.nT ?? 1;
  const anim = useAnimator(nT, { fps: 12 });
  const it = Math.min(anim.frame, nT - 1);

  if (!data) return <div className="loading">{es ? "Cargando campo…" : "Loading field…"}</div>;
  const { field, tArr, sArr, yRange, carpet } = data;
  const profile = tIdx === 1 ? field.map((col) => col[it]) : field[it];
  const ghost = tIdx === 1 ? field.map((col) => col[0]) : field[0];
  const tVal = tArr[it] ?? 0;

  return (
    <div className="te-kit">
      {manifest.outputs.length > 1 && (
        <div className="row" style={{ gap: 8, marginBottom: 8 }}>
          <span className="muted" style={{ fontSize: 13 }}>{es ? "Campo" : "Field"}</span>
          <div className="variant-chips">
            {manifest.outputs.map((o, i) => (
              <button key={o} type="button" className={"variant-chip" + (i === outIdx ? " active" : "")} onClick={() => setOutIdx(i)}>{o}</button>
            ))}
          </div>
        </div>
      )}
      <Transport anim={anim} lang={lang} axisLabel={tAxis} axisValue={tVal} />
      <div className="te-grid">
        <div className="te-hero">
          <LineProfile values={profile} ghost={ghost} spaceArr={sArr} yRange={yRange} spaceLabel={spaceAxis} outLabel={outName} />
        </div>
        <div className="te-carpet">
          <div className="te-carpet-label muted">
            {outName}({spaceAxis},{tAxis}) — {es ? "clic para saltar en el tiempo" : "click to seek in time"}
          </div>
          <HeatCanvas
            field={carpet}
            vCursor={nT > 1 ? it / (nT - 1) : 0}
            onSeekV={(f) => { anim.setPlaying(false); anim.setFrame(f * (nT - 1)); }}
            ariaLabel={`${outName} ${spaceAxis}-${tAxis} carpet`}
          />
        </div>
      </div>
      <p className="hint">
        {es
          ? "Reproducción animada de la solución horneada — Play / arrastra la barra o haz clic en el mapa para saltar en el tiempo. La pestaña Live re-evalúa el ONNX en vivo."
          : "Animated replay of the baked solution — Play / drag the bar or click the carpet to seek in time. The Live tab re-evaluates the ONNX live."}
      </p>
    </div>
  );
}
