import { useEffect, useMemo, useState } from "react";

import type { CaseManifest, TrainingData } from "../../lib/contract";
import { loadTraining } from "../../lib/data";
import { fmtCount, fmtTick } from "../../lib/plot";
import { snapshotElement } from "../../lib/snapshot";
import { HeatCanvas } from "./HeatCanvas";
import { Transport } from "./Transport";
import { useAnimator } from "./useAnimator";

/** TrainingKit - "WATCH IT LEARN" (issue #36). Replays the field at real training checkpoints for the naive
 *  and adapted lanes SIDE BY SIDE (one colour scale across every frame of both lanes), with the live L2 per lane
 *  and the L2-vs-iteration curve underneath. This makes the training pathology VISIBLE as dynamics: the naive
 *  Helmholtz lane stays a low-frequency blur across all of training (spectral bias), the naive Allen-Cahn lane
 *  slides into the metastable collapse - while the adapted lane converges. All frames are real baked training
 *  states; paused by default (no autoplay). */
export function TrainingKit({ manifest, lang }: { manifest: CaseManifest; lang: "en" | "es" }) {
  const es = lang === "es";
  const path = manifest.training?.path;
  const [d, setD] = useState<TrainingData | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    if (!path) return;
    let alive = true;
    setD(null);
    loadTraining(path).then((x) => alive && setD(x)).catch((e) => alive && setErr(String(e)));
    return () => { alive = false; };
  }, [path]);

  const nF = d?.checkpoints.length ?? 1;
  const anim = useAnimator(nF, { fps: 2 });
  const f = Math.min(anim.frame, nF - 1);

  const prep = useMemo(() => {
    if (!d) return null;
    const keys = Object.keys(d.lanes);
    let lo = Infinity;
    let hi = -Infinity;
    for (const k of keys) for (const fr of d.lanes[k].frames) for (const col of fr) for (const v of col) {
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    const range: [number, number] = [Number.isFinite(lo) ? lo : 0, hi > lo ? hi : lo + 1];
    return { keys, range };
  }, [d]);

  if (!path) return null;
  if (err) return <div className="banner error">⚠ {err}</div>;
  if (!d || !prep) return <div className="loading">{es ? "Cargando dinámica de entrenamiento…" : "Loading training dynamics…"}</div>;

  const { keys, range } = prep;
  const laneColor = (k: string) => (k === "naive" ? "var(--bad)" : "var(--good)");
  const cps = d.checkpoints;

  // the L2-vs-iteration chart (log y), with a cursor at the current checkpoint
  const W = 640;
  const H = 220;
  const pad = { l: 56, r: 14, t: 12, b: 36 };
  const allL2 = keys.flatMap((k) => d.lanes[k].l2).filter((v) => v > 0);
  const yMin = Math.log10(Math.max(1e-4, Math.min(...allL2)));
  const yMax = Math.log10(Math.max(...allL2, 1e-4));
  // INDEX-spaced x (checkpoints are 0/250/.../12k - raw-value spacing crushes the early ones into a corner)
  const px = (i: number) => pad.l + (i / Math.max(1, cps.length - 1)) * (W - pad.l - pad.r);
  const py = (v: number) => {
    const t = (Math.log10(Math.max(v, 1e-4)) - yMin) / (yMax - yMin || 1);
    return H - pad.b - t * (H - pad.t - pad.b);
  };
  // log-decade y ticks inside [yMin, yMax] (plus the ends)
  const decades: number[] = [];
  for (let e = Math.ceil(yMin); e <= Math.floor(yMax); e++) decades.push(Math.pow(10, e));
  const yTickVals = Array.from(new Set([Math.pow(10, yMin), ...decades, Math.pow(10, yMax)]));

  return (
    <div className="train-kit">
      <p className="dim-caption">
        {es
          ? "MÍRALO APRENDER: el campo de cada carril en checkpoints REALES de entrenamiento, lado a lado, misma escala de color. La patología de entrenamiento se ve como dinámica: el carril ingenuo no converge; el adaptado sí."
          : "WATCH IT LEARN: each lane's field at REAL training checkpoints, side by side, one colour scale. The training pathology becomes visible dynamics: the naive lane fails to converge; the adapted lane snaps on."}
      </p>
      <Transport anim={anim} lang={lang} axisLabel={es ? "iteración" : "iteration"} axisValue={cps[f] ?? 0} fmt={fmtCount} />
      <div className="train-panels">
        {keys.map((k) => (
          <figure key={k} className="train-panel">
            <figcaption className="train-cap" style={{ color: laneColor(k) }}>
              {es ? d.lanes[k].label_es : d.lanes[k].label_en}
              <span className="mono muted"> · L2 = {(d.lanes[k].l2[f] * 100).toFixed(1)}%</span>
              <button type="button" className="snap-btn" title="Save PNG"
                onClick={(e) => snapshotElement((e.currentTarget as HTMLElement).closest(".train-panel") as HTMLElement, `${manifest.case_id}-training-${k}`)}>⤓</button>
            </figcaption>
            <div className="train-map"><HeatCanvas field={d.lanes[k].frames[f]} range={range} ariaLabel={`${k} at iter ${cps[f]}`} /></div>
          </figure>
        ))}
      </div>
      <div className="diag-block">
        <h4>{es ? "L2 relativa vs iteración (escala log)" : "Relative L2 vs iteration (log scale)"}</h4>
        <svg viewBox={`0 0 ${W} ${H}`} className="diag-svg">
          <line x1={pad.l} y1={pad.t} x2={pad.l} y2={H - pad.b} stroke="var(--border)" strokeWidth="1" />
          <line x1={pad.l} y1={H - pad.b} x2={W - pad.r} y2={H - pad.b} stroke="var(--border)" strokeWidth="1" />
          {yTickVals.map((v, i) => (
            <g key={i}>
              <line x1={pad.l} y1={py(v)} x2={W - pad.r} y2={py(v)} stroke="var(--border)" strokeWidth="0.6" opacity="0.7" />
              <text x={pad.l - 6} y={py(v) + 3} textAnchor="end" className="lp-tick">{(v * 100) >= 1 ? `${fmtTick(v * 100)}%` : fmtTick(v)}</text>
            </g>
          ))}
          {cps.map((cp, i) => (
            <g key={cp}>
              <line x1={px(i)} y1={pad.t} x2={px(i)} y2={H - pad.b} stroke="var(--border)" strokeWidth="0.4" opacity="0.4" />
              <text x={px(i)} y={H - pad.b + 14} textAnchor="middle" className="lp-tick">{fmtCount(cp)}</text>
            </g>
          ))}
          <text x={(pad.l + W - pad.r) / 2} y={H - 4} textAnchor="middle" className="lp-axislabel">{es ? "iteración de entrenamiento" : "training iteration"}</text>
          {keys.map((k) => (
            <g key={k}>
              <polyline points={d.lanes[k].l2.map((v, i) => `${px(i)},${py(v)}`).join(" ")} fill="none" stroke={laneColor(k)} strokeWidth="2" />
              {d.lanes[k].l2.map((v, i) => <circle key={i} cx={px(i)} cy={py(v)} r={i === f ? 4 : 2.2} fill={laneColor(k)} />)}
            </g>
          ))}
          <line x1={px(f)} y1={pad.t} x2={px(f)} y2={H - pad.b} stroke="var(--accent-2)" strokeDasharray="3 2" strokeWidth="1" />
        </svg>
        <div className="diag-legend">
          {keys.map((k) => (
            <span key={k} className="diag-leg"><span className="diag-swatch" style={{ background: laneColor(k) }} />{es ? d.lanes[k].label_es : d.lanes[k].label_en}</span>
          ))}
        </div>
      </div>
    </div>
  );
}
