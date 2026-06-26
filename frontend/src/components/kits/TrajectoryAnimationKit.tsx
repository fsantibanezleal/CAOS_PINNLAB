import { useMemo } from "react";

import { PendulumCanvas } from "./PendulumCanvas";
import { TracePlot, type Series } from "./TracePlot";
import { Transport } from "./Transport";
import type { KitProps } from "./types";
import { useAnimator } from "./useAnimator";

const wrap = (a: number) => ((a + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
const rangeOf = (...arrs: number[][]): [number, number] => {
  let lo = Infinity;
  let hi = -Infinity;
  for (const a of arrs) for (const v of a) { if (v < lo) lo = v; if (v > hi) hi = v; }
  return Number.isFinite(lo) ? [lo, hi] : [0, 1];
};

/** TrajectoryAnimationKit — the `ode-dynamical` kit (ADR-0063). For a case whose solution is a t → state
 *  trajectory (the double pendulum), it animates the swinging linkage (PINN ghost over the RK45 anchor), a
 *  phase portrait, the butterfly twin-IC divergence, and the angle time-series — with the honest leave-time
 *  marked: a fixed network cannot follow a chaotic trajectory past its Lyapunov horizon. */
export function TrajectoryAnimationKit({ trace, lang }: KitProps) {
  const es = lang === "es";

  const d = useMemo(() => {
    if (!trace) return null;
    const t = (trace.axes.t ?? []) as number[];
    const f = trace.fields as Record<string, number[]>;
    const th1 = (f.th1 ?? []) as number[];
    const th2 = (f.th2 ?? []) as number[];
    const th1r = (f.th1_ref ?? th1) as number[];
    const th2r = (f.th2_ref ?? th2) as number[];
    const th1tw = (f.th1_twin ?? th1r) as number[];
    const th2tw = (f.th2_twin ?? th2r) as number[];
    if (!t.length || !th1.length) return null;
    const s = (trace.summary ?? {}) as Record<string, number>;
    const l1 = s.l1 ?? 1;
    const l2 = s.l2 ?? 1;
    const leaveTime = s.leave_time ?? t[t.length - 1];
    const leaveIdx = Math.max(0, t.findIndex((tt) => tt >= leaveTime));
    // butterfly separation |Δθ| between the two RK45 runs (wrapped)
    const sep = t.map((_, i) => Math.hypot(wrap(th1r[i] - th1tw[i]), wrap(th2r[i] - th2tw[i])));
    return {
      t, th1, th2, th1r, th2r, sep, l1, l2, leaveTime, leaveIdx,
      lyap: s.lyapunov_est, twin: s.twin_perturb ?? 1e-3, tol: s.leave_tol ?? 0.3,
      l2err: s.l2_relative,
      angRange: rangeOf(th1, th2, th1r, th2r),
      thetaRange: rangeOf(th1r, th2r),
    };
  }, [trace]);

  const nT = d?.t.length ?? 1;
  const anim = useAnimator(nT, { fps: 30 });
  const i = Math.min(anim.frame, nT - 1);

  if (!d) return <div className="loading">{es ? "Cargando trayectoria…" : "Loading trajectory…"}</div>;
  const { t, th1, th2, th1r, th2r, sep, l1, l2, leaveTime, leaveIdx, lyap, twin, tol, l2err, angRange, thetaRange } = d;
  const tNow = t[i];
  const tEnd = t[t.length - 1];
  const tracksFully = leaveTime >= tEnd - 1e-6;
  const l2pct = l2err != null ? (l2err * 100).toFixed(2) : null;

  // phase portrait paths up to the current frame
  const pinnPath: [number, number][] = th1.slice(0, i + 1).map((v, k) => [v, th2[k]]);
  const refPath: [number, number][] = th1r.slice(0, i + 1).map((v, k) => [v, th2r[k]]);
  const tsSeries: Series[] = [
    { points: t.map((tt, k) => [tt, th1r[k]]), color: "#4ea1ff", width: 1.6 },
    { points: t.map((tt, k) => [tt, th1[k]]), color: "rgba(120,220,170,0.95)", dashed: true, width: 1.4 },
    { points: t.map((tt, k) => [tt, th2r[k]]), color: "#8a6bff", width: 1.6 },
    { points: t.map((tt, k) => [tt, th2[k]]), color: "rgba(255,170,90,0.95)", dashed: true, width: 1.4 },
  ];

  return (
    <div className="traj-kit">
      <p className="muted" style={{ marginTop: 0 }}>
        {es
          ? "Trayectoria t → (θ₁, θ₂): la PINN (fantasma) sobre el integrador RK45 (sólido). Es caótica — mira la PINN seguir un rato y luego perder la trayectoria."
          : "Trajectory t → (θ₁, θ₂): the PINN (ghost) over the RK45 integrator (solid). It is chaotic — watch the PINN track for a while, then lose the trajectory."}
      </p>
      <Transport anim={anim} lang={lang} axisLabel="t" axisValue={tNow} />
      <div className="traj-grid">
        <div className="traj-hero">
          <PendulumCanvas th1Ref={th1r} th2Ref={th2r} th1Pinn={th1} th2Pinn={th2} frame={i} l1={l1} l2={l2} leaveIdx={leaveIdx} />
          <div className="traj-legend">
            <span><i className="sw" style={{ background: "#4ea1ff" }} /> RK45 {es ? "(referencia)" : "(reference)"}</span>
            <span><i className="sw" style={{ background: "rgba(120,220,170,0.95)" }} /> PINN</span>
            <span><i className="sw" style={{ background: "#ff5d5d" }} /> PINN {es ? "tras leave-time" : "after leave-time"}</span>
          </div>
        </div>
        <div className="traj-side">
          <TracePlot
            title={es ? "Retrato de fase θ₁–θ₂" : "Phase portrait θ₁–θ₂"}
            series={[{ points: refPath, color: "#4ea1ff", width: 1.4 }, { points: pinnPath, color: "rgba(120,220,170,0.95)", dashed: true, width: 1.4 }]}
            dots={[{ x: th1r[i], y: th2r[i], color: "#4ea1ff" }, { x: th1[i], y: th2[i], color: i >= leaveIdx ? "#ff5d5d" : "rgba(120,220,170,1)" }]}
            xRange={thetaRange} yRange={thetaRange} xLabel="θ₁ (rad)" yLabel="θ₂ (rad)" height={196}
          />
          <TracePlot
            title={es ? "Efecto mariposa: |Δθ| de dos inicios vecinos" : "Butterfly: |Δθ| of two nearby starts"}
            series={[{ points: t.map((tt, k) => [tt, sep[k]]), color: "#ff8a3d", width: 1.6 }]}
            cursorX={tNow}
            xRange={[t[0], t[t.length - 1]]} yRange={[twin, Math.PI]} xLabel="t (s)" yLabel="|Δθ|" yLog height={196}
          />
        </div>
      </div>
      <TracePlot
        title={es ? "Ángulos vs t — PINN (discontinua) vs RK45 (sólida)" : "Angles vs t — PINN (dashed) vs RK45 (solid)"}
        series={tsSeries}
        cursorX={tNow}
        markers={[{ x: leaveTime, label: `leave-time ${leaveTime.toFixed(2)}s`, color: "#ff5d5d" }]}
        xRange={[t[0], t[t.length - 1]]} yRange={angRange} xLabel="t (s)" yLabel="θ (rad)" height={188}
      />
      <p className="hint">
        {tracksFully
          ? (es
            ? `La PINN sigue al integrador RK45 en TODA la ventana${l2pct ? ` (L2 relativo = ${l2pct}%)` : ""} — una PINN sí resuelve esta EDO caótica aquí. El panel mariposa (dos inicios separados ${twin} rad) muestra la sensibilidad que hace inviable extender el horizonte. Ancla: RK45 rtol=atol=1e-10.`
            : `The PINN tracks the RK45 integrator across the WHOLE window${l2pct ? ` (relative-L2 = ${l2pct}%)` : ""} — a PINN really does solve this chaotic ODE here. The butterfly panel (two starts ${twin} rad apart) shows the sensitivity that makes extending the horizon infeasible. Anchor: RK45 rtol=atol=1e-10.`)
          : (es
            ? `Métrica honesta: leave-time = ${leaveTime.toFixed(2)} s (primer |Δθ| > ${tol} rad vs RK45) — la PINN sigue al RK45 con precisión${l2pct ? ` (L2 ~${l2pct}%)` : ""} y luego, pasado el horizonte de Lyapunov${lyap ? ` (λ≈${lyap.toFixed(2)} 1/s)` : ""}, pierde la trayectoria caótica (el brazo se pone rojo). Las dos curvas naranjas (inicios separados ${twin} rad) muestran por qué. Ancla: RK45 rtol=atol=1e-10.`
            : `Honest metric: leave-time = ${leaveTime.toFixed(2)} s (first |Δθ| > ${tol} rad vs RK45) — the PINN tracks RK45 accurately${l2pct ? ` (L2 ~${l2pct}%)` : ""} and then, past the Lyapunov horizon${lyap ? ` (λ≈${lyap.toFixed(2)} 1/s)` : ""}, loses the chaotic trajectory (the arm turns red). The two orange curves (starts ${twin} rad apart) show why. Anchor: RK45 rtol=atol=1e-10.`)}
      </p>
    </div>
  );
}
