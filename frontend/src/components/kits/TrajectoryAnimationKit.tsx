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
  return Number.isFinite(lo) ? [lo, hi]: [0, 1];
};

/** TrajectoryAnimationKit: the `ode-dynamical` kit (ADR-0063). For a case whose solution is a t → state
 *  trajectory (the double pendulum), it animates the swinging linkage (PINN ghost over the RK45 anchor), a
 *  phase portrait, the butterfly twin-IC divergence, and the angle time-series: with the honest leave-time
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
    // A twin exists only for the chaos case; when it does not (e.g. the structure-preserving case, which
    // compares a MODEL against the reference and has no butterfly), the "twin" fell back to the reference and
    // sep is identically 0. Detect that so the kit does not show a meaningless flat butterfly panel.
    const hasTwin = sep.some((v) => v > 1e-9);
    // The energy trace (present for the structure-preserving case): the model's total energy over time and
    // the constant true value. When present, it replaces the butterfly panel with the quantity that matters.
    const energy = (f.energy ?? null) as number[] | null;
    const energyRef = (f.energy_ref ?? null) as number[] | null;
    return {
      t, th1, th2, th1r, th2r, sep, hasTwin, energy, energyRef, l1, l2, leaveTime, leaveIdx,
      lyap: s.lyapunov_est, twin: s.twin_perturb ?? 1e-3, tol: s.leave_tol ?? 0.3,
      l2err: s.l2_relative,
      angRange: rangeOf(th1, th2, th1r, th2r),
      thetaRange: rangeOf(th1r, th2r),
    };
  }, [trace]);

  const nT = d?.t.length ?? 1;
  const anim = useAnimator(nT, { fps: 30 });
  const i = Math.min(anim.frame, nT - 1);

  if (!d) return <div className="loading">{es ? "Cargando trayectoria…": "Loading trajectory…"}</div>;
  const { t, th1, th2, th1r, th2r, sep, hasTwin, energy, energyRef, l1, l2, leaveTime, leaveIdx, lyap, twin, tol, l2err, angRange, thetaRange } = d;
  const tNow = t[i];
  const tEnd = t[t.length - 1];
  const tracksFully = leaveTime >= tEnd - 1e-6;
  const l2pct = l2err != null ? (l2err * 100).toFixed(2): null;
  // "compare a model against the reference" mode (structure-preserving case): no twin, so no butterfly and no
  // chaos/leave-time framing; the moving arm never turns red.
  const compareMode = !hasTwin;
  const modelLbl = compareMode ? (es ? "modelo" : "model") : "PINN";
  const eRange: [number, number] = energy && energyRef
    ? (() => { const all = [...energy, ...energyRef]; return [Math.min(...all), Math.max(...all)]; })()
    : [0, 1];

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
        {compareMode
          ? (es
            ? "Trayectoria t → (θ₁, θ₂): el modelo seleccionado (fantasma) sobre el integrador RK45 (sólido). Régimen acotado (no caótico): ambos siguen la trayectoria; lo que se mide es si la energía se conserva."
            : "Trajectory t → (θ₁, θ₂): the selected model (ghost) over the RK45 integrator (solid). Bounded regime (not chaotic): both track the trajectory; what is measured is whether the energy is conserved.")
          : (es
            ? "Trayectoria t → (θ₁, θ₂): la PINN (fantasma) sobre el integrador RK45 (sólido). Es caótica: mira la PINN seguir un rato y luego perder la trayectoria."
            : "Trajectory t → (θ₁, θ₂): the PINN (ghost) over the RK45 integrator (solid). It is chaotic: watch the PINN track for a while, then lose the trajectory.")}
      </p>
      <Transport anim={anim} lang={lang} axisLabel="t" axisValue={tNow} />
      <div className="traj-grid">
        <div className="traj-hero">
          <PendulumCanvas th1Ref={th1r} th2Ref={th2r} th1Pinn={th1} th2Pinn={th2} frame={i} l1={l1} l2={l2} leaveIdx={leaveIdx} />
          <div className="traj-legend">
            <span><i className="sw" style={{ background: "#4ea1ff" }} /> RK45 {es ? "(referencia)": "(reference)"}</span>
            <span><i className="sw" style={{ background: "rgba(120,220,170,0.95)" }} /> {modelLbl}</span>
            {!compareMode && <span><i className="sw" style={{ background: "#ff5d5d" }} /> {modelLbl} {es ? "tras leave-time": "after leave-time"}</span>}
          </div>
        </div>
        <div className="traj-side">
          <TracePlot
            title={es ? "Retrato de fase θ₁-θ₂": "Phase portrait θ₁-θ₂"}
            series={[{ points: refPath, color: "#4ea1ff", width: 1.4 }, { points: pinnPath, color: "rgba(120,220,170,0.95)", dashed: true, width: 1.4 }]}
            dots={[{ x: th1r[i], y: th2r[i], color: "#4ea1ff" }, { x: th1[i], y: th2[i], color: i >= leaveIdx ? "#ff5d5d": "rgba(120,220,170,1)" }]}
            xRange={thetaRange} yRange={thetaRange} xLabel="θ₁ (rad)" yLabel="θ₂ (rad)" height={196}
          />
          {compareMode && energy && energyRef ? (
            <TracePlot
              title={es ? "Energía total vs t (constante = conservada)": "Total energy vs t (constant = conserved)"}
              series={[
                { points: t.map((tt, k) => [tt, energyRef[k]]), color: "#4ea1ff", width: 1.4 },
                { points: t.map((tt, k) => [tt, energy[k]]), color: "rgba(120,220,170,0.95)", dashed: true, width: 1.6 },
              ]}
              cursorX={tNow}
              xRange={[t[0], t[t.length - 1]]} yRange={eRange} xLabel="t (s)" yLabel="E" height={196}
            />
          ) : (
            <TracePlot
              title={es ? "Efecto mariposa: |Δθ| de dos inicios vecinos": "Butterfly: |Δθ| of two nearby starts"}
              series={[{ points: t.map((tt, k) => [tt, sep[k]]), color: "#ff8a3d", width: 1.6 }]}
              cursorX={tNow}
              xRange={[t[0], t[t.length - 1]]} yRange={[twin, Math.PI]} xLabel="t (s)" yLabel="|Δθ|" yLog height={196}
            />
          )}
        </div>
      </div>
      <TracePlot
        title={es ? `Ángulos vs t: ${modelLbl} (discontinua) vs RK45 (sólida)`: `Angles vs t: ${modelLbl} (dashed) vs RK45 (solid)`}
        series={tsSeries}
        cursorX={tNow}
        markers={compareMode ? [] : [{ x: leaveTime, label: `leave-time ${leaveTime.toFixed(2)}s`, color: "#ff5d5d" }]}
        xRange={[t[0], t[t.length - 1]]} yRange={angRange} xLabel="t (s)" yLabel="θ (rad)" height={188}
      />
      <p className="hint">
        {compareMode
          ? (es
            ? `El modelo seleccionado sigue al RK45 en toda la ventana${l2pct ? ` (L2 relativo = ${l2pct}%)`: ""}. El panel de energía es lo que importa: una red hamiltoniana la conserva por construcción (dz/dt = J∇H, con J antisimétrica), una red sin estructura no. Régimen acotado de baja energía; el caso caótico es dyn-double-pendulum. Ancla: RK45 rtol=atol=1e-10.`
            : `The selected model tracks RK45 across the whole window${l2pct ? ` (relative-L2 = ${l2pct}%)`: ""}. The energy panel is the point: a Hamiltonian network conserves it by construction (dz/dt = J∇H, J antisymmetric), an unstructured one does not. Bounded low-energy regime; the chaotic case is dyn-double-pendulum. Anchor: RK45 rtol=atol=1e-10.`)
          : tracksFully
          ? (es
            ? `La PINN sigue al integrador RK45 en TODA la ventana${l2pct ? ` (L2 relativo = ${l2pct}%)`: ""}: una PINN sí resuelve esta EDO caótica aquí. El panel mariposa (dos inicios separados ${twin} rad) muestra la sensibilidad que hace inviable extender el horizonte. Ancla: RK45 rtol=atol=1e-10.`
           : `The PINN tracks the RK45 integrator across the WHOLE window${l2pct ? ` (relative-L2 = ${l2pct}%)`: ""}: a PINN really does solve this chaotic ODE here. The butterfly panel (two starts ${twin} rad apart) shows the sensitivity that makes extending the horizon infeasible. Anchor: RK45 rtol=atol=1e-10.`)
         : (es
            ? `Métrica honesta: leave-time = ${leaveTime.toFixed(2)} s (primer |Δθ| > ${tol} rad vs RK45): la PINN sigue al RK45 con precisión${l2pct ? ` (L2 ~${l2pct}%)`: ""} y luego, pasado el horizonte de Lyapunov${lyap ? ` (λ≈${lyap.toFixed(2)} 1/s)`: ""}, pierde la trayectoria caótica (el brazo se pone rojo). Las dos curvas naranjas (inicios separados ${twin} rad) muestran por qué. Ancla: RK45 rtol=atol=1e-10.`
           : `Honest metric: leave-time = ${leaveTime.toFixed(2)} s (first |Δθ| > ${tol} rad vs RK45): the PINN tracks RK45 accurately${l2pct ? ` (L2 ~${l2pct}%)`: ""} and then, past the Lyapunov horizon${lyap ? ` (λ≈${lyap.toFixed(2)} 1/s)`: ""}, loses the chaotic trajectory (the arm turns red). The two orange curves (starts ${twin} rad apart) show why. Anchor: RK45 rtol=atol=1e-10.`)}
      </p>
    </div>
  );
}
