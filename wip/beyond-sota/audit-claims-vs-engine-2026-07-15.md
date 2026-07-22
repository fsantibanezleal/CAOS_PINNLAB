# Audit: what the docs claim vs what the engine actually runs (2026-07-15)

Trigger: a domain expert commented publicly on the PINN-Lab diffusion post:

> "las PINN suelen ser muy buenas resolviendo sistemas suaves, lo unico si que las PINN tienden a tener
> tiempos de entrenamientos muy largos y explosiones de gradientes en sus estructuras. Han pensado en indagar
> en operadores neuronales PINO como variantes de FNO?"

Paraphrased: PINNs are good on smooth systems, but they train slowly and suffer gradient blow-up; have you
looked at PINO as an FNO variant? The owner then asked why the catalogue has nothing on PINO and FNO.

This file records the **verified** state, read from the repo and the manifests, not from memory.

## 1. The finding: an overclaim, and the case doc already knew

`bench-darcy-operator` ships a **real** 2D Fourier Neural Operator: `data-pipeline/pinnlab/cases/bench_darcy_operator.py`,
`method="operator-fno"`, `engine="fno-torch"`, trained on the Li et al. Darcy benchmark, exported to its own
field-in ONNX. That part is honest and real. So "we have nothing on FNO" is **not** accurate.

But it is trained **data-driven only**: there is no PDE residual anywhere in the operator lane.

Meanwhile the method and framework docs claim otherwise:

| Location | The claim | Reality |
|---|---|---|
| `docs/methods/operator-learning.md:200-202` | "The case **exercises all three methods on the same family**: DeepONet (branch/trunk), FNO (with the zero-shot super-resolution demo), and PINO (FNO + the Darcy residual at higher resolution, showing the data-reduction benefit)" | Only FNO is implemented. **DeepONet: absent. PINO: absent.** |
| `docs/methods/operator-learning.md:171-172` | PINO "is the second method exercised by `bench-darcy-operator` (after plain FNO)" | Not exercised. Not implemented. |
| `docs/frameworks/neuraloperator/README.md:339` | Primary case "exercises FNO + PINO" | FNO only. |
| `docs/frameworks/neuraloperator/README.md:340` | "`methods/fno.md` (#19) and `methods/pino.md` (#20) — both first land in `bench-darcy-operator`" | Only #19 landed. |
| `docs/cases/bench-darcy-operator.md:30` | "(The optional PINO physics-residual is documented as the upgrade; the shipped path is the data-driven FNO.)" | **Correct.** The case doc is the honest one. |

So the docs contradict each other, and the wrong side is the one a reader hits first. Per the standing rule
(audit claims against the ENGINE, not just cross-page consistency), this is a real defect, not a wording nit.
Grep evidence: `PINO` appears **zero** times in `data-pipeline/` and `frontend/src/`; `DeepONet` appears only
in `docs/frameworks/` describing what DeepXDE and neuraloperator can do, never in our own pipeline.

**Actions:** either implement PINO/DeepONet or correct these lines. This plan implements PINO (it is the
direct answer to the public critique) and corrects the DeepONet claim until DeepONet actually ships.

## 2. The full method inventory, read from the 21 manifests

| method (manifest field) | case | family |
|---|---|---|
| `hard-constraints` | bench-poisson2d, ctrl-zero-source | constraint handling |
| `hard-constraints-rar` | bench-allencahn, bench-burgers1d | + adaptive sampling |
| `time-dependent-hard-constraints` | bench-heat1d | constraint handling |
| `siren-hard-constraints` | bench-wave1d | architecture |
| `multioutput-loss-weighting` | bench-navier-cavity | loss weighting |
| `fourier-features` | ind-helmholtz | architecture / spectral bias |
| `ode-residual-hard-ic` | dyn-double-pendulum | plain residual ODE |
| `operator-fno` | bench-darcy-operator | **operator learning (data-driven only)** |
| `inverse-field` | ind-heat2d-inverse | inverse |
| `inverse-parameter-real` | env-soil-heat-real | inverse (real data) |
| `inverse-hidden-state` | ind-hidden-velocity | inverse |
| `bayesian-ensemble-uq` | poll-source-uq-bpinn | UQ |
| `domain-decomposition` | poll-soil-barrier | scaling |
| `advection-diffusion-reaction` | mine-heap-leach-rt | applied mining |
| `population-balance` | mine-comminution-pbe | applied mining |
| `parametric-kinetics` | mine-flotation-kinetics | applied mining |
| `nonlinear-settling` | mine-thickener-settling | applied mining |
| `advection-diffusion` | poll-ocean-transport | applied env |
| `richards-seepage` | poll-tailings-seepage | applied env |

## 3. What is genuinely absent (the gap the owner is pointing at)

1. **PINO / any physics loss on an operator.** Absent. The operator lane is data-driven, which is precisely
   the FNO weakness the PINO paper exists to fix.
2. **DeepONet.** Absent despite being claimed.
3. **Every geometric / structure-preserving method.** Zero occurrences of Hamiltonian, Lagrangian (as a
   network), symplectic, Lie group, or equivariant anywhere in the source. `dyn-double-pendulum` is a plain
   `ode-residual-hard-ic` residual PINN, so the one mechanical system in the catalogue does **not** conserve
   energy by construction and has no symplectic structure. This is the whole subject of the RSS 2026 workshop
   "The Geometry of Motion: Physics-Informed Structures for Learning and Control" (17 July 2026, Sydney;
   speakers Sachtler, Welde, Cohn, Perez Dattari), whose topic list is a fair map of what a 2026 catalogue
   should cover: Riemannian geometry, Lie groups/symmetries/invariances, geometric Lagrangian and Hamiltonian
   dynamics, structure-preserving model reduction, geometry-aware planning, learning on non-Euclidean spaces.
4. **The failure modes as a documented, demonstrated axis.** The critique (long training, gradient blow-up) is
   correct and citable, and the catalogue has partial fixes (Fourier features, hard constraints, RAR, loss
   weighting) but never states the failure modes as such, nor measures training cost.
5. **Subsurface groundwater.** No confined-aquifer / pumping-well case, despite it being a canonical applied
   target with an analytic anchor (Theis) and a natural PINO demonstration.

## 4. What must NOT be claimed

- Do not say the catalogue "has no FNO": it has a real one, trained and exported.
- Do not repeat "exercises DeepONet" until DeepONet is actually implemented.
- Do not claim a small-scale PINO reproduction reproduces the paper's headline numbers on Kolmogorov flow;
  what a modest reproduction can honestly show is the *mechanism* and the *direction* of the effect on a small
  benchmark, measured on our own grid.
