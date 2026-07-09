# PINN-Lab: per-case demonstration ladder design (all 20) - for validation before compute

Owner chose "design all 20 ladders first". This is the paper design. Each case's pipeline will COMPUTE and bake a
comparable set so the app can SHOW the science. Grounded in each case's actual definition (physics, method, reference
already in `cases/*.py`). **Honesty flag**: not every case has a dramatic "naive fails" story; where the naive PINN
essentially works (easy forward MMS cases), the ladder is standard-solver-vs-PINN + diagnostics, and I say so rather
than fabricate a failure. Strength = how strong the naive-vs-fix contrast genuinely is (STRONG / MODERATE / WEAK).

## Legend - what every case bakes
- **STD** = standard solution (ground truth): the analytic/MMS closed form AND/OR a classical numerical solver
  (finite-difference / finite-volume / spectral) run offline. The bar every PINN variant is measured against.
- **NAIVE** = the plain PINN without the case's fix (soft BC/IC, plain tanh, uniform sampling), run so its behaviour
  is visible next to STD.
- **ADAPTED** = the case's actual method (the fix).
- **DATA** = a physics+data (inverse / assimilation / operator / UQ) variant, where the case is about data.
- **DIAG** = diagnostics that explain WHY (error maps vs STD, convergence, spectral bias, parameter sweep where naive
  breaks, conservation).

---

## Group A - canonical benchmarks (8)

### 1. allencahn - STRONG
- STD: spectral reference `Allen_Cahn.npz` (already vendored) + optional FDM/IMEX check.
- NAIVE: soft-IC/BC PINN, uniform sampling, no RAR -> collapses to a metastable state, smears the sharp +/-1 layers.
- ADAPTED: hard-constraint IC (output transform) + RAR adaptive sampling chasing the moving interface.
- DIAG: error map at the transition layer (naive vs adapted vs STD); RAR collocation-point density over time; L2(t)
  naive vs adapted; training-loss curve.
- Story: watch the naive lane lose the interface while the RAR lane tracks it.

### 2. burgers1d - STRONG
- STD: Whitham traveling-shock closed form (any nu) + FDM (upwind/WENO) reference.
- NAIVE: soft-BC, uniform sampling, no RAR -> oscillates/smears at the small-nu front.
- ADAPTED: hard constraints (IC + both BCs) + RAR at the front.
- DATA: n/a.
- DIAG: front-thickness naive vs adapted vs STD; L2 vs nu sweep (naive degrades as nu shrinks); RAR points at the shock.

### 3. wave1d - STRONG (spectral bias)
- STD: manufactured standing wave sin(pi x) cos(c pi t) + FDM (leapfrog) reference.
- NAIVE: tanh activation, soft BC -> spectral bias: dispersion/amplitude decay at high c.
- ADAPTED: SIREN (sin) + hard-constraint transform (BC + both ICs exact).
- DIAG: SIREN-vs-tanh error vs c; dispersion/amplitude error over t; spectral content of the two solutions.

### 4. poisson2d - MODERATE
- STD: manufactured g(x;k)g(y;k) closed form + 5-point FDM / spectral reference.
- NAIVE: soft-Dirichlet-loss PINN -> boundary error, worse as k grows.
- ADAPTED: hard-constraint distance transform x(1-x)y(1-y)N (no BC loss).
- DIAG: boundary-error naive vs adapted vs k; interior L2.

### 5. heat1d - MODERATE
- STD: analytic exp(-alpha pi^2 t) sin(pi x) + FDM (Crank-Nicolson).
- NAIVE: soft IC/BC -> IC/BC drift, decay-rate error.
- ADAPTED: time-dependent hard-constraint transform.
- DIAG: IC/BC residual naive vs adapted; decay-rate error vs alpha.

### 6. navier-cavity - STRONG
- STD: Ghia-Ghia-Shin (1982) Re=100 centerlines (digitized) + a streamfunction-vorticity FDM cavity solver.
- NAIVE: equal loss weights, no corner regularization, no gauge up-weight -> v-centerline wrong, corner artifacts.
- ADAPTED: multi-output loss weighting (BC/gauge 10x) + regularized lid + pressure gauge.
- DIAG: u/v centerline overlay vs Ghia (naive vs adapted); loss-weighting ablation; continuity residual map.

### 7. darcy-operator - STRONG (operator / data)
- STD: finite-difference Darcy solve u_true for each held-out coefficient field a(x) (already computed).
- NAIVE: per-instance PINN (must retrain for each a(x)) - slow, no generalization; and/or a plain MLP operator.
- ADAPTED/DATA: the FNO operator G:a(x)->u(x), trained on a(x)->u pairs (data-driven by construction), one forward pass.
- DIAG: held-out test relative-L2 (generalization); one-pass FNO vs retrain-per-instance cost; error map vs STD.

### 8. zero-source (control) - WEAK (by design a control)
- STD: MMS u* = a*g(x,y) closed form.
- NAIVE vs ADAPTED: modest (this is the MMS/degenerate control). Keep the a=0 degenerate negative control (f=0 -> u=0).
- DIAG: MMS L2 vs a; the flat-zero at a=0. Honest label: this is a verification control, not a naive-fails showcase.

---

## Group A (dynamical) - 1

### 9. double-pendulum - STRONG (chaos)
- STD: high-accuracy RK45 integrator (rtol=atol=1e-10) - the classical ground-truth trajectory (already computed).
- NAIVE: soft-IC PINN (or a t^1 transform) -> leaves the trajectory almost immediately.
- ADAPTED: ODE-residual PINN + hard IC transform theta0 + t^2 N(t); tanh.
- DIAG: leave-time (when |error| exceeds a threshold); phase-error growth; twin-IC butterfly divergence.
- Story: the PINN tracks RK45 for ~1.99 s, then chaos peels it away - the honest limit.

---

## Group D - industrial-fluids-heat (2)

### 10. heat2d-inverse - STRONG (the data-driven showcase)
- STD: MMS T* = sin(pi x)sin(pi y), k* = 1 + 0.5 sin sin (closed form).
- NAIVE: pure-physics PINN with NO sensor data -> k(x,y) is underdetermined, recovery fails.
- DATA: physics + N sparse noisy sensors; sweep N in {0, 10, 25, 50, 100} and watch k* emerge.
- ADAPTED: PFNN 2-output, product-rule residual, PointSetBC, softplus(k) positivity.
- DIAG: recovered-k error vs #sensors; pure-vs-hybrid side-by-side; residual where |grad T| ~ 0 (hardest).

### 11. helmholtz - STRONG (THE reference / phase 1)
- STD: spectral (Fourier/Chebyshev) or 5-point FDM Helmholtz solve + analytic MMS sin(k0 x)sin(k0 y).
- NAIVE: plain tanh MLP -> spectral bias, high-k pattern blurred/flat.
- ADAPTED: random Fourier-feature input embedding (Tancik 2020; Wang-Wang-Perdikaris 2021) + soft Dirichlet.
- DIAG: spectral-energy vs frequency (the plateau naive can't cross); wavenumber sweep n=1..6 where naive collapses;
  error maps naive vs adapted vs STD.
- Story (Live): toggle Fourier on/off -> field goes wrong->right; raise n -> naive lane collapses, adapted holds.

---

## Group C - pollution-environmental (5)

### 12. soil-heat-real - STRONG (REAL data)
- STD: FDM heat solve driven by the real 5 cm / 100 cm boundaries, evaluated at the held-out 10/20/50 cm depths.
- NAIVE: assume a literature diffusivity (no inversion) -> misfits the interior sensors.
- DATA/ADAPTED: inverse PINN recovering alpha from the real boundaries; validate out-of-sample vs held-out real temps.
- DIAG: held-out RMSE (deg C) assumed-alpha vs recovered-alpha; recovered alpha (mm^2/s) vs the soil-physics range;
  residual vs depth.

### 13. source-uq-bpinn - STRONG (data + uncertainty)
- STD: analytic diffusion mode c* = e^{-D pi^2 t} sin(pi x).
- NAIVE: single deterministic PINN -> one answer, no error bars.
- DATA/ADAPTED: deep ensemble (Lakshminarayanan 2017) -> mean + std; std small near sensors/walls, grows where sparse.
- DIAG: mean L2 vs c*; 2-sigma calibration; std map vs sensor placement (add/remove a sensor -> watch std change).

### 14. soil-barrier - STRONG (FBPINN)
- STD: MMS layered series-resistance profile (closed form, exhibits the kink) + FDM with the D-jump.
- NAIVE: single-domain PINN -> cannot represent the discontinuous c_x at the barrier faces (rounds the kink).
- ADAPTED: FBPINN domain decomposition (partition-of-unity, 2 channels blended by sigmoid windows).
- DIAG: c_x at the interface single-domain vs FBPINN vs STD; interface error.

### 15. ocean-transport - MODERATE
- STD: advected-diffused Gaussian (2D Green's function, exact) + FDM advection-diffusion.
- NAIVE vs ADAPTED: modest (forward, well-posed). Honest contrast = STD vs PINN accuracy over t + the time-scrubber.
- DIAG: patch center/spread vs exact over t; mass conservation; L2(t).

### 16. tailings-seepage - MODERATE (nonlinear stiffness)
- STD: Kirchhoff-transform exact (alpha-dependent) + FDM Richards (Gardner).
- NAIVE: plain PINN on the raw nonlinear Richards operator -> slower/less accurate on the stiff K'(psi) terms.
- ADAPTED: the case's residual formulation; alpha as input.
- DIAG: head-profile error vs exact; convergence naive vs adapted; alpha dependence.

---

## Group B - mining-mineral-processing (4)

### 17. comminution-pbe - MODERATE
- STD: advected-diffused Gaussian / 1D Green's function (exact) + FDM.
- NAIVE vs ADAPTED: modest. Contrast = STD vs PINN + mass conservation of the size distribution.
- DIAG: distribution vs exact over t; mass conservation; grind-rate g sweep.

### 18. heap-leach-rt - MODERATE (coupled reaction)
- STD: MMS 2-species (cA*, cB*) + FDM advection-diffusion-reaction.
- NAIVE: plain PINN on the coupled bimolecular reaction -> struggles with the A*B coupling term.
- ADAPTED: the case's residual; downward Darcy advection.
- DIAG: per-species front error vs exact; reaction-coupling residual.

### 19. thickener-settling - MODERATE/STRONG (degenerate PDE)
- STD: MMS descending tanh front (exact through the genuine nonlinear flux + degenerate D) + a FV solver for the
  degenerate conservation law.
- NAIVE: plain PINN -> smears the sharp settling front / the degenerate diffusion switch.
- ADAPTED: the case's C^1-regularized residual.
- DIAG: front sharpness vs exact; R (descent-rate) sweep; bed-rise vs supernatant.

### 20. flotation-kinetics - WEAK (easy ODE, be honest)
- STD: exact C* = exp(-k t).
- NAIVE vs ADAPTED: minimal - this is an easy first-order ODE; a plain PINN already nails it. Honest label: the value
  here is the parametric k-family + exact validation + recovery R(t), NOT a naive-fails showcase.
- DIAG: C vs exact; rate-constant k sweep; recovery curve.

---

## Cross-cutting deliverables
- **Manifest schema**: add `comparisons` (std | naive | adapted | data) and `diagnostics` (named small JSON traces).
- **Pipeline**: `build(seed, lane)` trains each lane; a shared `standard_solver.py` (FDM/FV/spectral helpers) computes
  STD; diagnostics computed alongside. Heavy -> phased, artifacts committed as each case is proven.
- **Web**: a compare view (STD | naive | adapted | data + difference) + a diagnostics panel + mechanism-Live; the
  layout/interactive-graph work from #24 (dropdown nav, top context strip so the equation is never cut, rich-hover /
  snapshot / focus / detail-popup graphs, both-mode probe) becomes the shell that presents all this.

## Honest summary of contrast strength
- STRONG (11): allencahn, burgers1d, wave1d, navier-cavity, darcy-operator, double-pendulum, heat2d-inverse,
  helmholtz, soil-heat-real, source-uq-bpinn, soil-barrier.
- MODERATE (7): poisson2d, heat1d, ocean-transport, tailings-seepage, comminution-pbe, heap-leach-rt, thickener-settling.
- WEAK / verification (2): zero-source (control), flotation-kinetics (easy ODE).

For the WEAK/MODERATE forward cases the demonstration is honestly "standard solver vs PINN + convergence/conservation
+ parameter sweep", not a fabricated naive failure.
