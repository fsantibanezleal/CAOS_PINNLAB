# PINN-Lab GLOBAL review + fix (2026-07-09) — the offline pipeline must COMPUTE the demonstrations

## 0. Root cause (owner-identified, my failure)

The **offline pipeline is the product**; the web app is only a viewer of what it computes. Today every case's
pipeline bakes ONE thin result (Helmholtz bakes only the Fourier-feature field; Allen-Cahn one "Default" run), so
there is nothing rich to show. I treated the symptom (UI layout) instead of the origin: the pipeline does not
CALCULATE the comparative science that makes a PINN catalogue worth looking at. I did not understand that the
repo's relevance is the deep offline computation. This document reframes everything around that.

**Principle:** for every case the offline pipeline computes the METHOD LADDER and its DEMONSTRATION — the naive
baseline that fails, the fix that works, the ablations, the diagnostics, and (where relevant) the data-driven vs
pure-physics contrast — as comparable baked artifacts. The web app then SHOWS the contrast; it invents nothing.

## 1. What each pipeline must now COMPUTE (per case), not just one field

For every case, bake a small set of comparable runs + diagnostics so the science is visible. The core artifact is
the **head-to-head comparison: STANDARD PDE solution vs naive PINN vs adapted PINN vs data-driven PINN**, all on
the same grid, each with its error against the standard.
- **Standard PDE solution (the classical rung)** — a real classical numerical solver run offline (finite-difference
  / finite-element / spectral, per problem) plus the analytic MMS where it exists. This is the GROUND TRUTH the PINN
  variants are measured against. This does not exist today and is the single biggest missing artifact.
- **Naive PINN run** — the plain PINN the docs say FAILS, actually run, so its failure is visible next to the standard.
- **Adapted PINN run (the fix / SOTA)** — the real method (Fourier features, hard constraints, FBPINN, ...).
- **Data-driven PINN run (where it applies)** — physics + observed data (inverse / assimilation / operator).
- **(Where it applies) a novel/beyond-SOTA run** — per the product bar.
- **Difference + error maps** — each PINN variant minus the standard solution; where each is wrong.
- **Diagnostics** — the curve that explains WHY: convergence (loss vs epochs), spectral-bias energy vs frequency,
  the parameter sweep where the baseline breaks (e.g. wavenumber for Helmholtz), sampling/RAR point maps.
- **Data-driven contrast (where relevant)** — pure-physics vs data+physics, the sensor/data points, and how the
  result and its uncertainty change as data is added/removed.

## 2. Per-case demonstration ladder (the design)

| Case | The fix (method) | Naive baseline that FAILS (to bake) | Diagnostics to compute | Data-driven contrast |
|---|---|---|---|---|
| helmholtz | Fourier features | plain tanh MLP -> spectral bias, high-k blurred/flat | naive vs Fourier field + error; spectral-energy vs freq; wavenumber sweep n=1..6 where naive collapses | — |
| allencahn | hard-constraints + RAR | soft-BC uniform-sampling -> smears the sharp layers | error map at the transition layer; RAR point density; loss curve | — |
| burgers1d | hard-constraints + RAR | naive -> oscillates/smears the shock | shock front naive vs fix; error at the front; RAR points | — |
| wave1d | SIREN + hard constraints | tanh -> dispersion/spectral bias at large c | SIREN vs tanh dispersion error vs c | — |
| poisson2d | hard constraints | soft BC -> boundary error | boundary error naive vs fix vs k | — |
| navier-cavity | multi-output loss weighting | equal weights -> v-centerline off | Ghia centerline overlay; weighting ablation | — |
| darcy-operator | FNO operator | per-instance PINN (retrain per a(x)) | one-pass FNO vs retrain-per-instance; held-out L2 | learns from a(x)->u(x) data pairs |
| heat2d-inverse | inverse field (data+physics) | pure physics, NO data -> k unrecoverable | recovered k vs #sensors sweep; pure-vs-hybrid; residual | THE data-driven showcase |
| soil-heat-real | inverse parameter (REAL) | assumed alpha vs recovered-from-real-data | fit to REAL USCRN temps; recovered alpha vs assumed | REAL data |
| source-uq-bpinn | Bayesian ensemble UQ | single deterministic PINN (no uncertainty) | mean +- sigma; sigma grows in data-sparse regions; calibration | data + uncertainty |
| double-pendulum | PINN vs RK45 | (contrast IS PINN vs integrator) | leave-time; phase-error growth; twin divergence | — |
| soil-barrier | domain decomposition (FBPINN) | single-domain PINN -> fails at the barrier kink | single vs FBPINN; interface error | — |
| poisson/heat/ocean/heap-leach/thickener/comminution/flotation/tailings | the stated scheme | a plain-PINN / no-scheme baseline where meaningful | vs analytic/MMS error; conservation; parameter sweep | — |

(The 6-variant cases already sweep a parameter; the missing axis is the naive-vs-fix METHOD comparison + diagnostics.)

## 3. Pipeline re-architecture

- Each `cases/<case>.py` gains a **method axis** (e.g. `baseline` | `method` | `novel`) alongside the existing
  parameter regimes: `build(seed, method)` trains each; export each to ONNX; bake each field + the difference.
- Compute + persist the **diagnostics** (spectral energy, loss history, error maps, sweeps) as small JSON traces.
- Manifests gain a `comparisons` block (which runs are contrastable) and a `diagnostics` block.
- Keep the local `.venv` (3.13) precompute discipline; runs are heavy -> phase it (Section 5).

## 4. Web app (the VIEWER of the computed richness)

Only after the pipeline computes the above:
- **Compare mode**: naive vs fix side-by-side (or A/B swipe) + the difference map; the "regime" selector becomes a
  METHOD selector (baseline / fix / novel) crossed with the parameter regimes.
- **Diagnostics panel**: the spectral-bias curve, the convergence curve, the sweep where naive breaks — the WHY.
- **Live** demonstrates the mechanism: toggle Fourier features on/off and watch the field go wrong->right; raise the
  wavenumber and watch the naive lane collapse; add/remove data on the inverse case and watch k + sigma change.
- Layout that serves this: compact command bar (Domain+Case dropdowns, method+regime, view) + a full-width stage;
  equation/metrics in a top strip (never cut); interactive graphs (rich hover, snapshot, focus, detail popup);
  the both-mode field probe (follow / click-pin / double-click-release). Paused-by-default, no compute bombs.

## 5. Phased rollout (each phase = a validated, screenshot-QA'd, issue-tracked patch)
1. **Helmholtz as the reference**: bake naive tanh vs Fourier + error + spectral curve + wavenumber sweep; build the
   compare + diagnostics views; Live toggles Fourier on/off. Prove the pattern end-to-end on ONE case.
2. Propagate the baseline-vs-fix bake + diagnostics to the other canonical benchmarks (allencahn, burgers, wave,
   poisson, navier).
3. Data-driven cases (heat2d-inverse sensor sweep + pure-vs-hybrid; soil-heat-real; source-uq-bpinn).
4. Remaining mining/pollution cases + the FBPINN barrier.
5. Global layout + interactive-plot polish across all.

## 6. To validate before I compute anything
- **A.** Is "the pipeline must compute the naive-vs-fix ladder + diagnostics per case" the correct framing of the fix?
- **B.** Start with Helmholtz end-to-end as the reference template, then propagate (recommended), or design all 20
  ladders on paper first, then compute?
- **C.** Compute budget: these are real training runs on the local machine; OK to run them incrementally (phase by
  phase), committing baked artifacts as each case is proven?
