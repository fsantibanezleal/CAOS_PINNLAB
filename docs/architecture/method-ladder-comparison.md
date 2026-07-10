# The method ladder: computing and showing the comparison

The offline pipeline is the product; the web app is only a viewer of what it computes. For a long time each case
baked exactly **one** solution field, so the app could describe a method (naive fails, the fix works) it could never
**show**. This document is the capability that fixes that: the pipeline now **computes the comparison** the docs
describe, per case, and the app renders it. Added 2026-07-10 (issues #24, #25).

## The principle

For every case the pipeline computes, and the app shows, the honest **method ladder**:

| Lane | What it is | Role |
|------|------------|------|
| **standard** | the ground truth: a closed-form/MMS solution and/or a real classical numerical solver (finite-difference, spectral) | reference |
| **naive** | the plain PINN *without* the case's fix (soft BC, plain tanh, uniform sampling, no domain decomposition, no data) | baseline |
| **adapted** | the case's actual method (Fourier features, hard constraints + RAR, SIREN, FBPINN, loss weighting, FNO, ...) | fix |
| **data-driven** | physics + observed data, where the case is about assimilation / inversion / an operator | data |

plus **diagnostics** that explain *why* (spectral-bias curve, convergence, the parameter sweep where the naive lane
collapses, benchmark-centerline or held-out-sensor validation).

**Nothing is invented.** Every field and number is a real baked result. Where a fair test shows no real contrast
(e.g. wave1d: both SIREN and tanh land ~11% at the hard regime), the case keeps only its honest standard-vs-PINN
comparison rather than a fabricated one. Where a rushed solver produced garbage (a diverged FDM cavity), it was
reverted, not shipped.

## What is computed, per case

The pipeline / ladder tools bake a **comparison trace** (several fields on one grid) and an optional **diagnostics**
JSON, then patch the manifest with a `comparison` and/or `diagnostics` block.

- `data/derived/<case>/comparison.json` — `{ axes, fields: { standard, naive?, adapted, err_naive?, err_adapted, ... }, summary: { naive_vs_std?, adapted_vs_std, ... } }`
- `data/derived/<case>/diagnostics.json` — `{ wavenumber_sweep? , radial_spectrum? , line_comparisons? , rmse? }`
- manifest `comparison` — `{ trace, lanes: [{ key, label, role, err? }], onnx_naive?, note }`
- manifest `diagnostics` — `{ path }`

### The pipeline tools (`data-pipeline/`)

| Tool | What it does | Training |
|------|--------------|----------|
| `build_standard_comparisons.py` | universal: for every analytic case, bake standard (closed form) vs the already-baked adapted PINN + error | none |
| `build_helmholtz_ladder.py` | the reference: FDM standard + naive tanh + Fourier PINN + error maps + the wavenumber sweep + radial spectrum | naive + adapted, fresh |
| `build_allencahn_ladder.py` | spectral-reference standard + the naive soft-PINN collapse vs hard-constraint+RAR | naive, fresh |
| `build_naive_lane.py <case>` | generic: add a naive lane to a case that has `build_naive()` + an existing comparison | naive, fast |
| `build_ladder.py <case> [variant]` | generic, fair: train BOTH adapted + naive fresh at a chosen (hard) regime, bake the comparison | both, fresh |
| `build_navier_centerlines.py` | the Ghia 1982 Re=100 centerline validation (PINN vs benchmark points) | none |
| `build_soilheat_validation.py` | PINN reconstruction vs REAL measured USCRN temps at the held-out depths | none |

A case exposes `build_naive(seed)` (the plain net) and, where useful, `standard_field(coords)` (a classical solver)
alongside its `build(seed)` and `analytic()`.

**Fairness matters.** Reusing a pre-trained adapted ONNX while training the naive fresh is an unfair, budget-mismatched
comparison that can invert the result; `build_ladder.py` trains both lanes fresh with the same budget so the contrast
is the *method*, not the training.

## What is shown (the web viewer)

- **CompareKit** (`frontend/src/components/kits/CompareKit.tsx`) — renders the comparison lanes as a row of heatmaps
  (standard | naive | adapted | ...) on a shared colour scale, plus the error maps, plus a shared hover probe that
  reads *every* lane at a point, plus the real baked relative-L2 headline. Manifest-driven, so it lights up for any
  case that has a `comparison`. It is the **default view** when a comparison exists.
- **DiagnosticsKit** (`frontend/src/components/kits/DiagnosticsKit.tsx`) — the *why*: the wavenumber sweep (naive L2
  climbs, the fix stays low), the radial spectral energy, and generic **line comparisons** (an `XYChart` of benchmark
  points vs a model curve — used for the Ghia centerlines and the soil-heat held-out sensors).
- Both carry a **snapshot-to-PNG** button per panel/chart (`frontend/src/lib/snapshot.ts`).

## The results (real, baked)

- **Helmholtz** — naive tanh **120.8 %** vs Fourier **9.3 %** vs the classical FDM standard; the sweep shows the naive
  lane going from 3 % (n=1) to ~100 % (n>=2).
- **allencahn** — naive soft-PINN **95.4 %** (collapses to a metastable state, smears the sharp ±1 layers) vs
  hard-constraint+RAR **0.4 %**, vs the spectral reference.
- **heat2d-inverse** — pure physics with no data **356 %** (k underdetermined) vs physics + ~100 sensors **4.0 %**:
  the data is what makes the inverse solvable.
- **soil-barrier** — single-domain vs FBPINN, both ~19 % on the CPU lane (the edge is subtle, shown honestly).
- **darcy** — the FNO operator's one-pass prediction vs the finite-difference reference (~2.5 %).
- **navier** — the PINN velocity vs the Ghia (1982) Re=100 centerlines (u RMSE 0.053, v 0.029).
- **soil-heat-real** — the reconstruction vs REAL measured USCRN temperatures at the held-out 10/20/50 cm depths,
  out-of-sample (RMSE 1.24 / 1.05 / 0.75 °C).

## Adding a comparison to a new case

1. Give the case a `build_naive(seed)` (the plain net without the fix) and, if a closed form is not enough, a
   `standard_field(coords)` classical solver.
2. Run `build_ladder.py <case> [hard-variant]` (both lanes fresh) or, for a training-free standard, extend
   `build_standard_comparisons.py`.
3. The manifest gains `comparison`/`diagnostics`; the CompareKit/DiagnosticsKit render them with no per-case web code.
4. Screenshot-verify (light + dark, 0 console errors) before deploy. Never ship a fabricated or diverged result.
