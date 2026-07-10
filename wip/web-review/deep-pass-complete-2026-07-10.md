# PINN-Lab deep pass: COMPLETE (2026-07-10)

Record of the deep pass that answered the owner's root critique: "the app has nothing rich to show because the
offline pipeline never computed the real comparative content." State: complete + fully validated on the live site
(pinnlab.fasl-work.com, v0.20.008). Issues #24 (UI) + #25 (pipeline) closed.

## The root fix

The offline pipeline now COMPUTES the comparison the docs describe (it did not before: one thin field per case). The
app is the viewer. Design + capability persisted in:
- `docs/architecture/method-ladder-comparison.md` (the capability: lanes, tools, kits, schema, results).
- `wip/web-review/ux-redesign-2026-07-09.md` (the reframed plan) + `case-ladders-design-2026-07-09.md` (the 20-case design).

## What ships (all real, nothing invented)

**Every one of the 20 cases has a real comparison/validation.**
- 17 Compare views (standard vs PINN + error), incl. 4 naive-vs-fix ladders:
  - helmholtz: naive tanh 120.8% vs Fourier 9.3% + Diagnostics (wavenumber sweep + spectrum).
  - allencahn: naive soft-PINN 95.4% (collapse) vs hard-constraint+RAR 0.4%.
  - heat2d-inverse: pure-physics 356% (no data) vs physics+data 4.0% (the data-driven contrast).
  - soil-barrier: single-domain vs FBPINN, ~19% both (honest, subtle edge).
  - darcy: FNO one-pass vs finite-difference reference (2.5%).
- navier: Diagnostics = Ghia 1982 Re=100 centerline validation (u RMSE 0.053, v 0.029).
- soil-heat-real: Diagnostics = reconstruction vs REAL held-out USCRN sensors 10/20/50 cm (RMSE 1.24/1.05/0.75 degC).
- double-pendulum: its trajectory IS the PINN-vs-RK45 comparison.

**Pipeline tools** (`data-pipeline/`): build_standard_comparisons.py, build_helmholtz_ladder.py, build_allencahn_ladder.py,
build_naive_lane.py, build_ladder.py (fair: both lanes fresh), build_navier_centerlines.py, build_soilheat_validation.py.
Cases gained `build_naive()` / `standard_field()` where needed (helmholtz, allencahn, soil-barrier, wave1d, heat2d-inverse).

**Web** (`frontend/src/`): CompareKit + DiagnosticsKit (manifest-driven, generic XYChart for benchmark-vs-model),
lib/snapshot.ts (PNG export per panel/chart); contract `comparison`/`diagnostics` blocks. Compare is the default view.

**UI (#24)**: equation moved to a full-width top strip (never cut), both-mode field probe (graphs follow the cursor /
click pins / double-click releases), Highlights+Domain dropdowns, full-width stage, snapshot buttons.

## Honesty decisions (owner rule: never invent)

- wave1d: a FAIR test (SIREN + tanh both fresh at the hard regime c=2) landed both ~11% -> NO fabricated spectral-bias
  contrast; kept its honest standard-vs-PINN comparison.
- navier: a rushed finite-difference cavity solve DIVERGED (NaN) and its guard let it bake garbage -> reverted; replaced
  with the robust Ghia centerline validation. The bug (`nan > 0.06` is false) is a lesson: guard with `!(x < tol)`.

## Validation

Full validation on the LIVE site: all 20 cases render, 17 Compare + navier/soil-heat Diagnostics + pendulum trajectory,
light + dark, 0 console errors. Screenshot-verified per case.

## Optional future (non-blocking)

- Detail popup (click a Compare panel / chart -> enlarged modal) + change-focus/zoom on plots.
- Naive lanes for the easy forward cases are deliberately omitted (no real contrast at feasible budgets - would be invented).
