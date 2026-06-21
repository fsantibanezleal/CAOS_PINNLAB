# Workbench migration — `env-soil-heat-real`

**Component:** `SoilHeatRealContext` · **Context file:** `frontend/src/content/cases/SoilHeatRealContext.tsx`
**Case .py:** `data-pipeline/pinnlab/cases/env_soil_heat_real.py` · **Engine:** DeepXDE (inverse PINN) · **Status of existing bake:** PASS (manifest already has `l2_relative = 0.069`, `recovered_alpha_mm2_s = 0.303`, ONNX parity `1.37e-6`).

## Decision: **SINGLE honest validated-real variant** (no `variants()`, no `param_specs`)

`field_axes` stays the default `inputs = (z, t)` → the baked artifact is the single 2-D `T(z,t)` heatmap (damped, phase-lagged seasonal wave). The `Live` tab re-evaluates the trained net (no parameter slider).

### Rationale (adversarial)

- **Only REAL-data case.** Boundaries (5 cm, 100 cm) and the held-out anchor (10/20/50 cm) are real NOAA USCRN measurements. `real_or_synthetic = validated-real`. The score is the held-out RMSE vs the real interior temperatures, plus the recovered α in physical mm²/s.
- **It is an inverse problem.** The scalar α (effective thermal diffusivity) is the *unknown being recovered*, trained as `log κ` (a DeepXDE `Variable`). α is **not** a network input we could sweep.
- **No closed-form parametric family.** The surface forcing is a real, non-analytic time series; there is no MMS / Green's function that solves the PDE for "all α". A parametric-α sweep would have **no exact anchor to validate against** and would contradict the validated-real honesty. → Faking regimes here is exactly the ADR-0016 §9.A "toy" failure. The playbook (`wip/workbench-migration-state.md`, line 55) already prescribes "single validated-real variant".
- **Time-scrubber rejected.** Although the field is `(z, t)`, the physical story IS the joint depth-vs-time structure (amplitude damping + phase lag with depth). Splitting into t-snapshots would discard the depth axis that carries the whole inverse signal. Keep the full `(z, t)` heatmap as one field.

**Adversarial check — would a parametric version train+validate?** No. No exact solution to anchor it; α is recovered not prescribed; the boundary forcing has no closed form. Single variant is the conservative, correct choice.

## Analytic anchor / validation anchor

There is **no closed-form `analytic()`** — this is a `real-data-holdout` case (correct and expected for a dataset-anchored inverse). The "anchor" is the **held-out real interior temperatures** at 10/20/50 cm, never shown to the optimizer. Proof of validity is empirical, not by substitution:

- Recovered `α ≈ 0.303 mm²/s` sits inside the textbook mineral-soil band `0.2–0.8 mm²/s` (e.g. de Vries soil-thermal-properties; van Wijk & de Vries, *Physics of Plant Environment*).
- Held-out relative-L2 `= 0.069` and `holdout_rmse_c ≈ 1.05 °C` (per-depth: 10 cm 1.26, 20 cm 1.06, 50 cm 0.75 °C) — a sub-1.3 °C reconstruction of sensors the model never saw confirms the recovered α and the diffusive physics out-of-sample.

The forward physics that legitimises the diffusivity: heat conduction `T_t = α T_zz` admits a damped, phase-lagged wave per depth, `T(z,t) ~ A e^{-z√(ω/2α)} cos(ωt − z√(ω/2α))` for a sinusoidal surface forcing of angular frequency ω. The damping/lag rate `√(ω/2α)` is monotone in α, so the top-vs-bottom amplitude ratio identifies α — which is exactly the structure the inverse PINN exploits. (Closed form cited for intuition only; the real boundary forcing is not a single sinusoid, hence no `analytic()`.)

## Exact case `.py` edits (orchestrator)

The current `env_soil_heat_real.py` is **already structurally workbench-correct** (no `variants()`, `field_axes` defaults to `(z,t)`, the bake passes). The migration needs **no method change and no re-bake**. The only edits are clarity-hardening — make the single-variant intent and the physical constants explicit. Apply if desired; safe (no behavioural change to the model/field/ONNX):

1. **Docstring one-liner** — make the single-variant decision explicit. In the module docstring, after the `real_or_synthetic` sentence, add:

```python
# (append to the existing module docstring)
# Workbench: SINGLE validated-real variant — no parametric family (alpha is recovered, not a sweepable input;
# the real surface forcing has no closed form). field_axes defaults to (z,t): the baked field is the single
# damped/phase-lagged T(z,t) heatmap; Live re-evaluates the trained net (no parameter slider).
```

2. **Pin the field grid axis order explicitly** (no behaviour change — documents that `(z,t)` is the heatmap and matches `inputs`). The `CaseSpec` already has `inputs=("z","t")`, `grid={"z":49,"t":_NT}`, and no `param_specs`/`field_axes`, so `case.axes == ("z","t")` already. Optionally add the explicit default for readers:

```python
    # in CASE = CaseSpec(...), keep inputs/grid as-is; optionally state the default field axes for clarity:
    field_axes=("z", "t"),   # explicit: single 2-D heatmap over depth × time (== inputs; no parametric axis)
```

> Note: `__post_init__` requires `grid` axes == `axes`. With `field_axes=("z","t")` and `grid={"z":49,"t":_NT}` this holds. Do **not** add `param_specs` (would force a parameter axis that does not exist here).

3. **No change** to `build()`, `extra_metrics()`, `pde`, the point-set BCs, the `log_kappa` variable, the loss weights, or `train{}`. The existing inverse method (Adam-only, real Dirichlet point-set boundaries, recovered `α = κ L²/τ`) is preserved exactly.

If the orchestrator prefers a **zero-diff** migration, edits 1–2 can be skipped entirely — the case already bakes as the single workbench variant. The mandatory deliverables are the Context (written) and the registry line (below).

## Bake recipe (already validated — keep as-is)

- **Net:** `dde.nn.FNN([2] + [48]*4 + [1], "tanh", "Glorot normal")` (2 inputs `z,t` → 1 output `T`).
- **Inverse variable:** `log_kappa = dde.Variable(3.4)` (κ ≈ e³·⁴ ≈ 30 normalised), passed via `external_trainable_variables=[log_kappa]`.
- **Optimizer:** **Adam only**, `lr = 1e-3`, `adam = 18000`. **No L-BFGS** (`lbfgs=False`) — protects the trainable scalar from being dropped.
- **RAR:** none (not needed; the real-data fit is smooth seasonal, no moving front).
- **Sampling:** `num_domain=4000`, `num_boundary=0`, `num_initial=0`, `num_test=2000`; anchors = the real boundary/initial point sets.
- **Loss weights:** `[1, 60, 60, 20]` = `[pde, bc_top(5cm), bc_bot(100cm), bc_ic(t=0 profile)]` — heavy weight on the real boundaries.
- **Expected band:** held-out relative-L2 `~7e-2` (`~1 °C` RMSE); recovered `α ~0.2–0.8 mm²/s`. (Measured: L2 0.069, RMSE 1.05 °C, α 0.303 mm²/s.)
- **Runtime:** ~4.5 min on CPU (`run_ms ≈ 279 s`). `--quick` smoke first as always.

## Registry line

In `frontend/src/content/cases/registry.tsx` — add the import and the map entry:

```tsx
import { SoilHeatRealContext } from "./SoilHeatRealContext";
// ...
"env-soil-heat-real": SoilHeatRealContext,
```

## Risks / fallback

- **Risk:** adding `field_axes=("z","t")` literally would be a no-op but a careless add of `param_specs` would break `__post_init__` (parameter axis with no input). Fallback: ship the **zero-diff** migration (skip .py edits 1–2 entirely) — the case already bakes correctly as a single variant.
- **Risk:** the recovered α is an *effective* bulk value over a real, moisture-varying soil column; it is honest to label it "effective thermal diffusivity", not a pristine material constant. The Context does this.
- **Risk:** held-out RMSE ~1 °C is the genuine error, not 0 — that is the point of a real-data case; the Context labels the numbers as the honest figures, never as a perfect fit. Do not tune toward an unrealistically low RMSE (would imply leakage of the held-out sensors).
