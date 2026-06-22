# Case design — poll-source-uq-bpinn (deep-ensemble Bayesian PINN, epistemic UQ)

**Component:** `SourceUqBpinnContext`  ·  **Registry line:** `"poll-source-uq-bpinn": SourceUqBpinnContext`

## Decision: SINGLE honest benchmark variant (no parametric sweep, no discrete family)

The case ships as ONE variant — the deep-ensemble UQ demonstrator exactly as it already is. The Python file already
has the correct single-variant shape (no `variants()`, no `param_specs`, `field_axes` defaults to `("x","t")`), so the
**only** edits below are confirmation/hardening, not a restructure. The real deliverable is the deep bilingual Context.

### Why not PARAMETRIC

A parametric sweep would have to make a *network input* the swept knob. The only closed-form-exact knob is the
diffusivity `D`: `c*(x,t;D) = e^{-Dπ²t} sin(πx)` solves `c_t = D c_xx` for ALL `D` (proof below). But making `D` a
network input **destroys the reason this case exists** — the epistemic-UQ-from-sparse-sensors deep ensemble. The
engine is built around fixed-`D` sensor bagging; the std maps to "where sensors are sparse," which has no meaning once
`D` is a free input that the net must also interpolate. Bolting on a `D`-sweep to hit a chip count is exactly the
ADR-0016 §9.A "toy / fabricated-regime" failure. Rejected.

### Why not DISCRETE (noise / sparsity levels)

The natural UQ family is *sensor noise σ* and/or *sensor count N* — low noise ⇒ tight ensemble, small std; high noise
⇒ wide disagreement. Each level is genuinely meaningful. **But the pipeline cannot bake it honestly.** The orchestrator
trains the net ONCE and re-bakes one field trace per variant via `param_grid`, which only varies **network-input**
parameter axes. σ and N are *training-data* knobs, not network inputs — so every discrete-noise variant would re-bake
the **identical** field from the **one** trained ensemble (the `params` keys aren't in `case.inputs`, so `param_grid`
ignores them; see `model/analytic.py::param_grid`). Identical fields under different chips ⇒ fake. A genuine
discrete-noise family would require K separate trained ensembles ⇒ K separate ONNX exports, which the current
"train-once, share-ONNX" pipeline does not support. Rejected as un-bakeable-honestly here.

### Why not TIME-SCRUBBER

`t` is already a **field axis** — the heatmap is the full `x–t` plane, not a 2-D slice per `t`. There is nothing to
scrub out. Rejected.

### Conclusion

SINGLE honest benchmark, matching the `bench-allencahn` precedent (stiff/structural case with no closed-form
parametric family). Honesty over chip count.

## Analytic anchor + substitution proof

Reference field (fundamental diffusion mode), `D = 0.1`, `c|_{x=0,1} = 0`, `c(x,0) = sin(πx)`:

    c*(x,t) = e^{-D π² t} · sin(π x)

Substitution into `c_t = D c_xx`:

    c*_t  = ∂/∂t [ e^{-Dπ²t} sin(πx) ] = -D π² · e^{-Dπ²t} sin(πx) = -D π² · c*
    c*_xx = ∂²/∂x² [ e^{-Dπ²t} sin(πx) ] = -π² · e^{-Dπ²t} sin(πx) = -π² · c*
    ⇒ D · c*_xx = -D π² · c* = c*_t.   ✓  (exact for all (x,t), and for any D > 0)

Boundary: `sin(π·0) = sin(π·1) = 0` ⇒ `c*|_{x=0,1} = 0` ✓. The hard output transform `x(1-x)·N` reproduces this BC
exactly. The per-variant **relative-L2 of the mean μ vs c\*** is therefore the true mean error; UQ quality is the
**2σ calibration** in `extra_metrics`.

## Exact case .py edits (orchestrator applies)

The case is **already** a valid single-variant workbench case under `base.py::__post_init__`:
`inputs=("x","t")`, `field_axes=()` → `axes=("x","t")`; `grid={x,t}`==axes ✓; `param_axes = inputs−axes = {}` ==
`param_specs` keys `{}` ✓; `domain={x,t}`==inputs ✓. No `variants()` ⇒ registry ships the single `"default"` variant.

Keep the existing SOTA method (deep ensemble, hard walls, `[mean,std]` ONNX) verbatim. The only changes are a one-line
docstring note making the single-variant decision explicit, and confirming the `CaseSpec` stays exactly as-is. No
`variants()`, no `param_specs`, no `analytic`/`build`/output-transform change.

```python
# In poll_source_uq_bpinn.py — the CaseSpec stays EXACTLY as it is today (do NOT add variants() or param_specs):
#   - inputs=("x", "t"); outputs=("c", "c_std"); field_axes defaults to ("x","t")
#   - validation_anchor="analytic"; real_or_synthetic="synthetic-illustrative"
#   - train={"lr":1e-3,"adam":6000,"lbfgs":False,"num_domain":1500,"num_boundary":0,"num_test":2000}
# Only edit: append the single-variant rationale to `notes` so the decision is recorded in the artifact.

CASE = CaseSpec(
    # ... all fields unchanged ...
    notes="Deep ensemble of K=5 independently-initialized PINNs (approx. Bayesian); hard c=0 walls; N=24 sparse noisy "
          "sensors; exported as a single ONNX emitting [mean, std]. Custom-engine (prebuilt) case. "
          "Single honest benchmark variant: the UQ knob (sensor noise/sparsity) is a TRAINING-DATA knob, not a "
          "network input, so a discrete family would need K separate ONNX exports the share-one-ONNX pipeline does "
          "not support; and a D-input sweep would erase the sparse-sensor UQ story (ADR-0016 §9.A). Not parametric.",
)
```

No other edits. `analytic()`, `extra_metrics()`, `_Ensemble`, and `build(seed, quick=...)` are correct as written:
- `build()` returns `{"model": _Ensemble(nets), "input_dim": 2, "prebuilt": True}` → `train.py` takes the prebuilt
  branch, exports `model.net` (the `[mean,std]` torch graph) to ONNX, and `infer.py` reshapes both output columns
  (`outputs=("c","c_std")`) onto the `61×61` field grid. The parity check reshapes `(N,-1)` so it handles 2 columns.
- `evaluate.py` scores `outputs[0]` ("c" = mean) vs the analytic anchor and merges `extra_metrics` (2σ calibration).

## Bake recipe

- **Engine / method:** DeepXDE, deep ensemble (prebuilt custom-engine path) — unchanged.
- **Members:** `K=5` independently-initialised FNNs, distinct seed + bootstrap (bagging) of sensors per member.
- **Net per member:** `FNN([2] + [24]*3 + [1], "tanh", "Glorot normal")` (small is fine; UQ comes from the ensemble,
  not from a big net).
- **Hard constraint:** output transform `x(1-x)·y` ⇒ `c=0` walls exact (no BC loss); survives ONNX.
- **Optimizer:** Adam, `lr=1e-3`, `6000` iters per member, `loss_weights=[1, 30]` (pde, data). **No L-BFGS, no RAR** —
  L-BFGS would collapse the K members toward the same minimum and *destroy* the epistemic spread that is the whole
  point; keep them at the Adam basins they each found.
- **Sampling:** `num_domain=1500`, `num_boundary=0` (walls are hard), `num_test=2000`, `N_OBS=24`, `NOISE=0.02`.
- **Export:** ONE ONNX graph emitting `[mean, std]` (input_dim=2), opset 18. Parity `< 1e-6` (pure tensor stack/mean/std).
- **Quick smoke first:** `--quick` ⇒ `k=2`, `200` iters — exercises train→ONNX→parity→bake in ~30 s before the real bake.
- **Expected band:** mean relative-L2 vs `c*` **< 5e-2** (typically ~1–3e-2); ensemble std small at walls/sensors and
  larger in data-sparse interior; **~95% 2σ calibration** (`uq_calibration_2sigma ≈ 0.93–0.97`). Bake is light
  (5 small nets × 6k Adam, CPU ≈ a few minutes) — safe to pair with a heavy bake per the playbook.

## Registry / wiring (orchestrator, NOT this agent)

- `frontend/src/content/cases/registry.tsx`: import `SourceUqBpinnContext` and add
  `"poll-source-uq-bpinn": SourceUqBpinnContext` to `CASE_CONTEXT`.
- Add `poll-source-uq-bpinn` to the migrated list regenerated into `index.json`.
- `npm run build` → screenshot-verify (workbench + 4 sub-tabs, KaTeX renders, mean & std heatmaps correct, 0 console
  errors) via the visual-verify harness.

## Risks / fallback

- **Risk — ensemble collapse:** if members converge to the same field, std → 0 everywhere and calibration drops below
  ~95% (over-confident). Mitigation already in `build()`: distinct per-member seed (`+1000*(m+1)`) + bootstrap sensor
  bags + per-member noise realisation. If still too tight, *do not* add L-BFGS; instead reduce `N_OBS` slightly or
  widen the per-member seed offset to increase basin diversity.
- **Risk — std overconfidence (epistemic ≠ Bayesian posterior):** a deep ensemble *approximates* UQ; it is not the
  exact posterior. The Context labels this honestly ("ensemble proxy, not HMC/VI") and reports calibration as the
  honest score, not a guarantee.
- **Risk — ONNX 2-output reshape:** `infer.py` maps `pred[:, k]` to `outputs[k]`; if the exported graph column order
  (`cat([mean,std])`) ever flips, "c" would show std. Verified order is `[mean, std]` in `_Ensemble._Mod.forward`.
- **Fallback:** none needed — this is the conservative option. If a richer UQ story is wanted later, the honest upgrade
  is a *separate* case with K trained ensembles at discrete noise levels (multi-ONNX), not a fake sweep here.
```
