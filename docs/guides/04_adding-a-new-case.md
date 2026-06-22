# 04 · Adding a new case

A case is one documented PDE problem: a `CaseSpec`, a validation anchor, a trained PINN, and (in the web) a deep
bilingual Context. This guide is the end-to-end design flow.

## 1 — Define the `CaseSpec`

Create `data-pipeline/pinnlab/cases/<group>_<name>.py` exposing a module-level `CASE = CaseSpec(...)`. The key fields:

- `id`, `category`, `title`, `governing_equations` (LaTeX), `method`, `engine`.
- `inputs` / `outputs` — the network's input and output axis names.
- `domain` — the box per input; `grid` — the field-grid resolution per **field axis**.
- `field_axes` — the two axes the baked heatmap spans (omit ⇒ all inputs).
- `param_specs` — one `ParamSpec` per swept parameter (an input that is **not** a field axis).
- `validation_anchor` — `analytic` | `dataset` | `fem`.
- `train` — the optimizer schedule (`layers`, `activation`, `lr`, `adam`, `lbfgs`, sampling, optional RAR keys).
- `real_or_synthetic` — the honesty label.

The invariant the loader enforces: `domain keys == inputs`, `grid keys == field_axes`, and
`(inputs − field_axes) == param_specs keys`.

## 2 — Pick the method the physics demands

Choose the SOTA method that the solution structure needs — not a default. Hard-constraint output transforms for exact
BC/IC (and a clean ONNX), SIREN/Fourier features for oscillatory or high-frequency solutions, RAR adaptive sampling
for sharp fronts, domain decomposition for coefficient jumps, an operator net for a parametric family, a deep
ensemble for UQ. See [methods/](../methods/).

## 3 — Decide the parametrization (variants)

This is what turns a single solve into a workbench:

- **If a closed-form (or MMS) family exists** in a physical knob — wavenumber, diffusivity, speed, viscosity, time —
  make that knob a **network input** (in `inputs`, in `param_specs`, *not* in `field_axes`). One trained net + one
  ONNX covers the family; the Live tab sweeps it. Bake **≥ 6 variants** at representative values via `variants()`.
- **If no closed parametric form exists** (a stiff dataset-anchored benchmark whose dynamics can't be parametrized),
  ship a single `default` variant — honestly, not faked into artificial regimes.

Expose `analytic(X)` (the anchor) and `variants() -> list[Variant]`. Each `Variant` fixes the parameter(s) and carries
a bilingual label + note.

## 4 — Implement `build(seed)` (and optional `refine`)

`build(seed)` returns `{"model": <dde.Model>, "input_dim": d}`. For a hard-constraint case, apply the output transform
that bakes IC/BC exactly (pure tensor ops, so it survives the ONNX export). If the case uses RAR, also expose
`refine(model, case, seed)` — the pipeline calls it after the base fit.

## 5 — Smoke-test, then bake for real

```powershell
./scripts/precompute.ps1 <case-id> --quick     # ~30s: validates the plumbing, NOT the accuracy
./scripts/precompute.ps1 <case-id> --seed 42   # the real bake
```

Require a relative-L2 inside the case's `expected_band` and ONNX parity `< 1e-4` across all variants. If a parametric
net misses the band at the extremes, narrow the parameter range or add capacity — and re-bake.

## 6 — Write the Context and register it (web)

Add `frontend/src/content/cases/<Name>Context.tsx` — the deep bilingual write-up following the established structure:
**the problem → components & variables → formalization (KaTeX) → scope & assumptions → what each variant shows →
how to read & use the viz**. Register it in `frontend/src/content/cases/registry.tsx`.

## 7 — Add to the index, build, verify

Add the case to `data/derived/manifests/index.json` (or rebuild the index by running the pipeline with no case id).
Then:

```bash
cd frontend && npm run build      # copies artifacts + typechecks + builds
```

Screenshot-verify the case workbench (variant chips, the four sub-tabs, the cursor read-out, the Live sweep) **before**
committing. Then commit the case as one logical unit (case `.py` + Context + registry + manifest + traces + ONNX +
index).

## 8 — Document it

Add `docs/cases/<case-id>.md` (the deep case page) so the documentation tracks the catalogue. The docs are part of the
deliverable, not an afterthought.

---

**Next:** [05 · Interpreting results](05_interpreting-results.md)
