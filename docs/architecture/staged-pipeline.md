# The staged pipeline

`pipeline.py` runs each case through **six deterministic stages**. The whole chain is a pure function of
`(case, seed)` — same inputs, same artifacts, byte-for-byte (see [determinism](determinism.md)).

```
preprocess → feature_extraction → train → infer → evaluate → export
```

Each stage lives in `stages/<name>.py` and consumes the previous stage's dict, so the data flow is explicit and a
single stage can be reasoned about (and unit-tested) in isolation.

## What each stage owns

| Stage | File | Responsibility |
|-------|------|----------------|
| **preprocess** | `stages/preprocess.py` | Resolve the case spec; for inverse cases, load + validate observations through the [ingestion contract](data-contracts.md). Produces the run config. |
| **feature_extraction** | `stages/feature_extraction.py` | Build the evaluation grid (`linspace_grid`, or the case's own `eval_grid()` when it overrides sampling). Defines where the field is measured. |
| **train** | `stages/train.py` | The heavy SOTA stage — build the PINN, run Adam → L-BFGS (+ optional RAR refinement hook), export ONNX, verify parity. See [train → ONNX → web](train-export-onnx.md). |
| **infer** | `stages/infer.py` | Evaluate the trained net on the grid; reshape multi-output channels; time one full forward pass (`infer_ms`, the gate's interactivity proxy). |
| **evaluate** | `stages/evaluate.py` | The test stage — relative-L2 + max-abs error of the field vs the validation anchor (analytic / reference dataset / Ghia), plus any case `extra_metrics`. Leakage-safe: the anchor is never PINN output. |
| **export** | `stages/export.py` | Bake the `field.json` trace, assemble the manifest (with the [gate](the-gate.md) verdict + measured numbers), write artifacts under `data/derived/` + `models/`. |

## The `train` ↔ `evaluate` split is deliberate

`train` knows the PINN and the optimizer; it must **not** know the test metric, or accuracy numbers would be
optimistic by construction. `evaluate` only sees the trained field and an independent anchor. Keeping them in
separate stages enforces that separation structurally, not by convention.

## Case hooks the stages respect

A case module can opt into richer behavior without the stages special-casing it:

- `analytic(x)` / `reference_on_grid()` — the validation anchor `evaluate` compares against;
- `eval_grid()` — a custom measurement grid (overrides the default `linspace_grid`);
- `refine(model)` — a RAR / curriculum refinement run after the base Adam phase (sharp-front cases: Burgers,
  Allen–Cahn);
- `extra_metrics(pred, grid)` — domain-specific diagnostics (e.g. Ghia centerline error for the lid-driven cavity);
- multi-output reshape — `infer`/`evaluate` handle vector fields (e.g. `u,v,p` for Navier–Stokes) transparently.

## The `--quick` path

`pipeline.py --quick` runs a reduced recipe (≈ 300 iters, no L-BFGS) used by CI smoke and local plumbing checks. It
exercises the *entire* chain — including ONNX export + parity + gate — so a structural break is caught in seconds,
while the full recipe is reserved for the committed artifacts.
