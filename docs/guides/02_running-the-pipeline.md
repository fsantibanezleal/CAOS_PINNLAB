# 02 · Running the pipeline

The offline pipeline is the heavy, Python side of PINN-Lab. It is **deterministic given `(case, seed)`** and writes
the committed artifacts the web app replays.

## The command

Everything goes through one module, wrapped by the `precompute` scripts:

```powershell
./scripts/precompute.ps1 <case-id> --seed <n>      # one case
./scripts/precompute.ps1                            # ALL cases + rebuild index.json
```
```bash
./scripts/precompute.sh <case-id> --seed <n>
python -m pinnlab.pipeline <case-id> --seed <n>     # the raw form (inside .venv-pipeline)
```

Case ids are the catalogue names: `bench-poisson2d`, `bench-heat1d`, `bench-wave1d`, `bench-burgers1d`,
`bench-allencahn`, `poll-ocean-transport`, … (see [cases/](../cases/README.md)).

## The stages

Each case runs the full chain (one trained net, then one field trace **per variant**):

```
preprocess → feature/sampling → train → infer → evaluate → export
```

- **train** fits the PINN (Adam → L-BFGS, plus RAR adaptive sampling if the case defines `refine()`), exports the
  trained net to ONNX (opset 18) and checks parity against `model.predict` (`< 1e-4`).
- **infer/evaluate** bake one decimated field trace per variant (parameter regime) and measure the relative-L2 vs the
  case's validation anchor.
- **export** writes `data/derived/<case>/<variant>.json` traces + `data/derived/manifests/<case>.json` (schema v2).

## Variants & parametric cases

A **parametric** case makes its physical knob (e.g. wavenumber `k`, diffusivity `α`, wave speed `c`, viscosity `ν`,
or time `t`) a *network input*: one trained net + one ONNX covers the whole family, and the web Live tab sweeps it.
Each variant bakes a field at one fixed value of that knob. A **non-parametric** case (e.g. a stiff dataset-anchored
benchmark) ships a single `default` variant.

## Seeds & determinism

Same seed → same artifacts, bit-for-bit on the same platform. Change `--seed` to get an independent training run (for
a robustness check). The default is `42`. The seed is recorded in the manifest.

## The `--quick` smoke path

```powershell
./scripts/precompute.ps1 <case-id> --quick
```

`--quick` runs a few hundred iterations with **no L-BFGS and no RAR** — it exercises the entire
train → ONNX → parity → infer → manifest plumbing in ~30 s so you can catch a structural bug *before* committing to a
full bake. **Never** commit `--quick` artifacts: the relative-L2 will be far from converged. Use it only to validate
wiring.

## Rebuilding the index

`index.json` is the catalogue inventory the App tab reads. Running the pipeline with **no case id** rebuilds it from
all cases. To add a single freshly-baked case to the index without re-baking everything, regenerate the index from the
manifests on disk (see [04 · Adding a new case](04_adding-a-new-case.md)).

## GPU lane (optional)

The CPU lane bakes every Group-A case. For the geometry-heavy / 3-D cases at higher fidelity, layer the GPU
requirements on top of the pipeline venv:

```powershell
.venv-pipeline\Scripts\python.exe -m pip install -r requirements-gpu.txt --index-url https://download.pytorch.org/whl/cu128
```

---

**Next:** [03 · Using the web app](03_using-the-web-app.md) · [05 · Interpreting results](05_interpreting-results.md)
