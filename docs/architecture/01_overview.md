# Architecture overview

PINN-Lab is an instance of the CAOS product-repo archetype (ADR-0057): an **offline pipeline + static
deterministic-replay web app**. The pipeline is the product; the web app replays a validated subset and runs the
exported network live.

```
data-pipeline/pinnlab/   the offline engine (Python, .venv-pipeline) — trains PINNs, validates, exports ONNX, bakes artifacts
data/reference/          vendored numerical reference datasets (Burgers.npz, Allen_Cahn.npz) — validation anchors
data/derived/            committed artifacts: manifests/ (index + per-case) + <case>/field.json replay traces
models/                  exported <case>.onnx (the live lane)
frontend/                the SPA (Vite + React + TS) — onnxruntime-web live inference + replay of the baked traces
docs/                    this wiki (architecture / frameworks / methods / cases / guides)
```

## The two data contracts

1. **Ingestion (raw → pipeline)** — `io/contract.py`. For *inverse* cases the bring-your-own-data gate validates an
   observation table (`case_id, x0..x{d-1}, value[, weight]`) against schema + an explicit outlier policy: bad rows
   are rejected with a reason, extreme values flagged. Forward cases need no external data.
2. **Artifact (pipeline → web)** — `core/manifest.py` + `core/trace.py`. Every run writes a compact field trace
   (`pinnlab.field/v1`) and a manifest (`pinnlab.manifest/v1`) recording the governing equation, the SOTA method, the
   engine, the seed, the ONNX pointer + parity, the lane/gate verdict, and the evaluation metrics. The web loads ONLY
   these; `frontend/src/lib/contract.ts` mirrors the schema so any drift fails the build.

## The named staged pipeline

`pipeline.py` runs six deterministic stages per case (pure function of `(case, seed)`):

```
preprocess → feature_extraction → train → infer → evaluate → export
```

- **train** is the heavy SOTA stage: it builds the case's PINN (DeepXDE / PhysicsNeMo / a self-contained FNO), runs the
  training recipe (Adam → L-BFGS, plus a case-defined RAR refinement hook for sharp-front cases), exports the trained
  net to **ONNX (opset 18)**, and verifies **ONNX-vs-`model.predict` parity** (< 1e-4) — the train → web bridge.
- **evaluate** is the TEST stage: relative-L2 of the PINN field vs the validation anchor (analytic / reference
  dataset / Ghia benchmark), leakage-safe (the reference is never PINN output).

## The measured lane gate (extended for ONNX/ort-web)

`core/gate.py::classify_lane` decides, **from measurements**, whether a case runs live in the browser or only
replays its baked artifact:

```
LIVE  iff  onnx_bytes <= 4 MB  AND  infer_ms <= 120 ms (ort-web proxy)  AND  trace_bytes <= 1 MB
```

A LIVE case ships its `.onnx` and the SPA evaluates the field at arbitrary resolution / cursor probes in real time
via onnxruntime-web; a PRECOMPUTE case ships only the replay trace. The verdict + the measured numbers go into the
manifest, and CI fails on mislabeling. This generalizes the ADR-0054 gate from a Pyodide-wheel check to the ONNX
runtime that actually governs an in-browser PINN evaluation.

## Why PINNs export cleanly to the browser

Training a PINN is heavy (PyTorch/CUDA, offline). But the *result* is a small dense network: a trained PINN is a few
dense + activation layers (tens of KB of weights). Exported to ONNX it runs in real time client-side via
onnxruntime-web — so the interactive web app keeps every SOTA training technique (they are "baked into" the trained
weights) without shipping any of the training stack. Output transforms (hard constraints) and input transforms
(Fourier features) are pure tensor ops, so they are captured inside the exported graph; the parity check proves it.

See also: [the gate](the-gate.md) · [data contracts](data-contracts.md) · [train → ONNX → web](train-export-onnx.md)
· the per-engine guides in [`../frameworks/`](../frameworks/) and the SOTA methods in [`../methods/`](../methods/).
