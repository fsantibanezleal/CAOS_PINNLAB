# The measured lane gate

Every case is classified into one of two **lanes**, and the decision is made **from measurements taken during the
run**, never by hand. This is the single mechanism that keeps the web app honest: a case cannot *claim* to run live
unless the numbers say it actually can.

## The two lanes

| Lane | Ships | In the browser |
|------|-------|----------------|
| **live** | the `<case>.onnx` network **and** the baked `field.json` trace | onnxruntime-web evaluates the field at arbitrary resolution + cursor probes in real time |
| **precompute** | only the baked `field.json` trace | the SPA replays the precomputed field; no in-browser inference |

## The rule

`core/gate.py::classify_lane` applies a hard conjunction:

```
live  iff  onnx_bytes ≤ 4 MB  and  infer_ms ≤ 120 ms  and  trace_bytes ≤ 1 MB
```

- **`onnx_bytes`** — the size of the self-contained ONNX file. A PINN is a few dense layers, so this is typically
  20–40 KB; the 4 MB ceiling is the download budget for a snappy first paint.
- **`infer_ms`** — wall-clock for one full-grid forward pass, measured in the `infer` stage as an ort-web proxy. The
  120 ms ceiling is the interactivity budget (≈ 8 fps for a full re-evaluation; cursor probes are far cheaper).
- **`trace_bytes`** — the size of the baked `field.json`. The 1 MB ceiling bounds the replay payload; `MAX_AXIS = 81`
  in `core/trace.py` keeps the baked grid within it.

If any term fails, the case is precompute. The verdict **and the three measured numbers** are written into the
manifest (`lane`, `onnx.bytes`, `infer_ms`, `trace.bytes`), so the classification is auditable and reproducible.

## Why these three terms

A browser PINN evaluation is governed by exactly these costs: the model has to **download** (`onnx_bytes`),
**execute** (`infer_ms`), and the fallback replay has to **download** (`trace_bytes`). This generalizes the older
ADR-0054 lane gate — which only asked "does the compute kernel fit in a Pyodide wheel?" — to the ONNX runtime that
actually governs an in-browser PINN. The gate is about *delivered interactivity*, not about whether training
succeeded.

## What the gate does not decide

- **Correctness.** Accuracy is the `evaluate` stage's job (relative-L2 vs the validation anchor). A case can be live
  and inaccurate, or precompute and exact — the lane is purely about in-browser cost. The two are reported
  side-by-side so neither masquerades as the other.
- **Honesty of the data.** Whether a field is `synthetic-illustrative` or fit to a real dataset is the manifest's
  `real_or_synthetic` flag, set by the case — orthogonal to the lane.

## CI enforcement

CI re-derives the lane from the committed manifest's measured numbers and fails on any mismatch, so a hand-edited
"live" label that the measurements don't support cannot land. See [deploy](deploy.md).
