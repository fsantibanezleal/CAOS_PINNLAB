# train → ONNX → web

The bridge that lets a heavy offline PINN become a real-time in-browser field. Three steps inside the `train` stage:
**train the net**, **export to ONNX**, **prove parity**.

## 1 · Train the net (the SOTA recipe)

`stages/train.py` builds the case's PINN via its `build(seed)` and runs a two-phase recipe:

1. **Adam** — first-order, the bulk of the iterations, drives the loss into the right basin. Per-term `loss_weights`
   (PDE vs BC vs IC) come from the case spec where the residual scales differ.
2. **L-BFGS** — a quasi-Newton polish that tightens a well-conditioned PINN by another order of magnitude. Skipped on
   the `--quick` path.

A case may also expose `refine(model)` — a **RAR** (residual-adaptive resampling) or curriculum hook run between the
phases, used by sharp-front cases (Burgers shock, Allen–Cahn interface) to concentrate collocation points where the
residual is largest. Hard constraints (output transforms) and Fourier features (input transforms) are part of
`build`, so they are inside the network the optimizer sees — and, crucially, inside the graph that gets exported.

## 2 · Export to ONNX (the modern dynamo exporter)

```python
torch.onnx.export(
    net, (dummy,), str(onnx_path),
    input_names=["coords"], output_names=["u"],
    dynamic_axes={"coords": {0: "n"}, "u": {0: "n"}},
    opset_version=18, dynamo=True, verbose=False, external_data=False,
)
```

- **`dynamo=True`** — the TorchDynamo-based exporter (requires `onnxscript`). It traces the *actual* graph, including
  the input/output transforms, rather than the legacy tracer's best-effort. This was a deliberate choice: the
  dependency (`onnxscript`) is declared in `requirements`, not worked around.
- **`dynamic_axes` `{0: "n"}`** — batch dimension is dynamic, so the same ONNX evaluates 1 cursor probe or a full
  81×81 grid.
- **`external_data=False`** — force a **self-contained** `.onnx` (no sidecar `.onnx.data`), so the web ships a single
  file and the [trace/gate](the-gate.md) byte accounting is exact.
- **`verbose=False`** + `PYTHONUTF8=1` — the exporter prints unicode (✅) status; on a cp1252 Windows console that
  raised `UnicodeEncodeError`. Silencing verbose + forcing UTF-8 makes the export portable.

## 3 · Prove parity (the guarantee)

Right after export, the stage runs both the PyTorch `model.predict` and the ONNX session on the same probe batch and
checks:

```
parity_max_abs = max |u_torch − u_onnx|  <  1e-4
```

This is the contract's teeth: it **proves** the browser will evaluate the *same* network the pipeline validated —
hard constraints, Fourier features and all. The measured `parity_max_abs` goes into the manifest; in practice it sits
at ~1e-7. If parity ever failed, the case would not be allowed to claim the LIVE lane.

## Why this works at all

Training cost lives entirely offline (PyTorch/CUDA). The *artifact* is tiny — a few dense + activation layers, tens
of KB. onnxruntime-web runs that in real time client-side, so the SPA keeps every training technique (they are baked
into the weights) while shipping none of the training stack. That asymmetry — heavy to make, cheap to evaluate — is
exactly what makes an interactive PINN showcase possible as a static site.
