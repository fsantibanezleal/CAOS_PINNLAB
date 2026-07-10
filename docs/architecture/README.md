# Architecture

How PINN-Lab is put together: an **offline PINN pipeline** (the product) feeding a **static deterministic-replay web
app** (the showcase). Read in order, or jump to a theme.

| # | Doc | What it answers |
|---|-----|-----------------|
| 01 | [overview](01_overview.md) | The whole shape on one page — folders, the two contracts, the staged pipeline, the gate, why PINNs export to the browser. |
| 02 | [the gate](the-gate.md) | How a case is decided LIVE (ships `.onnx`, runs in-browser) vs PRECOMPUTE (replay-only), from measurements, not by hand. |
| 03 | [data contracts](data-contracts.md) | The two schemas that bracket the pipeline: ingestion (observations in) and artifact (manifest + trace out). |
| 04 | [staged pipeline](staged-pipeline.md) | The six deterministic stages (`preprocess → … → export`) and what each one owns. |
| 05 | [train → ONNX → web](train-export-onnx.md) | The bridge: training recipe, the modern dynamo ONNX export, and the parity check that guarantees the browser sees the same network. |
| 06 | [deploy](deploy.md) | Build + GitHub Pages (Actions), the `copy-data` step, base-path and SPA-routing constraints. |
| 07 | [determinism](determinism.md) | Why a run is a pure function of `(case, seed)`, and what would break replay. |
| 08 | [the method ladder](method-ladder-comparison.md) | How the pipeline COMPUTES the comparison (standard PDE vs naive/adapted/data-driven PINN + diagnostics) and the app shows it — the CompareKit/DiagnosticsKit, the ladder tools, and the real per-case results. |

This is the **product archetype** layer (ADR-0057). The PINN science — adaptive sampling, hard constraints, Fourier
features, domain decomposition, inverse/UQ — lives in [`../methods/`](../methods/); the per-engine usage guides
(DeepXDE, PhysicsNeMo, neuraloperator, …) in [`../frameworks/`](../frameworks/); and the per-problem write-ups in
[`../cases/`](../cases/).
