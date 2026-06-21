# Changelog

All notable changes to **PINN-Lab**. Format: `X.XX.XXX` (display) — see `pinnlab.__version__`. Keep `0.x` while on
synthetic/benchmark data. Tag every release.

## [0.02.000] — 2026-06-20

### Changed
- **Specialized the engine to Physics-Informed Neural Networks**, replacing the template's example SIR body. Primary
  engine: **DeepXDE 1.15.0** (PyTorch backend); trained PINNs export to **ONNX (opset 18)** for onnxruntime-web.
- The two data contracts re-cast for PINNs: ingestion = observation tables (inverse cases), artifact = solution
  fields. The lane gate now decides live-vs-precompute from **ONNX size + ort-web inference time + artifact bytes**.

### Added
- `cases/bench-poisson2d` — 2D Poisson (Dirichlet) via a **hard-constraint PINN**, analytic validation anchor.
  Verified end-to-end: **lane=live, relative-L2 = 5e-6 vs analytic, ONNX parity = 2.4e-7, onnx = 48 KB self-contained.**
- Core: `io/schema` (ObservationRow, SolutionField), `io/contract` (observation ingestion + outlier policy),
  `core/trace` (`pinnlab.field/v1`), `core/manifest` (`pinnlab.manifest/v1`), `core/gate` (ONNX/ort-web lane),
  `model/analytic` (numpy reference helpers — Pyodide-safe).
- Stages: preprocess / feature_extraction / **train (DeepXDE → ONNX + parity check)** / infer / evaluate
  (relative-L2 vs analytic) / export. CLI `python -m pinnlab.pipeline`.
- Precompute-lane requirements pinned + mapped to docs (numpy, deepxde, torch, onnx, onnxruntime, onnxscript);
  documented GPU lane (`requirements-gpu.txt`).
- Tests rewritten to PINN (contract, smoke, manifest, gate) — **10 passing**.

### Removed
- Stray `models/surrogate.json` (an SIR-example artifact the template copy carried in).

## [0.01.000] — 2026-06-20

### Added
- Initial instantiation from the CAOS product-repo template (ADR-0057): the frozen base — `data-pipeline` package,
  the two data contracts, the named staged pipeline, the seeded RNG, the manifest/trace, the measured lane gate, the
  cases-by-category registry, tests, and CI — before the PINN engine replaced the example body in 0.02.000.
