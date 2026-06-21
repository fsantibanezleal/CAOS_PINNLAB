# Changelog

All notable changes to **PINN-Lab**. Format: `X.XX.XXX` (display) — see `pinnlab.__version__`. Keep `0.x` while on
synthetic/benchmark data. Tag every release.

## [0.04.000] — 2026-06-21

The first REAL-data case. Everything prior validates against closed-form / reduced-model truth; this milestone adds a
case trained on, and validated against, real measured observations — plus the engine plumbing for it.

### Added — `env-soil-heat-real` (case #17, the flagship real-data inverse)
- Recovers **soil thermal diffusivity** from **NOAA USCRN** daily soil temperatures (station IL_Champaign_9_SW,
  2019–2021, depths 5/10/20/50/100 cm). The 1D heat equation $T_t=\alpha T_{zz}$ with the **5 cm + 100 cm sensors as
  real time-varying Dirichlet boundaries** and the diffusivity a **trainable scalar** (`dde.Variable`,
  `external_trainable_variables`; Adam-only so the generic L-BFGS recompile never drops it).
- **Out-of-sample validation**: the **10/20/50 cm sensors are held out** and scored in `evaluate` against the real
  interior temperatures. Measured (seed 42): recovered **α = 0.30 mm²/s** (textbook moist-mineral-soil range),
  held-out **RMSE = 1.05 °C** (10 cm 1.26 · 20 cm 1.06 · 50 cm 0.75), **relative-L2 = 6.9 %**, lane **live** (40 KB,
  0.7 ms, parity 1.4e-6). Honesty flag **`validated-real`** (new green "real data" tag, EN/ES).

### Added — engine + ingestion
- `datasets/uscrn_soil.py` — documented fetcher that vendors the real USCRN data offline
  (`data/reference/uscrn/soil_temp_il_champaign.json`, schema `pinnlab.dataset.uscrn/v1`), so training is reproducible
  and CI needs no network.
- **Data-fit validation mode** in `evaluate`: a case with `validation_anchor="real-data-holdout"` and no analytic
  anchor scores via `extra_metrics` (held-out interpolation of the baked field vs real observations) and reports the
  recovered physical parameter.
- `docs/cases/` (landing + `env-soil-heat-real.md`); management dossier `real-datasets.md` updated to record the
  shipped real-data case (and why USGS nested-piezometer heads were rejected as non-diffusive, USCRN soil heat chosen).

### Notes
- The USGS groundwater path was explored and dropped: the nested-piezometer head profile is decoupled by aquifer
  layering (not a homogeneous-diffusion fit). OpenAQ air-source inversion remains a documented future real-data case.

## [0.03.000] — 2026-06-21

The full case catalogue (16 cases), the interactive web app, the architecture wiki, and the live GitHub Pages
deployment. Every metric below is the committed manifest's measured value (relative-L2 vs the validation anchor);
lanes are derived from measurements (ONNX size · ort-web inference time · trace bytes), never hand-set.

### Added — engine generalizations (what made the catalogue possible)
- **Validation anchors**: analytic, vendored numerical reference (`reference_on_grid`), and benchmark (Ghia
  centerline) — all leakage-safe in `evaluate`. Custom `eval_grid()`, `extra_metrics()`, and multi-output reshape
  hooks so vector fields (u,v,p) and non-default grids flow through unchanged.
- **SOTA training surface**: hard constraints (output transforms), Fourier-feature input transforms, SIREN,
  RAR residual-adaptive refinement hook (`refine`), per-term `loss_weights`, **MMS** (method of manufactured
  solutions) sources for cases with no closed-form solution, **parametric PINNs** (a parameter as a network input),
  **FBPINN** 2-channel domain decomposition, and **inverse** cases (a 2nd network output for the unknown field +
  `PointSetBC` observations through the ingestion contract).

### Added — cases (8 canonical + 8 mining/pollution/industrial/control differentiators)
- **Canonical benchmark (6)**: `bench-poisson2d` (hard-constraint, **L2 5e-6**), `bench-heat1d` (**9.1e-5**),
  `bench-wave1d` (**4.9e-5**), `bench-burgers1d` (RAR + Burgers.npz, **9.7e-3**), `bench-allencahn`
  (RAR + Allen_Cahn.npz, **1.2e-3**), `bench-navier-cavity` (3-output, Ghia anchor, **L2 1.7e-1**, honest
  CPU-limited).
- **Industrial fluids/heat (2)**: `ind-helmholtz` (Fourier features, **1.0e-1**, CPU-limited), `ind-heat2d-inverse`
  (PFNN inverse k+T from observations, **4.0e-2**).
- **Mining / mineral-processing (4)**: `mine-heap-leach-rt` (2-species reactive transport, MMS, **7.9e-5**),
  `mine-thickener-settling` (Bürger–Concha degenerate flux, **5.1e-3**), `mine-flotation-kinetics` (parametric
  first-order kinetics C(k,t), **7.6e-4**), `mine-comminution-pbe` (size-transport reduced population balance,
  **1.8e-4**).
- **Pollution / environmental (3)**: `poll-ocean-transport` (advection–diffusion, **3.1e-4**), `poll-soil-barrier`
  (FBPINN 2-channel, D-contrast, **1.9e-1**, honest CPU-limited), `poll-tailings-seepage` (Richards/Gardner
  unsaturated seepage, **6.7e-4**).
- **Control (1)**: `ctrl-zero-source` (degenerate/zero-source sanity anchor, **2.1e-4**).
- All 16 cases classify **lane=live** (ONNX 20–40 KB, ort-web infer < 1 ms, parity ~1e-7); honesty flags
  (`synthetic` vs `synthetic-illustrative`) set per case; CPU-limited cases labeled, not hidden.

### Added — web app (Vite + React 19 + TS)
- Six pages (App · Introduction · Methodology · Implementation · Experiments · Benchmark), HashRouter, zustand,
  i18next (EN/ES), KaTeX equations, canvas viridis heatmap, **onnxruntime-web live inference** with baked-trace
  replay. `lib/contract.ts` mirrors the artifact schema so contract drift fails the build.

### Added — docs wiki + deploy
- `docs/architecture/` (overview · the-gate · data-contracts · staged-pipeline · train→ONNX→web · deploy ·
  determinism), `docs/frameworks/` (DeepXDE, PhysicsNeMo, neuraloperator, jax-pi, NeuralPDE.jl, PINA),
  `docs/methods/` (adaptive sampling · causal-curriculum · loss-weighting · architectures · domain-decomposition ·
  variational-scalable · optimization · operator-learning · inverse-UQ).
- **Live on GitHub Pages** (Actions): https://fsantibanezleal.github.io/CAOS_PINNLAB/ — `base: "./"`, `copy-data`
  prebuild, `github-pages` environment branch-policy cleared. CI smoke (`--quick`) + lane re-derivation guard.

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
