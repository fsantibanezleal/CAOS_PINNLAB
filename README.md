# PINN-Lab — a runnable catalogue of Physics-Informed Neural Networks

[![CI](https://img.shields.io/github/actions/workflow/status/fsantibanezleal/CAOS_PINNLAB/ci.yml?branch=main&label=CI)](https://github.com/fsantibanezleal/CAOS_PINNLAB/actions)
[![License](https://img.shields.io/github/license/fsantibanezleal/CAOS_PINNLAB)](LICENSE)
[![Version](https://img.shields.io/github/v/tag/fsantibanezleal/CAOS_PINNLAB?label=version&sort=semver)](https://github.com/fsantibanezleal/CAOS_PINNLAB/tags)
[![Live demo](https://img.shields.io/badge/demo-live-2ea44f)](https://pinnlab.fasl-work.com)

PINN-Lab is a **real, reproducible product**, not a demo: a catalogue of 20 differential-equation cases (PDEs + a chaotic ODE system),
each trained offline by a state-of-the-art Physics-Informed Neural Network engine, **validated against an analytic
or numerical reference**, exported to **ONNX**, and **replayed and re-inferred in the browser**. Every case ships its
governing equations, the SOTA method that solves it, an interactive visualization that reacts to the cursor and the
controls, and an honest benchmark vs. the reference — and the whole thing is deterministic given `(case, seed)`.

It is modelled on the validated exemplar **CAOS_SIMLAB**: a four-tab workbench per case, deep bilingual write-ups,
and a `docs/` tree that is written *as the code is versioned*.

> **Start here:** [docs/README.md](docs/README.md) — the documentation entry point. ·
> **Just run it:** [docs/guides/01_getting-started.md](docs/guides/01_getting-started.md).

## What it is (and is not)

- **It IS** a teaching-and-decision instrument and a **method catalogue**: each SOTA family — hard constraints, RAR
  adaptive sampling, Fourier features / SIREN, domain decomposition, operator learning (FNO), inverse + UQ — is
  genuinely *exercised* in at least one case, not merely named.
- **It is NOT** a replacement for FEM/FVM (a good classical solver beats a PINN on a single well-posed forward
  problem), and it is **not** an industrial digital twin: most mining/pollution cases are validated on analytic (MMS)
  anchors or faithful reduced models, each carrying an honest `synthetic` / `synthetic-illustrative` /
  `validated-real` label. Exactly one case (`env-soil-heat-real`) is trained against a real measured dataset.

## The two worlds, joined by a data contract

PINN-Lab is split into a **heavy offline world** (Python) and a **light web world** (this SPA), joined by an
**artifact contract** so the web never recomputes the physics:

1. **Offline pipeline (`data-pipeline/pinnlab/`).** A deterministic chain — `preprocess → feature/sampling → train →
   infer → evaluate → export` — trains each PINN (Adam → L-BFGS, + RAR where defined), exports the trained net to
   ONNX (parity-checked `< 1e-4`), and bakes a compact per-variant field trace + a `manifests/<case>.json` (schema
   v2: params, seed, metrics, lane, bytes). Inverse cases also enforce an **ingestion contract** on their input data.
2. **Artifact contract (`processing → web`).** The web app loads **only** the committed artifacts (index → manifest →
   trace → ONNX); a TypeScript type mirrors the manifest schema so a contract drift fails the build.

## The method ladder & the dynamics (what makes the catalogue worth looking at)

The pipeline does not bake one field per case: it **computes the comparison the docs describe**, and the app shows it
([docs/architecture/method-ladder-comparison.md](docs/architecture/method-ladder-comparison.md)):

- **Compare view** (default): the classical **standard** solution vs the **naive** PINN vs the **adapted** fix on one
  grid + error maps + a shared labeled colorbar. Real headline numbers: Helmholtz naive **120.8%** vs Fourier **9.3%**;
  Allen-Cahn soft **95.4%** (metastable collapse) vs hard-constraint+RAR **0.4%**; heat2d-inverse with no data
  **356%** vs physics+data **4.0%**.
- **Dynamics everywhere the content is motion**: an animated evolution hero on every time case, the ladder's lanes
  animating together, 2-D frame sequences (ocean, heap-leach), and **Training — "watch it learn"**: the field at real
  training checkpoints, naive vs adapted side by side (spectral bias made visible as a training pathology).
- **The story**: an 8-chapter when-PINNs-win/lose selector, each chapter deep-linking (`#/?case=…&view=…`) to the
  case + view that DEMONSTRATES it. Chapter 1 is honest: on the easy forward problem the classical solver wins.
- Ladder tools live in `data-pipeline/` (`build_standard_comparisons.py`, `build_ladder.py`, `build_naive_lane.py`,
  `build_training_dynamics.py`, `build_evolution_frames.py`, `build_identifiability_sweep.py`, …); artifacts are
  contract-tested (`tests/test_dynamics_artifacts.py`).

## Quickstart

```powershell
# 1. build the isolated environments (offline lane: numpy, DeepXDE/PyTorch, onnxruntime) — no global installs
./scripts/setup.ps1                          # or ./scripts/setup.sh

# 2. bake a case (train → validate → ONNX → manifest)
./scripts/precompute.ps1 bench-poisson2d --seed 42      # omit the id to bake ALL cases + rebuild the index

# 3. run the web app (predev overlays the baked artifacts, then Vite serves them)
cd frontend && npm install && npm run dev
```

The browser opens to the **App** tab: pick a case, walk its variant chips, and open the **Live** sub-tab to move the
physical parameter and watch the exported ONNX re-evaluate in your browser. See
[docs/guides/03_using-the-web-app.md](docs/guides/03_using-the-web-app.md).

## Repository layout

| Path | What |
|------|------|
| `data-pipeline/pinnlab/` | The offline pipeline: `cases/` (one module per case), `stages/`, `core/` (manifest, gate, trace), `model/`, `pipeline.py`. |
| `data/derived/` | The committed artifacts: per-case field traces + `manifests/` (per-case + `index.json`). |
| `models/` | The exported `<case>.onnx` for the live lane. |
| `frontend/` | The Vite + React + TypeScript SPA (the workbench, the content pages). |
| `docs/` | The documentation tree — **the core of the repo**: `architecture/`, `cases/`, `methods/`, `frameworks/`, `guides/`. |
| `scripts/` | `setup`, `precompute`, `dev`, `smoke` — PowerShell + bash parity. |
| `requirements*.txt` | Pinned per-lane requirements (`data-pipeline/requirements.txt` = the offline engine; root `requirements.txt` = the minimal live lane; `requirements-gpu.txt` = the optional GPU fidelity lane). |

## Hard rules this repo bakes in

- **The deep research is binding.** Every engine the research selected lives in `docs/frameworks/<tool>/` *and* the
  pinned requirements, and the pipeline actually uses it — no hand-rolled substitute for a prescribed SOTA engine.
- **Honesty by design.** Each case declares its validation anchor and data label; the Benchmark page publishes the
  measured relative-L2 and ONNX parity, undressed.
- **Reproducible.** Pinned requirements per lane; `scripts/setup`; deterministic given `(case, seed)`; the ONNX the
  browser runs is exactly the validated network.
- **Versioned** (X.XX.XXX, CHANGELOG + tags from day 1) with license/attribution hygiene.
- **Tests never write the canonical artifacts.** The whole pytest suite is sandboxed (an autouse `conftest.py`
  fixture redirects every pipeline write target to a tmp dir) — a quick-mode smoke once clobbered committed bakes;
  never remove that fixture.

See [docs/architecture/01_overview.md](docs/architecture/01_overview.md) for the full rationale and
[STRUCTURE.md](STRUCTURE.md) for the file-by-file map.
