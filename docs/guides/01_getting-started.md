# 01 · Getting started

From a fresh clone to a running web app in four steps. Nothing installs globally — both Python and Node dependencies
live inside the repo.

## Prerequisites

- **Python 3.12** (the pinned, verified interpreter) on `PATH`.
- **Node.js ≥ 20** (for the Vite web app).
- A C/C++ toolchain only if your platform builds any wheel from source (rare; the pins are wheels).
- Optional: a local **NVIDIA GPU** for the higher-fidelity lane (see `requirements-gpu.txt`); not required to build.

## 1 — Build the environments

One idempotent script creates the isolated Python virtual environment(s) and installs the per-lane requirements plus
the editable package. No global installs.

```powershell
# PowerShell (Windows)
./scripts/setup.ps1
```
```bash
# bash (Linux/macOS)
./scripts/setup.sh
```

This creates `.venv-pipeline/` (the offline/precompute lane: numpy, DeepXDE on the PyTorch backend, onnxruntime, the
dev tools) and installs the `pinnlab` package editable. It is safe to re-run.

## 2 — Bake a case (offline pipeline)

Train one case, validate it, export ONNX, and write its artifacts. The script passes its arguments straight through
to the pipeline:

```powershell
./scripts/precompute.ps1 bench-poisson2d --seed 42
```
```bash
./scripts/precompute.sh bench-poisson2d --seed 42
```

You will see the Adam→L-BFGS training, the ONNX parity check, and a final line like:

```
precomputed bench-poisson2d: lane=live variants=6 l2=[...] parity=1.2e-07 onnx=...B infer=...ms
```

To bake the **whole catalogue**, omit the case id (this is long — hours on CPU):

```powershell
./scripts/precompute.ps1            # all cases + rebuild the index
```

See [02 · Running the pipeline](02_running-the-pipeline.md) for seeds, determinism, lanes and the `--quick` smoke path.

## 3 — Run the web app

```bash
cd frontend
npm install
npm run dev        # predev copies data/derived + models into public/data, then starts Vite
```

Open the printed local URL. The **predev** hook (`copy-data.mjs`) overlays the committed pipeline artifacts into the
SPA so Vite serves them; the app loads **only** those artifacts and never recomputes the physics. For a production
build:

```bash
npm run build      # prebuild copies data, then tsc --noEmit + vite build → dist/
```

## 4 — Verify

- The **App** tab lists every baked case as a workbench. Pick one, walk its variant chips, and open the **Live**
  sub-tab to move the physical parameter and watch the exported ONNX re-evaluate in your browser.
- The **Benchmark** page shows the measured relative-L2 and ONNX parity per case — the honest numbers.

If a case is missing from the App tab, it is not in the index yet — bake it (step 2) and rebuild the index
(`./scripts/precompute.ps1` with no args, or see guide 02).

---

**Next:** [02 · Running the pipeline](02_running-the-pipeline.md) · [03 · Using the web app](03_using-the-web-app.md)
