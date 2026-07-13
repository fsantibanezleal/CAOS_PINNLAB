# PINN-Lab documentation

> The navigable entry point to the PINN-Lab docs. Start here.

PINN-Lab is a runnable catalogue of 20 differential-equation cases (PDEs + a chaotic ODE system), each trained offline by a
state-of-the-art Physics-Informed Neural Network engine, validated against an analytic or numerical reference, exported to
ONNX, and replayed/inferred in the browser. This `docs/` tree is the **core of the repository**: the physics, the
methods, the architecture and the how-to guides all live here, written *as the code is versioned* — not bolted on at
the end.

If you only read one other page, read **[guides/01_getting-started.md](guides/01_getting-started.md)**.

---

## How the docs are organised

The tree is **one folder per theme**, each with its own landing `README.md` and deep pages inside:

| Folder | What it answers | Start at |
|--------|-----------------|----------|
| **[architecture/](architecture/README.md)** | How is PINN-Lab built? The two worlds (offline/web), the staged pipeline, the data contracts, the train→ONNX bridge, the lane gate, determinism, deploy, and the **method ladder & dynamics** (how the standard-vs-naive-vs-adapted comparisons, training-dynamics and evolution animations are computed and shown). | [architecture/README.md](architecture/README.md) |
| **[cases/](cases/README.md)** | What problems does it solve? One page per case: the PDE, the method, the validation anchor, the honesty label, the measured numbers. | [cases/README.md](cases/README.md) |
| **[methods/](methods/adaptive-sampling.md)** | Which SOTA methods, and why? One page per method family — the idea, the formulation, the cases that exercise it, the primary reference. | [methods/](methods/) |
| **[frameworks/](frameworks/deepxde/README.md)** | Which engines, and how to use them? One guide per framework (DeepXDE, PhysicsNeMo, neuraloperator, jaxpi, PINA, NeuralPDE.jl) with install, API surface and a runnable example. | [frameworks/deepxde/README.md](frameworks/deepxde/README.md) |
| **[guides/](guides/01_getting-started.md)** | How do I *do* things? Clone-and-run, add a case, drive the pipeline, read the web app, interpret the results. | [guides/01_getting-started.md](guides/01_getting-started.md) |

---

## The mental model in one minute

PINN-Lab is split into **two worlds** joined by a **data contract**:

- **Offline (Python, heavy).** A deterministic pipeline — `preprocess → feature/sampling → train → infer → evaluate
  → export` — trains each PINN with a SOTA engine, validates it (relative-L2 vs the reference), exports the trained
  net to ONNX (parity-checked < 1e-4), and bakes a compact per-variant field trace + a manifest.
- **Web (this SPA, light).** It **never recomputes the physics**: it loads the index → manifest → trace and draws an
  interactive heatmap; in the **Live** lane it runs the exported ONNX in your browser (onnxruntime-web) to sweep the
  physical parameter.

Each case is presented as an identical **workbench**: a variant bar (parameter regimes) + four sub-tabs —
**Field / Live / Charts / Context**. See [architecture/staged-pipeline.md](architecture/staged-pipeline.md) and
[architecture/the-gate.md](architecture/the-gate.md).

---

## Honesty contract

Every case declares its validation anchor (`analytic` · `dataset` · `fem`) and a data-honesty label
(`synthetic` · `synthetic-illustrative` · `validated-real`). The **Benchmark** page publishes the measured
relative-L2 and ONNX parity per case, undressed. Exactly one case (`env-soil-heat-real`) is trained and validated
against a **real measured dataset**; the rest are closed-form/MMS or faithful reduced models. See
[cases/README.md](cases/README.md#honesty-what-real-means-here).

---

## Conventions

- **Equations** are written in LaTeX and render in both the docs (GitHub/MathJax) and the web app (KaTeX).
- **Numbers** quoted in case pages are the committed manifest's measured values, not aspirations.
- **References** are peer-reviewed primaries with DOIs (see [methods/](methods/) and each framework page).
- **Reproducibility:** every artifact is deterministic given `(case, seed)`; the ONNX the browser runs is exactly the
  validated network.
