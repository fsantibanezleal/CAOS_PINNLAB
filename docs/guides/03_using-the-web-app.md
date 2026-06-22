# 03 · Using the web app

The web app is a faithful **viewer** of the baked artifacts. It never recomputes the physics; it replays committed
field traces and, in the Live lane, runs the exported ONNX in your browser.

## The pages

- **App** — the catalogue. One tab per case; each case is a workbench (below).
- **Introduction** — what PINN-Lab is, the PINN loss, the honest scope.
- **Methodology** — the SOTA method families, each with its formulation, the cases that exercise it, and a DOI.
- **Implementation** — the architecture and the three flows (offline precompute / web / design).
- **Experiments** — a per-category summary of the baked cases.
- **Benchmark** — the honesty table: measured relative-L2, ONNX parity, and the data label per case.

## The case workbench

Every case is presented identically:

### The variant bar
A row of **chips**, one per variant (parameter regime), plus a **lane badge** (`live` or `precomputed`) and a
bilingual one-line note describing what the active variant shows. Click a chip to load that regime.

### The four sub-tabs

| Sub-tab | What it does |
|---------|--------------|
| **Field** | The baked field as an interactive viridis heatmap. Hover for the exact value at the cursor (`x, y, u`), a colorbar with a cursor tick, and two line-cut profiles through the cursor. The governing equation renders above it. |
| **Live** | Re-evaluates the exported **ONNX in your browser** (onnxruntime-web). Move the parameter slider(s) and the field recomputes live — this is the real network, not a pre-baked image. A resolution slider trades detail for speed. Non-parametric cases show a banner (no knob to sweep). |
| **Charts** | A per-variant relative-L2 comparison bar. Click a bar to jump to that variant. |
| **Context** | The case's deep write-up: the problem, the components & variables, the formalization (with equations), the scope & assumptions, what each variant shows, and **how to read & use the viz**. |

## Reading the heatmap

- **Axes** are the field axes (e.g. space `x` × time `t`, or `x × y` for a 2-D field). The colorbar maps value → colour.
- **Hover** anywhere to read the exact value; the crosshair and the colorbar tick follow the cursor.
- **Line-cuts** show the field along each axis through the cursor — e.g. a sine in space and an exponential decay in
  time for the heat case.

## The Live lane in practice

1. Open the **Live** sub-tab. The first time, the ONNX loads (a moment).
2. Drag the physical-parameter slider — the wavenumber, diffusivity, speed, viscosity, or a **time scrubber** for
   the transport cases.
3. The heatmap re-evaluates instantly from the network. This is the same function that was validated offline (parity
   `< 1e-4`), so what you see is the model, faithfully.

## Language & theme

The header toggles **EN/ES** (English is the default) and **light/dark**. All case Context and page prose are fully
bilingual.

---

**Next:** [04 · Adding a new case](04_adding-a-new-case.md) · [05 · Interpreting results](05_interpreting-results.md)
