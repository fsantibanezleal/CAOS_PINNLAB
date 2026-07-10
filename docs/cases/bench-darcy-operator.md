# bench-darcy-operator — Darcy-flow operator learning (Fourier Neural Operator)

The case that exercises the **operator-learning** method family. Every other case trains one network for one
boundary-value problem; this one learns the **solution operator** $\mathcal{G}: a(\mathbf{x}) \mapsto u(\mathbf{x})$
over a whole *family* of coefficient fields, so a single trained network maps any new permeability field to its
pressure field in one forward pass — no per-instance retraining.

## Problem

Steady **Darcy flow** (second-order elliptic) on the unit square, with constant forcing $f \equiv 1$:

$$ -\nabla\!\cdot\big(a(\mathbf{x})\,\nabla u(\mathbf{x})\big) = 1, \quad \mathbf{x}\in(0,1)^2, \qquad u|_{\partial\Omega}=0. $$

The **input function** $a(\mathbf{x})>0$ is a two-value (piecewise-constant) thresholded Gaussian random field
(values $\{3, 12\}$ — sharp material interfaces, the canonical Li-et-al. benchmark); the **output function** $u$ is
the pressure/head field. The learning target is the operator itself:

$$ \mathcal{G}_\theta:\; a \longmapsto u \approx \big(-\nabla\!\cdot(a\nabla\cdot)\big)^{-1} 1. $$

Reference solutions come from a finite-difference solve (5-point scheme, harmonic-mean face conductivities, sparse
direct solve) on a $32\times32$ grid. The dataset (256 train + 64 test pairs) is generated in-build, seeded.

## Method

`method = operator-fno`, `engine = fno-torch`. A compact, self-contained **2D Fourier Neural Operator** (Li et al.,
2021): lift (1×1 conv) → 4 Fourier layers (a truncated **spectral convolution** — FFT, keep the lowest $10\times10$
modes, multiply by learnable complex weights, inverse FFT — plus a 1×1 skip, GELU) → projection (1×1 convs). The
spectral convolution is implemented in **real arithmetic** (split real/imag, real weights for each part) so the whole
operator exports cleanly to ONNX. Data-driven training: relative-L2 loss on $(a,u)$ pairs, Adam, 120 epochs.
(The optional PINO physics-residual is documented as the upgrade; the shipped path is the data-driven FNO.)

`neuraloperator` is the documented reference library; this minimal controlled implementation is the genuine method
(Fourier layers = spectral conv + skip), kept dependency-free and ONNX-exportable.

## Result (measured, seed 42)

| metric | value |
|--------|-------|
| **held-out test relative-L2** (the operator-generalization metric, 64 unseen $a$) | **5.5 %** |
| per-sample relative-L2 (FNO vs FD reference) | mostly **2–10 %** |
| ONNX parity (max abs) | **1.8e-6** |
| lane | **precompute** (field-IO operator — not browser-coordinate-drivable) |

One trained FNO reproduces unseen pressure fields to ~5.5 % across 64 held-out coefficient fields — genuine operator
generalization, not a per-instance fit. The workbench ships **6 discrete variants** — six held-out coefficient fields
the FNO never saw at training — and the chip selector switches between them; each shows three switchable fields (the
input coefficient $a$, the FNO prediction $u_{\text{pred}}$, the FD reference $u_{\text{true}}$). The headline metric
stays the held-out **test-set** mean (the true operator-generalization number); each chip also reports its own sample
L2.

## Why precompute (honest lane)

This is a **field-in / field-out** operator: it maps a coefficient *field* to a solution *field*, not coordinates to
a value like the PINN cases. The SPA drives the live lane with coordinate queries, which an FNO cannot answer, so the
case ships **lane=precompute** — the browser replays the baked representative result. The real FNO is trained and
validated **offline**; its ONNX is shipped and parity-checked (1.7e-6), ready for a future field-input web lane.

## Honesty

`real_or_synthetic = synthetic`. The analytic-coefficient Darcy dataset is the field-standard FNO benchmark; the
reference $u$ is a finite-difference numerical solve (a numerical anchor, not measured field data). The operator and
its generalization metric are genuine; nothing about the method is faked.

## Reproduce

```bash
python -m pinnlab.pipeline bench-darcy-operator --seed 42
```


## Comparison (the app's Compare view)

The FNO operator's one-pass prediction on a held-out permeability field vs the classical **finite-difference**
reference (~**2.5 %** relative-L2) - the generalization of a learned operator to a NEW input in a single forward pass,
not a per-instance retrain. See [the method ladder](../architecture/method-ladder-comparison.md).
