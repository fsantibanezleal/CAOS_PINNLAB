# bench-poisson2d — 2D Poisson (Dirichlet), parametric source mode (hard-constraint PINN)

The catalogue's cleanest canonical benchmark, as a **parametric family**: the source mode / wavenumber $k$ is a network
input, so one trained net + one ONNX covers the whole mode family and the web **Live** tab sweeps the spatial structure
from a single bump to a finer ripple. The Dirichlet BC is baked in exactly (no boundary loss term). This is the case
the train → ONNX → web contract was hardened on.

## Problem

2D Poisson with homogeneous Dirichlet boundaries on the unit square, with a manufactured exact solution (the anchor)
that vanishes on the boundary for **any** $k$:

$$ -\nabla^2 u = f(x,y;k)\ \text{on}\ (0,1)^2,\quad u|_{\partial\Omega}=0,\qquad
   u^*(x,y;k)=x(1-x)\sin(k\pi x)\,\cdot\,y(1-y)\sin(k\pi y), $$

with the forcing $f=-\nabla^2 u^*$ derived in closed form. The **source mode** $k\in[1,3]$ is the swept knob: small $k$
is a smooth low-frequency field, larger $k$ adds finer oscillatory structure. Field evaluated on a $\sim121\times121$
grid over $x,y\in[0,1]$.

## Method — hard constraints (distance-function output transform)

The Dirichlet condition is enforced **structurally** by multiplying the raw network by a factor that vanishes on every
wall:

$$ \hat u(x,y) = x(1-x)\,y(1-y)\,\mathcal{N}(x,y,k). $$

So $\hat u|_{\partial\Omega}=0$ for any weights, the loss is **only** the PDE residual $-u_{xx}-u_{yy}-f$ (no
boundary-vs-interior weight to tune). Net $[3,\dots,1]$ tanh (DeepXDE), Adam (12000) → L-BFGS, `num_boundary=0`. $k$ is
a network input, so the trained net spans the family and the Live tab sweeps it.

## Result (measured, seed 42)

Validation anchor: the **closed-form MMS** $u^*(x,y;k)$ (any $k$). Six variants across $k\in[1,3]$:

| metric | value |
|--------|-------|
| relative-L2 vs analytic | **≤ 0.16 %** across all 6 variants |
| ONNX parity (max abs) | 1.2e-7 |
| lane | **live** (one shared ONNX; Live sweeps $k$) |

Tight across the whole mode range — the hard-constraint formulation on this smooth family is about as favourable as a
PINN gets.

## Honesty

`real_or_synthetic = synthetic` — the truth is the closed-form manufactured solution (exact for every $k$), not measured
data. A numerical-correctness benchmark for the hard-constraint method + the export pipeline. For a real-data
counterpart see [env-soil-heat-real](env-soil-heat-real.md).

## Reproduce

```bash
python -m pinnlab.pipeline bench-poisson2d --seed 42
```
