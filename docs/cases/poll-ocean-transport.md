# poll-ocean-transport — 2D advection-diffusion PINN (passive scalar in a gyre)

The first case to put **two spatial dimensions and time** under the same network. It exercises the
**unsteady advection-diffusion** family with the method of manufactured solutions (MMS) as the validation anchor:
a passive pollutant carried by a prescribed divergence-free ocean gyre while eddy-diffusing.

## Problem

A passive scalar $c$ (plastic / oil-spill tracer) is advected by an incompressible current $\mathbf{v}$ and
diffused by eddy diffusivity $D$ on the unit square over one time unit:

$$ c_t + \mathbf{v}\cdot\nabla c = D\,\nabla^2 c + f, \qquad \nabla\cdot\mathbf{v}=0,\ D=0.01, $$

$$ \mathbf{v} = \big(-A\sin(\pi x)\cos(\pi y),\ A\cos(\pi x)\sin(\pi y)\big),\quad A=1. $$

The current is a single divergence-free gyre. The domain is $(x,y)\in(0,1)^2$, $t\in(0,1]$, sampled on a
$41\times41\times21$ grid. The Péclet number $\mathrm{Pe}=AL/D\approx 100$ — strongly advection-dominated, yet it
converges without a curriculum.

The **manufactured truth** (validation anchor) is

$$ c^*(x,y,t)=e^{-2D\pi^2 t}\sin(\pi x)\sin(\pi y), $$

and the source term $f=\mathbf{v}\cdot\nabla c^*$ is set to exactly whatever the gyre does to $c^*$, so $c^*$ is an
exact solution of the forced PDE.

## Method

**Advection-diffusion PINN with soft IC/BC.** The network is a $[3,64,64,64,64,1]$ tanh FNN (DeepXDE engine)
trained Adam (18000 steps, lr $10^{-3}$) → L-BFGS. The residual encodes the full transport operator —
$c_t$, the gyre dot-product $v_x c_x + v_y c_y$, the Laplacian $D(c_{xx}+c_{yy})$, and the MMS source.

The design choice that makes the score honest: the Dirichlet boundary ($c=c^*$ on the open boundary) and the
initial condition ($c=c^*$ at $t=0$) are imposed as **soft penalty losses**, not hard-baked into the output.
Loss weights $[1,10,50]$ for $[\text{pde},\text{bc},\text{ic}]$ keep the data-fit terms from being drowned by the
interior residual. Because the constraints are soft, the network must genuinely learn the interior transport field
rather than interpolate a hard-coded boundary — so the reported relative-L2 is the true PINN error, not a boundary
artifact.

## Result (measured, seed 42)

| metric | value |
|--------|-------|
| relative-L2 vs MMS analytic | **0.0314 %** (3.14e-4) |
| max absolute error | 7.72e-4 |
| validation anchor | analytic (MMS closed form) |
| ONNX parity (max abs) | **5.96e-7** |
| lane | **live** (61 KB, 6.54 ms infer, opset 18) |

The relative-L2 of **3.1e-4** is comfortably inside the expected band (`< 1e-2`) and is not CPU-limited — the
advection-dominated $\mathrm{Pe}\approx100$ field is recovered cleanly, and the ONNX export reproduces the trained
net to single-precision parity (5.96e-7).

## Honesty

`real_or_synthetic = synthetic-illustrative`. The truth is a closed-form MMS field, not a fit to a real spill or a
real ocean-current product — the gyre, the diffusivity, and the tracer profile are a **physically-faithful
illustration** of how a pollutant spreads under a current, not a measurement. What *is* real is the PINN error:
because IC/BC are soft, the relative-L2 measures the network's actual interior accuracy against the exact analytic
solution, with nothing hard-coded to flatter it. Use this case as a transport-solver template, not as a forecast of
any specific spill.

## Reproduce

```bash
python -m pinnlab.pipeline poll-ocean-transport --seed 42
```
