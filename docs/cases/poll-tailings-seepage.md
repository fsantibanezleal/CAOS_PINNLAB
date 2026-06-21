# poll-tailings-seepage — unsaturated seepage through a tailings dam (Richards equation)

The case that exercises a **strongly nonlinear, degenerate-parabolic PDE** — Richards' equation for unsaturated
flow. Unlike the linear-diffusion cases, here the storage and conductivity coefficients depend on the solution
itself, so the PINN must satisfy a nonlinear residual rather than a constant-coefficient one.

## Problem

1D vertical unsaturated seepage through a tailings deposit ($z$ up, $\psi$ = pressure head, $\psi<0$ in the
unsaturated zone, $t$ = time). The mixed-form Richards equation with a Gardner exponential closure:

$$ C(\psi)\,\psi_t = \partial_z\!\big[K(\psi)(\psi_z+1)\big], \qquad
   C\,\psi_t - K\,\psi_{zz} - K'(\psi)(\psi_z^2+\psi_z) = f. $$

The **Gardner** constitutive model keeps the manufactured source analytic:

$$ K(\psi)=K_s e^{\alpha\psi},\quad K'(\psi)=\alpha K,\quad
   \theta(\psi)=\theta_r+(\theta_s-\theta_r)e^{\alpha\psi},\quad
   C(\psi)=\frac{d\theta}{d\psi}=(\theta_s-\theta_r)\,\alpha\,e^{\alpha\psi}. $$

Parameters: $\theta_s=0.43$, $\theta_r=0.078$, $\alpha=2.0$, $K_s=0.25$. The MMS anchor is a smooth,
strictly-unsaturated head profile drying in time on $z\in[0,1]$, $t\in[0,1]$:

$$ \psi^*(z,t)=-A+(A-\psi_{top})\,e^{-t/\tau}(1-z), \qquad A=1.0,\ \psi_{top}=-0.1,\ \tau=1.0, $$

with the source $f=\mathcal{L}[\psi^*]$ derived in closed form. Grid: $101\times51$ ($z\times t$).

## Method

**Richards seepage with the Method of Manufactured Solutions (MMS).** Because no analytic solution of the
nonlinear Richards equation exists for an arbitrary profile, the case *manufactures* a physically faithful drying
head field $\psi^*$ and computes the exact source $f$ that makes it a solution of the full nonlinear operator. The
PINN (a `tanh` FNN, $[2,40,40,40,40,1]$, Glorot-normal init) then minimizes the **nonlinear** residual
$C(\psi)\psi_t - K\psi_{zz} - K'(\psi)(\psi_z^2+\psi_z) - f$, with the Gardner exponential evaluated on the network
output so $C$, $K$, $K'$ all couple back to $\psi$. The manufactured profile is enforced as soft Dirichlet BC and
IC (`solution=analytic`). Training is **Adam (18000 steps, lr 1e-3) → L-BFGS** with loss weights $[1,10,10]$
(PDE, BC, IC) to keep the boundary/initial constraints tight against the dominant interior residual. This is the
standard rigorous-verification path: MMS gives a known truth for a PDE that has none, so relative-L2 is an honest
convergence metric.

## Result (measured, seed 42)

| metric | value |
|--------|-------|
| relative-L2 vs MMS analytic $\psi^*$ | **6.7e-4** (0.067 %) |
| max absolute error | 1.18e-3 |
| validation anchor | **analytic** (MMS closed-form) |
| ONNX parity (max abs) | 6.56e-7 |
| lane | **live** (31 KB ONNX, 0.81 ms infer) |

The validation anchor is the manufactured analytic field $\psi^*$. The relative-L2 of **0.067 %** is well inside
the `< 1e-2` expected band — the nonlinear residual is resolved cleanly on CPU, no honesty caveat is needed here.

## Honesty

`real_or_synthetic = synthetic-illustrative` — a faithful reduced model, **not** fit to data. The physics
(Richards + Gardner) and the operator are real; the truth field is manufactured via MMS, so the L2 measures how
well the PINN solves a known-answer PDE, not how well it matches the world. This is the honest stance because **no
open unsaturated-zone $\psi(z,t)$ tailings dataset exists** (`real-datasets.md`). Only the saturated-zone Darcy
head inverse could be driven by real USGS data — documented as the Tier-C extension. Likewise the fuller
**van Genuchten–Mualem** closure is documented as an extension; Gardner is used here precisely because it keeps the
MMS source analytic. Plainly: the unsaturated lane is modeled, not measured.

## Reproduce

```bash
python -m pinnlab.pipeline poll-tailings-seepage --seed 42
```
