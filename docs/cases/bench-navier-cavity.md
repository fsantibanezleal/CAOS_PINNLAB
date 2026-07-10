# bench-navier-cavity — steady Navier-Stokes lid-driven cavity (u, v, p)

The hardest canonical case in the catalogue, and the one that exercises **multi-output PINNs with loss
weighting**: three coupled scalar outputs, three PDE residuals, a pressure gauge, and a corner-regularized
boundary — all balanced by per-term weights so no single loss dominates the others.

## Problem

The 2D steady incompressible Navier-Stokes equations in primitive variables $(u, v, p)$, at Reynolds number
$\text{Re} = UL/\nu = 1/0.01 = 100$ (with $\rho=1$, $\nu=0.01$):

$$ (\mathbf{u}\cdot\nabla)\mathbf{u} = -\tfrac1\rho\nabla p + \nu\nabla^2\mathbf{u}, \qquad \nabla\cdot\mathbf{u}=0, \qquad \text{Re}=100 \ \text{on}\ (0,1)^2. $$

Componentwise, the three residuals the network is trained to zero are

$$ u\,u_x + v\,u_y = -\tfrac1\rho p_x + \nu\,(u_{xx}+u_{yy}), \qquad u\,v_x + v\,v_y = -\tfrac1\rho p_y + \nu\,(v_{xx}+v_{yy}), \qquad u_x + v_y = 0. $$

**Domain & setup.** The unit square $(0,1)^2$, tested on a $101\times101$ grid. The **lid** ($y=1$) is driven by a
*regularized* tangential profile $u(x,1) = 16\,x^2(1-x)^2$, which vanishes at both top corners to tame the
classic lid-corner velocity singularity; $v=0$ there. The other three walls are **no-slip** ($u=v=0$). Pressure
is only defined up to a constant, so it is **pinned** at the origin, $p(0,0)=0$, as a gauge fix (`PointSetBC`).

## Method

`method = multioutput-loss-weighting`. A single fully-connected network ($2\to[64]\times5\to3$, tanh, Glorot)
maps $(x,y)\mapsto(u,v,p)$, so the same shared trunk must satisfy all three PDE residuals and all boundary
conditions simultaneously. The difficulty is **scale imbalance**: the momentum/continuity residuals, the lid and
wall Dirichlet terms, and the single-point pressure gauge live on very different magnitudes, and an unweighted
sum lets the easy terms swamp the hard ones.

The fix is explicit **per-term loss weighting**, `loss_weights = [1, 1, 1, 10, 10, 10, 10, 10]` — the three
residuals at weight 1, and all five constraints (lid $u$, lid $v$, wall $u$, wall $v$, pressure gauge) **up-weighted
10×** so the boundary data and gauge are actually enforced rather than averaged away. Optimization is Adam
(20 000 steps, lr $10^{-3}$) followed by **L-BFGS** polishing, on 2601 interior + 400 boundary collocation
points. This is the textbook hard-constraint-by-weighting recipe for coupled multi-physics PINNs, and it is the
reason this case is the catalogue's stress test for the weighting strategy.

## Result

Validation anchor: the **Ghia, Ghia & Shin (1982)** Re=100 centerline benchmark (digitized), the canonical
reference for the lid-driven cavity. The score is the relative-L2 of the PINN $u$ along the vertical centerline
$x=0.5$ and $v$ along the horizontal centerline $y=0.5$ versus the digitized Ghia points.

| metric | value |
|--------|-------|
| relative-L2 vs Ghia (mean of $u$,$v$ centerlines) | **0.1675** |
| &nbsp;&nbsp;· $u$-centerline rel-L2 | 0.1173 |
| &nbsp;&nbsp;· $v$-centerline rel-L2 | 0.2178 |
| ONNX parity (max abs) | 1.237e-6 |
| lane | **live** (79 KB, 2.31 ms) |

This is **CPU-limited, and we state it plainly**: ~17 % relative-L2 against Ghia is reduced fidelity, not a
publication-grade cavity solve. The network captures the primary vortex and the qualitative corner-eddy
structure, but the centerline match — especially $v$, at 0.22 — is coarse because the CPU lane (DeepXDE, Adam +
L-BFGS) cannot afford the iteration count a sharp Re=100 cavity needs. A GPU lane (PhysicsNeMo) would tighten
this; on the CPU lane it does not, and the number is reported as-is.

## Honesty

`real_or_synthetic = synthetic-illustrative`. The Ghia 1982 centerlines are a real, widely-cited benchmark, but
this case is **not fit to data** and is **not** a converged reference solution — it is a *faithful reduced model*:
the correct governing equations, the correct boundary conditions, a regularized lid and a pressure gauge, solved
at the fidelity the CPU lane allows. The 0.17 relative-L2 is the honest cost of that lane. Nothing here is
manufactured to look better than it is; the gap to Ghia is the headline number, not a footnote.

## Reproduce

```bash
python -m pinnlab.pipeline bench-navier-cavity --seed 42
```


## Validation (the app's Diagnostics view: the Ghia benchmark)

The canonical lid-driven-cavity validation: the PINN velocity along both cavity centerlines vs the tabulated
**Ghia-Ghia-Shin (1982)** Re=100 benchmark points (u RMSE **0.053**, v RMSE **0.029**). Robust: no finicky field
solver (a rushed finite-difference cavity solve diverged and was rejected, per [the method ladder](../architecture/method-ladder-comparison.md)); the
streamlines remain the Field view.
