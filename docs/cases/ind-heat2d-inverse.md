# ind-heat2d-inverse — 2D inverse heat conduction, recovering the conductivity field from sparse sensors

The case that exercises the **field-inverse** method: not a forward solve, and not a scalar-parameter inverse, but the
recovery of an entire spatially-varying coefficient $k(x,y)$ from ~100 sparse noisy temperature readings. This is the
canonical sparse-data field-inverse problem where PINNs beat classical FEM/FVM — the PDE prior fills the gaps where
there are no sensors.

## Problem

Steady-state heat conduction in a medium with unknown, spatially-varying conductivity $k(x,y)$ on the unit square,
with the source $q$ given and the temperature pinned to zero on the boundary:

$$ \nabla\!\cdot\big(k(x,y)\,\nabla T\big) = q \quad\text{on}\ (0,1)^2, \qquad T|_{\partial\Omega}=0. $$

We are given the source $q$ and ~100 sparse, noisy interior samples $T(x_i,y_i)$, and asked to infer the **whole**
conductivity field $k(x,y)$ — an ill-posed, under-determined problem away from the sensors. The manufactured ground
truth (MMS) is

$$ T^* = \sin\pi x\,\sin\pi y, \qquad k^* = 1 + \tfrac12\sin\pi x\,\sin\pi y, \qquad q = \nabla\!\cdot(k^*\nabla T^*), $$

with $q$ derived in closed form by SymPy. Domain $(0,1)^2$, evaluated on an $81\times81$ grid; $100$ sensors at
interior locations with $\sigma=0.01$ Gaussian noise on $T$.

## Method

This is an **inverse field PINN**, and the design choices are what make the ill-posed recovery tractable:

- **Two-output PFNN.** A parallel fully-connected network `PFNN([2,[40,40],[40,40],[40,40],2])` emits both fields,
  `[k, T]`. The unknown conductivity is the **first network output** — a *field*, not a `dde.Variable` scalar — so the
  optimizer recovers a value of $k$ at every point, with the PDE residual interpolating between sensors.
- **Product-rule residual.** The divergence is expanded explicitly,
  $k(T_{xx}+T_{yy}) + k_x T_x + k_y T_y - q$, using DeepXDE jacobians/hessians, so the spatial gradients of the
  *recovered* $k$ enter the residual directly rather than through an opaque autodiff of a product.
- **Hard constraints via output transform.** $k = \mathrm{softplus}(\cdot)+10^{-3}$ enforces positivity exactly
  (a negative conductivity is unphysical), and $T = x(1-x)\,y(1-y)\,(\cdot)$ enforces the zero-boundary condition
  exactly, removing the boundary loss term entirely.
- **PointSetBC observations + loss weighting.** The 100 noisy sensors enter as a `PointSetBC` on the $T$ component, and
  the data term is up-weighted `loss_weights=[1, 100]` so the sparse measurements actually pull the field. The sensor
  locations are also added as `anchors` to the collocation set. Training is Adam ($2\times10^4$ steps, lr $10^{-3}$)
  then L-BFGS.

The primary score is the relative-L2 of the recovered $k$ vs $k^*$; the $T$ error is reported as an extra metric.

## Result (measured, seed 42)

| metric | value |
|--------|-------|
| recovered-$k$ relative-L2 vs $k^*$ | **4.0 %** (0.040295) |
| max abs error in $k$ | 0.439 |
| forward $T$ relative-L2 vs $T^*$ | **0.77 %** (0.007664) |
| ONNX parity (max abs) | **4.8e-7** |
| lane | **live** (54 KB, 2.15 ms) |

Validation anchor is **analytic** (the manufactured $k^*$). The 4.0 % field recovery is genuinely good for a sparse
field-inverse: the temperature field is reconstructed to under 1 %, and the conductivity to a few percent. The error
is honestly concentrated where it must be — the max-abs error of 0.44 sits where $|\nabla T|$ is small (near the
boundary and at the field's peak), because there $k$ barely influences the residual and is therefore weakly
identifiable from temperature data alone. That is the physics of the problem, not a tuning failure.

## Honesty

`real_or_synthetic = synthetic`. The truth here is a **manufactured solution (MMS)**: the triple $(T^*, k^*, q)$ is
closed-form, with $q$ derived symbolically so that $k^*$ is the exact conductivity producing $T^*$. No open 2D
thermal-field inverse dataset exists (see `real-datasets.md §6`), so there is nothing real to fit; the value of the
case is the *method* — demonstrating that a PINN recovers a full coefficient field from ~100 noisy point samples — and
the score is an exact comparison against a known answer, not a curve fit to data. The sensor noise ($\sigma=0.01$) is
synthetic but realistic, so the recovery is not trivially exact.

## Reproduce

```bash
python -m pinnlab.pipeline ind-heat2d-inverse --seed 42
```


## Comparison (the app's Compare view: the data-driven contrast)

standard k* (MMS) | **pure physics, NO data** | **physics + ~100 sensors** (see [the method ladder](../architecture/method-ladder-comparison.md)). With no
sensor data the conductivity field k is underdetermined and the recovery fails (**356 %** vs the analytic standard);
with the sparse noisy sensors it is recovered (**4.0 %**). The DATA is what makes the inverse solvable.


## Identifiability: the computed information budget (Diagnostics view)

How much data does the inverse need? The sweep (one fixed fast training budget, seeded sensors, first-n subsets)
answers with real runs: recovered-k relative-L2 vs number of sensors = **356 % at n=0** (pure physics: the field
unknown is underdetermined), **17.3 % at n=10**, 16.3 % at n=25, 13.6 % at n=50, **12.6 % at n=100**. The cliff is
between 0 and 10: ANY anchoring data restores identifiability, then returns diminish (the smooth k* is already
pinned by few points; the fully-trained n=100 run reaches 4.0 %, see Compare). Baked by
`build_identifiability_sweep.py`; shown as the log-y curve in the app's Diagnostics view.
