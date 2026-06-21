# mine-flotation-kinetics — parametric first-order flotation PINN $C(k,t)$

The case that answers *"can one network cover a whole family of operating points?"* — it exercises the
**parametric-PINN** method family. Instead of training one net per flotation rate constant, a single net takes the
rate constant $k$ as a second input and returns the entire 2D field $C(k,t)$ in one pass.

## Problem

Batch froth flotation of a single floatable species follows lumped first-order kinetics: the floatable mineral
concentration $C$ decays at a rate proportional to itself,

$$ \partial_t C = -k\,C, \qquad C(k,0)=1 \ \Rightarrow\ C^* = e^{-k t}, \qquad R = 1 - C, $$

where $k$ is the flotation **rate constant** (1/min) and $R$ is the recovery. Rather than fix $k$, we promote it to a
coordinate: the domain is $k\in[0.5,5]$ (the "spatial" axis is the rate constant) and $t\in[0,1]$, on an
$81\times81$ grid. One trained net gives the concentration — and the derived recovery $R=1-C$ — for **any** rate
constant in the range without retraining, exactly the lumped first-order model used to compare flotation circuits.

## Method — parametric PINN with a hard IC

The net is an FNN `[2] → [32]×3 → [1]` with `tanh` activations and Glorot-normal init. The PINN-specific choices:

- **Parametric input.** Input 0 is the parameter $k$, input 1 is the time $t$. The residual
  $\partial_t C + k\,C$ reads $k$ directly off the input column, so the optimizer learns the solution operator over the
  whole $k$-family at once — not a single trajectory.
- **Hard initial condition.** $C(k,0)=1$ is imposed *exactly* by an output transform $y \mapsto 1 + t\cdot N(k,t)$:
  the $t$ factor vanishes the network contribution at $t=0$, so the IC is never a soft penalty and the IC weight cannot
  fight the residual. The loss list is therefore PDE-only (`num_boundary=0`).
- **Adam → L-BFGS.** 10 000 Adam steps at `lr=1e-3` over 2000 domain + 200 initial collocation points, then an L-BFGS
  polish; scored against the analytic `solution=exp(-k t)` on 4000 test points via `l2 relative error`.

Recovery $R=1-C$ is a pure post-processing identity, derived in the App rather than learned.

## Result (measured, seed 42)

| metric | value |
|--------|-------|
| relative-L2 vs analytic $C^*=e^{-kt}$ | **7.6e-4** (0.076 %) |
| max absolute error | 2.22e-3 |
| validation anchor | **analytic** ($C^*=e^{-kt}$, exact) |
| ONNX parity (max abs) | 7.15e-7 |
| lane | **live** (19.4 KB, 0.77 ms) |

The relative-L2 of **0.076 %** clears the case's own band (`< 5e-3`) by an order of magnitude — this is a smooth,
low-dimensional field with an exact analytic anchor, so the CPU lane resolves it essentially to optimizer precision;
no accuracy caveat is needed here. The single 19.4 KB ONNX reproduces the trained net to 7e-7 and infers in under a
millisecond, so the App can sweep $k$ interactively and read $C(k,t)$ and $R(k,t)$ live.

## Honesty

`real_or_synthetic = synthetic-illustrative`. The lumped first-order kinetics is the *standard* flotation model and
the field is a faithful reduced model of it — but it is a clean analytic illustration, **not fitted to a plant or lab
assay**. The available open dataset (Kaggle iron-ore flotation) is a 0-D process time-series with no rate-constant
axis (see `real-datasets.md`), so it cannot supply the $C(k,t)$ surface this case learns. The validation truth is the
closed-form $C^*=e^{-kt}$, which is exact; what is illustrative is the *scenario* (a manufactured $k$-sweep), not the
physics or the error metric.

## Reproduce

```bash
python -m pinnlab.pipeline mine-flotation-kinetics --seed 42
```
