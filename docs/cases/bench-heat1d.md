# bench-heat1d — 1D transient heat/diffusion, parametric diffusivity (time-dependent hard constraints)

The catalogue's cleanest **time-dependent** PINN, as a **parametric family**: the thermal diffusivity $\alpha$ is a
network input, so one trained net + one ONNX covers the whole diffusivity family and the web **Live** tab sweeps how
fast the profile relaxes. The IC and both Dirichlet BCs are baked in exactly (no IC/BC loss term).

## Problem

The 1D heat (diffusion) equation with an initial sine profile and homogeneous Dirichlet walls:

$$ \partial_t u = \alpha\,\partial_{xx} u \ \text{on}\ (0,1)\times(0,1],\qquad u(x,0)=\sin(\pi x),\ u(0,t)=u(1,t)=0. $$

For the fundamental mode this has a closed-form solution, valid for **any** $\alpha$ — the validation anchor:

$$ u^*(x,t;\alpha) = e^{-\alpha\pi^2 t}\,\sin(\pi x). $$

The **diffusivity** $\alpha\in[0.1,1.0]$ is the swept knob: small $\alpha$ barely decays over the window, large $\alpha$
collapses the sine to near zero by $t=1$. Domain $x\in[0,1]$, $t\in[0,1]$ ($161\times101$ field grid).

## Method — time-dependent **hard constraints**

The IC and both BCs are imposed exactly by an output transform on the raw network $\mathcal{N}(x,t,\alpha)$:

$$ \hat{u} = t\,x(1-x)\,\mathcal{N}(x,t,\alpha) + \sin(\pi x). $$

At $t=0$ it leaves $\sin(\pi x)$ (the IC); at $x=0,1$ the $x(1-x)$ factor vanishes (the BCs). With the constraints baked
in, the `TimePDE`/`PDE` data carries an **empty BC list** and the loss is the bare PDE residual. Net
$[3,48,48,48,48,1]$ tanh (DeepXDE), Adam → L-BFGS. $\alpha$ is a network input, so the trained net spans the family and
the Live tab sweeps it.

## Result (measured, seed 42)

Validation anchor: the **closed-form** $u^*(x,t;\alpha)$ (any $\alpha$). Six variants ($\alpha=0.1,0.2,0.4,0.6,0.8,1.0$):

| metric | value |
|--------|-------|
| relative-L2 vs analytic | **≤ 0.15 %** across all 6 variants |
| ONNX parity (max abs) | ~5e-7 |
| lane | **live** (one shared ONNX; Live sweeps $\alpha$) |

Excellent accuracy across the whole diffusivity range — hard constraints remove the IC/BC loss-balancing problem,
leaving a single well-conditioned residual.

## Honesty

`real_or_synthetic = synthetic`. The truth is the closed-form separable solution (exact for every $\alpha$), not a
measured dataset — a textbook benchmark chosen to validate the method. The network only ever sees the PDE residual; the
IC/BC are construction-enforced.

## Reproduce

```bash
python -m pinnlab.pipeline bench-heat1d --seed 42
```
