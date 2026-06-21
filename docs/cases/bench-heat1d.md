# bench-heat1d — 1D transient heat/diffusion with time-dependent hard constraints

The catalogue's first **time-dependent** PINN, and the cleanest demonstration of **exact constraint enforcement**:
the initial condition and both Dirichlet boundaries are baked into the network's output transform, so there is *no*
IC/BC loss term at all — the optimizer only ever sees the PDE residual.

## Problem

The 1D heat (diffusion) equation with $\alpha=1$, an initial sine profile, and homogeneous Dirichlet walls:

$$ \partial_t u = \alpha\,\partial_{xx} u \ \text{on}\ (0,1)\times(0,1],\ \alpha=1,\qquad u(x,0)=\sin(\pi x),\ u(0,t)=u(1,t)=0. $$

The domain is $x\in[0,1]$, $t\in[0,1]$, gridded $201\times101$. This setup has a manufactured exact solution that
serves as the validation anchor:

$$ u^*(x,t) = e^{-\alpha\pi^2 t}\,\sin(\pi x), $$

i.e. the sine profile decays exponentially in time while keeping its shape — a monotone, well-conditioned target.

## Method — time-dependent **hard constraints**

Rather than penalizing IC/BC violations with extra loss terms (the "soft constraint" default), the IC and both BCs are
imposed *exactly* by an output transform applied to the raw network $N(x,t)$:

$$ \hat{u}(x,t) = t\,x\,(1-x)\,N(x,t) + \sin(\pi x). $$

At $t=0$ the first term vanishes and $\hat{u}=\sin(\pi x)$ (the IC); at $x=0$ and $x=1$ the $x(1-x)$ factor vanishes and
$\hat{u}=0$ (the BCs). Because these are satisfied by construction, the `TimePDE` data carries an **empty BC list** and
the loss is the PDE residual alone. The net is a $[2,32,32,32,1]$ tanh FNN (Glorot normal), trained 15 000 Adam steps at
$10^{-3}$ then polished with L-BFGS, on 2540 collocation points.

This is also where the **causal-training** concept is introduced on the docs, but it is *not load-bearing here*: the
exponential decay is monotone, so naive time-uniform weighting already converges. Causality becomes essential only on
the stiff, multi-scale cases (Allen–Cahn, Navier–Stokes) — stating that honestly is the point of placing the concept
on the easy case first.

## Result (measured, seed 42)

| metric | value |
|--------|-------|
| relative-L2 vs analytic $u^*$ | **9.1e-5** (well inside the $<5\times10^{-3}$ band) |
| max abs error vs $u^*$ | 5.6e-5 |
| validation anchor | **analytic** (closed-form $e^{-\pi^2 t}\sin\pi x$) |
| ONNX parity (max abs) | 5.36e-7 |
| lane | **live** (24 KB ONNX, 4.11 ms infer) |

The accuracy here is genuinely excellent — five-significant-figure agreement with the exact solution — not a
CPU-limited compromise. That is expected: the target is smooth and monotone, and hard constraints remove the
IC/BC loss-balancing problem entirely, leaving a single, well-conditioned residual to minimize.

## Honesty

`real_or_synthetic = synthetic`. The truth is a **closed-form** solution of the PDE, not a measured dataset and not a
reduced surrogate — the relative-L2 above is computed against $u^*(x,t)$ evaluated analytically on the test grid. So
every number is real and reproducible, but the *problem* is a textbook benchmark chosen to validate the method, not a
real-world observation. Nothing here is fit to data; the network only ever sees the PDE residual and the
construction-enforced IC/BC.

## Reproduce

```bash
python -m pinnlab.pipeline bench-heat1d --seed 42
```
