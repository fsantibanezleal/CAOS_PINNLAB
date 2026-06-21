# bench-wave1d — 1D wave equation with a SIREN + hard-constraint PINN

The canonical hyperbolic benchmark of the catalogue. It exercises the two techniques a PINN needs to nail an
*oscillatory* solution: a **SIREN (sine activation)** backbone to beat spectral bias, and a **hard-constraint output
transform** that satisfies the IC and BC exactly so there is no IC/BC loss term — and therefore no loss weighting to
tune. It is the cleanest, most stable trainer in the suite and the reference point for "what good looks like".

## Problem

The 1D wave (second-order hyperbolic) equation with wave speed $c=1$, a single-mode initial displacement, zero
initial velocity, and fixed (Dirichlet) ends:

$$ u_{tt} = c^2\,u_{xx} \quad\text{on}\ (0,1)\times(0,1],\ c=1, $$
$$ u(x,0)=\sin(\pi x),\quad u_t(x,0)=0,\quad u(0,t)=u(1,t)=0. $$

The manufactured exact solution — the validation anchor — is a standing wave that completes half a temporal period
over $t\in[0,1]$:

$$ u^*(x,t) = \sin(\pi x)\,\cos(c\pi t). $$

Domain $x\in[0,1]$, $t\in[0,1]$; single output $u(x,t)$; evaluation grid $201\times 201$.

## Method — SIREN + exact hard constraints

Two SOTA choices, each addressing a specific failure mode:

- **SIREN (sine activation).** A standard `tanh` MLP suffers spectral bias and learns the oscillatory $\cos(\pi t)$
  factor slowly and inaccurately. The network here is a `[2, 64, 64, 64, 1]` FNN with **`sin` activation**, the
  spectral-bias remedy for periodic/wave solutions.
- **Hard-constraint output transform.** Instead of penalising the IC and BC with loss terms, the raw network output
  $N(x,t)$ is wrapped so the constraints hold *by construction*:

$$ \hat u(x,t) = \sin(\pi x) + t^2\,x\,(1-x)\,N(x,t). $$

  At $t=0$ this collapses to $\sin(\pi x)$ (IC#1); every term of $\partial_t\hat u$ carries a factor of $t$, so
  $u_t(x,0)=0$ (IC#2); the factors $x(1-x)$ and $\sin(\pi x)$ vanish at $x=0,1$ (BC). With `num_boundary=0` and the
  ICs absorbed into the architecture, the **only** loss term is the PDE residual — no multi-objective weighting to
  balance, which is exactly why this case trains so stably.

Engine: DeepXDE, residual $u_{tt}-c^2 u_{xx}$ via `dde.grad.hessian`, 2540 domain collocation points, Adam
(15k steps, lr $10^{-3}$) followed by L-BFGS, scored against `solution=analytic` on 4000 test points.

## Result (measured, seed 42)

| metric | value |
|--------|-------|
| relative-L2 vs analytic $u^*$ | **4.9e-5** (anchor: **analytic**) |
| max absolute error | 7.1e-5 |
| ONNX parity (max abs) | **6.56e-7** |
| lane | **live** (50 KB ONNX, 30.1 ms infer) |

This clears the expected band (relative-L2 $<5\times10^{-3}$) by two orders of magnitude. Unlike the CPU-limited cases
in the suite, there is nothing to apologise for here — hard constraints plus SIREN on a smooth single-mode wave is the
benchmark's best-behaved regime, and the field is reproduced to ~$10^{-5}$.

## Honesty

`real_or_synthetic = synthetic`. The validation anchor $u^*(x,t)=\sin(\pi x)\cos(\pi t)$ is the **closed-form exact
solution** of this IBVP, not a measured dataset and not a reduced surrogate. "Synthetic" (vs `synthetic-illustrative`)
is the strongest of the non-real tags: the truth is exact, so the relative-L2 is a genuine solution-accuracy number,
not a fit-quality proxy. Nothing here is dressed up — there is no real-world data in this case, and the comparison is
purely against analysis.

## Reproduce

```bash
python -m pinnlab.pipeline bench-wave1d --seed 42
```
