# bench-burgers1d — 1D viscous Burgers, parametric viscosity (hard-constraint + RAR)

The canonical nonlinear advection-diffusion benchmark, as a **parametric family**: an exact traveling-shock front
whose thickness is set by the viscosity $\nu$ — a network input — so one trained net + one ONNX covers the whole
viscosity family and the web **Live** tab sweeps the shock from sharp to diffuse.

## Problem

The 1D viscous Burgers equation on $x\in[-1,1]$, $t\in[0,1]$:

$$ u_t + u\,u_x = \nu\,u_{xx}. $$

Burgers admits an exact **traveling-wave (tanh) solution** — a front that translates without changing shape, valid
for *any* $\nu$ (Whitham, *Linear and Nonlinear Waves*). With left/right states $u_L=1,\ u_R=0$ (so $\Delta=u_L-u_R=1$,
shock speed $s=\tfrac{u_L+u_R}{2}=\tfrac12$) and an initial front position $x_0=-0.4$:

$$ u^*(x,t;\nu) = s - \frac{\Delta}{2}\tanh\!\Big(k\,(x-x_0-s\,t)\Big), \qquad k=\frac{\Delta}{4\nu}. $$

The viscosity $\nu\in[0.02,0.08]$ sets the front **thickness** ($\sim 4\nu$): small $\nu$ → a razor-thin internal
layer; large $\nu$ → a diffuse ramp. The front translates right at speed $s$ and stays interior to the domain.

## Method — hard constraints + RAR

The IC and both Dirichlet BCs are imposed **exactly** by an output transform (no IC/BC loss term):

$$ u_\theta = g(x;\nu) + t\,(1-x^2)\,\mathcal{N}_\theta(x,t,\nu), \qquad g(x;\nu)=u^*(x,0;\nu), $$

which leaves the initial front at $t=0$ and the (time-constant) Dirichlet states at $x=\pm1$. On top of the base fit,
**RAR** (residual-based adaptive refinement, Wu et al., *CMAME* 2023) adds the highest-residual collocation points —
which land on the moving front — over several rounds. Net `[3,96,96,96,96,96,1]` tanh (DeepXDE), Adam → L-BFGS.
$\nu$ is a network input, so the trained net spans the family and the Live tab sweeps it.

## Result (measured, seed 42)

Validation anchor: the **exact traveling-shock** $u^*(x,t;\nu)$ (closed form, all $\nu$). Six baked variants
($\nu=0.02,0.03,0.04,0.05,0.06,0.08$):

| metric | value |
|--------|-------|
| relative-L2 vs analytic | **≤ 1.2 %** across all 6 variants (sharpest $\nu=0.02$ → 0.08 %; $\nu=0.08$ → 1.2 %) |
| ONNX parity (max abs) | 6.5e-7 |
| lane | **live** (one shared ONNX; Live sweeps $\nu$) |

The hard-constraint baseline encodes the front, so the net only learns the small interior translation — the *sharpest*
viscosity is the *most* accurate. Honest CPU-trained accuracy, all variants inside the `< 2e-2` band.

## Honesty

`real_or_synthetic = synthetic`. The anchor is the exact traveling-shock solution (closed-form for every $\nu$), not
measured data. The classic Raissi sine-IC benchmark ($u(x,0)=-\sin\pi x$, $\nu=0.01/\pi$) has no closed-form
*parametric* family, so the traveling-wave family is used here — it gives a genuinely $\nu$-dependent field and an
exact anchor for the whole sweep.

## Reproduce

```bash
python -m pinnlab.pipeline bench-burgers1d --seed 42
```
