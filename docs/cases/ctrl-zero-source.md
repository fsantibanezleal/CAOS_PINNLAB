# ctrl-zero-source — MMS verification family, parametric source amplitude a (contains the degenerate control)

A manufactured-solution (MMS) **verification family** for the Poisson operator, parametric in the **source amplitude**
$a$ — a network input. It exercises the **hard-constraint** technique (a zero-Dirichlet output transform, exact by
construction) and *contains* the catalogue's mandatory degenerate negative control as its $a=0$ limit. The web **Live**
tab sweeps $a$ and the field fades to flat zero.

## Problem

Poisson with a hard-zero boundary and a two-mode manufactured solution, on $(0,1)^2$:

$$ -\nabla^2 u = f(x,y;a),\quad u|_{\partial\Omega}=0,\qquad
   u^*(x,y;a)=a\big(\sin\pi x\sin\pi y+\tfrac12\sin 2\pi x\sin 2\pi y\big). $$

$u^*$ vanishes on the whole boundary for every $a$; substituting into $-\nabla^2$ gives the imposed source
$f=a\,(2\pi^2\sin\pi x\sin\pi y+4\pi^2\sin 2\pi x\sin 2\pi y)$. The amplitude $a\in[0,1]$ is a network input:
**at $a=0$ this is the archetype's degenerate negative control** ($f\equiv0\Rightarrow u\equiv0$ — the engine must run
and return flat zero); at $a=1$ a two-mode field (a dominant fundamental lobe + a finer second-mode ripple). (This
generalises the old degenerate-only control into an honest parametric family whose $a=0$ limit *is* the control.)

## Method

**Hard constraints.** The zero Dirichlet boundary is satisfied identically by the output transform
$u_\theta = x(1-x)\,y(1-y)\cdot\mathcal{N}_\theta(x,y,a)$ — no BC loss, no boundary collocation. The PDE residual is the
bare $-u_{xx}-u_{yy}-f(x,y;a)$. Net $[3,48,48,48,48,1]$ tanh (DeepXDE), Hypercube $(x,y,a)$, Adam (12000, lr $10^{-3}$)
→ L-BFGS, anchor via `solution=analytic`. $a$ is a network input, so the trained net spans the family and the Live tab
sweeps it.

## Result (measured, seed 42)

Validation anchor: the manufactured $u^*(x,y;a)$. Six variants ($a=0,0.2,0.4,0.6,0.8,1.0$):

| metric | value |
|--------|-------|
| relative-L2 vs $u^*$ | **≤ 0.15 %** for $a\ge0.2$; $a=0$ → $\lVert\text{pred}\rVert=2.1\%$ (the degenerate control: $u^*\equiv0$, so the metric is the field norm — essentially flat zero) |
| ONNX parity (max abs) | 4.2e-7 |
| lane | **live** (one shared ONNX; Live sweeps $a$) |

The $a=0$ chip reproduces the mandatory negative control as the limit of the family; the engine returns a near-flat-zero
field there, and recovers the two-mode field cleanly as $a\to1$.

## Honesty

`real_or_synthetic = synthetic` — the truth is the closed-form manufactured $u^*$, not measured data. Nothing is fit to
data; the case is a rigorous MMS verification of the Poisson operator (and the pipeline's degenerate-control sanity
check) — the relative-L2 is an exact-by-construction error.

## Reproduce

```bash
python -m pinnlab.pipeline ctrl-zero-source --seed 42
```
