# bench-allencahn — Allen-Cahn (stiff reaction-diffusion), hard-constraint + RAR PINN

The canonical PINN *failure case* turned working recipe. A plain soft PINN collapses to a metastable
state on this stiff bistable equation; this case exercises the **hard-constraint ansatz + residual-adaptive
refinement (RAR)** family — the standard DeepXDE technique for resolving sharp, slowly-moving interfaces.

## Problem

The 1D Allen-Cahn equation: a tiny diffusion competing with a bistable reaction,

$$ u_t = d\,u_{xx} + 5(u - u^3), \qquad d = 0.001, \quad x \in [-1,1], \ t \in [0,1], $$

with initial condition and endpoint-matched (periodic-ish) boundary

$$ u(x,0) = x^2 \cos(\pi x), \qquad u(\pm 1, t) = \text{matched at the endpoints}. $$

Because $d$ is small relative to the $5(u-u^3)$ reaction, the solution forms **metastable $\pm 1$ plateaus
separated by sharp, slowly-moving transition layers**. The reaction is stiff, the interface is thin, and a
naive PINN with the IC/BC as soft loss terms simply settles into the wrong (metastable) attractor — which is
exactly why Allen-Cahn is the textbook benchmark for "PINNs need more than the vanilla loss".

## Method — hard constraints + RAR

Two SOTA ingredients make this case train, both implemented in DeepXDE (`engine=deepxde`):

- **Hard-constraint IC ansatz.** The initial condition is **baked into the output transform**, not penalized.
  The network output $N$ is wrapped as
  $$ \hat{u}(x,t) = x^2\cos(\pi x) \;+\; t\,(1 - x^2)\,N(x,t), $$
  so $\hat u$ equals the IC exactly at $t=0$ (the second term vanishes) and is endpoint-matched at $x=\pm 1$
  (the $(1-x^2)$ factor vanishes there). There is **no IC/BC loss term at all** — the constraints are
  satisfied by construction, which removes the loss-balancing that lets a soft PINN drift to the metastable state.
- **RAR-G adaptive sampling.** After the initial Adam→L-BFGS fit, the case runs **4 rounds of greedy residual
  refinement**: each round draws a 100 000-point pool, evaluates the PDE residual, **adds the top-600
  highest-residual points** as anchors, then re-fits (5 000 Adam iters); a **single final L-BFGS** polishes after the
  loop. This chases the moving interface — collocation density concentrates exactly where the residual is largest,
  i.e. on the thin layer.

Architecture: a `[2, 64, 64, 64, 1]` tanh FNN, 8 000 domain / 400 boundary / 800 initial points, Adam (lr 1e-3,
20 000 iters) then L-BFGS, then the RAR loop. This case ships as a **single honest benchmark variant** — the
symmetric Allen-Cahn front is *stationary*, so there is no closed-form parametric family to sweep. The **SOTA ceiling is cited, not claimed**: PirateNets / jaxpi
reach ~2e-5 relative-L2 on this problem; this case targets a working DeepXDE recipe, not that frontier.

## Result (measured, seed 42)

Validation anchor is the **spectral reference `Allen_Cahn.npz`** (DeepXDE/Raissi, MIT) on a 201×101 (x,t) grid —
a *numerical* truth, not real-world data.

| metric | value |
|--------|-------|
| relative-L2 vs spectral reference | **0.41 %** (0.004106) |
| ONNX parity (max abs) | 1.16e-06 |
| lane | **live** (49 KB ONNX, opset 18) |

The 0.41 % relative-L2 clears the expected band (`< 1e-2`) by ~2.4× — the hard-constraint + RAR recipe genuinely
resolves the transition layers rather than collapsing to the metastable state. (The trimmed RAR — 4 rounds + one
final L-BFGS instead of 6 per-round L-BFGS — bakes far faster for the same honest sub-1 % accuracy.)

## Honesty

`real_or_synthetic = synthetic`. The truth is a **closed-form / spectral numerical reference**, not measured data:
the equation is exact, the IC is analytic, and the validation field comes from a high-fidelity spectral solver
(`Allen_Cahn.npz`). Nothing here is fit to real-world observations — the case demonstrates that a PINN can *match
an established numerical benchmark*, which is the point of a canonical-benchmark case. The SOTA ~2e-5 frontier is
referenced for context and explicitly **not** claimed as this case's result.

## Reproduce

```bash
python -m pinnlab.pipeline bench-allencahn --seed 42
```


## Comparison (the app's Compare view)

The pipeline bakes the spectral-reference **standard** vs the **naive soft PINN** vs the **hard-constraint + RAR** PINN
(see [the method ladder](../architecture/method-ladder-comparison.md)). The naive lane collapses to a metastable state and smears the sharp +/-1 transition
layers (**95.4 %** relative-L2 vs the spectral standard); the hard-constraint + RAR fix tracks them (**0.4 %**). The
textbook PINN failure, computed and shown side by side with the error maps.
