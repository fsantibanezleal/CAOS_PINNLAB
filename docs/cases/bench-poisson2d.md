# bench-poisson2d — 2D Poisson (Dirichlet) by a hard-constraint PINN

The catalogue's cleanest canonical benchmark. It exercises the **hard-constraints** method family: a Dirichlet boundary
condition baked into the network output so it holds *exactly* for any weights — no boundary-loss term to weight. This is
the case used to harden the train → ONNX → web contract on, because a closed-form solution gives an unambiguous error.

## Problem

2D Poisson with homogeneous Dirichlet boundaries on the unit square,

$$ -\nabla^2 u = 2\pi^2 \sin(\pi x)\sin(\pi y)\ \text{on}\ (0,1)^2,\qquad u|_{\partial\Omega}=0. $$

The forcing $f(x,y)=2\pi^2\sin(\pi x)\sin(\pi y)$ is manufactured so the exact solution — the validation anchor — is

$$ u^*(x,y) = \sin(\pi x)\sin(\pi y), $$

a single smooth bump with $\max u \approx 1$ at the centre. The field is evaluated on a $101\times101$ grid over
$x,y\in[0,1]$.

## Method — hard constraints (distance-function output transform)

The Dirichlet condition $u=0$ on $\partial\Omega$ is enforced **structurally**, not by a penalty. The raw network
$N(x,y)$ is multiplied by a distance-like factor that vanishes on every wall:

$$ \hat u(x,y) = x(1-x)\,y(1-y)\,N(x,y). $$

Because $x(1-x)y(1-y)=0$ on all four edges, $\hat u$ satisfies the BC *exactly for any network weights*. The
consequence is that the loss has **only the PDE residual** $-u_{xx}-u_{yy}-f$ — there is no BC term, hence no
boundary-vs-interior loss-weight to tune, which removes the single most common PINN failure mode. The net is a
`[2,64,64,64,1]` tanh FNN (Glorot init), trained with Adam (12k steps, lr $10^{-3}$) then L-BFGS, with 2000 interior
collocation points, **0 boundary points** (none are needed), and 4000 test points. Engine: DeepXDE.

## Result (measured, seed 42)

| metric | value |
|--------|-------|
| relative-L2 vs analytic $u^*$ | **5e-6** |
| max abs error vs $u^*$ | 8e-6 |
| validation anchor | **analytic** (closed-form $\sin\pi x\sin\pi y$) |
| ONNX parity (max abs) | 2.38e-7 |
| lane | **live** (48 KB ONNX, 2.81 ms infer, opset 18) |

The relative-L2 of **5e-6** is three-plus orders of magnitude inside the `< 1e-2` band and is genuinely tight — this
case is **not** CPU-limited the way the harder cavity/Helmholtz benchmarks are. The hard-constraint formulation on a
smooth single-mode field is about as favourable as a PINN gets, and the number reflects that honestly.

## Honesty

`real_or_synthetic = synthetic` — the truth is **closed-form (manufactured solution / MMS)**, not measured data. The
forcing was chosen so that an exact analytic field exists, and that field is the anchor every metric is scored against.
Nothing here is fit to or validated against real-world measurements, and nothing claims to be: this is a numerical
correctness benchmark for the hard-constraint method and the export pipeline, and the 5e-6 relative-L2 is meaningful
*only* in that sense. For a real-data counterpart see [env-soil-heat-real](env-soil-heat-real.md).

## Reproduce

```bash
python -m pinnlab.pipeline bench-poisson2d --seed 42
```
