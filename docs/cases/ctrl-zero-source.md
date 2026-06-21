# ctrl-zero-source — degenerate control: zero source, zero field

The catalogue's mandatory **negative control**. It exercises the **hard-constraint** technique (an output transform that
makes the Dirichlet boundary condition exact by construction) on the most trivial PDE there is, to prove the pipeline
survives the degenerate case without crashing and returns a flat-zero field.

## Problem

Homogeneous Laplace problem on the unit square with a zero source and a hard-zero boundary:

$$ -\nabla^2 u = 0\ \text{on}\ (0,1)^2,\quad u|_{\partial\Omega}=0\ \Rightarrow\ u\equiv 0. $$

The domain is the square $\Omega=(0,1)^2$, sampled on a $41\times41$ grid (inputs $x,y$; output $u$). With no forcing and
homogeneous boundaries, the unique solution is identically zero — so the analytic truth is $u^*(x,y)=0$ everywhere, and
the relative-L2 collapses to $\lVert\text{pred}\rVert$ (the norm of the prediction itself), which must be tiny.

## Method

**Hard constraints.** Rather than penalizing the boundary residual in the loss (a soft constraint), the network output is
multiplied by a function that vanishes on $\partial\Omega$ by construction:

$$ u(x,y)=x(1-x)\,y(1-y)\cdot \mathcal{N}_\theta(x,y), $$

via DeepXDE's `apply_output_transform`. The factor $x(1-x)y(1-y)$ is exactly zero on all four edges, so $u|_{\partial\Omega}=0$
is satisfied identically — no BC loss term, no BC collocation points (`num_boundary=0`). The PDE residual is the bare
Laplacian $-u_{xx}-u_{yy}$ (zero source) evaluated at 500 interior collocation points; the net is a small
`[2, 16, 16, 1]` tanh FNN trained with Adam (lr $10^{-3}$, 2000 steps, no L-BFGS). With both the source and the boundary
pinned to zero, the only fixed point of the residual is the trivial field — the test is whether the engine reaches it.

## Result (measured, seed 42)

| metric | value |
|--------|-------|
| relative-L2 vs analytic ($u^*\equiv0$) | **2.07e-4** ($\lVert\text{pred}\rVert$, anchor = analytic) |
| max abs error | 1.4e-5 |
| ONNX parity (max abs) | 4e-9 |
| lane | **live** (13 KB ONNX, 0.24 ms infer) |

Validation anchor is **analytic**: the closed-form truth is exactly zero, so the score is the magnitude of the field the
PINN actually produces. A relative-L2 of 2.07e-4 (max pointwise 1.4e-5) means the predicted field is essentially flat
zero to four decimals — the engine handles the degenerate case cleanly. The 4e-9 ONNX parity and 0.24 ms inference put it
comfortably on the live lane.

## Honesty

`real_or_synthetic = synthetic` — the truth is closed-form ($u\equiv0$), not a measured dataset. Nothing is fit to data
and nothing is manufactured to flatter the model: this is a sanity check, not a science result. Its only job in the
catalogue is to confirm that with zero forcing and a hard-zero boundary the pipeline (PDE residual → train → bake →
ONNX export → parity → gate) runs end-to-end and returns the correct trivial field. If this case ever drifts off zero,
something upstream is broken.

## Reproduce

```bash
python -m pinnlab.pipeline ctrl-zero-source --seed 42
```
