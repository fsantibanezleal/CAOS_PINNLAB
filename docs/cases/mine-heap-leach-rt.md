# mine-heap-leach-rt — heap-leach reactive transport (2-species advection-diffusion-reaction)

The mining-mineral-processing entry. It exercises the **coupled multi-output PINN with a nonlinear bimolecular
reaction term** under downward Darcy advection — the catalogue's stress test for a single network that must solve two
PDEs simultaneously, each carrying a nonlinear coupling, validated against a method of manufactured solutions (MMS)
anchor.

## Problem

A saturated porous heap is percolated downward by lixiviant. Two aqueous reactants $c_A, c_B$ advect with the Darcy
velocity, disperse, and react bimolecularly $A+B\to C$ at rate $k_f c_A c_B$. For each species $i\in\{A,B\}$:

$$ \partial_t c_i + \mathbf{v}\cdot\nabla c_i = D\,\nabla^2 c_i - k_f\,c_A c_B + f_i,\quad
\mathbf{v}=(0,1),\ D=0.05,\ k_f=1\ \text{on}\ (0,1)^2\times(0,1]. $$

The velocity is purely downward ($v_z=1$), dispersion is isotropic with $D=0.05$, and the reaction is second-order. The
MMS truth (offset by $+1.5$ to keep concentrations positive) is

$$ c_A^\* = e^{-t}\sin(\pi x)\cos(\pi z) + 1.5,\qquad c_B^\* = e^{-t/2}\cos(\pi x)\sin(\pi z) + 1.5, $$

and the source terms $f_A, f_B = \mathcal{L}[c^\*]$ are the analytic residuals of the operator applied to these fields,
so the manufactured pair is an exact solution. Soft Dirichlet boundary conditions and the initial condition are set to
$c^\*$. Domain grid: $41\times41\times21$ in $(x,z,t)$.

## Method

**Single shared FNN, two outputs, MMS-anchored.** The architecture is a `[3]→[40]×4→[2]` tanh network: one body
predicts both $c_A$ and $c_B$, so the nonlinear coupling $k_f c_A c_B$ is differentiated through a single autodiff
graph rather than two separate models. The PDE residual returns both equations as a list, and the **loss is weighted**
`[1, 1, 10, 10, 10, 10]` over `[eqA, eqB, bcA, bcB, icA, icB]` — the boundary and initial fits are up-weighted 10×
relative to the interior residual so the soft Dirichlet/IC anchors hold while the bimolecular interior is learned.

The **method of manufactured solutions** is what makes this case honest and well-posed: rather than guess a closed-form
solution to a nonlinear reactive transport system (which generally has none), we pick smooth $c_A^\*, c_B^\*$ and derive
the exact source $f=\mathcal{L}[c^\*]$ analytically, giving an unambiguous L2 anchor for a nonlinear coupled PDE.
Optimisation is Adam (20 000 steps, lr $10^{-3}$) followed by L-BFGS. Method precedent: reactive-transport PIML for
critical minerals (arXiv:2506.15960).

## Result (measured, seed 42)

| metric | value |
|--------|-------|
| $c_A$ relative-L2 vs MMS analytic | **7.9e-05** |
| $c_B$ relative-L2 vs MMS analytic | **8.5e-05** |
| $c_A$ max abs error | 0.001684 |
| validation anchor | **analytic** (MMS) |
| ONNX parity (max abs) | 1.192e-06 |
| lane | **live** (31 KB ONNX, 4.82 ms infer) |

The validation anchor is the analytic MMS field. Both species reach relative-L2 below 1e-4 — comfortably inside the
case's `< 2e-2 per species` target band — so this is a genuinely well-converged result, not a CPU-limited compromise.
The single-network multi-output design resolves the nonlinear $k_f c_A c_B$ coupling without sacrificing accuracy on
either channel; $c_A$ is the primary output and $c_B$'s error is reported via `extra_metrics`.

## Honesty

`real_or_synthetic = synthetic-illustrative`. The MMS field is a closed-form manufactured truth, so the L2 numbers
above measure solver accuracy against an exact reference — not a fit to data. The physics is **Chilean-Cu/REE-relevant**
(parameter ranges drawn from the heap/bioleach literature) but is **not fitted to any column-test or plant dataset**.
The real process is more involved — a shrinking-core dissolution sink, dual-porosity mass transfer, and a spatially
variable Darcy velocity. The single bimolecular $k_f c_A c_B$ term here is a deliberate, well-posed teaching
simplification of that chemistry, chosen so the MMS anchor stays exact and the case demonstrates the coupled-PINN
technique cleanly rather than claiming plant fidelity.

## Reproduce

```bash
python -m pinnlab.pipeline mine-heap-leach-rt --seed 42
```
