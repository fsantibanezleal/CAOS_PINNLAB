# bench-darcy-superres — Zero-shot super-resolution: one operator, any grid

**Method:** `operator-superres` · **Engine:** `fno-torch` · **Category:** canonical-benchmark ·
**Label:** synthetic · **Lane:** precompute

The fourth Darcy operator case. It exercises the FNO's signature property, the one that makes a neural
*operator* different from a grid-to-grid network (a CNN, a U-Net): **discretisation invariance**.

> An operator trained only at 32x32 runs at 64x64 and 96x96 it never saw, with no retraining and no change to
> its weights.

## Why it works

Each Fourier layer applies `(K v)(x) = F^-1( R . F(v) )(x)`, where `R` are learned weights over a truncated
set of low-frequency modes. Because `R` indexes **modes**, not grid positions, the same transformation applies
to an input sampled at any resolution: the FFT and its inverse adapt to the grid, the weights do not change. A
CNN, whose filters are local in pixels, cannot even be evaluated off its training grid without changing
architecture.

Reference: Li et al., *Fourier Neural Operator for Parametric PDEs*, arXiv:2010.08895 (ICLR 2021) — "the first
ML-based method to successfully model turbulent flows with zero-shot super-resolution."

## Results (measured, trained only at 32x32)

| evaluation grid | held-out relative-L2 | vs the 32x32 training grid |
|---|---|---|
| **32x32** (training) | 0.038 | baseline |
| **64x64** (unseen) | 0.057 | 1.5x, never seen |
| **96x96** (unseen) | 0.066 | 1.7x, never seen |

Verified before building (spike): the same 32→64 degradation landed at ~1.8x. The operator just runs; the
error rises gracefully.

## The build

The coefficient random field is generated with a **resolution-consistent correlation length** (the
Gaussian-filter sigma scales with the grid), so all three resolutions are the *same physical process* sampled
more finely, not three different families. Each variant is evaluated at its native resolution (an `eval_grid()`
hook returns the per-variant grid) against a finite-difference reference solved at that grid. The shown field
therefore genuinely sharpens from 32 to 96.

## What this case does NOT claim

- **Not free accuracy.** The error rises with resolution: finer grids resolve features the coarse training
  never taught the operator. It is "one operator serves many grids", not "the operator gets better on finer
  grids".
- **Not an unbounded property.** Far past the training resolution the error degrades further; discretisation
  invariance transfers the operator across grids, it does not make it a multigrid solver.
- **The exported ONNX is the 32x32 operator.** The finer-grid results are baked field artifacts (the same
  torch operator produced them at each grid; the FNO's mode-scatter does not export cleanly with dynamic
  spatial ONNX axes, but discretisation invariance is a property of the torch model, demonstrated by the baked
  64/96 fields).

## References

- Li, Kovachki, Azizzadenesheli, Liu, Bhattacharya, Stuart, Anandkumar. *Fourier Neural Operator for
  Parametric Partial Differential Equations.* arXiv:2010.08895, ICLR 2021.
- Duruisseaux, Kossaifi, Anandkumar. *Fourier Neural Operators Explained.* arXiv:2512.01421.

The four Darcy operator cases together: `bench-darcy-operator` (data-driven FNO), `bench-darcy-pino` (physics
in the loss), `bench-darcy-conformal` (a distribution-free error bar), and this one (discretisation invariance).
