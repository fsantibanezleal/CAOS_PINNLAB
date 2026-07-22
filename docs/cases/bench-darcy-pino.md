# bench-darcy-pino — PINO: the operator that also knows the equation

**Method:** `operator-pino` · **Engine:** `pino-torch` · **Category:** canonical-benchmark ·
**Label:** synthetic · **Lane:** precompute

The companion to [`bench-darcy-operator`](bench-darcy-operator.md). That case learns the Darcy solution
operator from solved pairs alone. This one puts the governing equation into the operator's own training loss
and asks the question that decides whether any of this is useful in practice:

> **how many solved instances do you need, if the operator also knows the equation?**

## The question this case answers

Generating training data for an operator means running the classical solver once per instance. On a real
problem that is the expensive part. PINO (Li et al., arXiv:2111.03794, ACM/IMS J. Data Science 1(3) 2024,
doi:[10.1145/3648506](https://doi.org/10.1145/3648506)) adds the PDE residual to the operator's loss, so the
equation can substitute for labels. This case measures **by how much, on our own grid** — including where it
does not work.

## The problem

Steady Darcy flow on the unit square, the canonical FNO benchmark:

$$-\nabla\!\cdot\big(a(\mathbf{x})\,\nabla u\big) = 1, \qquad u|_{\partial\Omega} = 0, \qquad \Omega=(0,1)^2$$

with $a(\mathbf{x})$ a two-value thresholded Gaussian random field taking the values 3 and 12 (sharp material
interfaces). The reference $u$ is a 5-point finite-volume solve with harmonic-mean face conductivities and a
sparse direct solution. Grid 32x32; 128 labelled instances available; 32 held out for test.

## What is held identical

Both lanes use the same FNO backbone (width 20, 10 Fourier modes, 4 layers), the same seed, the same 80
epochs, the same held-out test set. **Only the loss differs:**

| lane | loss |
|---|---|
| data-only FNO | $\mathcal{L}_{\text{data}}$ |
| PINO | $\mathcal{L}_{\text{data}} + \lambda\,\mathcal{L}_{\text{pde}}$, plus $\mathcal{L}_{\text{pde}}$ on **virtual** instances |

The virtual instances are the mechanism that matters: a fresh coefficient field $a'\sim\mu$ costs a Gaussian
filter and a threshold with **no solve**, so the equation supplies unlimited training signal. This is the
paper's Algorithm 1 — "we have access to the unlimited dataset by sampling new $a_j$ in each iteration".

## Results (measured, 2026-07-15)

Held-out mean relative-L2 over 32 unseen coefficient fields. The variants in the app **are** these budgets.

| labels | PINO | data-only FNO | PINO better by | $\lambda$ | train PINO | train FNO |
|---|---|---|---|---|---|---|
| 0 | 0.6885 | 1.0452 | **+34.1%** | 1.0 | 26 s | 0 s |
| 8 | 0.4708 | 0.1696 | **-177.6%** | 1.9 | 83 s | 27 s |
| 32 | **0.0784** | 0.1433 | **+45.3%** | 0.5 | 250 s | 115 s |
| 128 | **0.0371** | 0.0715 | **+48.1%** | 1.6 | 766 s | 377 s |

**Read it honestly, in three parts.**

1. **With no labels at all**, the data-only operator is untrained by definition; PINO reaches 0.69 from the
   equation alone. That is a qualitative point, not a good absolute number.
2. **With enough labels (32, 128)** the physics term is a large, consistent win: **45-48% lower error at the
   same label budget**.
3. **With 8 labels it fails, badly, and we ship that.** $\lambda$ was 1.9 there, so this is not an
   over-weighted physics term. PINO lands at 0.471, near its own zero-label result: eight labels cannot pin
   the solution, so the outcome is dominated by a physics term that has not converged in 80 epochs. The
   equation is the right signal but a **slow** one. This is precisely the standing objection to
   physics-informed training — long training times — reproduced and measured inside our own operator.

## Two implementation facts that decide whether this works at all

**The residual on a grid output.** An operator emits $u$ on a grid, so there is no autograd in $x$; the paper
(§3.3) offers numerical differentiation or a query-function. We use the finite-volume divergence form with
harmonic-mean faces, **verified before anything was built on it**: fed the reference solution it returns rms
$3.6\times10^{-14}$; fed $u=0$ it returns exactly $f=1$. The FFT route is *not* the default here — this
problem is non-periodic with a discontinuous coefficient, and measured on a reference field the spectral
Laplacian misses by ~6x the source term, exactly the failure the paper warns about ("spectral methods require
smoothness and uniform grids"). Fourier continuation (FC-PINO, arXiv:2211.15960) is the published fix; not
implemented here.

**The boundary condition must be hard.** The residual is interior-only, and the Darcy solution is unique only
*with* its boundary condition. With a merely soft boundary penalty, a strong physics weight drives the
interior residual to zero around the *wrong* boundary values and converges to a different member of the
solution family — measured: at 32 labels the soft-BC configurations scored -0.5% and -68.9%, and enforcing
$u|_{\partial\Omega}=0$ exactly (multiplying the output by $16\,x(1-x)\,y(1-y)$) moved the same budget to
**+45.3%**. The exported ONNX applies the same constraint, so the artifact and the trained model agree.

## What this case does NOT claim

- **Not a discretisation-accuracy result.** The residual uses the same stencil as the reference solver, so
  driving it to zero *is* solving the reference's discrete system. What the physics term buys, and all we
  claim, is data efficiency.
- **Not the paper's numbers.** PINO's headline results (20x lower error and 25x speedup vs PINN on Kolmogorov
  flow; 400x vs a GPU pseudo-spectral solver; 3000x on Darcy inversion) are the paper's, on the paper's
  problems, at the paper's scale. Ours are the table above.
- **Not a speed claim.** Measured on this machine, the classical reference solve at 32x32 takes **42.4 ms**
  and the exported operator's ONNX inference takes **35-39 ms**. At this grid size the operator has
  essentially no speed advantage, and the classical side is an unoptimised Python assembly loop. The
  many-query amortisation argument is real in the literature but is not demonstrated at our grid size.
- **Not cheaper to train.** The PINO lane costs about 2x the data-only lane (766 s vs 377 s at 128 labels).
  The saving is in **solver calls avoided**, never in training time.

## References

- Li, Zheng, Kovachki, Jin, Chen, Liu, Azizzadenesheli, Anandkumar. *Physics-Informed Neural Operator for
  Learning Partial Differential Equations.* arXiv:2111.03794; ACM/IMS J. Data Science 1(3), 2024.
  doi:10.1145/3648506
- Li et al. *Fourier Neural Operator for Parametric Partial Differential Equations.* arXiv:2010.08895, ICLR 2021.
- Duruisseaux, Kossaifi, Anandkumar. *Fourier Neural Operators Explained: A Practical Perspective.*
  arXiv:2512.01421 (Dec 2025, rev. Jan 2026).
- Maust et al. *Fourier Continuation for Exact Derivative Computation in PINOs (FC-PINO).* arXiv:2211.15960.

Working notes, including all three configurations tried and why the first two failed:
`wip/beyond-sota/plan-2026-07-15.md` §B.
