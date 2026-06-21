# mine-comminution-pbe — comminution population balance (size-transport reduced PINN)

The **mining-mineral-processing** entry in the catalogue. It exercises the **population-balance / transport-PDE**
family with the **method of manufactured solutions (MMS)**: a drift-diffusion in particle-size space stands in for the
full comminution population-balance equation, and a closed-form manufactured field gives an exact anchor to score
against.

## Problem

Grinding (SAG / ball milling) evolves the particle-size distribution $n(s,t)$: breakage continuously shifts mass
toward smaller sizes. The full population-balance equation (PBE) is an integro-differential equation coupling
**selection** and **breakage** kernels. This case ships the **reduced size-transport surrogate** — a drift-diffusion
in normalized size space that captures the net downward shift (drift) and the spread (dispersion) **without** the
breakage integral:

$$ n_t + G\,n_s = D\,n_{ss} + f \qquad s \in [0,1],\ t \in [0,1], $$

with constant grind drift $G = 0.6$ (transport toward smaller $s$) and size dispersion $D = 0.03$. The manufactured
(MMS) solution, strictly positive so it reads as a distribution, is

$$ n^*(s,t) = 1 + \tfrac{1}{2}\,e^{-t}\sin(\pi s), $$

and the source $f = \mathcal{L}[n^*] = n^*_t + G\,n^*_s - D\,n^*_{ss}$ is therefore **closed-form**. The domain is the
unit size–time square $[0,1]^2$, discretized on an $81 \times 81$ grid. Inputs are $(s,t)$; the single output is the
size density $n$.

## Method

The governing operator is linear, so the SOTA play here is **MMS**: pick a smooth target $n^*$, push it through the
operator to obtain the exact forcing $f$, and train the PINN to satisfy the PDE residual plus the boundary/initial
data that $n^*$ implies. This turns an otherwise truth-less surrogate into a case with a **rigorous analytic anchor** —
the relative-L2 against $n^*$ is a real error, not a self-consistency check.

The network is an FNN $[2, 48, 48, 48, 48, 1]$ with `tanh` activations and Glorot-normal init (DeepXDE engine). The
constraints are imposed **softly**: a `DirichletBC` and an `IC` both evaluate $n^*$ on the boundary and at $t=0$, and
training uses **multi-output loss weighting `[1, 10, 10]`** — the PDE-residual term at weight 1, the boundary and
initial terms up-weighted 10× so the soft constraints actually bind. Optimization is **Adam (15 000 steps, lr 1e-3)
followed by L-BFGS** polishing, sampling 3 000 collocation, 200 boundary and 200 initial points, scored on 6 000 test
points.

## Result (measured, seed 42)

| metric | value |
|--------|-------|
| relative-L2 vs MMS analytic $n^*$ | **1.79e-4** (0.018 %) |
| max abs error vs $n^*$ | 6.83e-4 |
| validation anchor | **analytic** (manufactured solution) |
| ONNX parity (max abs) | **4.77e-7** |
| lane | **live** (39.0 KB, 0.88 ms infer) |

The validation anchor is the closed-form MMS field $n^*$. The relative-L2 of **1.79e-4** clears the
`expected_band` target (< 1e-2) by nearly two orders of magnitude — a clean, fully converged fit, not a CPU-limited
one. The 39 KB ONNX reproduces the trained network to **4.77e-7** max abs deviation and infers in **0.88 ms**, so the
case lands on the **live** lane with no gate reasons.

## Honesty

`real_or_synthetic = synthetic-illustrative`. This is a **faithful reduced model, not a fit to data**: the
drift-diffusion is a deliberate simplification of the comminution PBE — it keeps the physically correct behaviour
(mass drifting and spreading toward smaller sizes) but drops the selection/breakage integral that the complete model
carries. The truth it scores against is the **manufactured** field $n^*$, so the error is exact and honest, but it is
**not** a measured mill PSD. There is no open SAG / ball-mill particle-size dataset to validate against (see
`real-datasets.md`); the full breakage-kernel integro-differential PBE is documented as the complete model. Nothing
here is dressed up as real measurement.

## Reproduce

```bash
python -m pinnlab.pipeline mine-comminution-pbe --seed 42
```
