# mine-thickener-settling — Bürger-Concha degenerate settling (nonlinear flux + MMS)

The mining-mineral-processing flagship for **nonlinear conservation laws**. It exercises a strongly-degenerate
convection-diffusion PDE — the Bürger-Concha thickener model — where the residual carries a genuinely nonlinear
Richardson-Zaki flux and a diffusion term that switches on only above a gel concentration. A sharp settling front
must descend while a consolidated bed rises.

## Problem

1D batch settling of a flocculated suspension: the solid volume fraction $\phi(z,t)$ obeys the
strongly-degenerate convection-diffusion conservation law (Kynch hindered settling + sediment consolidation)

$$ \phi_t + \partial_z f_{bk}(\phi) = \partial_z\!\big(D(\phi)\,\phi_z\big), \qquad
f_{bk} = v_0\,\phi\,(1-\phi/\phi_{max})^C, \qquad D\ \text{degenerate above}\ \phi_c. $$

The batch flux $f_{bk}$ is Richardson-Zaki hindered settling ($v_0=-1$, $\phi_{max}=0.66$, $C=5$); the diffusion
$D(\phi)$ is **degenerate** — zero in the dilute regime and switching on only above the gel concentration
$\phi_c=0.23$, regularized by a tanh switch so the residual stays $C^1$. The domain is scaled $z\in[0,1]$,
$t\in[0,1]$ on a $101\times51$ grid. Physically: a clear supernatant forms above a descending settling front, while a
consolidating bed thickens from the bottom — the canonical "applied" moving-front case.

## Method

**Forward solve via the Method of Manufactured Solutions (MMS) on the true nonlinear operator.** There is no public
measured thickener field, so the validation anchor is a manufactured descending-front $\phi^*(z,t)$ — a tanh front
$s=z_0-r\,t$ with $z_0=0.9$, $r=0.8$, width $0.06$ — whose analytic source $f=\mathcal{L}[\phi^*]$ is derived in closed
form and subtracted in the PDE residual. Crucially the source is computed through the **genuine** $f_{bk}'(\phi)$ and
the regularized degenerate $D(\phi)$, $D'(\phi)$ — not a linearization — so the network must learn against the real
nonlinear flux and switch, not a toy.

The net is a 4×64 tanh FNN (DeepXDE), trained Adam (18k, lr $10^{-3}$) → L-BFGS. The IC and Dirichlet boundaries are
soft-imposed at $\phi^*$, and **multi-output loss weighting** `[1, 10, 10]` up-weights the BC/IC against the interior
residual to lock the front position. RAR (residual-based adaptive refinement) is noted as the front-sharpening
upgrade; the shipped run relies on the weighted soft constraints plus the manufactured source.

## Result (measured, seed 42)

| metric | value |
|--------|-------|
| relative-L2 vs MMS analytic $\phi^*$ | **0.51 %** (0.005127) |
| max absolute error | 0.006319 |
| validation anchor | **analytic** (manufactured descending front) |
| ONNX parity (max abs) | **2.7e-7** |
| lane | **live** (61 KB, 1.33 ms) |

The relative-L2 of 0.51 % clears the expected band (< 2e-2) by ~4×: the front position and the dilute/consolidated
plateaus are both captured cleanly. This is **not** a CPU-limited result — the degenerate term is mild enough on this
scaled domain that Adam→L-BFGS resolves the front without adaptive refinement, and the sub-1 % L2 reflects that.

## Honesty

`real_or_synthetic = synthetic-illustrative`. The Bürger-Concha PDE, the Richardson-Zaki flux, and the degenerate
consolidation diffusion are the real settling physics, with parameters from the literature ($\phi_{max}$, $\phi_c$, the
hindered-settling exponent). What is **not** real is the field itself: there is no public measured $(z,t,\phi)$
thickener dataset (see `real-datasets.md`), so the truth is a Method-of-Manufactured-Solutions descending front, not
data. So this is a faithful reduced model of thickener settling validated against a closed-form anchor — it is honest
about exercising the genuine nonlinear operator, and equally honest that it is not fit to a real thickener.

## Reproduce

```bash
python -m pinnlab.pipeline mine-thickener-settling --seed 42   # train + validate vs MMS + bake ONNX (deterministic)
```
