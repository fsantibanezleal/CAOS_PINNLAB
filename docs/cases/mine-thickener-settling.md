# mine-thickener-settling — Bürger-Concha degenerate settling, parametric descent rate R

The mining-mineral-processing flagship for **nonlinear conservation laws**, as a **parametric family**: a
strongly-degenerate convection-diffusion PDE (the Bürger-Concha thickener model) with an exact descending-front anchor
whose settling speed $R$ is a network input — so the web **Live** tab sweeps the mud-line from slow to fast.

## Problem

1D batch settling of a flocculated suspension: the solid volume fraction $\phi(z,t)$ obeys the strongly-degenerate
convection-diffusion conservation law (Kynch hindered settling + sediment consolidation)

$$ \phi_t + \partial_z f_{bk}(\phi) = \partial_z\!\big(D(\phi)\,\phi_z\big), \qquad
   f_{bk}=v_0\,\phi\,(1-\phi/\phi_{max})^C,\qquad D\ \text{degenerate above}\ \phi_c. $$

The batch flux is Richardson-Zaki hindered settling ($v_0=-1$, $\phi_{max}=0.66$, $C=5$); the diffusion $D(\phi)$ is
**degenerate** — zero in the dilute regime, switching on only above the gel concentration $\phi_c=0.23$ (tanh-regularized
so the residual stays $C^1$). The validation anchor is an exact descending **tanh front**

$$ \phi^*(z,t;R)=\phi_{lo}+(\phi_{hi}-\phi_{lo})\,\tfrac12\big(1-\tanh\tfrac{z-s}{W}\big),\qquad s=z_0-R\,t, $$

with $z_0=0.9$, front width $W=0.10$. The **front descent rate** $R\in[0.3,0.9]$ is the swept knob: a faster-settling
suspension drops its mud-line faster. The MMS source $f=\mathcal{L}[\phi^*]$ is recomputed analytically *as a function of
$R$* through the genuine $f_{bk}'$ and the degenerate $D,D'$, so the anchor stays exact for every $R$.

## Method

**Hard-front MMS family on the true nonlinear operator.** Net $[3,64,64,64,64,1]$ tanh (DeepXDE), Adam (24000, lr
$10^{-3}$) → L-BFGS, loss weights $[1,10]$; the exact $\phi^*$ is soft-imposed on the whole $(z,t,R)$ cube boundary
(incl. the $t=0$ IC). $R$ is a network input so the trained net spans the settling-rate family and the Live tab sweeps
it. (RAR was dropped here — it de-stabilised the stiff degenerate-diffusion residual; widening the front to $W=0.10$ +
Adam→L-BFGS converges cleanly instead.)

## Result (measured, seed 42)

Validation anchor: the **exact descending front** $\phi^*(z,t;R)$. Six variants ($R=0.30,0.42,0.54,0.66,0.78,0.90$):

| metric | value |
|--------|-------|
| relative-L2 vs analytic | **≤ 0.41 %** across all 6 variants |
| ONNX parity (max abs) | 5.1e-7 |
| lane | **live** (one shared ONNX; Live sweeps $R$) |

Well inside the `< 2e-2` band — the front position and the dilute/consolidated plateaus are captured across the whole
descent-rate range, against the genuine nonlinear flux + degenerate switch.

## Honesty

`real_or_synthetic = synthetic-illustrative`. The Bürger-Concha PDE, the Richardson-Zaki flux and the degenerate
consolidation diffusion are the real settling physics (literature parameters); the field is an exact MMS illustration,
**not** a measured $(z,t,\phi)$ thickener dataset (none is public — `real-datasets.md`). Honest about exercising the
genuine nonlinear operator, and equally honest that it is not fit to a real thickener.

## Reproduce

```bash
python -m pinnlab.pipeline mine-thickener-settling --seed 42
```
