# poll-tailings-seepage — unsaturated seepage (Richards/Gardner), parametric sorptive number α

A **strongly nonlinear, degenerate-parabolic PDE** — Richards' equation for unsaturated flow — as a **parametric
family**: the Gardner sorptive number $\alpha$ is a network input, so one trained net + one ONNX covers the whole
sorptivity family and the web **Live** tab sweeps how deep the deposit dries.

## Problem

1D vertical unsaturated seepage through a tailings deposit ($z$ up, $\psi$ = pressure head $<0$, $t$ = time), with a
Gardner exponential closure:

$$ C(\psi)\,\psi_t = \partial_z\!\big[K(\psi)(\psi_z+1)\big], \quad
   K(\psi)=K_s e^{\alpha\psi},\ \ \theta(\psi)=\theta_r+(\theta_s-\theta_r)e^{\alpha\psi},\ \ C(\psi)=\theta'(\psi). $$

The **Kirchhoff transform** $m=e^{\alpha\psi}$ linearises the nonlinear operator *exactly* into a constant-coefficient
advection-diffusion in $m$, which admits an exact separable mode. The head solution, valid for **any** $\alpha$, is the
anchor:

$$ \psi^*(z,t;\alpha)=\frac1\alpha\ln\!\Big(M_0+A\,e^{-\lambda(\alpha)t}\,e^{-\kappa z}\Big),\quad
   \lambda(\alpha)=\frac{K_s}{\theta_s-\theta_r}\,\frac{\kappa(\alpha-\kappa)}{\alpha}. $$

With $M_0+A<1$, $M_0>0$, the argument $m\in(0,1)$ everywhere so $\psi<0$ **strictly** (always unsaturated — the
physical invariant; verified, plus a finite-difference residual $\le10^{-6}$). $\alpha\in[1.0,2.5]$: smaller $\alpha$
(broader pores) → deeper, more stratified suction. Constants $\theta_s=0.43,\ \theta_r=0.078,\ K_s=0.25,\ \kappa=0.9$.
(This replaces the earlier $\alpha$-independent manufactured-source MMS, whose field did not change with $\alpha$.)

## Method

**Richards seepage with a Kirchhoff exact family.** Net $[3,48,48,48,48,1]$ tanh (DeepXDE), Adam (18000, lr $10^{-3}$)
→ L-BFGS, loss weights $[1,10]$. The PDE residual evaluates the *genuine* nonlinear Gardner operator
($C(\psi)\psi_t-K\psi_{zz}-K'(\psi_z^2+\psi_z)$, source-free) on the network output; the exact $\psi^*$ is imposed as a
soft Dirichlet anchor on the $(z,t,\alpha)$ cube boundary (which includes the $t=0$ IC face). $\alpha$ is a network
input, so the trained net spans the family and the Live tab sweeps it.

## Result (measured, seed 42)

Validation anchor: the **exact Kirchhoff family** $\psi^*(z,t;\alpha)$. Six variants ($\alpha=1.0,1.3,1.6,1.9,2.2,2.5$):

| metric | value |
|--------|-------|
| relative-L2 vs exact | **≤ 0.26 %** across all 6 variants |
| ONNX parity (max abs) | 3.6e-7 |
| lane | **live** (one shared ONNX; Live sweeps $\alpha$) |

Well inside the `< 1e-2` band — the nonlinear residual is resolved cleanly across the whole sorptivity range, $\psi<0$
strictly throughout.

## Honesty

`real_or_synthetic = synthetic-illustrative` — the Richards + Gardner physics and the operator are real, but the field
is an exact illustration, **not** fit to a deposit: no open unsaturated-zone $\psi(z,t)$ tailings dataset exists
(`real-datasets.md`). The van Genuchten–Mualem closure and a real saturated-zone inverse are documented as extensions;
Gardner is used because the Kirchhoff transform makes the family exactly solvable for every $\alpha$.

## Reproduce

```bash
python -m pinnlab.pipeline poll-tailings-seepage --seed 42
```
