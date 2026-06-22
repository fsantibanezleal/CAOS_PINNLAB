# poll-ocean-transport — 2D advection-diffusion PINN (time-scrubber over an advected pollutant patch)

Two spatial dimensions and time under one network, presented as a **time-scrubber**: a pollutant patch drifts with a
coastal current and spreads by eddy diffusion. **Time is the swept parameter** — `field_axes=(x,y)`, six time
snapshots — so the web **Live** tab scrubs $t$ and replays the spill drifting and diluting.

## Problem

A passive scalar $c$ (plastic / oil tracer) advects with a uniform current $\mathbf{v}$ and diffuses with eddy
diffusivity $D$ on the unit square over one time unit:

$$ c_t + \mathbf{v}\cdot\nabla c = D\,\nabla^2 c, \qquad \mathbf{v}=(0.45,\,0.35),\ D=0.01. $$

For a Gaussian point release this has an **exact solution** (the advected-diffused 2D Green's function — a genuine
solution, *not* a manufactured source):

$$ c^*(x,y,t)=\frac{s_0^2}{s_0^2+2Dt}\,\exp\!\Big(-\frac{(x-x_0-v_x t)^2+(y-y_0-v_y t)^2}{2\,(s_0^2+2Dt)}\Big). $$

The patch **center** moves with the current ($\mathbf{x}_0+\mathbf{v}t$), its **variance** grows linearly
($s^2=s_0^2+2Dt$), and its **peak** decays as $s_0^2/s^2$ (mass conserved). Péclet $\mathrm{Pe}=|\mathbf{v}|L/D\approx45$
— advection-dominated. (The earlier gyre-MMS field was a decaying eigenmode that did not visibly move; it was replaced
by this genuinely-translating exact solution.)

## Method

**Advection-diffusion PINN with soft IC/BC.** Net $[3,64,64,64,64,1]$ tanh (DeepXDE), Adam (18000, lr $10^{-3}$) →
L-BFGS, loss weights $[1,10,50]$ for $[\text{pde},\text{bc},\text{ic}]$. The Dirichlet boundary ($c=c^*$) and IC
($c=c^*$ at $t=0$) are **soft penalties**, so the net genuinely learns the interior transport and the relative-L2 is
the true PINN error. Time is a network input; the six variants fix $t\in\{0,0.2,0.4,0.6,0.8,1.0\}$ and the Live tab
sweeps it continuously (a time scrubber over the shared ONNX).

## Result (measured, seed 42)

Validation anchor: the **exact advected-diffused Gaussian** $c^*$. Six time-snapshot variants:

| metric | value |
|--------|-------|
| relative-L2 vs exact | **≤ 0.19 %** across all 6 snapshots ($t=0$ → 0.06 %; $t=1$ → 0.19 %) |
| ONNX parity (max abs) | 4.8e-7 |
| lane | **live** (one shared ONNX; Live = time scrubber) |

## Honesty

`real_or_synthetic = synthetic-illustrative`. The truth is a closed-form exact solution, not a fit to a real spill or
a real ocean-current product — a **physically-faithful illustration** of advective-diffusive transport. The uniform
current is a deliberate simplification (no gyre / time-varying flow). What is real is the PINN error: with soft IC/BC,
the relative-L2 measures the network's actual interior accuracy against the exact solution.

## Reproduce

```bash
python -m pinnlab.pipeline poll-ocean-transport --seed 42
```
