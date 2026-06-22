# mine-comminution-pbe — comminution population balance (size-transport), parametric grind rate g

The **mining-mineral-processing** entry exercising the **population-balance / transport-PDE** family as a **parametric
family**: a drift-diffusion in particle-size space stands in for the full comminution PBE, with the grind rate $g$ a
network input — so the web **Live** tab sweeps the size distribution drifting toward fines.

## Problem

Grinding (SAG / ball milling) evolves the particle-size distribution $n(s,t)$: fragmentation continuously shifts mass
toward smaller sizes. The full population-balance equation is an integro-differential equation coupling **selection**
and **breakage** kernels; this case ships the **reduced size-transport surrogate** — a drift-diffusion in normalized
size space (the Fokker-Planck reduction of the breakage operator):

$$ n_t + (-g)\,n_s = D\,n_{ss}, \qquad s\in[0,1]\ (1=\text{coarse}),\ t\in[0,1]. $$

For a narrow coarse-feed Gaussian release this has an **exact solution** (the advected-diffused 1D Green's function,
valid for any $g$):

$$ n^*(s,t;g)=\sqrt{\frac{s_0^2}{s_0^2+2Dt}}\;\exp\!\Big(-\frac{(s-(s_0-g t))^2}{2(s_0^2+2Dt)}\Big), $$

with $s_0=0.8$, $s_0^2=0.01$, $D=0.012$. The **grind rate** $g\in[0,0.6]$ is the swept knob: the feed centered at $s_0$
drifts down in size at speed $g$ and spreads by dispersion $D$, its peak decaying as mass is conserved. (This replaces
the earlier $g$-independent manufactured-source MMS, whose field did not move with the knob.)

## Method

**Drift-diffusion PINN with a hard-IC output transform** (the burgers/flotation pattern): $u_\theta = g_0(s) + t\,
\mathcal{N}_\theta(s,t,g)$ where $g_0(s)=n^*(s,0)$ is the (grind-rate-independent) initial Gaussian, so the IC is exact
and the net learns the interior evolution. Net $[3,48,48,48,48,1]$ tanh (DeepXDE), Hypercube $(s,t,g)$, Adam (15000, lr
$10^{-3}$) → L-BFGS, anchor via `solution=analytic`. $g$ is a network input, so the trained net spans the family and the
Live tab sweeps it.

## Result (measured, seed 42)

Validation anchor: the **exact advected-diffused Gaussian** $n^*(s,t;g)$. Six variants ($g=0,0.12,0.24,0.36,0.48,0.6$):

| metric | value |
|--------|-------|
| relative-L2 vs exact | **≤ 2.0 %** across all 6 variants |
| ONNX parity (max abs) | 9.5e-7 |
| lane | **live** (one shared ONNX; Live sweeps $g$) |

The high-grind corner $g=0.6$ is advection-leaning (Péclet $\approx50$) and sits at ~2 % — labeled honestly in the
expected band; the lower-grind variants are well under 1 %.

## Honesty

`real_or_synthetic = synthetic-illustrative`. The drift-diffusion is a deliberate reduction of the comminution PBE — it
keeps the correct behaviour (mass drifting + spreading toward fines) but drops the selection/breakage integral. The
truth it scores against is the exact Green's-function field, **not** a measured mill PSD (no open SAG/ball-mill
size-distribution dataset with a grind-rate axis — `real-datasets.md`). The full breakage-kernel integro-differential
PBE is documented as the complete model.

## Reproduce

```bash
python -m pinnlab.pipeline mine-comminution-pbe --seed 42
```
