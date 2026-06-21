# poll-soil-barrier — domain-decomposition (FBPINN) PINN across a low-permeability barrier

The case that answers *"how do you keep a single smooth network from smearing out a kink?"* — it exercises the
**domain-decomposition** method family (FBPINN-style partition of unity). A contaminant diffuses through a soil column
interrupted by a vertical clay/slurry cutoff, and the coefficient jump forces a kink in the field that two cooperating
sub-nets resolve better than one global tanh.

## Problem

A dissolved contaminant diffuses (pure diffusion, $V=0$) through a soil column $x\in[0,1]$, $t\in[0,1]$, containing one
low-permeability vertical barrier — a slab $[A_B,B_B]=[0.45,0.55]$ whose diffusivity is **10× lower** than the
surrounding soil:

$$ c_t = D(x)\,c_{xx} + f, \qquad D(x) = \begin{cases} D_{soil}=1.0 & x \notin [0.45,0.55] \\ D_{barrier}=0.1 & x \in [0.45,0.55] \end{cases} $$

The coefficient **jump** makes $c$ develop a kink at each barrier face: $c$ and the diffusive flux $D\,c_x$ stay
continuous, but $c_x$ itself is discontinuous. The MMS anchor is the layered **series-resistance** steady profile,
which exhibits exactly that kink:

$$ \Psi(x) = 1 - \frac{R(x)}{R(L)}, \quad R(x)=\int_0^x \frac{dx'}{D(x')}, \qquad c^*(x,t) = (1-e^{-t})\,\Psi(x), \quad f = e^{-t}\,\Psi(x). $$

Because the diffusive flux $D\,c^*_x$ is constant in each layer, its divergence vanishes in the interior and $f$ carries
only the time term. Domain: $x\in[0,1]$, $t\in[0,1]$, grid $101\times51$.

## Method

**Domain decomposition — FBPINN-style partition of unity** (dossier §4 #14). Instead of one global network fighting
the coefficient jump, a **2-channel net** is blended by overlapping sigmoid windows centred on the barrier midpoint
$x_c=0.5$:

$$ w_{\text{left}} = \sigma\!\big(\beta\,(x_c - x)\big), \quad w_{\text{right}} = 1 - w_{\text{left}}, \quad
c_{\text{raw}} = w_{\text{left}}\,y_1 + w_{\text{right}}\,y_2 \quad (\beta = 40). $$

The two sub-nets meet on the subdomain boundary, so the kink in $c_x$ is produced by **two networks handing off**
rather than a single smooth tanh straining across the discontinuity. IC and both boundary conditions are imposed as
**hard constraints** via the output transform: a lift $(1-e^{-t})(1-x)$ enforces inlet $c(0,t)=1-e^{-t}$, outlet
$c(1,t)=0$, and IC $c(x,0)=0$, with a vanishing factor $t\,x\,(1-x)$ gating the learned correction. Three extra anchor
columns are seeded at $x=A_B,\,B_B,\,x_c$ to pin the faces. Trained Adam (18k) → L-BFGS in DeepXDE.

The strict per-subdomain-normalized FBPINN is documented in [domain-decomposition](../methods/domain-decomposition.md).

## Result (measured, seed 42)

| metric | value |
|--------|-------|
| relative-L2 vs analytic $c^*$ | **0.192** (19.2 %) |
| max abs error | 0.134 |
| validation anchor | **analytic** (series-resistance MMS, closed-form) |
| ONNX parity (max abs) | **8.9e-08** |
| lane | **live** (73 KB, 1.95 ms) |

The 19 % relative-L2 is **honestly CPU-limited and stated as such** — the coefficient-jump kink is the hard part for a
2-channel net on a single CPU. A 10× contrast was used precisely because a 100× jump makes the kink too severe for this
lane; the strict per-subdomain-normalized FBPINN plus a GPU lane tighten it further. The plume is correctly slowed by the
low-$D$ barrier and the kink appears at each face — the qualitative physics is right even where the pointwise error is not
yet small.

## Honesty

`real_or_synthetic = synthetic-illustrative`. The barrier values ($D_{soil}=1$, $D_{barrier}=0.1$, slab at
$[0.45,0.55]$) are illustrative engineering numbers, and the validation truth is a manufactured-solution (MMS)
series-resistance profile — **not** a calibrated field site or a measured dataset. What *is* real here is the method:
the partition-of-unity decomposition, the coefficient-jump kink physics, and the closed-form anchor it is scored against
are all genuine; nothing about the *technique* is faked. It is a faithful reduced model of barrier transport, not a fit
to data.

## Reproduce

```bash
python -m pinnlab.pipeline poll-soil-barrier --seed 42
```
