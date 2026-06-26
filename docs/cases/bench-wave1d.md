# bench-wave1d — 1D wave equation, parametric speed (SIREN + hard constraints)

The canonical hyperbolic benchmark, as a **parametric family**: the wave speed $c$ is a network input, so one trained
net + one ONNX covers the whole speed family and the web **Live** tab makes the standing wave oscillate faster or
slower. It exercises the two techniques a PINN needs for an *oscillatory* solution: a **SIREN (sine activation)**
backbone against spectral bias, and a **hard-constraint output transform** for the two ICs + BCs (no IC/BC loss term).

## Problem

The 1D wave equation with a single-mode initial displacement, zero initial velocity, and fixed ends:

$$ u_{tt} = c^2\,u_{xx} \quad\text{on}\ (0,1)\times(0,1],\qquad u(x,0)=\sin(\pi x),\ u_t(x,0)=0,\ u(0,t)=u(1,t)=0. $$

The standing-wave solution is exact for **any** $c$ — the validation anchor:

$$ u^*(x,t;c) = \sin(\pi x)\,\cos(c\pi t). $$

The **wave speed** $c\in[0.5,2.0]$ is the swept knob: $c=0.5$ advances a quarter period over the window; $c=1$ half a
period; $c=2$ a full period. Domain $x,t\in[0,1]$ ($161\times161$ field grid).

## Method — SIREN + exact hard constraints

- **SIREN.** A `tanh` MLP suffers spectral bias on the oscillatory $\cos(c\pi t)$; the net uses **`sin` activation**.
- **Hard constraints.** The raw network is wrapped so both ICs and the BCs hold by construction:

$$ \hat u = \sin(\pi x) + t^2\,x(1-x)\,\mathcal{N}(x,t,c). $$

  At $t=0$ → $\sin(\pi x)$ (IC#1); every $\partial_t$ term carries a factor $t$, so $u_t(x,0)=0$ (IC#2); $x(1-x)$
  vanishes at the walls (BC). With `num_boundary=0` the only loss is the PDE residual.

The oscillatory parametric family needed capacity: net $[3,96,96,96,96,96,1]$ `sin` (DeepXDE), Adam (22000) → L-BFGS.
$c$ is a network input, so the trained net spans the family and the Live tab sweeps it.

## Result (measured, seed 42)

Validation anchor: the **exact standing wave** $u^*(x,t;c)$ (any $c$). Six variants ($c=0.5,0.75,1.0,1.25,1.5,2.0$):

| metric | value |
|--------|-------|
| relative-L2 vs analytic | **≤ 0.32 %** across all 6 variants (incl. the 4× speed extremes) |
| ONNX parity (max abs) | ~7e-7 |
| lane | **live** (one shared ONNX; Live sweeps $c$) |

The bigger SIREN net nails the full 4× speed family — including the fast $c=2$ corner where a smaller net stalled.

## Honesty

`real_or_synthetic = synthetic`. The anchor is the closed-form standing-wave solution (exact for every $c$), not a
measured dataset — the relative-L2 is a genuine solution-accuracy number, not a fit proxy.

## Reproduce

```bash
python -m pinnlab.pipeline bench-wave1d --seed 42
```
