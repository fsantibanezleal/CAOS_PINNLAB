# dyn-double-pendulum — the double pendulum (chaotic ODE), a PINN as a t → state map

> Category: **canonical-benchmark** · system_type: **ode-dynamical** · view_kit: **TrajectoryAnimationKit** ·
> lane: **precompute** (replay) · data: **synthetic-illustrative** · anchor: **integrator-ref (RK45)**.

The flagship of the `ode-dynamical` class and the one case with **no spatial field**: the network maps **time
`t` → the two angles `(θ₁, θ₂)`** of a planar double pendulum. The "solution" is an animated *trajectory*, not a
heatmap — rendered by `TrajectoryAnimationKit` (ADR-0063).

## Governing equations

Two coupled nonlinear second-order ODEs from the Lagrangian (point masses `m₁=m₂=1`, massless rigid arms
`ℓ₁=ℓ₂=1`, `g=9.81`), released from rest at `θ₁(0)=θ₂(0)=120°`:

```
θ₁'' = f₁(θ₁, θ₂, θ₁', θ₂'),   θ₂'' = f₂(θ₁, θ₂, θ₁', θ₂')
```

(the standard explicit accelerations; identical code drives the torch residual and the numpy integrator).

## Method — `ode-residual-hard-ic` (soft IC)

- **Physics loss** = squared residual of the two ODEs at collocation times (`r_i = θ_i''_net − f_i`, autodiff in `t`).
- **Initial condition = SOFT** (`dde.icbc.IC` for `θ_i(0)` + `OperatorBC` for `θ_i'(0)=0`), weighted 100× above the
  residual. A `t²` hard-constraint ansatz was tried first and **rejected**: it makes `∂θ̂/∂params ∝ t² → 0` near
  `t=0`, killing the gradient signal so the net never learns the initial acceleration (it moved θ₁ the *wrong way*).
  Soft IC is well-conditioned and the standard DeepXDE IVP approach.
- **Net**: `[1, 96×4, 2]`, tanh, Adam(25k) → L-BFGS. Engine: DeepXDE (PyTorch). Exported to ONNX (parity ~1.6e-6).

## Validation anchor — RK45

A high-accuracy `scipy.solve_ivp` RK45 reference (`rtol=atol=1e-10`) is baked alongside the PINN, plus a second RK45
run with a `+1e-2 rad` perturbed `θ₁(0)` for the butterfly.

## Measured result (committed manifest, seed 42)

- **Relative-L2 = 0.0934 (9.3%)** over `t ∈ [0, 3] s`.
- **leave-time = 1.99 s** — the headline honest metric: the PINN tracks RK45 to ~0.02 rad for ~2 seconds, then
  `|Δθ|` crosses 0.30 rad and the trajectory peels away (the App turns the PINN arm **red** past this point).
- Twin-IC divergence rate (crude largest-Lyapunov estimate) `λ ≈ 0.02 1/s` over the early growth window.

## Honesty: chaos has a hard wall

A double pendulum is chaotic — no fixed network can follow the true trajectory past a finite horizon. We do **not**
claim a long-term match: the leave-time *is* the result, and the butterfly panel (two starts `1e-2 rad` apart,
diverging) shows why. Data is `synthetic-illustrative` (no real measurement); the case is didactic — it shows how a
PINN attacks a dynamical system (not a field) and exactly where chaos defeats it.

## Reproduce

```powershell
./scripts/precompute.ps1 dyn-double-pendulum --seed 42
```
