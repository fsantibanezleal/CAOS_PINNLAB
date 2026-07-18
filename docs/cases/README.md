# Cases

Each case is one documented PINN problem: a governing equation, a SOTA method, a validation anchor, and a baked
artifact the web app replays. Cases carry a **category** (the registry groups by it) and an honesty flag
(`synthetic` · `synthetic-illustrative` · `validated-real`). Every number below is the committed manifest's measured
value; lanes are derived from measurements (see [the gate](../architecture/the-gate.md)).

## The catalogue (20 cases)

| Category | Cases |
|----------|-------|
| **canonical-benchmark** | poisson2d · heat1d · wave1d · burgers1d · allencahn · navier-cavity · darcy-operator (FNO) · **double-pendulum (chaotic ODE)** |
| **industrial-fluids-heat** | helmholtz · heat2d-inverse · **hidden-velocity (HFM flagship)** |
| **mining / mineral-processing** | heap-leach-rt · thickener-settling · flotation-kinetics · comminution-pbe |
| **pollution / environmental** | ocean-transport · soil-barrier · tailings-seepage · **soil-heat-real (real data)** · source-uq-bpinn (UQ) |
| **control** | zero-source |

## Honesty: what "real" means here

Most cases are **synthetic** (closed-form/MMS truth) or **synthetic-illustrative** (a faithful reduced model of a
real process, but the field is illustrative, not fit to a measured dataset). Exactly one case is trained and
validated against a **real measured dataset**:

- [**env-soil-heat-real**](env-soil-heat-real.md) — recovers soil thermal diffusivity from NOAA USCRN soil
  temperatures, validated out-of-sample against held-out depths. **This is the flagship real-data case.**

The mining/pollution differentiators are honest about their status: the reduced models (flotation first-order
kinetics, comminution size-transport, Bürger–Concha thickening, Richards/Gardner seepage) are the standard
engineering closures, but no open plant/field dataset exists for them (documented in the management dossier's
`real-datasets.md`), so they are labeled `synthetic-illustrative`, never dressed up as fit-to-data.

## Per-case write-ups

Every case has a full page (governing equations, the SOTA method, the measured result, the honesty rationale,
the reproduce command).

**Canonical benchmark**
- [bench-poisson2d](bench-poisson2d.md) — 2D Poisson, hard-constraint BC.
- [bench-heat1d](bench-heat1d.md) — 1D heat, time-dependent hard constraints.
- [bench-wave1d](bench-wave1d.md) — 1D wave, SIREN + hard constraints.
- [bench-burgers1d](bench-burgers1d.md) — viscous Burgers shock, RAR adaptive sampling.
- [bench-allencahn](bench-allencahn.md) — Allen–Cahn interface, hard constraints + RAR.
- [bench-navier-cavity](bench-navier-cavity.md) — Navier–Stokes lid-driven cavity (u,v,p), multi-output + loss weighting.
- [bench-darcy-operator](bench-darcy-operator.md) — Darcy-flow **operator learning** (Fourier Neural Operator; lane=precompute).
- [dyn-double-pendulum](dyn-double-pendulum.md) — **chaotic double pendulum**, a PINN as a `t → state` map vs an RK45 anchor (the `ode-dynamical` category; honest leave-time).

**Industrial fluids / heat**
- [ind-helmholtz](ind-helmholtz.md) — Helmholtz, Fourier-feature inputs.
- [ind-heat2d-inverse](ind-heat2d-inverse.md) — inverse conductivity field from sparse sensors.
- [ind-hidden-velocity](ind-hidden-velocity.md) — **the HFM flagship**: the whole current estimated from sparse dye samples alone (no velocity data, no IC/BC), with the dye-swept identifiability split.

**Mining / mineral-processing**
- [mine-heap-leach-rt](mine-heap-leach-rt.md) — 2-species reactive transport.
- [mine-thickener-settling](mine-thickener-settling.md) — Bürger–Concha nonlinear settling.
- [mine-flotation-kinetics](mine-flotation-kinetics.md) — parametric first-order kinetics C(k,t).
- [mine-comminution-pbe](mine-comminution-pbe.md) — comminution population balance (size transport).

**Pollution / environmental**
- [poll-ocean-transport](poll-ocean-transport.md) — advection–diffusion.
- [poll-soil-barrier](poll-soil-barrier.md) — FBPINN domain decomposition across a low-permeability barrier.
- [poll-tailings-seepage](poll-tailings-seepage.md) — Richards/Gardner unsaturated seepage.
- [env-soil-heat-real](env-soil-heat-real.md) — **real-data** inverse (USCRN soil temperatures).
- [poll-source-uq-bpinn](poll-source-uq-bpinn.md) — Bayesian PINN (deep ensemble) with epistemic uncertainty.

**Control**
- [ctrl-zero-source](ctrl-zero-source.md) — degenerate zero-source sanity anchor.
