# Buildable specs: groundwater (Theis) + conformal prediction (2026-07-15)

Two units researched and, where possible, VERIFIED, but not yet built. Written so the next session builds on
checked ground, not from memory. Both follow the completeness rule PINO and HNN followed: pipeline + estimate
+ bilingual context + verdict + docs + fit gate + deploy, in one landing.

---

## Unit 3 — Groundwater: a pumping-test inverse, Theis anchor (subsurface gap)

### Status: ANCHOR VERIFIED, case not built.

`wip/beyond-sota/verify_theis_anchor.py` checks the Theis analytic solution against a numerical radial
finite-difference solve and PASSES: max relative error **0.06%** at observation radii 10-1000 m. So the
analytic anchor and the well-flux boundary condition are both correct. Run it with the pipeline venv.

### The physics (verified, confined aquifer ONLY)

Radial transient flow to a fully-penetrating well in a homogeneous, isotropic, INFINITE, CONFINED aquifer:

    S ds/dt = T (1/r) d/dr( r ds/dr ),   s(r,0)=0,   lim_{r->0} r ds/dr = -Q/(2 pi T),   s(inf,t)=0

Exact solution (Theis 1935):

    s(r,t) = Q/(4 pi T) * W(u),   u = r^2 S / (4 T t),   W(u) = E1(u)  (exponential integral, scipy.special.exp1)

**Confined is not optional.** Theis assumes a confined aquifer of constant thickness. For an UNCONFINED
aquifer the saturated thickness changes and you need Boulton/Neuman or at least the Dupuit correction
s' = s - s^2/(2b); using Theis there produces a plausible-looking WRONG anchor. The case must state "confined"
and mean it.

### Verified realistic parameters (SI)

- Transmissivity T = 500 m^2/day (moderately transmissive sand), range ~50-2000 m^2/day.
- Storativity S = 2e-4 (confined: 1e-5 .. 1e-3, dimensionless).
- Pumping Q = 2000 m^3/day.
- Resulting drawdowns are physical: 3.5 m at r=10 m, 0.58 m at r=1000 m, at t=1 day.

### The case (recommended: the INVERSE, because it is the real task)

`env-aquifer-test` (or `poll-...`), `method="inverse-parameter"`, category pollution-environmental or a new
subsurface bucket. The engineering question: **you cannot measure transmissivity or storativity directly; a
pumping test infers them from drawdown over time.** So:

- Generate synthetic drawdown at a few observation radii over time from the exact Theis solution, add
  realistic noise (a few cm), with KNOWN true (T, S).
- PINN maps (r, t) -> s, with T and S as `dde.Variable` unknowns (follow `env_soil_heat_real.py`: the two
  variables in `external_trainable_variables`, observation points as `dde.icbc.PointSetBC`, the PDE residual
  with the radial 1/r term). Non-dimensionalise r and t so the network sees O(1) inputs; the 1/r term needs
  care near the well, so keep the inner radius rw ~ 0.1 m and do NOT sample r=0.
- **Three-way validation (all honest):** recovered (T, S) vs the true synthetic values; the PINN drawdown vs
  the noisy observations; and the analytic Theis curve through the recovered parameters vs through the true
  ones. Cross-check against the classical Cooper-Jacob straight-line method (drawdown vs log t) as a second
  independent estimate.
- **Honesty:** for the idealised confined-infinite-homogeneous problem Theis is exact, so the FORWARD problem
  does not need a PINN; the value is the inverse (T, S are unobservable) and the generalisation once the
  aquifer is bounded or heterogeneous, where no closed form exists. Say both plainly.

### The forward alternative (simpler, weaker point)

A forward radial PINN validated against Theis is easy to verify but invites "why not just evaluate the closed
form". Only build it if the inverse proves too finicky; even then, frame it as the honest "here a classical
solution wins; the PINN earns its keep off these assumptions" case.

---

## Unit 4 — Conformal prediction on the operator (research: highest value-per-hour gap)

### Status: researched, not built. No verification spike yet.

The survey ranks this the single worst gap by value: we ship the Darcy FNO/PINO operators with a headline
error number and NO bound on the next instance. Split conformal prediction gives a distribution-free coverage
guarantee at almost no cost and is a dense 2026 literature (arXiv:2606.09923, 2606.08654, 2606.17513,
2607.17297).

### The method (split conformal, the honest minimal version)

1. Hold out a CALIBRATION set of coefficient fields the operator never trained on.
2. For each, compute a nonconformity score, e.g. the per-pixel absolute error |G(a) - u| (or a normalised
   score s(x) = |G(a)(x) - u(x)| / sigma(x) if a spread estimate exists).
3. For target coverage 1 - alpha, take the ceil((n+1)(1-alpha))/n empirical quantile q of the calibration
   scores.
4. On a NEW instance, the prediction band is G(a)(x) +/- q (or +/- q sigma(x)). Guarantee: marginal coverage
   >= 1 - alpha, distribution-free, under exchangeability only.

### The case

A companion to `bench-darcy-operator` / `bench-darcy-pino`: same family, add a calibration split, bake the
band. The workbench shows, per held-out field: the prediction, the true field, and the conformal band, plus
an **empirical coverage** readout (fraction of pixels inside the band) that should land near the target
1 - alpha. The estimate/verdict states the honesty: conformal gives MARGINAL coverage under exchangeability,
NOT per-pixel or per-instance certainty, and it says nothing about out-of-distribution inputs (a different
coefficient family breaks exchangeability and voids the guarantee) - which is exactly the caveat an operator
surrogate needs.

### Verify first (before building)

Spike: split conformal on the existing Darcy FNO. Compute the band on a calibration split, then measure
empirical coverage on a fresh test split; it must land near 1 - alpha (e.g. 90%). If it does not, the score or
the exchangeability assumption is wrong - find out before building the case, the same way the PINO residual and
the Hamiltonian were checked first.

---

## Why these two are separate landings, not one

Each is a full unit (verify -> case -> estimate -> context -> verdict -> docs -> fit gate -> deploy), the same
shape as the PINO and HNN units that shipped as v0.27.000 and v0.28.000. Neither should be half-built in the
app: an unlanded case with no context renders a hollow panel, which is the exact defect the completeness rule
exists to prevent.
