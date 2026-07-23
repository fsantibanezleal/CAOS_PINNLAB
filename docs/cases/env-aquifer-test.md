# env-aquifer-test — Pumping test: recover a confined aquifer's T and S

**Method:** `inverse-aquifer-test` (Cooper-Jacob) · **Engine:** `analytic-cooper-jacob` ·
**Category:** pollution-environmental · **Label:** synthetic-illustrative · **Lane:** precompute

The subsurface applied case the catalogue was missing, and an honest **"know when a PINN is NOT the tool"**
case.

## The problem

An aquifer's transmissivity T and storativity S cannot be measured directly. The standard field method is a
**pumping test**: pump a well at constant rate Q, watch the drawdown s fall in an observation well over time,
and infer T and S from the curve. The physics is the Theis (1935) solution of radial transient flow in a
**confined** aquifer:

    s(r,t) = Q/(4 pi T) W(u),   u = r^2 S/(4 T t),   W(u) = E1(u)

Confined is not optional: for an unconfined aquifer the saturated thickness changes and Theis is the wrong
model (Boulton/Neuman is needed).

## The method that works: Cooper-Jacob

For late time (small u), W(u) ~= -0.5772 - ln u, so s is **linear in ln t**:
`s ~= (Q/4 pi T) ln(2.25 T t / r^2 S)`. Fit a straight line to s vs ln t: the slope gives T, the zero-drawdown
intercept gives S.

## Results (measured)

| aquifer | T recovered vs true | T error | S recovered vs true | S error |
|---|---|---|---|---|
| fine sand | 124 vs 120 m²/day | 3.2% | 4.5e-4 vs 5.0e-4 | 9.4% |
| coarse sand | 504 vs 500 m²/day | 0.7% | 1.9e-4 vs 2.0e-4 | 2.9% |
| gravel | 1497 vs 1500 m²/day | 0.2% | 8.4e-5 vs 8.0e-5 | 4.8% |

The analytic Theis anchor was verified against a numerical radial finite-difference solve to **0.06%**
(`wip/beyond-sota/verify_theis_anchor.py`); Cooper-Jacob was verified against Theis across these three
aquifers before the case was built.

## Why a PINN is NOT the tool here

A PINN was spiked on the same data and did far worse:

- A **forward** radial PINN collapsed to **82%** relative error (the stiff near-well transient plus the
  well-flux boundary is the classic PINN gradient pathology).
- A PINN **parameter-inverse** recovered T = 2335 vs 500 m²/day (**367%** off): the data fit alone does not
  pin (T, S), and the log-radius Laplacian is tiny away from the well, so the PDE constraint that should
  identify the parameters is too weak where the observations live.

The PINN earns its keep only once the aquifer is bounded or heterogeneous, where no closed form exists and
Cooper-Jacob is invalid. On this well-posed problem, reaching for a PINN is the wrong call. That is the point
of the case, in the catalogue's honesty tradition.

## What this case does NOT claim

- **The exported network is a small surrogate fit to the drawdown field**, kept only so the case has an ONNX
  artifact. The honest parameter inverse is Cooper-Jacob, not the network.
- **Idealised.** Confined, homogeneous, infinite, fully-penetrating well, constant rate. Real aquifer tests
  add boundaries, partial penetration, wellbore storage and delayed yield; the case is the textbook baseline.

## References

- Theis, C.V. *The relation between the lowering of the piezometric surface and the rate and duration of
  discharge of a well using groundwater storage.* Trans. AGU, 1935.
- Cooper, H.H. & Jacob, C.E. *A generalized graphical method for evaluating formation constants and
  summarizing well-field history.* Trans. AGU, 1946.

Working notes, the Theis/Cooper-Jacob verification and the PINN spikes:
`wip/beyond-sota/spec-groundwater-and-conformal-2026-07-15.md`.
