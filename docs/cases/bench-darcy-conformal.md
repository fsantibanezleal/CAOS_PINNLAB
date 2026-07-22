# bench-darcy-conformal — Conformal prediction: a distribution-free error bar for the operator

**Method:** `operator-conformal-uq` · **Engine:** `fno-torch` · **Category:** canonical-benchmark ·
**Label:** synthetic · **Lane:** precompute

The third Darcy operator case. [`bench-darcy-operator`](bench-darcy-operator.md) ships a data-driven FNO and
[`bench-darcy-pino`](bench-darcy-pino.md) adds the equation; both report a single held-out error number and
say nothing about the **next** instance. This case adds the guarantee a deployed surrogate actually needs:

> for a new coefficient field, a band `pred(x) +/- q` that contains the true field with a stated probability,
> assuming no error distribution beyond exchangeability.

## Split conformal prediction

The recipe (Vovk; Lei et al. 2018; and the 2026 neural-operator conformal literature) is three steps and
retrains nothing:

1. Hold out a **calibration** set the operator never trained on.
2. Score each calibration instance by its worst pixel error, `s_i = max_x |G(a_i)(x) - u_i(x)|`.
3. Take the finite-sample quantile `q = ceil((n+1)(1-alpha))/n` of the scores. Then for a new instance the
   band `G(a)(x) +/- q` contains the whole true field with probability `>= 1 - alpha`.

The guarantee is **marginal** (over the draw of calibration + test) and **distribution-free**.

## Verified before building

A spike on this exact Darcy FNO measured the empirical coverage against each target before the case was
written. The case build reproduces it:

| target (1 - alpha) | band width q | empirical coverage (200 unseen fields) | holds? |
|---|---|---|---|
| 80% | 0.0017 | **87.5%** | yes |
| 90% | 0.0021 | **96.5%** | yes |
| 95% | 0.0024 | **99.5%** | yes |

At or above target every time, exactly as the finite-sample guarantee requires: it is a lower bound, so the
over-coverage is expected from a whole-field (worst-pixel) score.

## The variants and the visualization

The chips are the coverage targets. For one held-out field the case bakes four maps: the FNO prediction, the
true pressure, the absolute error, and the **in-band mask** (1 where the band contains the truth). The mask
makes the guarantee visible: the fraction of in-band pixels tracks the target, and the readout reports the
achieved whole-field coverage over the test set.

## What this case does NOT claim

- **Not per-pixel or per-instance certainty.** Coverage is marginal. The band is a single width `q` sized for
  the hardest pixel, so it over-covers the easy interior; a normalised per-pixel score would tighten it, at
  the cost of a less transparent guarantee.
- **Not valid out of distribution.** Coverage holds only under exchangeability. A different coefficient family
  (different roughness, a different value pair) breaks it and voids the guarantee. That caveat is exactly what
  an operator surrogate needs attached to it, so the case states it rather than hiding it.
- **Not a correctness fix.** Conformal quantifies the operator's own error against the reference solver; it
  cannot make a wrong operator right.

## References

- Vovk, Gammerman, Shafer. *Algorithmic Learning in a Random World.* Springer 2005 (conformal prediction).
- Lei, G'Sell, Rinaldo, Tibshirani, Wasserman. *Distribution-Free Predictive Inference for Regression.* JASA 2018.
- Angelopoulos, Bates. *A Gentle Introduction to Conformal Prediction.* arXiv:2107.07511.
- Neural-operator conformal (2026): arXiv:2606.09923, 2606.17513.

Working notes and the pre-build spike: `wip/beyond-sota/spec-groundwater-and-conformal-2026-07-15.md`.
