# poll-source-uq-bpinn — Bayesian PINN (deep ensemble) with epistemic uncertainty

The case that answers *"how sure is the PINN?"* — it exercises the **uncertainty-quantification** method family. A
single PINN returns one field with no error bars; this one returns a **predictive mean and a calibrated uncertainty**.

## Problem

A dissolved pollutant diffuses in 1D, $c_t = D\,c_{xx}$ on $x\in[0,1]$, $t\in[0,1]$, with $c=0$ at the walls and the
analytic field $c^*(x,t)=e^{-D\pi^2 t}\sin(\pi x)$ ($D=0.1$). We are given **only 24 sparse, noisy sensor readings**
— not the full initial condition — so the inverse field is genuinely under-determined away from the sensors.

## Method — deep ensemble ≈ Bayesian

We train **K = 5 independently-initialized PINNs** (Lakshminarayanan et al. 2017: deep ensembles are the cheap,
well-calibrated approximation to Bayesian inference). Diversity comes from two sources:

- **random initialization** (distinct seed per member), and
- **bagging** — each member bootstraps its own subset of sensors from a shared pool, with its own noise realization.

Where the sensors constrain the field, the members agree; where data is sparse, they disagree — and **that
disagreement is the epistemic uncertainty**. The predictive mean is the ensemble average; the uncertainty is the
ensemble standard deviation.

## One ONNX, two outputs

The whole ensemble is wrapped in a single torch module that emits **`[mean, std]`** and exported as **one
self-contained ONNX** (the [prebuilt-engine](../architecture/staged-pipeline.md) path: the case trains its own net, so
the generic Adam→L-BFGS loop is skipped, but export + parity + the gate still apply). The live lane therefore ships a
single 101 KB file that returns both the field and its error bars in one pass.

## Result (measured, seed 42)

| metric | value |
|--------|-------|
| mean relative-L2 vs $c^*$ | **1.2 %** |
| 2σ calibration (truth within mean ± 2 std) | **100 %** (well-calibrated, slightly conservative) |
| mean / max ensemble std | 0.0068 / 0.0186 (uncertainty ~2.7× higher in data-sparse regions) |
| lane | **live** (101 KB, 3.8 ms, parity 2.4e-7) |

The mean is accurate, and the uncertainty is honest: it is small near the 24 sensors and the $c=0$ walls, and grows
where the field is unconstrained — the entire point of a Bayesian PINN. `c_std` is baked as a second output channel
in the field trace and reported via `extra_metrics` (calibration, mean/max std); the App renders the mean field.

`real_or_synthetic = synthetic-illustrative` — a UQ demonstrator on a manufactured field, not a measured dataset.
