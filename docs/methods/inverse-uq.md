# Inverse problems & uncertainty quantification

> **Method group `inverse-uq`** — recovering unknown coefficients, fields and sources from
> sparse/noisy data, and attaching honest error bars to the answer.
> First exercised in: `poll-air-source-inv` (real OpenAQ data), `poll-groundwater-rt`,
> `ind-heat2d-inverse`, `poll-source-uq-bpinn`.

---

## What this group is, and why PINNs shine here

A *forward* PDE problem is "I know the equation, coefficients, sources and boundary conditions —
give me the field." An *inverse* problem is the opposite and far more useful in practice: "I have a
handful of noisy sensor readings; tell me the coefficient / source / boundary I cannot measure
directly." Examples in PINN-Lab: where is the pollutant being emitted (`poll-air-source-inv`), what
is the spatially-varying hydraulic conductivity of an aquifer (`poll-groundwater-rt`), what is the
unknown thermal conductivity field of a part (`ind-heat2d-inverse`).

Inverse problems are **ill-posed**: many parameter fields explain the same sparse data, and small
data perturbations cause large parameter swings. The classical remedy is to add a regulariser. A
PINN's regulariser *is the physics*. By forcing the network to satisfy the governing PDE at every
collocation point in the domain — not just where sensors sit — the PDE residual fills in the
enormous gaps between measurements with physically admissible behaviour. This is exactly where PINNs
beat classical numerics, which they do **not** beat on raw forward-solve speed (a good FEM/FVM solver
is usually faster and more accurate for a single well-posed forward problem; the ocean-transport case
in this repo records ~1168 s PINN training vs ~0.26 s for an FDM forward solve). On inverse and
sparse-data assimilation the calculus flips: the same mesh-free, differentiable, PDE-constrained
machinery that is *overhead* on a forward solve becomes the *whole point*. The seminal demonstration
is Hidden Fluid Mechanics (Raissi, Yazdani & Karniadakis, *Science* 2020), which recovered full
velocity and pressure fields from nothing but snapshots of a passive dye concentration — quantities
never directly observed, reconstructed because the Navier–Stokes residual tied them to what *was*
observed.

But a point estimate of a hidden coefficient is dangerous without an error bar — an under-determined
inverse problem that reports a single confident number is lying by omission. Hence the second half of
this group: **uncertainty quantification (UQ)**. Vanilla (deterministic) PINNs are known to be badly
*overconfident* / under-covering on inverse problems (Psaros et al. 2023). The four methods below
range from the rigorous-but-expensive (B-PINN via HMC) to the cheap-but-jointly-overconfident (deep
ensembles), with EKI as the gradient-free middle ground.

---

## 1. Inverse PINNs via trainable coefficients (`dde.Variable`)

**What it is.** Promote the unknown PDE quantity to a *trainable parameter* and optimise it jointly
with the network weights against the combined loss
$\mathcal{L} = \mathcal{L}_{\text{PDE}} + \mathcal{L}_{\text{data}}$, where
$\mathcal{L}_{\text{data}}$ penalises mismatch at the sparse sensor points and
$\mathcal{L}_{\text{PDE}}$ enforces the governing equation everywhere. An unknown *scalar* (e.g. a
reaction rate $k$, a diffusivity $D$) becomes a single learnable constant; an unknown *field* (e.g.
$k(x)$, a source $S(x)$) becomes a *second network output* trained simultaneously with the solution
output. The PDE residual is what couples them: it propagates information from the few labelled points
out to the unknown coefficient through the physics.

**Key equation.** For a parametric operator $\mathcal{N}[u;\lambda]$ with unknown $\lambda$ and sparse
observations $\{(x_i, u_i^{\text{obs}})\}_{i=1}^{N_d}$, jointly minimise over network weights $\theta$
*and* $\lambda$:

$$
(\theta^\star,\lambda^\star) = \arg\min_{\theta,\lambda}\;
\underbrace{\frac{1}{N_r}\sum_{j=1}^{N_r}\big\|\mathcal{N}[u_\theta;\lambda](x_j)\big\|^2}_{\text{PDE residual (the regulariser, evaluated everywhere)}}
\;+\;
\underbrace{\frac{1}{N_d}\sum_{i=1}^{N_d}\big\|u_\theta(x_i)-u_i^{\text{obs}}\big\|^2}_{\text{data misfit (sparse)}}
$$

For the inverse advection–diffusion source problem in `poll-air-source-inv`,
$\mathcal{N}[c;S] = c_t + \mathbf{u}\cdot\nabla c - \nabla\cdot(D\nabla c) - S$, and the unknown is the
source field $S(x)$ (a second network output), recovered from OpenAQ station concentrations.

**Canonical reference.** Raissi, Perdikaris & Karniadakis, *Physics-informed neural networks…*,
J. Comput. Phys. **378** (2019) 686–707, DOI [10.1016/j.jcp.2019.05.045](https://doi.org/10.1016/j.jcp.2019.05.045)
(inverse formulation, §4). Hidden-field reconstruction: Raissi, Yazdani & Karniadakis, *Hidden Fluid
Mechanics*, *Science* **367** (2020) 1026–1030, DOI [10.1126/science.aaw4741](https://doi.org/10.1126/science.aaw4741).

**Framework / API.** **DeepXDE** (primary), with **TorchPhysics** as the Apache-licensed PyTorch
alternative for `poll-air-source-inv`. DeepXDE scalar-inverse pattern (verified against the
`reaction.inverse` demo):

```python
import deepxde as dde

kf = dde.Variable(0.05)          # unknown scalar, initial guess
D  = dde.Variable(1.0)

def pde(x, y):                   # use kf, D inside the residual
    ...

model.compile("adam", lr=1e-3, external_trainable_variables=[kf, D])
var_cb = dde.callbacks.VariableValue([kf, D], period=1000, filename="variables.dat")
model.train(iterations=80000, callbacks=[var_cb])
```

For an unknown *field*, use a multi-output net (`dde.nn.PFNN`) and supply the sparse observations as a
`dde.icbc.PointSetBC`; no `dde.Variable` is needed because the unknown lives in the network outputs.

**PINN-Lab cases.** `poll-air-source-inv` (source field $S$, real OpenAQ data),
`poll-groundwater-rt` (reactive coupling + transport parameters), `ind-heat2d-inverse` (conductivity
field $k(x)$), `mine-flotation-kinetics` (scalar rate constant $k$). The `real_or_synthetic` flag on
each case flips to `validated` only where wired to committed real data (OpenAQ, USGS NWIS); otherwise
it stays `synthetic-illustrative` and says so on the Benchmark page.

---

## 2. Bayesian PINNs (B-PINN)

**What it is.** Instead of one best-fit weight vector, treat the network weights $\theta$ (and any
unknown physical parameters $\lambda$) as *random variables* and infer their full posterior given the
data. The PDE residual and the noisy observations together define a likelihood; a prior over weights
regularises. Prediction becomes a *posterior predictive distribution*, so every output carries a
credible interval. B-PINN is the most principled UQ option here and the only one that propagates both
*aleatoric* (data-noise) and *epistemic* (model/weight) uncertainty in one consistent framework. Two
posterior estimators are offered in the original paper: **Hamiltonian Monte Carlo (HMC)** — more
reliable, the recommended default, but expensive — and **variational inference (VI)** — cheaper but
can underestimate variance.

**Key equation.** By Bayes' rule, the posterior over weights and parameters given data $\mathcal{D}$
(sparse observations + the requirement that the PDE residual be zero, both with Gaussian noise models):

$$
p(\theta,\lambda \mid \mathcal{D}) \;\propto\;
\underbrace{p(\mathcal{D}\mid\theta,\lambda)}_{\text{likelihood: data misfit + PDE residual}}\;
\underbrace{p(\theta)\,p(\lambda)}_{\text{priors}}
$$

with a typical likelihood

$$
p(\mathcal{D}\mid\theta,\lambda)\propto
\exp\!\Big(-\tfrac{1}{2\sigma_d^2}\textstyle\sum_i \|u_\theta(x_i)-u_i^{\text{obs}}\|^2
-\tfrac{1}{2\sigma_r^2}\textstyle\sum_j \|\mathcal{N}[u_\theta;\lambda](x_j)\|^2\Big).
$$

HMC then draws samples $\{\theta^{(s)},\lambda^{(s)}\}$ from this posterior; the predictive mean and
variance at a query $x$ are the sample mean and variance of $u_{\theta^{(s)}}(x)$.

**Canonical reference.** Yang, Meng & Karniadakis, *B-PINNs: Bayesian Physics-Informed Neural
Networks for Forward and Inverse PDE Problems with Noisy Data*, J. Comput. Phys. **425** (2021)
109913, arXiv:[2003.06097](https://arxiv.org/abs/2003.06097),
DOI [10.1016/j.jcp.2020.109913](https://doi.org/10.1016/j.jcp.2020.109913).

**Framework / API.** **NeuralUQ** (Crunch-UQ4MI — the companion library to the UQ review, implements
B-PINN/HMC/VI; SIAM/ASA J. UQ, DOI [10.1137/22M1518189](https://doi.org/10.1137/22M1518189)) is the
reference UQ engine; **NeuralPDE.jl** also ships a native BPINN solver (no ONNX→web path, so docs-only
here). For the PINN-Lab pipeline, the HMC sampling runs offline; only the *posterior predictive mean
field* (and optionally a baked variance field) is exported to the web lane — sampling itself never
runs client-side.

**PINN-Lab case.** `poll-source-uq-bpinn` (Bayesian source identification with credible intervals;
the headline honesty case, where a deterministic PINN's point estimate is shown to badly under-cover).

---

## 3. Deep ensembles

**What it is.** The cheap, embarrassingly-parallel UQ baseline: train $M$ independent PINNs from
different random initialisations (and optionally different collocation samples), then read the spread
of their predictions as the uncertainty. No sampler, no Bayesian machinery; each member is an ordinary
deterministic inverse PINN. It captures epistemic uncertainty arising from optimisation
non-uniqueness and is trivially parallel. **Honest limitation:** ensembles of PINNs can be *jointly
overconfident* — the members are pulled toward the same PDE-satisfying solutions and so agree even
where the data does not constrain the answer, under-covering the true uncertainty (documented in the
Psaros et al. review). Repulsive-ensemble and evidential variants partly address this by explicitly
diversifying members.

**Key equation.** With $M$ members $\{\theta^{(m)}\}_{m=1}^{M}$, the ensemble predictive mean and
(epistemic) variance at $x$ are

$$
\bar{u}(x)=\frac{1}{M}\sum_{m=1}^{M} u_{\theta^{(m)}}(x),
\qquad
\widehat{\sigma}^2(x)=\frac{1}{M-1}\sum_{m=1}^{M}\big(u_{\theta^{(m)}}(x)-\bar{u}(x)\big)^2 .
$$

For an inverse problem, the same statistics over the per-member recovered coefficient
$\lambda^{(m)}$ give a confidence band on the *parameter*.

**Canonical reference.** Lakshminarayanan, Pritzel & Blundell, *Simple and Scalable Predictive
Uncertainty Estimation using Deep Ensembles*, NeurIPS 2017, arXiv:[1612.01474](https://arxiv.org/abs/1612.01474)
(the ensembles-as-UQ foundation); analysed in the PINN setting by Psaros et al. (see below). Repulsive
ensembles for PINNs: arXiv:[2505.17308](https://arxiv.org/abs/2505.17308).

**Framework / API.** Any engine — implemented as an outer loop over **DeepXDE** trainings with
distinct seeds; **NeuralUQ** also wraps ensembles as a first-class UQ method. Cheap to add to any
inverse case already running on DeepXDE.

**PINN-Lab case.** `ind-heat2d-inverse` (ensemble UQ on the recovered conductivity field $k(x)$);
also the cheap UQ cross-check on `poll-air-source-inv`.

---

## 4. Ensemble Kalman Inversion (EKI)

**What it is.** A **gradient-free** Bayesian inference scheme for B-PINNs. Maintain an ensemble of
parameter vectors (network weights + physical parameters) and iteratively nudge them toward the data
using Kalman-style update steps driven only by the ensemble's empirical covariances — no
back-propagation through the sampler. Its cost is *linear* in the number of unknown parameters, so it
scales to large/overparameterised networks far better than HMC (whose per-sample cost and mixing
degrade badly in high dimensions), while still delivering an approximate posterior (hence error bars),
unlike a single deterministic fit. The trade-off: EKI's Gaussian-update assumptions make the
uncertainty *approximate* (it can collapse for strongly non-Gaussian posteriors), so it is the
pragmatic middle ground between rigorous-but-slow HMC and cheap-but-overconfident ensembles.

**Key equation.** Let $u_n^{(j)}$ be the $j$-th ensemble member's parameter vector at iteration $n$,
$\mathcal{G}$ the forward map (PINN residual + observation operator), $y$ the data with noise
covariance $\Gamma$. The EKI update is

$$
u_{n+1}^{(j)} = u_n^{(j)} + C^{up}_n\big(C^{pp}_n+\Gamma\big)^{-1}\Big(y - \mathcal{G}(u_n^{(j)})\Big),
$$

where $C^{pp}_n$ is the empirical covariance of the predicted outputs $\{\mathcal{G}(u_n^{(j)})\}$ and
$C^{up}_n$ the empirical cross-covariance between parameters and predictions across the ensemble — both
estimated from the members, so no derivative of $\mathcal{G}$ is ever required.

**Canonical reference.** Pensoneault & Zhu, *Efficient Bayesian Physics Informed Neural Networks for
inverse problems via Ensemble Kalman Inversion*, J. Comput. Phys. **508** (2024) 113006,
arXiv:[2303.07392](https://arxiv.org/abs/2303.07392),
DOI [10.1016/j.jcp.2024.113006](https://doi.org/10.1016/j.jcp.2024.113006). EKI foundations: Iglesias,
Law & Stuart, *Inverse Problems* **29** (2013) 045001.

**Framework / API.** No turnkey PINN library exposes EKI as a one-liner; it is implemented as an
offline ensemble-update loop wrapping a **DeepXDE** residual evaluation (the forward map
$\mathcal{G}$). Documented here as the gradient-free scaling option; in PINN-Lab it is an alternative
inference backend for the B-PINN case rather than its own case.

**PINN-Lab case.** `poll-source-uq-bpinn` (alternative gradient-free inference backend, contrasted
against HMC on the same source-identification problem).

---

## Honest limitations of the whole group

- **Inverse PINNs are still ill-posed.** The PDE prior reduces — does not remove — non-uniqueness. If
  the data carry no information about a coefficient (e.g. a source far from every sensor), no method
  here invents it; UQ should then report a *wide* band, and a deterministic PINN that reports a narrow
  one is wrong.
- **UQ calibration is not free.** Deterministic and ensembled PINNs systematically *under-cover* on
  inverse problems (Psaros et al. 2023). B-PINN/HMC is the most trustworthy but is the slowest and
  needs careful step-size/mass-matrix tuning; VI and EKI trade rigour for speed and can collapse
  variance on non-Gaussian posteriors.
- **Cost.** HMC scales poorly with parameter dimension (the motivation for EKI); ensembles multiply
  training cost by $M$. None of this runs client-side — all inference is offline and only the baked
  mean/variance fields ship to `onnxruntime-web`.
- **Real data is scarce.** Per the coverage map, only `poll-air-source-inv` (OpenAQ, CC BY 4.0) and
  the groundwater/seepage head pair (USGS NWIS, US public domain) are wired to *real* public data;
  every other inverse case is synthetic and labelled as such.

---

## References (consolidated)

1. Raissi, Perdikaris, Karniadakis. *Physics-informed neural networks.* J. Comput. Phys. 378 (2019) 686–707. DOI [10.1016/j.jcp.2019.05.045](https://doi.org/10.1016/j.jcp.2019.05.045)
2. Raissi, Yazdani, Karniadakis. *Hidden Fluid Mechanics.* Science 367 (2020) 1026–1030. DOI [10.1126/science.aaw4741](https://doi.org/10.1126/science.aaw4741)
3. Yang, Meng, Karniadakis. *B-PINNs.* J. Comput. Phys. 425 (2021) 109913. arXiv:[2003.06097](https://arxiv.org/abs/2003.06097) · DOI [10.1016/j.jcp.2020.109913](https://doi.org/10.1016/j.jcp.2020.109913)
4. Psaros, Meng, Zou, Guo, Karniadakis. *Uncertainty Quantification in Scientific Machine Learning.* J. Comput. Phys. 477 (2023) 111902. arXiv:[2201.07766](https://arxiv.org/abs/2201.07766) · NeuralUQ: DOI [10.1137/22M1518189](https://doi.org/10.1137/22M1518189)
5. Lakshminarayanan, Pritzel, Blundell. *Deep Ensembles.* NeurIPS 2017. arXiv:[1612.01474](https://arxiv.org/abs/1612.01474)
6. Pensoneault, Zhu. *Efficient Bayesian PINNs via Ensemble Kalman Inversion.* J. Comput. Phys. 508 (2024) 113006. arXiv:[2303.07392](https://arxiv.org/abs/2303.07392) · DOI [10.1016/j.jcp.2024.113006](https://doi.org/10.1016/j.jcp.2024.113006)
7. Iglesias, Law, Stuart. *Ensemble Kalman methods for inverse problems.* Inverse Problems 29 (2013) 045001.

See also `methods/optimizers.md` (the Adam→L-BFGS baseline every inverse fit uses) and the
`docs/frameworks/deepxde/` guide for the full inverse-problem workflow and the ONNX-export contract.
