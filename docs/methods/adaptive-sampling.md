# Adaptive sampling (RAR / RAR-G / RAR-D / RAD)

> Method group **A — Adaptive sampling** in the PINN-Lab SOTA catalogue.
> Residual-based adaptive refinement and distribution: *put collocation points where the PDE residual is large.*

## What it is

A physics-informed neural network (PINN) minimises the PDE residual on a finite set
of **collocation points** sampled inside the domain. The quality of the solution is bounded
by *where* those points sit: a uniform (or Latin-hypercube / Sobol / Halton) sample wastes
capacity on smooth regions and starves sharp features — shocks, interfaces, boundary layers —
exactly where the residual is worst and the solution most interesting. **Adaptive sampling**
closes that gap by moving or adding collocation points toward high-residual regions *during*
training, using the network's own residual field as the error indicator.

This is the cheapest, highest-leverage trick in the PINN toolbox: it touches only the
sampler, is orthogonal to loss-weighting (group C) and architecture (group D) choices, and
costs one extra forward pass over a dense candidate pool per refinement step. The canonical
reference is the comprehensive study of Wu, Zhu, Tan, Kartha & Lu (CMAME 2023,
[arXiv:2207.10289](https://arxiv.org/abs/2207.10289)), which benchmarked ten sampling schemes
across 6000+ runs and introduced the **RAD** and **RAR-D** distribution-based methods; the
**RAR-G** greedy variant predates it in the original DeepXDE paper
([Lu et al., SIAM Review 2021](https://doi.org/10.1137/19M1274067), [arXiv:1907.04502](https://arxiv.org/abs/1907.04502)).

### The four members of the family

All four share one error indicator — the absolute PDE residual at a point. Let the PINN
solution be $\hat{u}_\theta$ and the differential operator $\mathcal{F}$, so the residual on
the interior is

$$
r_\theta(\mathbf{x}) \;=\; \mathcal{F}\!\left[\hat{u}_\theta\right](\mathbf{x}),
\qquad
\varepsilon(\mathbf{x}) \;=\; \left| r_\theta(\mathbf{x}) \right|.
$$

The methods differ only in **how** the residual field $\varepsilon(\mathbf{x})$ is turned back
into a new point set.

---

### 1. RAR-G — Residual-based Adaptive Refinement, Greedy

The original RAR. At each refinement step, evaluate the residual on a large pool of fresh
candidate points $\mathcal{S}$, then **greedily add the $m$ points of highest residual** to the
training set; never remove points. The set grows monotonically and concentrates where the
residual is largest. Refinement stops when the mean residual over the pool drops below a
tolerance:

$$
\mathcal{X}^{(j+1)} \;=\; \mathcal{X}^{(j)} \,\cup\, \underset{\mathbf{x}\in\mathcal{S}}{\operatorname{top\text{-}}m}\;\varepsilon(\mathbf{x}),
\qquad
\text{stop when } \frac{1}{|\mathcal{S}|}\sum_{\mathbf{x}\in\mathcal{S}}\varepsilon(\mathbf{x}) < \tau .
$$

Greedy and deterministic: it pins points at the single worst locations. This is excellent for
a stationary sharp feature (a fixed shock) but can over-concentrate and leave the rest of the
domain under-sampled as the set grows.

- **Canonical reference:** Lu, Meng, Mao & Karniadakis, *DeepXDE: A Deep Learning Library for
  Solving Differential Equations*, SIAM Review 63(1), 2021 — [DOI:10.1137/19M1274067](https://doi.org/10.1137/19M1274067),
  [arXiv:1907.04502](https://arxiv.org/abs/1907.04502).
- **Framework:** **DeepXDE** — first-class via `data.add_anchors(...)` in a manual loop (see API below).
- **PINN-Lab case:** `bench-burgers1d` (shock formation) and `mine-thickener-settling`
  (sharp settling front) per the coverage map.

---

### 2. RAR-D — Residual-based Adaptive Refinement with Distribution

Introduced by Wu et al. (2023). Instead of greedily picking the top-$m$ residual points,
RAR-D **adds** a batch of new points *sampled from* a probability density proportional to a
power of the residual (see RAD below), then keeps accumulating. It is the stochastic,
distribution-driven cousin of RAR-G: the training set still grows monotonically (refinement),
but the new points are drawn from $p(\mathbf{x})$ rather than placed at the maxima. This avoids
RAR-G's over-concentration while retaining the cumulative-budget behaviour.

$$
\mathcal{X}^{(j+1)} \;=\; \mathcal{X}^{(j)} \,\cup\, \big\{\, \mathbf{x}_i \sim p_\theta(\mathbf{x}) \,\big\}_{i=1}^{m},
\qquad
p_\theta(\mathbf{x}) \;\propto\; \frac{\varepsilon^{k}(\mathbf{x})}{\mathbb{E}\!\left[\varepsilon^{k}(\mathbf{x})\right]} + c .
$$

- **Canonical reference:** Wu, Zhu, Tan, Kartha & Lu, *A comprehensive study of non-adaptive and
  residual-based adaptive sampling for physics-informed neural networks*, CMAME 403:115671, 2023 —
  [DOI:10.1016/j.cma.2022.115671](https://doi.org/10.1016/j.cma.2022.115671),
  [arXiv:2207.10289](https://arxiv.org/abs/2207.10289).
- **Framework:** **DeepXDE** (`add_anchors` with residual-weighted sampling; the reference
  implementation is published with the paper).
- **PINN-Lab case:** `bench-burgers1d` (RAR adaptive-sampling showcase), exercised alongside `bench-allencahn`.

---

### 3. RAD — Residual-based Adaptive Distribution

The headline method of Wu et al. (2023), and the strongest general default of the family. RAD
**resamples the entire collocation set** at each step (it does *not* accumulate) by drawing
$N$ fresh points from a density proportional to the residual:

$$
p_\theta(\mathbf{x}) \;\propto\; \frac{\varepsilon^{k}(\mathbf{x})}{\mathbb{E}\!\left[\varepsilon^{k}(\mathbf{x})\right]} + c ,
\qquad k \ge 0,\; c \ge 0 .
$$

Here $\mathbb{E}[\varepsilon^{k}(\mathbf{x})]$ is the mean of $\varepsilon^{k}$ over the domain
(estimated on the candidate pool), which normalises the residual term so the two hyperparameters
have an interpretable meaning:

- **$k$** controls how aggressively the density concentrates on high-residual regions
  ($k=0$ recovers uniform sampling; large $k$ → near-greedy mass on the maxima).
- **$c$** is a uniform floor that guarantees every region keeps *some* probability mass, so the
  network does not forget already-converged regions when it resamples.

Wu et al. report from their 6000+-run sweep that **$k=1,\,c=1$ is a robust default** across
problems, with the optimum being mildly problem-dependent. Because RAD resamples the whole set,
it tracks features that *move* during training (e.g. a forming shock, a propagating front) far
better than the accumulating refinement variants, and in the paper it gives the best accuracy /
point-budget trade-off overall.

- **Canonical reference:** Wu, Zhu, Tan, Kartha & Lu, CMAME 403:115671, 2023 —
  [DOI:10.1016/j.cma.2022.115671](https://doi.org/10.1016/j.cma.2022.115671),
  [arXiv:2207.10289](https://arxiv.org/abs/2207.10289).
- **Framework:** **DeepXDE** (full resample from the residual density; reference code with the paper).
- **PINN-Lab case:** `bench-allencahn` (moving interface, $u_t = \varepsilon^2 u_{xx} + u - u^3$),
  with `bench-burgers1d` as the secondary anchor.

---

## The unifying equation

Every distribution-based member (RAD, RAR-D) draws from the **same** residual-weighted density;
the greedy member (RAR-G) is the $k \to \infty,\; c=0$ deterministic limit of it. The single
governing object is therefore

$$
\boxed{\;p_\theta(\mathbf{x}) \;\propto\; \dfrac{\big|\mathcal{F}[\hat{u}_\theta](\mathbf{x})\big|^{k}}{\mathbb{E}\big[\,|\mathcal{F}[\hat{u}_\theta](\mathbf{x})|^{k}\,\big]} + c\;}
$$

| Method | New points drawn from | Set grows? | Limit / relation |
|---|---|---|---|
| **RAR-G** | $\operatorname{top\text{-}}m\;\varepsilon(\mathbf{x})$ (greedy) | yes (accumulate) | $k\to\infty,\,c=0$ of $p_\theta$ |
| **RAR-D** | $p_\theta(\mathbf{x})$ | yes (accumulate) | distribution-driven refinement |
| **RAD** | $p_\theta(\mathbf{x})$ | no (full resample) | tracks moving features best |

(Uniform / Sobol / Halton / Latin-hypercube non-adaptive sampling is the $k=0$ baseline against
which all three are measured in the paper.)

---

## Why / when to use it

- **Sharp gradients, shocks, interfaces, boundary layers.** This is the primary win: viscous
  Burgers develops a near-discontinuity, Allen–Cahn forms a thin diffuse interface, convection-
  dominated transport develops fronts. Uniform sampling under-resolves these; residual-based
  sampling automatically piles points there. Wu et al. show 1–2 orders of magnitude lower
  relative-$L^2$ error at equal point budget on exactly these problems.
- **Tight point budgets.** When each collocation point is expensive (high-order derivatives,
  large nets), spending points where they reduce error most is strictly better than spreading
  them uniformly.
- **As a free add-on.** It composes with everything else in PINN-Lab — loss weighting (NTK,
  grad-norm), causal training, hard constraints, Fourier features. Reach for it first on any
  case whose Benchmark page shows error localised at a feature.

**When *not* to bother:** smooth, low-frequency solutions (e.g. `bench-poisson2d` with a smooth
source) gain little — the residual is roughly uniform, so $p_\theta(\mathbf{x})$ is roughly
uniform and adaptive sampling reduces to the baseline. Use it where the residual field is
*peaked*, not where it is flat.

---

## Honest limitations

- **Candidate-pool cost.** Both refinement and resampling need a dense pool of candidate points
  to evaluate the residual on (DeepXDE's demo uses $10^5$). Each refinement step is an extra
  full forward pass over that pool; for very large nets or high dimension this is non-trivial,
  though still cheap relative to training.
- **Hyperparameter sensitivity.** $k$ and $c$ are problem-dependent. $k=c=1$ is the robust
  default, but a poor choice (e.g. large $k$, $c=0$) can over-concentrate, starve the rest of the
  domain, and destabilise training — the same failure mode as naive greedy RAR-G.
- **It fixes sampling, not optimisation.** Adaptive sampling cannot rescue a problem whose
  *loss landscape* is ill-conditioned (stiff Allen–Cahn, advection-dominated transport). Those
  need causal/curriculum training (group B) and better architectures (group D) on top; the
  paper itself stacks RAD with those, it does not replace them. The residual indicator is also a
  proxy — a low residual does **not** guarantee a low solution error, so an honest Benchmark
  page must still compare against an analytic / numerical reference, not just report the residual.
- **Resampling vs. refinement trade-off.** RAD's full resample tracks moving features but
  discards converged points each step (relying on the $c$ floor to retain them); RAR-G/RAR-D
  accumulate and never forget but can over-grow and over-concentrate. There is no universally
  best member — the paper's verdict is "RAD is the best general default", not "RAD always wins".
- **Greedy ≠ exploration.** Pure RAR-G is deterministic and can repeatedly target the same
  maximum, missing secondary features. The distribution methods exist precisely to fix this.

---

## How it maps into PINN-Lab

**Engine.** Adaptive sampling is implemented in the **DeepXDE** precompute lane
(`requirements-precompute.txt`, PyTorch backend). DeepXDE exposes RAR through
`Model.predict(X, operator=pde)` to get the residual field and `data.add_anchors(...)` to grow
the collocation set; RAD/RAR-D wrap that with the residual-weighted draw from $p_\theta$.

**Cases that exercise it (per the coverage map):**

| case | feature it resolves | member used |
|---|---|---|
| `bench-burgers1d` | viscous shock at $x=0$ | RAR-G / RAR-D (the canonical RAR demo) |
| `bench-allencahn` | thin moving diffuse interface | RAD (full resample tracks the front) |
| `mine-thickener-settling` 🟠 | sharp Kynch settling front | RAR (front localisation) |

**Pipeline stage.** It lives in the **`feature_extraction` → `train`** stages of
`pinnlab/stages/`: collocation sampling is part of feature extraction, and the refinement loop
runs inside training. The resulting fitted PINN is exported to ONNX exactly as any other case —
adaptive sampling changes *only* the training point set, never the exported network, so the
train→ONNX→onnxruntime-web contract is unaffected.

**Documentation honesty.** Because the residual is a proxy for error, the per-case Benchmark
page must report relative-$L^2$ against the analytic/numerical reference
(`burgers_shock.mat` / `AC.mat`, Raissi's MIT-licensed Chebfun spectral solutions — a
**numerical reference, not real data**), and show the point-budget vs. error curve that adaptive
sampling improves — not the residual alone.

### Minimal DeepXDE RAR loop (verified API)

The greedy RAR-G loop from the DeepXDE Burgers demo
([deepxde.readthedocs.io](https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/burgers.rar.html)):

```python
import deepxde as dde
import numpy as np

# ... define geomtime, pde, data, net, model and pre-train as usual ...

X = geomtime.random_points(100000)          # dense candidate pool
err = 1.0
while err > 0.005:
    f = model.predict(X, operator=pde)        # residual field on the pool
    err_eq = np.absolute(f)
    err = np.mean(err_eq)
    print("Mean residual: %.3e" % err)

    x_id = np.argmax(err_eq)                   # RAR-G: highest-residual point
    print("Adding new point:", X[x_id], "\n")
    data.add_anchors(X[x_id])                  # grow the collocation set

    early_stopping = dde.callbacks.EarlyStopping(min_delta=1e-4, patience=2000)
    model.compile("adam", lr=1e-3)
    model.train(iterations=10000, disregard_previous_best=True,
                callbacks=[early_stopping])
    model.compile("L-BFGS")
    losshistory, train_state = model.train()
```

For **RAD / RAR-D**, replace the `np.argmax` line with a draw of `m` points from the
residual-weighted density
$p_\theta(\mathbf{x}) \propto \varepsilon^{k}(\mathbf{x}) / \mathbb{E}[\varepsilon^{k}(\mathbf{x})] + c$
(with $k=c=1$), then `data.add_anchors(...)` (RAR-D, accumulate) or
`data.replace_with_anchors(...)` / rebuild the point set (RAD, full resample). The reference
implementations of RAD and RAR-D are published with [Wu et al. 2023](https://github.com/lu-group/sampling-pinn).

---

## Key references

1. **C. Wu, M. Zhu, Q. Tan, Y. Kartha, L. Lu.** *A comprehensive study of non-adaptive and
   residual-based adaptive sampling for physics-informed neural networks.* Computer Methods in
   Applied Mechanics and Engineering (CMAME) **403**, 115671, 2023.
   [DOI:10.1016/j.cma.2022.115671](https://doi.org/10.1016/j.cma.2022.115671) ·
   [arXiv:2207.10289](https://arxiv.org/abs/2207.10289). — *RAD, RAR-D; the $p(\mathbf{x})$ density;
   $k=c=1$ default; the 10-method / 6000-run benchmark.*
2. **L. Lu, X. Meng, Z. Mao, G. E. Karniadakis.** *DeepXDE: A Deep Learning Library for Solving
   Differential Equations.* SIAM Review **63**(1):208–228, 2021.
   [DOI:10.1137/19M1274067](https://doi.org/10.1137/19M1274067) ·
   [arXiv:1907.04502](https://arxiv.org/abs/1907.04502). — *original RAR (RAR-G) and the
   `add_anchors` API.*
3. **DeepXDE documentation** — Burgers equation with residual-based adaptive refinement:
   [deepxde.readthedocs.io/.../burgers.rar.html](https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/burgers.rar.html).
   — *the verified `model.predict(operator=pde)` + `data.add_anchors` loop.*
