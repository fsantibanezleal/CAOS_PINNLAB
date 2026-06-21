# Loss / gradient weighting

## What this group is

A physics-informed neural network is trained against a **composite loss** that sums several
heterogeneous terms — the PDE residual on interior collocation points, plus boundary-condition
(BC), initial-condition (IC), and (for inverse problems) data-misfit terms:

$$
\mathcal{L}(\theta) \;=\; \lambda_r\,\mathcal{L}_r(\theta) \;+\; \lambda_{bc}\,\mathcal{L}_{bc}(\theta) \;+\; \lambda_{ic}\,\mathcal{L}_{ic}(\theta) \;+\; \lambda_{data}\,\mathcal{L}_{data}(\theta),
$$

where each $\mathcal{L}_k = \frac{1}{N_k}\sum_i \big| \,\cdot\, \big|^2$ is a mean-squared term and the
$\lambda_k$ are scalar (or per-point) weights. The methods in this group answer one question:
**how do you choose the $\lambda_k$?**

This is not a cosmetic hyperparameter. The terms are physically and dimensionally different —
$\mathcal{L}_r$ measures how well a *differential operator* is annihilated (it involves first and
second derivatives of the network, so its scale depends on the PDE coefficients and the domain
size), whereas $\mathcal{L}_{bc}$ measures a *function-value* mismatch on a lower-dimensional set.
When these terms enter a single sum with equal weights, the optimizer minimises whichever term
produces the largest back-propagated gradient and effectively ignores the others. The empirical
symptom is a network that drives the residual to near-zero in the interior but **ignores the
boundary conditions**, yielding a smooth field that solves the PDE but is the *wrong* solution
(the BCs are what select the unique solution). Equivalently, the inverse failure: the BCs are
satisfied but the interior residual stagnates.

### Why residual vs BC/IC terms need balancing (the mechanism)

Two complementary diagnoses explain the imbalance, and each motivates one weighting scheme:

1. **Gradient pathology (Wang, Teng & Perdikaris 2020).** Numerical stiffness in the residual
   term makes its back-propagated gradient $\nabla_\theta \mathcal{L}_r$ dominate
   $\nabla_\theta \mathcal{L}_{bc}$ by one to two orders of magnitude. During gradient descent the
   parameter update is driven almost entirely by the residual, so the BC term never converges.

2. **NTK spectral imbalance (Wang, Yu & Perdikaris 2020).** Through the Neural Tangent Kernel
   lens, each loss term converges at a rate governed by the eigenvalues of *its own* NTK block.
   Those eigenvalue spectra differ by orders of magnitude between the residual and BC operators,
   so under equal weights the terms converge at radically different rates — exactly the observed
   pathology, now with a precise spectral cause.

In practice the cheapest fix is a **heuristic constant multiplier** on the BC/IC term (PINN-Lab's
lid-driven-cavity case uses a $10\times$ weight on the boundary loss — see
[`bench-navier-cavity`](../cases/bench-navier-cavity.md)). That works but is fragile and
problem-specific. The four methods below replace the hand-tuned constant with **principled,
adaptive** weighting.

---

## NTK weighting

**One-paragraph explanation.** Wang, Yu & Perdikaris analyse PINN training through the Neural
Tangent Kernel and prove that, in the infinite-width limit, the kernel converges to a deterministic
operator whose block structure ties each loss term to its own convergence rate. They show the
discrepancy between the residual and BC NTK eigenvalue spectra is the *cause* of the training
failure, and propose setting each term's weight so that all terms converge at a **balanced rate**.
Concretely, the weight of term $k$ is set from the trace of its NTK block relative to the total
trace, which equalises the effective learning rates. This is the most *principled* (non-heuristic)
weighting scheme in the group: it is derived from the training dynamics rather than from gradient
bookkeeping, at the cost of computing (an approximation of) the NTK each update.

**Key equation.** Writing $\mathbf{K}_{rr}$, $\mathbf{K}_{bc\,bc}$ for the diagonal NTK blocks of
the residual and boundary terms, the balanced weights are

$$
\lambda_r \;=\; \frac{\operatorname{Tr}\mathbf{K}}{\operatorname{Tr}\mathbf{K}_{rr}},
\qquad
\lambda_{bc} \;=\; \frac{\operatorname{Tr}\mathbf{K}}{\operatorname{Tr}\mathbf{K}_{bc\,bc}},
\qquad
\operatorname{Tr}\mathbf{K} = \operatorname{Tr}\mathbf{K}_{rr} + \operatorname{Tr}\mathbf{K}_{bc\,bc},
$$

so a term whose NTK block has a *small* trace (slow convergence) is *up-weighted*. The weights are
recomputed (or refreshed every $N$ steps) as the kernel drifts during training.

**Canonical reference.** S. Wang, X. Yu, P. Perdikaris, *"When and why PINNs fail to train: A
neural tangent kernel perspective,"* Journal of Computational Physics **449** (2022) 110768.
Preprint: [arXiv:2007.14527](https://arxiv.org/abs/2007.14527).

**Framework that implements it.** **jaxpi** (`PredictiveIntelligenceLab/jaxpi`) — the reference
implementation of NTK weighting, with the kernel trace estimated per term inside the JAX training
loop. It is the technique source of truth in PINN-Lab (not a shipping engine, because JAX→ONNX
export is fragile).

**PINN-Lab case that exercises it.** [`bench-navier-cavity`](../cases/bench-navier-cavity.md) —
the steady lid-driven cavity (incompressible Navier–Stokes), where the divergence-free constraint,
the no-slip walls, and the moving-lid BC must be balanced against the momentum residual. This is
the hard NS benchmark where NTK weighting (paired with the [modified-MLP](modified-mlp.md)
backbone) earns its cost.

---

## Gradient-pathology / grad-norm weighting (learning-rate annealing)

**One-paragraph explanation.** This is the first widely adopted *automatic* loss-weighting scheme
and the cheapest. Wang, Teng & Perdikaris identify the failure as **unbalanced back-propagated
gradients**: the residual term's gradient swamps the BC term's. Their remedy — "learning-rate
annealing" — rescales each term's weight from running **gradient-magnitude statistics** so that the
gradients of all terms have comparable scale. At each update (or every few updates) the target
weight for term $k$ is the ratio of the maximum residual-gradient magnitude to the mean
gradient magnitude of term $k$; the live weight is then an exponential moving average toward that
target. It is noisier than NTK weighting (it equalises *gradients* rather than *convergence rates*)
but costs only one extra backward pass per balanced term, and it lifted reported accuracy by
$50$–$100\times$ in the original paper.

**Key equation.** With $\bar{\lambda}_k$ the instantaneous target and $\alpha$ a moving-average
factor (e.g. $0.9$),

$$
\hat{\lambda}_k \;=\; \frac{\displaystyle\max_{\theta}\big|\nabla_\theta \mathcal{L}_r(\theta)\big|}{\displaystyle\overline{\big|\nabla_\theta \mathcal{L}_k(\theta)\big|}},
\qquad
\lambda_k \;\leftarrow\; \alpha\,\lambda_k \;+\; (1-\alpha)\,\hat{\lambda}_k ,
$$

where $\max|\nabla_\theta\mathcal{L}_r|$ is the max over parameters of the residual-term gradient
and $\overline{|\nabla_\theta\mathcal{L}_k|}$ is the mean-magnitude of the term-$k$ gradient. A term
with small gradients (e.g. the BC term) is up-weighted until its gradient scale matches the
residual's.

**Canonical reference.** S. Wang, Y. Teng, P. Perdikaris, *"Understanding and mitigating gradient
pathologies in physics-informed neural networks,"* SIAM Journal on Scientific Computing **43**(5)
(2021) A3055–A3081. Preprint: [arXiv:2001.04536](https://arxiv.org/abs/2001.04536).

**Framework that implements it.** **jaxpi** (grad-norm weighting alongside NTK weighting); also
**PINA** exposes gradient-based adaptive weighting. The scheme is simple enough to drop into any
custom PyTorch/DeepXDE loop by computing per-term gradient norms.

**PINN-Lab case that exercises it.** [`poll-air-source-inv`](../cases/poll-air-source-inv.md) — the
atmospheric advection–diffusion **inverse source-localisation** problem, where the sparse-sensor
data term, the PDE residual, and the BCs have wildly different gradient scales and must be balanced
for the unknown source $S$ to be recovered. This case consumes **real** public data (OpenAQ +
EPA AQS/AirNow), so robust gradient balancing matters for a noisy, ill-posed inverse.

---

## Self-Adaptive PINNs (SA-PINN)

**One-paragraph explanation.** Instead of one scalar weight per loss *term*, SA-PINN attaches a
trainable weight to **every individual training point** and trains those weights by gradient
**ascent** while the network weights descend — a minimax (saddle-point) problem. The point weights
act as a **soft self-attention mask**: a point whose residual stays large gets its weight pushed up,
so the optimiser is forced to attend to the hardest regions (sharp fronts, boundary layers, shock
locations) without the user having to know in advance where they are. The mechanism is principled
through the NTK: McClenny & Braga-Neto show SA-PINN produces a *smooth equalisation of the NTK
eigenvalues* across loss terms — the same spectral cure as NTK weighting, but learned per point and
end-to-end rather than computed from the kernel.

**Key equation.** With per-point mask weights $\lambda_r^{(i)} \ge 0$ on the residual points (and
analogous masks on BC/IC points), the objective is the minimax

$$
\min_{\theta}\;\max_{\boldsymbol{\lambda}\ge 0}\;
\mathcal{L}(\theta,\boldsymbol{\lambda})
\;=\;
\frac{1}{N_r}\sum_{i=1}^{N_r} m\!\big(\lambda_r^{(i)}\big)\,\big| r(x_i;\theta)\big|^2
\;+\;
\frac{1}{N_{bc}}\sum_{j} m\!\big(\lambda_{bc}^{(j)}\big)\,\big| u(x_j;\theta) - g_j\big|^2 ,
$$

where $r(\cdot)$ is the PDE residual, $g_j$ the BC target, and $m(\cdot)$ a non-negative,
monotonically increasing **mask function** (e.g. $m(\lambda)=\lambda$ or $m(\lambda)=\lambda^2$).
The masks are updated by gradient ascent, $\lambda^{(i)} \leftarrow \lambda^{(i)} + \eta_\lambda\,
\partial \mathcal{L}/\partial \lambda^{(i)}$, so weights grow where the loss is large.

**Canonical reference.** L. McClenny, U. Braga-Neto, *"Self-Adaptive Physics-Informed Neural
Networks using a Soft Attention Mechanism,"* Journal of Computational Physics **474** (2023) 111722.
Preprint: [arXiv:2009.04544](https://arxiv.org/abs/2009.04544).

**Framework that implements it.** **PINA** (`SelfAdaptivePINN` solver) is the production path used
by PINN-Lab; the authors' reference TensorFlow 2 repository is cited for provenance.

**PINN-Lab case that exercises it.** [`mine-sag-thermal`](../cases/mine-sag-thermal.md) — the
SAG-mill operability case, a hybrid supervised + physics problem with heterogeneous, partly
data-driven terms where per-point self-adaptive weights let the optimiser auto-focus on the
hard operating-region points. (This is one of the synthetic / illustrative mining cases — labelled
`synthetic` on its Benchmark page.)

---

## gPINN (gradient-enhanced)

**One-paragraph explanation.** gPINN is a *constraint-enrichment* method rather than a weight-tuning
method, but it lives in this group because it adds new weighted terms to the composite loss and
shifts the residual-vs-BC balance. The observation: if the PDE residual $r(x;\theta)$ is identically
zero at the true solution, then its **gradient** $\nabla_x r$ is also identically zero. gPINN adds
that gradient as extra supervision — for each spatial dimension $i$ a term penalising
$\partial r/\partial x_i$ — which tightens the residual in a Sobolev (derivative-inclusive) norm and
typically reaches the same accuracy with **fewer collocation points**. The extra terms come with
their own weights $w_{g_i}$ that must be balanced against the base residual and BC terms (so gPINN
is most effective combined with adaptive weighting and with [RAR adaptive
sampling](adaptive-sampling.md) on steep-gradient solutions, where the extra derivative constraint
pays off most). The cost is real: each gradient term raises the AD order by one, increasing
per-step compute.

**Key equation.** The gPINN loss augments the standard composite loss with one residual-gradient
term per spatial dimension $d$:

$$
\mathcal{L} \;=\; w_f\,\mathcal{L}_f \;+\; w_b\,\mathcal{L}_b \;+\; w_i\,\mathcal{L}_i
\;+\; \sum_{i=1}^{d} w_{g_i}\,\mathcal{L}_{g_i},
\qquad
\mathcal{L}_{g_i} \;=\; \frac{1}{N}\sum_{n=1}^{N}\left|\frac{\partial\, r(x_n;\theta)}{\partial x_i}\right|^2 ,
$$

where $\mathcal{L}_f$ is the standard residual loss, $\mathcal{L}_b,\mathcal{L}_i$ the BC/IC losses,
and each $\mathcal{L}_{g_i}$ drives the residual's $i$-th partial derivative toward zero.

**Canonical reference.** J. Yu, L. Lu, X. Meng, G. E. Karniadakis, *"Gradient-enhanced
physics-informed neural networks for forward and inverse PDE problems,"* Computer Methods in Applied
Mechanics and Engineering **393** (2022) 114823.
Preprint: [arXiv:2111.02801](https://arxiv.org/abs/2111.02801).

**Framework that implements it.** **DeepXDE** (the lu-group `gpinn` reference accompanies the
paper); **PINA** exposes a `GradientPINN` variant.

**PINN-Lab case that exercises it.** [`bench-burgers1d`](../cases/bench-burgers1d.md) — 1D viscous
Burgers, where the shock-forming steep gradient is exactly the regime gPINN targets, paired with
RAR adaptive sampling. Also used in [`mine-comminution-pbe`](../cases/mine-comminution-pbe.md)
(integro-differential population balance).

---

## How this group maps into PINN-Lab

| method | key idea | weight granularity | cost | reference | framework | first case |
|---|---|---|---|---|---|---|
| NTK weighting | balance NTK convergence rates | per term | high (NTK trace) | [arXiv:2007.14527](https://arxiv.org/abs/2007.14527) | jaxpi | `bench-navier-cavity` |
| grad-norm / LR-annealing | equalise gradient magnitudes | per term | low (extra backward) | [arXiv:2001.04536](https://arxiv.org/abs/2001.04536) | jaxpi, PINA | `poll-air-source-inv` |
| SA-PINN | minimax soft-attention mask | per **point** | medium (mask params) | [arXiv:2009.04544](https://arxiv.org/abs/2009.04544) | PINA | `mine-sag-thermal` |
| gPINN | add residual-gradient terms | per term (+new terms) | high (extra AD order) | [arXiv:2111.02801](https://arxiv.org/abs/2111.02801) | DeepXDE, PINA | `bench-burgers1d` |

**Where it lives in the pipeline.** Loss weighting is configured in the `train` stage
(`pinnlab/stages/train.py`) per case — the `Case` dataclass carries the chosen weighting scheme in
its `method` field. The baseline for every case is the cheap **heuristic constant** (e.g. the
$10\times$ BC weight in the cavity case); cases flagged in the [coverage matrix](../cases/README.md)
upgrade to NTK, grad-norm, or SA-PINN where the imbalance is severe enough to warrant the cost.

**Honest limitations of the whole group.**

- **None of these guarantee convergence.** They fix the *relative* scaling of loss terms; they do
  not cure the deeper ill-conditioning of the residual loss landscape (that needs
  [causal/curriculum training](causal-training.md), better [architectures](modified-mlp.md), or
  [second-order optimisers](optimizers.md)). Weighting is necessary, not sufficient.
- **NTK weighting is expensive.** Computing or approximating the NTK trace each refresh is costly on
  wide networks; in practice it is refreshed every $N$ steps, trading accuracy of the balance for
  speed.
- **grad-norm is noisy.** Equalising gradient magnitudes is a coarser proxy than equalising
  convergence rates; the EMA smoothing helps but the weights can oscillate.
- **SA-PINN multiplies parameters.** One trainable weight per collocation point is a large extra
  parameter set, and the minimax objective is harder to optimise (saddle-point dynamics) than a
  plain minimisation; the Gaussian-process map of weights mitigates but adds machinery.
- **gPINN raises cost super-linearly.** Each residual-gradient term needs one more order of
  automatic differentiation; on a second-order PDE the gradient term needs third derivatives, which
  is expensive and can be numerically delicate. Its benefit is concentrated on steep-gradient
  solutions and shrinks (or reverses) on smooth ones.
- **The exported ONNX is unaffected.** Loss weighting changes *training only*; the baked
  `.onnx` is the same plain feed-forward net regardless of the scheme, so this group imposes no
  constraint on the train→ONNX→web contract.

### References (consolidated)

1. S. Wang, X. Yu, P. Perdikaris, "When and why PINNs fail to train: A neural tangent kernel
   perspective," *J. Comput. Phys.* **449** (2022) 110768. [arXiv:2007.14527](https://arxiv.org/abs/2007.14527).
2. S. Wang, Y. Teng, P. Perdikaris, "Understanding and mitigating gradient pathologies in
   physics-informed neural networks," *SIAM J. Sci. Comput.* **43**(5) (2021) A3055–A3081.
   [arXiv:2001.04536](https://arxiv.org/abs/2001.04536).
3. L. McClenny, U. Braga-Neto, "Self-Adaptive Physics-Informed Neural Networks using a Soft
   Attention Mechanism," *J. Comput. Phys.* **474** (2023) 111722.
   [arXiv:2009.04544](https://arxiv.org/abs/2009.04544).
4. J. Yu, L. Lu, X. Meng, G. E. Karniadakis, "Gradient-enhanced physics-informed neural networks for
   forward and inverse PDE problems," *Comput. Methods Appl. Mech. Engrg.* **393** (2022) 114823.
   [arXiv:2111.02801](https://arxiv.org/abs/2111.02801).
