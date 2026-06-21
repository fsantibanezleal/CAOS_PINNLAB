# Optimization — Adam→L-BFGS two-stage + gradient-conflict second-order (SOAP / ConFIG)

> Method group · the optimizer recipe that trains *every* PINN-Lab case.
> Part of the PINN-Lab `docs/methods/` set. Companion pages: `adaptive-sampling`, `causal-training`, `ntk-weighting`, `gradnorm-weighting`, `self-adaptive`.

This page covers the **optimization** group of the §4 method catalogue: the reliable two-stage **Adam → L-BFGS** recipe used in every case, and the 2025-SOTA **second-order / gradient-alignment** family (SOAP, ConFIG, Dual Cone Gradient Descent) that targets the deeper pathology — the directional *conflict* between the PDE-residual gradient and the boundary/initial gradients. It explains why the PINN residual loss is stiff and ill-conditioned enough that first-order gradient descent alone underperforms, what each method does about it, the key equation, the canonical reference, which framework implements it, and which PINN-Lab case exercises it.

---

## Why the PINN residual loss needs second-order polish

A PINN minimizes a composite loss in which the dominant term is the squared PDE residual evaluated by automatic differentiation of the network. For a generic PDE $\mathcal{N}[u](x)=0$ on $\Omega$ with conditions $\mathcal{B}_k[u]=g_k$ on $\partial\Omega$,

$$
\mathcal{L}(\theta) \;=\; \underbrace{\frac{1}{N_r}\sum_{i=1}^{N_r}\big|\mathcal{N}[u_\theta](x_i)\big|^2}_{\mathcal{L}_r:\ \text{PDE residual}} \;+\; \sum_k \lambda_k\,\underbrace{\frac{1}{N_k}\sum_{j}\big|\mathcal{B}_k[u_\theta](x_j)-g_k(x_j)\big|^2}_{\mathcal{L}_{b,k}:\ \text{BC/IC terms}} .
$$

This loss is much harder to optimize than a standard supervised loss, for two compounding reasons:

**1. The residual loss is stiff and ill-conditioned.** $\mathcal{N}$ contains differential operators, so $\mathcal{L}_r$ depends on *derivatives* of $u_\theta$ with respect to its inputs. A diffusion term $u_{xx}$, for example, turns the loss into a function of second input-derivatives of the network; differentiating that with respect to the weights $\theta$ produces a Hessian whose eigenvalue spectrum spans many orders of magnitude. The loss landscape is a long, narrow, curved valley. First-order methods (plain gradient descent, Adam) take tiny steps along the stiff directions and zig-zag, so they stall at a moderate accuracy floor — typically relative-$L^2$ around $10^{-2}$–$10^{-3}$ — well above what the problem actually admits. Krishnapriyan et al. (NeurIPS 2021) showed this ill-conditioning *is* the failure mode for stiff/advection-dominated PINNs, not a lack of network capacity ([arXiv:2109.01050](https://arxiv.org/abs/2109.01050)).

**2. The loss terms conflict.** $\mathcal{L}_r$ and the $\mathcal{L}_{b,k}$ are a **multi-task** objective. Empirically their gradients $g_r=\nabla_\theta\mathcal{L}_r$ and $g_b=\nabla_\theta\mathcal{L}_{b,k}$ are (i) wildly different in magnitude — the residual gradient is routinely tens to hundreds of times larger than the boundary gradient (Wang, Teng & Perdikaris, [arXiv:2001.04536](https://arxiv.org/abs/2001.04536)) — and (ii) frequently *negatively correlated*, $g_r\!\cdot\!g_b<0$, so a step that reduces the residual increases the boundary error and vice-versa (Hwang et al., [arXiv:2409.18426](https://arxiv.org/abs/2409.18426)). A single averaged gradient then makes little net progress on either.

Two families of fix follow directly. **Loss-weighting** methods (NTK, grad-norm, self-adaptive — separate pages) rescale the $\lambda_k$ to fix the *magnitude* imbalance. The **optimization** group on this page attacks the *curvature and direction*: L-BFGS uses an implicit Hessian approximation to step correctly through the stiff valley; SOAP/Shampoo precondition with a structured second-order estimate; ConFIG and Dual Cone Gradient Descent geometrically *de-conflict* the combined update so it never increases any single loss term. These are complementary — PINN-Lab layers a weighting scheme and an optimizer in the same run.

---

## Method 1 — Adam → L-BFGS (the reliable two-stage recipe)

**Explanation.** The standard, near-universal recipe runs training in two stages. **Stage 1 — Adam** (Kingma & Ba, [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)): a first-order, per-parameter adaptive-moment method that is robust to the initial chaotic landscape and explores globally on minibatches or the full collocation set; it gets the network into the right basin but plateaus on the stiff valley. **Stage 2 — L-BFGS** (Liu & Nocedal, *Math. Program.* 1989, [doi:10.1007/BF01589116](https://doi.org/10.1007/BF01589116)): a full-batch quasi-Newton method that builds a limited-memory inverse-Hessian approximation from the last $m$ gradient/step pairs, so it follows the curved valley floor and converges to a much higher accuracy (often 1–3 extra digits of relative-$L^2$). The order matters: L-BFGS from a cold start lands in a poor local minimum, while Adam-then-L-BFGS reliably beats either alone. This is precisely *because* the residual loss is stiff — the quasi-Newton step is what the ill-conditioning demands, and the dossier records it is used in **every** PINN-Lab case as the baseline optimizer.

**Key equation.** Adam's update with first/second moment estimates $m_t,v_t$ (bias-corrected $\hat m_t,\hat v_t$):

$$
\theta_{t+1} = \theta_t - \eta\,\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon},\qquad
m_t=\beta_1 m_{t-1}+(1-\beta_1)g_t,\quad v_t=\beta_2 v_{t-1}+(1-\beta_2)g_t^2 .
$$

L-BFGS takes a quasi-Newton step with a limited-memory inverse-Hessian approximation $H_t$ assembled from the last $m$ pairs $(s_k,y_k)=(\theta_{k+1}-\theta_k,\ g_{k+1}-g_k)$:

$$
\theta_{t+1}=\theta_t-\alpha_t\,H_t\,g_t,\qquad
H_t \ \text{built by the two-loop recursion over}\ \{(s_k,y_k)\}_{k=t-m}^{t-1},
$$

with $\alpha_t$ from a line search. The $H_t g_t$ product is the second-order correction that a first-order method lacks.

**Canonical reference.** Kingma & Ba, *Adam: A Method for Stochastic Optimization*, ICLR 2015, [arXiv:1412.6980](https://arxiv.org/abs/1412.6980); Liu & Nocedal, *On the limited memory BFGS method for large scale optimization*, *Mathematical Programming* 45:503–528, 1989, [doi:10.1007/BF01589116](https://doi.org/10.1007/BF01589116). The two-stage pairing for PINNs originates with Raissi, Perdikaris & Karniadakis, *JCP* 2019, [doi:10.1016/j.jcp.2019.05.045](https://doi.org/10.1016/j.jcp.2019.05.045).

**Framework.** First-class in essentially every engine: **DeepXDE** (`model.compile("adam")` then `model.compile("L-BFGS")`), **PINA** (chained optimizers), **NeuralPDE.jl** (`Adam` then `LBFGS(BackTracking())` via Optimization.jl), **PhysicsNeMo** (PyTorch `torch.optim.Adam` + `torch.optim.LBFGS`).

```python
# DeepXDE — the two-stage recipe, identical in every PINN-Lab case
model = dde.Model(data, net)
model.compile("adam", lr=1e-3, metrics=["l2 relative error"])
model.train(iterations=15000)        # stage 1: global exploration
model.compile("L-BFGS")              # stage 2: full-batch quasi-Newton polish
losshistory, train_state = model.train()
```

**PINN-Lab case.** **Every case** — it is the baseline optimizer for all 20 + the control (coverage map §B, method 17, marked ✅ for Adam→L-BFGS). The anchor `bench-poisson2d` exercises it end-to-end; `mine-flotation-kinetics` (inverse $k$) and `bench-burgers1d` are explicitly tagged with it in the case table.

**Honest limitations.** L-BFGS is **full-batch** — it needs the whole collocation set in memory each step, so on the largest cases (3D heap-leach, lid-driven cavity at high resolution) it is memory-bound and a per-case point budget caps it. Its line search can stall on a very rough loss surface, in which case capping L-BFGS iterations and falling back to more Adam is the pragmatic fix. Crucially, the two-stage recipe fixes *curvature* but does **not** resolve gradient *conflict*: if $g_r$ and $g_b$ point in opposing directions, even a perfect Newton step on the summed loss can degrade one term. That residual pathology is exactly what the next two methods target.

---

## Method 2 — SOAP / gradient alignment (second-order, the 2025 SOTA)

**Explanation.** The 2025 state of the art reframes PINN training as a multi-task problem and asks *why* second-order methods help beyond curvature. Wang et al. (2025) prove that the directional conflict between loss-term gradients is the core bottleneck, define a **gradient alignment score** to diagnose it, and show that a second-order preconditioner *implicitly* aligns the conflicting gradients — a Newton-like step in the right metric rotates the combined update so it makes progress on every term at once. They obtain this cheaply with **SOAP** (*ShampoO with Adam in the Preconditioner's eigenbasis*; Vyas et al., [arXiv:2409.11321](https://arxiv.org/abs/2409.11321)), a quasi-Newton optimizer that runs Adam inside the eigenbasis of Shampoo's structured (Kronecker-factored) second-moment preconditioner — capturing curvature that diagonal Adam cannot, at a fraction of full-Newton cost. On 10 challenging PDE benchmarks this delivers 2–10× accuracy improvements and the **first** successful PINN solution of turbulent flows at Reynolds number up to $10^4$. This is the headline 2025 SOTA optimizer for PINNs and the reason the dossier flags SOAP as the next step beyond Adam→L-BFGS.

**Key equation.** The diagnosis is a **gradient alignment** measure: for loss-term gradients $g_i=\nabla_\theta\mathcal{L}_i$, conflict is the negative-cosine regime

$$
\cos\angle(g_i,g_j)=\frac{g_i\cdot g_j}{\lVert g_i\rVert\,\lVert g_j\rVert}<0 .
$$

SOAP preconditions the gradient in the eigenbasis $Q$ of Shampoo's Kronecker-factored second-moment matrix $L\otimes R$ (left/right factors $L=\mathbb{E}[GG^\top]$, $R=\mathbb{E}[G^\top G]$ for the layer gradient matrix $G$), running Adam on the rotated coordinates:

$$
\tilde g_t = Q^\top g_t,\qquad
\theta_{t+1}=\theta_t-\eta\,Q\,\frac{\hat{\tilde m}_t}{\sqrt{\hat{\tilde v}_t}+\epsilon},
$$

where $\hat{\tilde m}_t,\hat{\tilde v}_t$ are Adam's bias-corrected moments accumulated on $\tilde g_t$. The structured preconditioner $QQ^\top$ is the implicit second-order metric that aligns the conflicting per-term gradients.

**Canonical reference.** Wang, Bhartt, et al., *Gradient Alignment in Physics-informed Neural Networks: A Second-Order Optimization Perspective*, NeurIPS 2025, [arXiv:2502.00604](https://arxiv.org/abs/2502.00604); SOAP optimizer: Vyas, Morwani, Zhao, Shapira, Brandfonbrener, Janson, Kakade, *SOAP: Improving and Stabilizing Shampoo using Adam*, [arXiv:2409.11321](https://arxiv.org/abs/2409.11321).

**Framework.** **jaxpi** (the JAX/Optax technique-donor stack where the gradient-alignment + SOAP recipe is implemented and where PINN-Lab mines it); a standalone SOAP optimizer is available as a drop-in PyTorch `torch.optim`-style module. PINN-Lab studies it via `docs/frameworks/jaxpi/` and ports the schedule rather than shipping on JAX (the JAX→ONNX path is fragile).

**PINN-Lab case.** **`bench-navier-cavity`** (steady lid-driven Navier–Stokes) is the natural showcase — it is the hard NS benchmark where the SOAP/gradient-alignment paper demonstrates its turbulence result, and the coverage map already pairs it with modified-MLP + NTK/grad-norm weighting (method 4/9). SOAP is documented here as the optimizer upgrade for that case and any stiff high-Reynolds variant.

**Honest limitations.** SOAP carries extra state (the Kronecker factors and their periodic eigendecomposition), so it costs more memory and per-step compute than Adam — justified only on the hard, stiff cases, not the easy benchmarks. The turbulence result is at $Re\le10^4$ on canonical 2D problems, not industrial 3D turbulence. And because PINN-Lab's web lane needs an ONNX artifact, SOAP lives in the **offline** training lane only — it changes how the net is *trained*, never how it is *served*.

---

## Method 3 — ConFIG (conflict-free inverse gradients)

**Explanation.** ConFIG attacks the *direction* problem head-on with a closed-form geometric construction rather than a preconditioner. Given the per-loss-term gradients $\{g_i\}$, it computes a single update direction $g_{\text{ConFIG}}$ that is guaranteed to have a **non-negative inner product with every** $g_i$ — i.e. the update decreases (or at least does not increase) *all* loss terms simultaneously, eliminating the destructive conflict where reducing the residual inflates the boundary error. It further enforces equal projected components along each $g_i$ so all terms optimize at a consistent rate, and adds a momentum variant that alternates which term is back-propagated to cut cost. It is a generic multi-task method (also for continual/multi-task learning) but was introduced for and validated on PINNs, consistently beating loss-weighting baselines in both accuracy and runtime. **Dual Cone Gradient Descent** (Hwang et al., NeurIPS 2024) is the closely related cousin: it projects the averaged gradient into the *dual cone* of $\{g_r,g_b\}$ — the set of directions with non-negative inner product against both — achieving the same conflict-free guarantee by a different construction.

**Key equation.** ConFIG seeks an update $g_{\text{ConFIG}}$ with $g_{\text{ConFIG}}\!\cdot\!g_i\ge 0\ \forall i$, equalizing the per-term projections. For two terms (residual $g_r$, boundary $g_b$) the conflict-free direction is the unit vector along the sum of the *normalized* gradients, rescaled so each term's projection is equal:

$$
g_{\text{ConFIG}} \;=\; \Big(g_r^\top \hat u + g_b^\top \hat u\Big)\,\hat u,
\qquad
\hat u \;=\; \frac{\mathcal{U}(g_r)+\mathcal{U}(g_b)}{\big\lVert \mathcal{U}(g_r)+\mathcal{U}(g_b)\big\rVert},
\qquad
\mathcal{U}(v)=\frac{v}{\lVert v\rVert},
$$

whose general $m$-term form solves $g_{\text{ConFIG}}\!\cdot\!\mathcal{U}(g_i)=$ const for all $i$ via the pseudo-inverse of the stacked normalized gradients. Dual Cone Gradient Descent instead returns the projection of the mean gradient onto the cone $\{v:\ v\!\cdot\!g_r\ge0,\ v\!\cdot\!g_b\ge0\}$.

**Canonical reference.** Liu, Chu, Thuerey, *ConFIG: Towards Conflict-free Training of Physics Informed Neural Networks*, ICLR 2025 (Spotlight), [arXiv:2408.11104](https://arxiv.org/abs/2408.11104); Hwang, Lim, et al., *Dual Cone Gradient Descent for Training Physics-Informed Neural Networks*, NeurIPS 2024, [arXiv:2409.18426](https://arxiv.org/abs/2409.18426).

**Framework.** Official implementation `conflictfree` ([PyPI](https://pypi.org/project/conflictfree/), [tum-pbs/ConFIG](https://github.com/tum-pbs/ConFIG)) — framework-agnostic, drops into any PyTorch (and JAX) training loop by replacing the gradient before the optimizer step, so it composes with the Adam→L-BFGS recipe and with DeepXDE/PINA/PhysicsNeMo loops. Dual Cone Gradient Descent has a reference PyTorch repo ([youngsikhwang/Dual-Cone-Gradient-Descent](https://github.com/youngsikhwang/Dual-Cone-Gradient-Descent)).

**PINN-Lab case.** Best exercised on the **conflict-heavy, stiff cases** where the residual and BC/IC gradients fight hardest: `bench-allencahn` (stiff reaction–diffusion) and `mine-heap-leach-rt` (coupled reactive transport, where the chemistry residual and the boundary/interface terms are strongly imbalanced). It is documented as the drop-in de-conflicting upgrade layered on top of the baseline Adam→L-BFGS, alternative to (and stackable with) the loss-weighting pages.

**Honest limitations.** ConFIG and Dual Cone require a **separate back-propagation per loss term** to obtain the individual $g_i$ (the momentum variant amortizes but does not remove this), so a step costs roughly $m\times$ a single backward pass for $m$ terms — cheap for the 2–3 terms of a typical PINN, more for many-constraint problems. The conflict-free guarantee is *directional*: it ensures no term gets worse, not that convergence is fastest, and it does not by itself fix the curvature/stiffness that L-BFGS/SOAP address — which is why PINN-Lab treats these as **composable** (a weighting scheme, a de-conflicting gradient surgeon, and a second-order optimizer can all run in the same training loop), not as mutually exclusive choices.

---

## How this group maps into PINN-Lab

| method | what it fixes | framework | first PINN-Lab case | lane |
|---|---|---|---|---|
| **Adam → L-BFGS** | curvature / stiffness (quasi-Newton polish) | DeepXDE · PINA · NeuralPDE.jl · PhysicsNeMo | **all cases** (baseline; anchor `bench-poisson2d`) | offline |
| **SOAP / gradient alignment** | implicit gradient alignment via structured 2nd-order preconditioner | jaxpi (Optax) · standalone SOAP | `bench-navier-cavity` (hard NS) | offline |
| **ConFIG / Dual Cone** | explicit conflict-free update direction | `conflictfree` (PyTorch/JAX) · DCGD repo | `bench-allencahn`, `mine-heap-leach-rt` | offline |

All three live strictly in the **offline precompute lane** (`requirements-precompute.txt`, `.venv-pipeline`): they govern how each case's PINN is *trained*. The browser never optimizes — it runs the exported `.onnx` via `onnxruntime-web` (see `frameworks/deepxde/` ONNX-export notes and `architecture/train-export-onnx`). The optimizer choice is recorded per case in its manifest so the Benchmark page can state, honestly, which recipe produced the reported relative-$L^2$.

The progression is deliberate and cumulative: **start every case on Adam→L-BFGS** (the proven baseline), reach for a **loss-weighting** scheme (NTK / grad-norm / self-adaptive pages) when terms are magnitude-imbalanced, and escalate to **SOAP** (curvature + implicit alignment) or **ConFIG/Dual Cone** (explicit de-conflicting) only on the stiff, conflict-dominated cases where the baseline plateaus. None of these is a silver bullet; each is the right tool for a specific failure signature of the residual loss.

---

## Key references

- Kingma, Ba. **Adam: A Method for Stochastic Optimization.** ICLR 2015. [arXiv:1412.6980](https://arxiv.org/abs/1412.6980).
- Liu, Nocedal. **On the limited memory BFGS method for large scale optimization.** *Mathematical Programming* 45:503–528, 1989. [doi:10.1007/BF01589116](https://doi.org/10.1007/BF01589116).
- Raissi, Perdikaris, Karniadakis. **Physics-informed neural networks.** *JCP* 378:686–707, 2019. [doi:10.1016/j.jcp.2019.05.045](https://doi.org/10.1016/j.jcp.2019.05.045).
- Krishnapriyan, Gholami, Zhe, Kirby, Mahoney. **Characterizing possible failure modes in PINNs.** NeurIPS 2021. [arXiv:2109.01050](https://arxiv.org/abs/2109.01050).
- Wang, Teng, Perdikaris. **Understanding and mitigating gradient pathologies in PINNs.** *SISC* 43(5), 2021. [arXiv:2001.04536](https://arxiv.org/abs/2001.04536).
- Wang et al. **Gradient Alignment in Physics-informed Neural Networks: A Second-Order Optimization Perspective.** NeurIPS 2025. [arXiv:2502.00604](https://arxiv.org/abs/2502.00604).
- Vyas, Morwani, Zhao, Shapira, Brandfonbrener, Janson, Kakade. **SOAP: Improving and Stabilizing Shampoo using Adam.** 2024. [arXiv:2409.11321](https://arxiv.org/abs/2409.11321).
- Liu, Chu, Thuerey. **ConFIG: Towards Conflict-free Training of Physics Informed Neural Networks.** ICLR 2025 (Spotlight). [arXiv:2408.11104](https://arxiv.org/abs/2408.11104) · code [tum-pbs/ConFIG](https://github.com/tum-pbs/ConFIG) (`conflictfree` on PyPI).
- Hwang, Lim, et al. **Dual Cone Gradient Descent for Training Physics-Informed Neural Networks.** NeurIPS 2024. [arXiv:2409.18426](https://arxiv.org/abs/2409.18426).

---

*Authored as PINN-Lab is built (ADR-0056 docs-as-you-version). Update this page when a case's optimizer recipe changes or a newer conflict-resolution method is adopted.*
