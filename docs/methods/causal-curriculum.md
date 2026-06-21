# Causal & curriculum training

**Group:** `causal-curriculum` · **Method family:** training-schedule fixes for time-dependent PDEs
**PINN-Lab cases that exercise it:** `bench-heat1d`, `bench-allencahn`, `poll-ocean-transport`, `poll-groundwater-rt`, `poll-tailings-seepage`

---

## What it is

Vanilla PINNs minimise a single, time-aggregated residual loss over the whole space–time domain at
once. For a time-dependent PDE this silently **breaks causality**: the optimiser is free to drive the
residual to zero by fitting the late-time behaviour *before* the early-time solution is correct,
producing a network that satisfies the PDE in a least-squares sense yet converges to a wrong (often
trivial or smeared) field. The pathology is most violent on **stiff** problems (Allen–Cahn, reaction
terms) and **advection-dominated** problems (transport with large Péclet number), where the loss
landscape is so ill-conditioned that gradient descent stalls in a bad basin and the relative L2 error
plateaus at O(1).

The `causal-curriculum` group collects the two complementary, well-validated remedies that both work by
**imposing a temporal ordering on what the optimiser is allowed to fit**:

1. **Causal training** (Wang, Sankaran & Perdikaris, CMAME 2024) — a soft, *continuous* re-weighting of
   the residual loss so that each time slice is only weighted once all earlier slices have already
   converged. A single extra hyperparameter, no architectural change.
2. **Curriculum regularisation + sequence-to-sequence / time-marching** (Krishnapriyan et al.,
   NeurIPS 2021) — a *discrete* schedule: ramp a difficulty knob (e.g. the convection or reaction
   coefficient) from easy to hard while warm-starting each stage, and/or solve the time interval in
   short successive segments instead of all at once.

Both are **training schedules, not architectures** — they compose freely with Fourier features,
modified-MLP, PirateNets, adaptive sampling and NTK/grad-norm loss weighting. Neither is a built-in
flag in DeepXDE; both are implemented as a **manual schedule** around the standard training loop (see
"How it maps into PINN-Lab" below).

---

## Method 1 — Causal training

### Explanation

Partition the temporal domain $[0, T]$ into $N_t$ ordered bins $t_1 < t_2 < \dots < t_{N_t}$ and let
$\mathcal{L}_r(t_i, \theta)$ be the PDE-residual loss restricted to bin $i$ (averaged over the spatial
collocation points sampled at that time). The standard PINN minimises the unweighted sum
$\frac{1}{N_t}\sum_i \mathcal{L}_r(t_i,\theta)$, which has no notion of ordering. Causal training instead
minimises a **weighted** residual loss

$$
\mathcal{L}_r(\theta) \;=\; \frac{1}{N_t}\sum_{i=1}^{N_t} w_i \,\mathcal{L}_r(t_i, \theta),
\qquad
w_i \;=\; \exp\!\left(-\,\epsilon \sum_{k=1}^{i-1} \mathcal{L}_r(t_k, \theta)\right),
\quad w_1 \equiv 1 .
$$

The weight $w_i$ is large (near 1) only when the **cumulative** residual of all earlier bins
$\sum_{k<i}\mathcal{L}_r(t_k,\theta)$ is small — i.e. bin $i$ is only "switched on" in the loss once the
solution up to $t_{i-1}$ is already accurate. As training proceeds the front of converged bins sweeps
forward in time, exactly mimicking how a time-stepping numerical solver marches the solution. The
weights $w_i$ are treated as **constants** for the gradient step (stop-gradient / `detach`); they are
recomputed from the current residuals at each iteration, not back-propagated through.

The causality parameter $\epsilon$ sets the steepness: large $\epsilon$ gives a sharp decay (strong
causality, emphasis on early times, slower); small $\epsilon$ approaches the uniform-weight vanilla PINN.
Because no single value is robust, the authors **anneal** $\epsilon$ over an increasing schedule (e.g.
$\epsilon \in \{10^{-2}, 10^{-1}, 1, 10, 10^2\}$): start permissive, tighten as training stabilises. A
practical convergence diagnostic is the **stopping criterion**

$$
\min_i w_i \;>\; \delta \qquad (\text{typically } \delta \approx 0.99),
$$

which holds only when *every* bin's cumulative-earlier residual is small — i.e. the whole time interval
has been resolved in causal order. The paper reports 10–100× lower relative L2 error on chaotic/stiff
benchmarks (Allen–Cahn, Kuramoto–Sivashinsky, Navier–Stokes) that vanilla PINNs cannot solve at all.

### Key equation

$$
\boxed{\;w_i = \exp\!\left(-\epsilon \sum_{k=1}^{i-1} \mathcal{L}_r(t_k, \theta)\right),
\qquad
\theta^\star = \arg\min_\theta \frac{1}{N_t}\sum_{i=1}^{N_t} w_i\,\mathcal{L}_r(t_i,\theta)\;}
$$

with $w_i$ held constant (stop-gradient) within each optimisation step, $\epsilon$ annealed upward, and
training stopped when $\min_i w_i > \delta$.

### Reference

Sifan Wang, Shyam Sankaran, Paris Perdikaris. *"Respecting causality is all you need for training
physics-informed neural networks."* Computer Methods in Applied Mechanics and Engineering (CMAME),
vol. 421, 116813, 2024. arXiv:2203.07404. DOI: 10.1016/j.cma.2024.116813.

### Which framework implements it

- **jaxpi** (`PredictiveIntelligenceLab/jaxpi`, JAX) — the canonical reference implementation; causal
  weighting is a first-class config option and the original equations come from this group.
- **PINA** (`mathLab/PINA`, PyTorch Lightning, MIT) — exposes it as the `CausalPINN` solver.
- **DeepXDE** — **no built-in flag.** Implemented in PINN-Lab as a manual residual-bin re-weighting on
  the PyTorch backend (a custom loss that bins collocation points by time, computes per-bin residuals,
  forms $w_i$ with `torch.no_grad()`, and returns the weighted sum).

### Which PINN-Lab case exercises it

- **Primary anchor:** `bench-heat1d` (1D transient heat/diffusion $u_t = \alpha u_{xx}$) — the smallest
  time-dependent case, used to validate the causal-weighting harness against the analytic solution
  before it is reused on harder cases (per coverage map row 2).
- **Reused by:** `bench-allencahn` (stiff $u_t = \varepsilon^2 u_{xx} + u - u^3$, combined with
  PirateNets), `poll-ocean-transport`, `poll-groundwater-rt` (inverse + reactive coupling),
  `poll-tailings-seepage` (Richards equation, hard-constraints + causal).

---

## Method 2 — Curriculum regularisation + sequence-to-sequence (time-marching)

### Explanation

Krishnapriyan et al. dissect *why* PINNs fail on convection/reaction/diffusion problems and show it is
**not** an approximation-capacity failure (a trained-then-frozen network can represent the true solution)
but an **optimisation** failure: as the convection or reaction coefficient grows, the residual-loss
landscape develops sharp, ill-conditioned minima that gradient descent cannot reach from a cold start.
They propose two schedule-based fixes that need no new architecture:

- **Curriculum regularisation.** Solve an easy version of the PDE first — small convection coefficient
  $\beta$, small reaction rate $\rho$, or large diffusion — then **warm-start** the next, slightly harder
  stage from the previous stage's weights, ramping the coefficient up to its target value over a sequence
  of stages. Each stage starts in a well-conditioned basin near the previous solution, so the optimiser
  never has to cross the bad landscape in one jump.

- **Sequence-to-sequence / time-marching.** Instead of predicting the entire space–time field at once,
  split $[0,T]$ into short successive sub-intervals and train them in order, each sub-interval taking the
  previous one's terminal state as its initial condition. This is the discrete, "hard" analogue of causal
  training — the network only ever sees a short, well-conditioned horizon at a time, which directly
  restores temporal causality.

Both cut error by **one to two orders of magnitude** on advection-dominated and stiff benchmarks where
the baseline PINN error is O(1). Curriculum is the natural choice when there is a single physical
"difficulty" coefficient to ramp (Péclet/Damköhler number); time-marching is the natural choice when the
horizon $T$ is long and the dynamics are stiff.

### Key equation

Curriculum regularisation solves a sequence of problems indexed by a difficulty coefficient
$\beta^{(0)} < \beta^{(1)} < \dots < \beta^{(M)} = \beta_{\text{target}}$, warm-starting each from the
last:

$$
\theta^{(m)} \;=\; \arg\min_\theta \; \mathcal{L}\big(\theta;\, \beta^{(m)}\big),
\qquad
\theta^{(m)} \;\text{initialised from}\; \theta^{(m-1)} ,
$$

where $\mathcal{L}(\theta;\beta)$ is the usual PINN loss for the PDE at coefficient $\beta$ (e.g. the
convection equation $u_t + \beta\,u_x = 0$). Time-marching is the same idea applied to the temporal
support: over sub-intervals $[T_{j-1}, T_j]$,

$$
\theta^{(j)} = \arg\min_\theta \; \mathcal{L}_{[T_{j-1},T_j]}(\theta)
\quad\text{s.t.}\quad u_\theta(x, T_{j-1}) = u_{\theta^{(j-1)}}(x, T_{j-1}),
$$

each segment inheriting its initial condition from the previous segment's terminal state.

### Reference

Aditi S. Krishnapriyan, Amir Gholami, Shandian Zhe, Robert M. Kirby, Michael W. Mahoney.
*"Characterizing possible failure modes in physics-informed neural networks."* Advances in Neural
Information Processing Systems (NeurIPS) 34, 2021. arXiv:2109.01050.
Reference code: `github.com/a1k12/characterizing-pinns-failure-modes`.

### Which framework implements it

- **No framework ships a turnkey curriculum/time-marching flag** — both methods are *recipes* over the
  standard training loop. The authors' reference code (above) is the canonical implementation.
- **DeepXDE / jaxpi** — implemented as a manual schedule: an outer Python loop that (a) for curriculum,
  rebuilds the PDE residual with an increasing coefficient and re-`compile`/re-`train`s from the prior
  state; (b) for time-marching, advances a sliding `TimeDomain` window, baking the previous window's
  terminal field as the next window's initial condition.

### Which PINN-Lab case exercises it

- **Primary anchor:** `bench-allencahn` (stiff Allen–Cahn) — the case that most needs time-marching /
  curriculum to converge at all; documented as "PirateNets; causal / time-marching" in the coverage map
  (row 4).
- **Naturally applicable to:** `poll-ocean-transport` and `poll-groundwater-rt` (advection-dominated
  transport, large Péclet), and any long-horizon transient case where a single causal pass is unstable.

---

## Why temporal causality matters for stiff / advective time-dependent PDEs

For an evolution PDE the true solution at time $t$ is *determined by* the solution at all earlier times.
A loss that averages the residual uniformly over space–time discards this dependency: the optimiser can
lower the global loss by trading early-time error for late-time error, and because stiff/advective
operators make the late-time residual extremely sensitive, gradient descent is pulled toward solutions
that are smooth-but-wrong (or that collapse onto a trivial steady state). Concretely:

- **Stiff problems** (Allen–Cahn, reaction terms) have widely-separated time scales; the residual
  gradient is dominated by the fast scale and the network never resolves the slow front evolution unless
  forced to fit early times first.
- **Advection-dominated problems** (high Péclet transport) have nearly-hyperbolic, sharp-front dynamics;
  information propagates along characteristics, so fitting downstream/late values before upstream/early
  values is physically meaningless and numerically unstable.

Causal training fixes this *continuously* (soft weights that sweep a converged front forward);
curriculum/time-marching fixes it *discretely* (warm-started difficulty ramp, or short well-conditioned
sub-horizons). In practice they are complementary and often stacked: time-march the long horizon into
windows, and apply causal weighting *within* each window.

---

## Honest limitations

- **Extra hyperparameters, no free lunch.** Causal training adds $\epsilon$ (and its annealing schedule),
  the number of time bins $N_t$, and the stopping $\delta$ — all problem-dependent and requiring tuning.
  A poorly chosen $\epsilon$ either reverts to the vanilla PINN (too small) or stalls progress to a crawl
  (too large). Curriculum adds the difficulty schedule $\{\beta^{(m)}\}$ and the per-stage iteration
  budget.
- **More expensive than a single solve.** Both methods extend wall-clock time: causal weighting requires
  per-bin residual bookkeeping every iteration; time-marching and curriculum run *multiple sequential
  trainings*. This compounds the headline PINN limitation (PINN-Lab dossier §1): for a single well-posed
  *forward* problem, a classical FEM/FVM solver is usually faster and more accurate. These schedules make
  hard time-dependent PINNs *converge at all* — they do not make PINNs beat a good numerical solver on
  forward-solve speed.
- **Causal weighting can over-emphasise early times.** With large $\epsilon$ the late-time bins are
  starved of gradient until very late in training, so a fixed iteration budget may leave the final time
  slices under-resolved. The annealing schedule and the $\min_i w_i > \delta$ check are there precisely to
  detect this.
- **Time-marching propagates error.** Each window inherits the previous window's terminal state as its
  IC, so any error compounds across windows — short windows (more transfers, faster error growth) vs long
  windows (harder each, but fewer transfers) is a genuine tradeoff, and the IC-transfer must be evaluated
  consistently (re-bake the previous net's terminal field on the new window's spatial grid).
- **Not a remedy for capacity or spectral-bias failures.** If the failure is high-frequency
  representation (spectral bias) rather than temporal ordering, you need Fourier features / SIREN /
  PirateNets *as well* — causality fixes *when* the optimiser fits, not *what* the network can represent.

---

## How it maps into PINN-Lab

- **Implemented as manual schedules on the DeepXDE PyTorch backend** (DeepXDE has no built-in causal or
  curriculum flag). Two reusable helpers live in the training stage:
  - a **causal residual loss** that bins collocation points by time, computes per-bin residual losses,
    forms the weights $w_i = \exp(-\epsilon\sum_{k<i}\mathcal{L}_r(t_k))$ under `torch.no_grad()`, returns
    the weighted sum, and ramps $\epsilon$ over training (the `causal-training` recipe);
  - a **time-marching / curriculum driver** that wraps `model.compile`/`model.train` in an outer loop over
    either a difficulty coefficient or a sliding `TimeDomain` window, warm-starting each stage from the
    previous fitted state (the `curriculum-timemarching` recipe).
- **First exercised by `bench-heat1d`** (causal) and **`bench-allencahn`** (curriculum / time-marching),
  then reused across the advective/stiff pollution cases (`poll-ocean-transport`, `poll-groundwater-rt`,
  `poll-tailings-seepage`).
- The trained net still exports through the standard **train → ONNX → onnxruntime-web** contract
  unchanged: these methods only alter *how the weights are fit*, not the final `model.net`, so the baked
  artifact and ONNX export are identical to any other case (parity check against `model.predict` still
  applies).
- **jaxpi** (technique-donor framework page) and **PINA** (`CausalPINN`) are documented as the reference
  implementations of causal weighting; PINN-Lab ports the technique rather than shipping on them, because
  the web pipeline requires the DeepXDE→ONNX path.

---

## References

1. S. Wang, S. Sankaran, P. Perdikaris. *Respecting causality is all you need for training
   physics-informed neural networks.* CMAME 421:116813, 2024. arXiv:2203.07404 ·
   DOI:10.1016/j.cma.2024.116813.
2. A. S. Krishnapriyan, A. Gholami, S. Zhe, R. M. Kirby, M. W. Mahoney. *Characterizing possible failure
   modes in physics-informed neural networks.* NeurIPS 34, 2021. arXiv:2109.01050 ·
   code: github.com/a1k12/characterizing-pinns-failure-modes.

*See also:* `methods/piratenets.md` (stacked with these schedules on `bench-allencahn`),
`methods/adaptive-sampling.md` and `methods/ntk-weighting.md` (orthogonal, composable),
`frameworks/jaxpi.md` (reference implementation of causal weighting),
`frameworks/deepxde.md` (the engine these schedules are implemented around).
