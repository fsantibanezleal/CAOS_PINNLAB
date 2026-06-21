# NeuralPDE.jl — symbolic PDE authoring + Bayesian PINN UQ (Julia / SciML)

> **Status in PINN-Lab:** *documented framework page only.* There is **no Julia toolchain in the
> precompute lane** and **no ONNX → web path** from NeuralPDE.jl. This page exists as a
> **symbolic-authoring showcase** and as the reference for the project's Bayesian-PINN uncertainty
> story. The cases that actually train → export ONNX → run in the browser are built with DeepXDE
> (PyTorch backend); see [`docs/frameworks/deepxde`](../deepxde/README.md) and
> [`docs/architecture/train-export-onnx`](../../architecture/train-export-onnx.md).

---

## What & why

[NeuralPDE.jl](https://github.com/SciML/NeuralPDE.jl) is the physics-informed neural network (PINN)
solver of the Julia [SciML](https://sciml.ai) ecosystem. Its distinguishing feature is **symbolic
authoring**: you write the PDE the way it appears on paper — declaring independent variables, an
unknown field, and differential operators — and the library *automatically derives the PINN loss
function* from that symbolic description. There is no hand-coded residual; `Dxx(u(x,y)) + Dyy(u(x,y))
~ -sin(pi*x)*sin(pi*y)` *is* the program. This is fundamentally different from the Python frameworks
(DeepXDE, PhysicsNeMo, PINA), where you write the residual as an explicit function of autodiff calls.

The mechanism is the SciML symbolic stack:
[ModelingToolkit.jl](https://github.com/SciML/ModelingToolkit.jl) +
[Symbolics.jl](https://github.com/JuliaSymbolics/Symbolics.jl) parse the equation into an expression
tree; NeuralPDE walks that tree, replaces each derivative with an automatic-differentiation rule on a
[Lux.jl](https://github.com/LuxDL/Lux.jl) neural network, samples collocation points according to a
*training strategy*, and emits a standard [Optimization.jl](https://github.com/SciML/Optimization.jl)
`OptimizationProblem`. Solving that problem trains the network.

### The PINN objective it builds

For a PDE $\mathcal{N}[u](\mathbf{x}) = f(\mathbf{x})$ on a domain $\Omega$ with boundary
conditions $\mathcal{B}[u](\mathbf{x}) = g(\mathbf{x})$ on $\partial\Omega$, the network
$u_\theta(\mathbf{x})$ (a Lux `Chain`) is trained to minimise the standard soft-constraint composite
loss

$$
\mathcal{L}(\theta) \;=\;
\lambda_{r}\,\underbrace{\frac{1}{N_r}\sum_{i=1}^{N_r}\bigl\|\mathcal{N}[u_\theta](\mathbf{x}_i) - f(\mathbf{x}_i)\bigr\|^2}_{\text{PDE residual}}
\;+\;
\lambda_{b}\,\underbrace{\frac{1}{N_b}\sum_{j=1}^{N_b}\bigl\|\mathcal{B}[u_\theta](\mathbf{x}_j) - g(\mathbf{x}_j)\bigr\|^2}_{\text{boundary / initial}} .
$$

The derivatives inside $\mathcal{N}$ and $\mathcal{B}$ are taken by automatic differentiation of
$u_\theta$ with respect to its inputs $\mathbf{x}$. NeuralPDE's contribution is that **you never
write the term inside the residual norm by hand** — it is generated from the symbolic `eq` and `bcs`.
The collocation points $\{\mathbf{x}_i\}$ are drawn by the *training strategy* you pass to
`PhysicsInformedNN` (grid, quasi-random, quadrature, or adaptive). This is the Raissi–Perdikaris–
Karniadakis PINN formulation (J. Comput. Phys. 2019,
[doi:10.1016/j.jcp.2019.05.045](https://doi.org/10.1016/j.jcp.2019.05.045)); NeuralPDE automates its
construction and is itself published in the DifferentialEquations.jl / SciML line of work
(Rackauckas & Nie, *J. Open Research Software* 2017,
[doi:10.5334/jors.151](https://doi.org/10.5334/jors.151)).

### Why it earns a page

1. **Best-in-class symbolic ergonomics.** For *authoring* a PDE — especially coupled systems and
   higher-order operators — the symbolic form is the cleanest of any PINN library. It is the natural
   place to *teach* "this is the equation, this is the loss it becomes."
2. **First-class Bayesian PINN UQ.** NeuralPDE ships a `BayesianPINN` discretizer that turns the
   PDE+data into a likelihood and samples the posterior over network weights with Hamiltonian Monte
   Carlo ([AdvancedHMC.jl](https://github.com/TuringLang/AdvancedHMC.jl)) — i.e. the B-PINN of Yang,
   Meng & Karniadakis (J. Comput. Phys. 2021,
   [arXiv:2003.06097](https://arxiv.org/abs/2003.06097)) as a maintained library feature, not a
   research script. It returns calibrated credible intervals, which vanilla deterministic PINNs do
   not provide. This is directly relevant to PINN-Lab's honest-uncertainty case
   (`poll-source-uq-bpinn`).
3. **SciML coupling.** Because it emits a standard `OptimizationProblem`, it composes with the rest of
   SciML (DifferentialEquations.jl, Optimization.jl optimizers, automatic-differentiation backends),
   which is unmatched for hybrid ODE/PDE + parameter-estimation workflows.

### Why it is **not** a build engine here

PINN-Lab's product contract is *train offline → export ONNX → run client-side with
onnxruntime-web* (see [`docs/architecture/the-gate`](../../architecture/the-gate.md)). NeuralPDE.jl
has **no native ONNX exporter**. A trained model is a Lux parameter `NamedTuple` you serialise with
[JLD2.jl](https://github.com/JuliaIO/JLD2.jl) (Julia-only) or, via
[Reactant.jl](https://github.com/EnzymeAD/Reactant.jl), compile to **StableHLO** for the XLA/JAX
world — neither of which is the ONNX graph that `onnxruntime-web` consumes. Bridging it would mean
adding a Julia toolchain to the build *and* hand-porting the MLP weights into an ONNX graph. That is
out of scope; see [ONNX-export notes](#onnx-export-notes). So NeuralPDE is included as a **symbolic-
authoring and UQ showcase**, decoupled from the shipping pipeline.

---

## Install (verified)

NeuralPDE.jl is a registered Julia package. From the Julia REPL, press `]` to enter the **Pkg**
prompt and add the package plus the satellite packages used in a real run:

```julia-repl
pkg> add NeuralPDE Lux ModelingToolkit DomainSets
pkg> add Optimization OptimizationOptimisers OptimizationOptimJL
pkg> add LineSearches
```

or, scripted (non-interactive), in a project-local environment — the Julia equivalent of a `.venv`,
so nothing touches a global install:

```julia
import Pkg
Pkg.activate("pinnlab-julia")          # project-local env (Project.toml/Manifest.toml)
Pkg.add(["NeuralPDE", "Lux", "ModelingToolkit", "DomainSets",
         "Optimization", "OptimizationOptimisers", "OptimizationOptimJL",
         "LineSearches"])
```

For the Bayesian PINN path, also add the sampler/measurement packages:

```julia
Pkg.add(["AdvancedHMC", "MonteCarloMeasurements", "Distributions"])
```

**Verified facts (mid-2026):**

| | |
|---|---|
| Latest version | **v6.0.0** (2026-05-08) |
| License | **MIT** |
| Neural-network library | **Lux.jl** (Flux.jl supported as legacy) |
| GPU | yes — via `LuxCUDA` / the SciML GPU stack and `Reactant` |
| Julia | ≥ 1.10 LTS recommended |

> The first `using NeuralPDE` after install triggers Julia's precompilation; on a clean machine this
> is several minutes (the time-to-first-solve cost of the Julia stack). Pin `Project.toml` /
> `Manifest.toml` for reproducibility, exactly as we pin Python `requirements-*.txt`.

Sources: package + install — [github.com/SciML/NeuralPDE.jl](https://github.com/SciML/NeuralPDE.jl);
API — [docs.sciml.ai/NeuralPDE](https://docs.sciml.ai/NeuralPDE/stable/).

---

## Core API / configure

The workflow is a fixed five-step pipeline. Each step maps to one SciML object.

| Step | What you do | API |
|---|---|---|
| 1. Declare symbols | independent vars, unknown field, differential operators | `@parameters`, `@variables u(..)`, `Differential(x)^2` |
| 2. Write the PDE + BCs | symbolic equation(s) with `~`, list of boundary/initial equations | `eq = … ~ …`, `bcs = [ … ]` |
| 3. Define domains | per-variable intervals | `x ∈ Interval(a, b)` (from `ModelingToolkit`/`DomainSets`) |
| 4. Build net + discretizer | a `Lux.Chain` and a training strategy | `Chain(...)`, `PhysicsInformedNN(chain, strategy; …)` |
| 5. Assemble + solve | name the system, lower to an `OptimizationProblem`, optimise | `@named pde_system = PDESystem(...)`, `discretize`, `solve` |

### `PhysicsInformedNN(chain, strategy; kwargs...)`

The discretizer. Key arguments:

- **`chain`** — a `Lux.Chain` (or a `Vector{Chain}` for coupled systems, one net per dependent
  variable). Input dimension = number of independent variables; output dimension = 1 per field.
- **`strategy`** — the collocation **training strategy** that decides how residual points are sampled.
  The main choices:
  - `QuadratureTraining(; batch, abstol, reltol)` — integrates the loss with an adaptive quadrature
    rule (high accuracy on low-dimensional domains; the recommended default for smooth problems).
  - `GridTraining(dx)` — fixed tensor grid of points (simple, deterministic).
  - `StochasticTraining(points)` / `QuasiRandomTraining(points; sampling_alg)` — (quasi-)random
    resampling each iteration (scales better in higher dimensions).
  - adaptive strategies (e.g. residual-/importance-weighted variants) for sharp-gradient solutions.
- **`adaptive_loss`** — optional automatic loss-term weighting (NTK-/gradient-based), the SciML
  analogue of the loss-weighting schemes in [`docs/methods/ntk-weighting`](../../methods/ntk-weighting.md)
  and [`docs/methods/gradnorm-weighting`](../../methods/gradnorm-weighting.md).
- **`param_estim = true`** — switches on **inverse mode**: unknown PDE coefficients declared as
  `@parameters` and listed in `PDESystem(...; defaults=…)` become trainable alongside the network
  (the Julia counterpart of DeepXDE's `dde.Variable`).

### `discretize(pde_system, discretization)`

Lowers the symbolic `PDESystem` to a concrete `OptimizationProblem` whose decision variables are the
flattened network parameters. (`symbolic_discretize` returns the *un-compiled* symbolic loss
functions instead — useful for inspecting exactly which residual NeuralPDE built from your equation.)

### `solve(prob, optimizer; maxiters, callback)`

Standard Optimization.jl solve. The validated PINN recipe is **Adam → (L-)BFGS**: `Adam` from
`OptimizationOptimisers` for robust global exploration of the stiff residual landscape, then
`BFGS`/`LBFGS` (often with `LineSearches.BackTracking()`) from `OptimizationOptimJL` for
high-accuracy local refinement. See [`docs/methods/optimizers`](../../methods/optimizers.md).

### Evaluating the trained field

`discretization.phi` is the **trained field evaluator**: `phi(point, res.u)` returns
$u_\theta(\text{point})$ for the optimised parameters `res.u`. This is how you produce the solution
grid for plotting or for baking a replay artifact.

---

## Runnable minimal example

A complete, self-contained 2D Poisson solve, transcribed from the current
[NeuralPDE.jl PDESystem tutorial](https://docs.sciml.ai/NeuralPDE/stable/tutorials/pdesystem/). The
PDE is

$$
\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}
= -\sin(\pi x)\,\sin(\pi y), \qquad (x,y)\in[0,1]^2,
$$

with $u=0$ on all four edges; the analytic solution is
$u(x,y) = \dfrac{\sin(\pi x)\sin(\pi y)}{2\pi^2}$, giving a clean validation anchor.

```julia
using NeuralPDE, Lux, Optimization, OptimizationOptimJL, LineSearches
using ModelingToolkit: Interval

# 1. symbols + differential operators
@parameters x y
@variables u(..)
Dxx = Differential(x)^2
Dyy = Differential(y)^2

# 2. the PDE + boundary conditions (symbolic — this IS the residual definition)
eq  = Dxx(u(x, y)) + Dyy(u(x, y)) ~ -sin(pi * x) * sin(pi * y)
bcs = [u(0, y) ~ 0.0, u(1, y) ~ 0.0,
       u(x, 0) ~ 0.0, u(x, 1) ~ 0.0]

# 3. domains
domains = [x ∈ Interval(0.0, 1.0),
           y ∈ Interval(0.0, 1.0)]

# 4. network + discretizer (training strategy)
dim   = 2
chain = Chain(Dense(dim, 16, σ), Dense(16, 16, σ), Dense(16, 1))
discretization = PhysicsInformedNN(
    chain, QuadratureTraining(; batch = 200, abstol = 1e-6, reltol = 1e-6))

# 5. assemble symbolic system -> OptimizationProblem -> solve
@named pde_system = PDESystem(eq, bcs, domains, [x, y], [u(x, y)])
prob = discretize(pde_system, discretization)

opt = LBFGS(linesearch = BackTracking())
res = solve(prob, opt, maxiters = 1000)

# evaluate the trained field
phi = discretization.phi
xs = ys = 0.0:0.01:1.0
u_predict = reshape([first(phi([x, y], res.u)) for x in xs for y in ys],
                    (length(xs), length(ys)))
```

> An Adam warm-up before LBFGS is the production recipe; add
> `using OptimizationOptimisers` and call
> `res = solve(prob, Adam(0.01); maxiters = 2000)`, then continue from `res.u` with `LBFGS(...)`.

### Bayesian PINN variant (the UQ showcase)

To obtain calibrated uncertainty instead of a point estimate, swap the discretizer for
`BayesianPINN` and sample the posterior with `ahmc_bayesian_pinn_pde`. Verified signature from the
[BayesianPINN manual](https://docs.sciml.ai/NeuralPDE/dev/manual/bpinns/):

```julia
using NeuralPDE, Lux, AdvancedHMC, MonteCarloMeasurements

# dataset = vector of matrices; each matrix's first column is the dependent variable,
# remaining columns the independent variables (here: noisy observations of u at (x,y)).
discretization = BayesianPINN([chain]; dataset = [u_obs_matrix])

sol = ahmc_bayesian_pinn_pde(
    pde_system, discretization;
    draw_samples = 1000,
    bcstd  = [0.01],            # boundary-loss noise std
    l2std  = [0.05],            # data-loss noise std
    phystd = [0.05],            # physics-(residual)-loss noise std
    priorsNNw = (0.0, 2.0),     # Gaussian prior (mean, std) over NN weights
    param  = [],                # priors on unknown PDE params (inverse problems)
    nchains = 1,
    Kernel = HMC(0.1, 30),
    saveats = [1 / 100.0])

# sol.ensemblesol      :: Particles (MonteCarloMeasurements) — solution distribution
# sol.estimated_nn_params, sol.estimated_de_params — posterior over weights / PDE params
```

The posterior over weights induces a posterior over the field; the spread of `sol.ensemblesol`
(`Particles` from MonteCarloMeasurements.jl) gives pointwise credible intervals — the honest
uncertainty that deterministic PINNs lack. For inverse problems, supply distributions in `param=` and
read `sol.estimated_de_params`.

---

## ONNX-export notes

**There is no native ONNX export from NeuralPDE.jl.** This is the load-bearing caveat for PINN-Lab and
the reason Julia stays out of the build pipeline.

What you *can* persist after training:

| Target | Mechanism | Reaches `onnxruntime-web`? |
|---|---|---|
| Julia reuse | **JLD2.jl** — serialise the Lux parameter `NamedTuple` (`res.u`) + model | No (Julia-only `.jld2`) |
| XLA / JAX | **Reactant.jl** → **StableHLO** MLIR; runs on XLA, convertible toward the JAX/TF world | Not directly; StableHLO ≠ ONNX |
| ONNX | *none built in* | — |

A trained NeuralPDE model is a small dense MLP (`Dense → σ → … → Dense`), so in principle one could
**hand-port the weight matrices** into an equivalent ONNX graph (the inference math is just
`σ(Wx+b)` stacked). But that is bespoke glue, untraced and error-prone for any hard-BC ansatz or
feature transform, and it would still require a Julia install in CI to produce the weights. The
project's verified train→web bridge instead lives entirely in PyTorch:
`torch.onnx.export(model.net, …)` from DeepXDE — see
[`docs/architecture/train-export-onnx`](../../architecture/train-export-onnx.md) and the parity gate in
[`docs/architecture/the-gate`](../../architecture/the-gate.md). **Do not** wire NeuralPDE output into
`models/<case>.onnx`.

---

## Role in PINN-Lab (which cases / lane)

| Aspect | Verdict |
|---|---|
| Build lane | **None.** No Julia in `requirements-precompute*.txt`; no `models/<case>.onnx` produced from NeuralPDE. |
| Live/web lane | **None.** No ONNX → `onnxruntime-web` path. |
| Documentation role | **Symbolic-authoring showcase** + **Bayesian-PINN UQ reference.** |
| Runnable example | The 2D-Poisson script above (CPU; `]add` env), provided to *read and run*, not to feed the pipeline. |

Concretely, NeuralPDE.jl appears in PINN-Lab as:

- **A `docs/frameworks/neuralpde-jl/` page** (this file) — the cleanest illustration of "the equation
  *is* the loss," used on the **Methodology** page to contrast symbolic authoring (NeuralPDE) against
  explicit-residual authoring (DeepXDE / PhysicsNeMo).
- **The Bayesian-PINN reference** behind the UQ narrative for `poll-source-uq-bpinn` (the honest-
  uncertainty case). NeuralPDE's `BayesianPINN` is shown as a maintained-library realisation of the
  B-PINN method; the *shipped* version of that case is built on the Python UQ stack
  (DeepXDE inverse + NeuralUQ / B-PINN, per [`docs/methods/inverse-uq`](../../methods/inverse-uq.md))
  so it can export ONNX. NeuralPDE is the conceptual anchor, not the producing engine.

It deliberately does **not** own any of the 20 trained cases. Every case that reaches the browser is
authored in Python and exported through the PyTorch → ONNX bridge.

---

## References

- L. Lu, X. Meng, Z. Mao, G. E. Karniadakis. *DeepXDE: A deep learning library for solving differential
  equations.* SIAM Review 63(1), 2021. [doi:10.1137/19M1274067](https://doi.org/10.1137/19M1274067) —
  framing of the PINN loss this page summarises.
- M. Raissi, P. Perdikaris, G. E. Karniadakis. *Physics-informed neural networks…* J. Comput. Phys.
  378, 2019. [doi:10.1016/j.jcp.2019.05.045](https://doi.org/10.1016/j.jcp.2019.05.045) — the PINN
  formulation NeuralPDE automates.
- C. Rackauckas, Q. Nie. *DifferentialEquations.jl…* J. Open Research Software 5(1), 2017.
  [doi:10.5334/jors.151](https://doi.org/10.5334/jors.151) — the SciML differential-equation
  substrate.
- L. Yang, X. Meng, G. E. Karniadakis. *B-PINNs: Bayesian physics-informed neural networks for forward
  and inverse PDE problems with noisy data.* J. Comput. Phys. 425, 2021.
  [arXiv:2003.06097](https://arxiv.org/abs/2003.06097) — the Bayesian PINN realised by `BayesianPINN`.
- A. F. Psaros, X. Meng, Z. Zou, L. Guo, G. E. Karniadakis. *Uncertainty quantification in scientific
  machine learning.* J. Comput. Phys. 477, 2023. [arXiv:2201.07766](https://arxiv.org/abs/2201.07766) —
  context for the UQ methods, incl. HMC vs. ensembles.
- **NeuralPDE.jl** — repository: [github.com/SciML/NeuralPDE.jl](https://github.com/SciML/NeuralPDE.jl);
  docs: [docs.sciml.ai/NeuralPDE](https://docs.sciml.ai/NeuralPDE/stable/); PINN manual:
  [docs.sciml.ai/NeuralPDE/.../manual/pinns](https://docs.sciml.ai/NeuralPDE/stable/manual/pinns/);
  BayesianPINN manual: [docs.sciml.ai/NeuralPDE/dev/manual/bpinns](https://docs.sciml.ai/NeuralPDE/dev/manual/bpinns/).
- Supporting libraries: [Lux.jl](https://github.com/LuxDL/Lux.jl),
  [ModelingToolkit.jl](https://github.com/SciML/ModelingToolkit.jl),
  [Optimization.jl](https://github.com/SciML/Optimization.jl),
  [Reactant.jl](https://github.com/EnzymeAD/Reactant.jl) (StableHLO),
  [JLD2.jl](https://github.com/JuliaIO/JLD2.jl) (Julia serialisation).
