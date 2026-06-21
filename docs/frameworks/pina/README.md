# PINA — framework guide

> **PINA** (*Physics-Informed Neural networks for Advanced modeling*) — the MIT-licensed,
> PyTorch-Lightning-idiomatic engine in PINN-Lab's stack. It carries the **richest single-library
> catalogue of PINN variants** (SelfAdaptive, RBA, Causal, Competitive, Gradient, DeepEnsemble)
> alongside supervised solvers, neural operators (DeepONet / FNO / Averaging) and ROM/GAROM — all
> behind one `Problem → Condition → Solver → Trainer` API, with a Lightning `Trainer` for free
> multi-GPU and a clean `torch.onnx.export` path because every solver's `.model` is a plain
> `torch.nn.Module`.

- **Repo:** <https://github.com/mathLab/PINA>
- **Docs:** <https://mathlab.github.io/PINA/>
- **PyPI:** `pina-mathlab` — latest **0.2.6** (verified 2026-06-21)
- **License:** **MIT** (verified against `LICENSE.rst` on `master`)
- **Built on:** PyTorch · PyTorch Lightning · PyTorch Geometric (PyG)
- **Authors / group:** mathLab, SISSA (Rozza group)

---

## 1. What & why

### What it is

PINA is a **high-level SciML framework** that sits on top of PyTorch, PyTorch Lightning and PyTorch
Geometric. You describe a problem **declaratively** — its output variables, its spatial/temporal
domains, and a set of **conditions** (boundary/initial values and the PDE residual itself) — and
then hand that `Problem` to a **`Solver`**. The solver wraps a `torch.nn.Module` and a
`LightningModule` training loop; a Lightning **`Trainer`** runs it. Because the heavy lifting
(device placement, optimizer stepping, logging, checkpointing, distributed strategy) is delegated
to Lightning, PINA's own surface area stays small and the same `Problem` can be solved by *any* of
its solvers by swapping one class name.

For PINNs, the defining idea is unchanged from Raissi, Perdikaris & Karniadakis (2019): embed the
governing PDE in the loss via automatic differentiation. For a PDE
$\mathcal{N}[u](x) = 0$ on a domain $\Omega$ with boundary/initial operators
$\mathcal{B}[u] = g$ on $\partial\Omega$, a network $u_\theta$ is trained to minimize

$$
\mathcal{L}(\theta) = \underbrace{\frac{1}{N_r}\sum_{i=1}^{N_r}\big\lVert \mathcal{N}[u_\theta](x_i^r)\big\rVert^2}_{\text{PDE residual (interior condition)}}
\;+\; \underbrace{\frac{1}{N_b}\sum_{j=1}^{N_b}\big\lVert \mathcal{B}[u_\theta](x_j^b) - g(x_j^b)\big\rVert^2}_{\text{boundary / initial conditions}} .
$$

In PINA, **each term of that sum is a `Condition`**: the residual term is a `Condition` on the
interior domain `"D"` carrying an `Equation`, and each boundary/initial term is a `Condition` on a
boundary domain carrying a `FixedValue` (or another `Equation`). The collocation points
$\{x_i^r\}$, $\{x_j^b\}$ are produced by `problem.discretise_domain(...)`. The *solver* decides how
those condition losses are combined and weighted — and that is where PINA's variants differ.

### Why it earns a place next to DeepXDE

PINA and DeepXDE solve overlapping problems, but their value in PINN-Lab is complementary:

| | **DeepXDE** | **PINA** |
|---|---|---|
| License | LGPL-2.1 | **MIT** (fully permissive) |
| Training loop | own `Model.train` loop | **PyTorch Lightning** (`Trainer`) |
| Multi-GPU / DDP | manual / backend-dependent | **free** via Lightning `accelerator`/`devices`/`strategy` |
| Variant catalogue | one canonical PINN + RAR/gPINN/hard-BC recipes | **6 PINN solvers in-library**: `PINN`, `GradientPINN`, `CausalPINN`, `CompetitivePINN`, `SelfAdaptivePINN`, `RBAPINN` (+ `DeepEnsemblePINN`) |
| Operators / ROM | DeepONet | DeepONet, **FNO**, **Averaging**, **ROM / GAROM** |
| ONNX | export raw `model.net` | export raw `solver.model` (plain `nn.Module`) |
| API style | imperative, declarative PDE function | **declarative `Problem`/`Condition`** classes |

The headline reason PINN-Lab keeps PINA is the **variant catalogue**: methods that in DeepXDE are
hand-rolled recipes (self-adaptive per-point weights, residual-based attention, causal time
weighting, competitive/minimax training) are **first-class solver classes** here. Swapping
`PINN(...)` for `SelfAdaptivePINN(...)` or `CausalPINN(...)` changes the training dynamics with no
change to the `Problem`. That makes PINA the natural engine for PINN-Lab's *variant-study* and
*method-comparison* cases, and an MIT-licensed fallback if a fully permissive dependency tree is
ever required.

### When to reach for PINA (vs DeepXDE)

- **You want a specific advanced variant out of the box** — `SelfAdaptivePINN`, `RBAPINN`,
  `CausalPINN`, `CompetitivePINN`, `GradientPINN`, or a deep-ensemble PINN. *(In PINN-Lab this is
  the `mine-sag-thermal` SA-PINN study and any method-comparison case.)*
- **You want Lightning ergonomics** — `Trainer`-level callbacks, multi-GPU/DDP with one flag,
  Lightning logging/checkpointing, mixed precision — without writing a training loop.
- **You need an MIT-licensed engine** for a permissive dependency tree.
- **You're mixing PINN and operator/ROM solvers** in the same library (DeepONet, FNO, GAROM).

Reach for **DeepXDE instead** when you want the lowest-friction canonical PINN with CSG geometry,
built-in RAR, and a battle-tested inverse path — which is why DeepXDE remains PINN-Lab's *primary*
engine and PINA the *secondary / variant* engine.

### Honest limitations

- **0.x API churn.** PINA is pre-1.0; module paths and signatures have moved between minor
  releases (e.g. `pina.solvers` → **`pina.solver`** singular; `pina.geometry` → **`pina.domain`**;
  `LabelTensor`/`Condition` keywords have shifted). **Pin the exact version** (`pina-mathlab==0.2.6`)
  and treat any example you find online as version-specific. Everything in this guide is verified
  against **0.2.6**.
- **Pulls PyTorch Geometric even for plain PINNs.** `torch_geometric` is a hard dependency, so a
  trivial 1-D ODE drags in PyG (and its compiled extensions). Acceptable in the **pipeline venv**,
  but a reason PINA never belongs in the live/Pyodide lane — only the *exported ONNX* ships.
- **Less geometry tooling than DeepXDE.** Domains are `CartesianDomain` / `EllipsoidDomain` /
  `SimplexDomain` and set operations; there is no CSG-grade constructive geometry like DeepXDE's.
- **Per-point-weight variants need care at export.** `SelfAdaptivePINN`/`RBAPINN` keep auxiliary
  weight tensors *outside* `solver.model`. Those are a **training device**; the deployable artifact
  is still the bare `solver.model`. Export only the model, and verify parity (see §5).
- **Research-grade, not an industrial twin.** PINA is a research library; like all PINNs it is
  usually slower/less accurate than a good FVM/FEM solver on a single well-posed forward problem.
  Its edge is inverse problems, variant studies, and producing a differentiable artifact.

### Key references

- M. Raissi, P. Perdikaris, G. E. Karniadakis. *Physics-informed neural networks.* **J. Comput.
  Phys.** 378 (2019) 686–707. DOI: [10.1016/j.jcp.2019.05.045](https://doi.org/10.1016/j.jcp.2019.05.045).
- D. Coscia, A. Ivagnes, N. Demo, G. Rozza. *Physics-Informed Neural networks for Advanced modeling
  (PINA).* **J. Open Source Software** 8(87), 5352 (2023). DOI: [10.21105/joss.05352](https://doi.org/10.21105/joss.05352).
- Variant solvers implement, respectively:
  L. McClenny & U. Braga-Neto, *Self-Adaptive PINNs*, **J. Comput. Phys.** 474 (2023) 111722,
  [arXiv:2009.04544](https://arxiv.org/abs/2009.04544) → `SelfAdaptivePINN`;
  S. J. Anagnostopoulos et al., *Residual-based attention (RBA) PINNs*,
  [arXiv:2307.00379](https://arxiv.org/abs/2307.00379) → `RBAPINN`;
  S. Wang, S. Sankaran, P. Perdikaris, *Respecting causality (causal training)*, **CMAME** 421 (2024),
  [arXiv:2203.07404](https://arxiv.org/abs/2203.07404) → `CausalPINN`;
  Q. Zeng et al., *Competitive PINNs*, [arXiv:2204.11144](https://arxiv.org/abs/2204.11144) → `CompetitivePINN`;
  J. Yu et al., *gradient-enhanced PINNs (gPINN)*, **CMAME** 393 (2022) 114823 → `GradientPINN`.

---

## 2. Install (verified)

PINA is on PyPI as **`pina-mathlab`** (the import name is `pina`). Verified 2026-06-21 against the
project README, `LICENSE.rst`, and the PyPI JSON metadata.

```bash
# PINN-Lab pipeline venv (.venv-pipeline), Python >= 3.10
python -m pip install "pina-mathlab==0.2.6"

# with tutorial / extra deps if you want to run the upstream notebooks
python -m pip install "pina-mathlab[tutorial]==0.2.6"

# ONNX export + verification (not pulled by PINA itself)
python -m pip install onnx onnxruntime
```

**Available extras:** `dev`, `test`, `doc`, `tutorial`.
**Hard dependencies** (unpinned upstream): `torch`, `lightning`, `torch_geometric`, `matplotlib`.
**Python:** `>=3.10`.

> CAOS rule: install into the project's isolated `.venv-pipeline`, never a global environment, and
> pin the version. `torch_geometric` brings compiled extensions sensitive to the torch/CUDA combo;
> if PyG wheels fail to resolve, install `torch` first to match your CUDA, then `pina-mathlab`.

GPU works through Lightning — no PINA-specific CUDA install. Pick the accelerator at `Trainer` time
(`accelerator="gpu"`); the same code runs on CPU with `accelerator="cpu"`.

---

## 3. Core API / configure

PINA's workflow is four declarative steps. Module paths below are the **0.2.x singular** paths
(verified against the current README).

```python
from pina import Trainer, Condition          # orchestration + the loss-term container
from pina.problem import SpatialProblem       # also: TimeDependentProblem, ParametricProblem, InverseProblem
from pina.domain   import CartesianDomain     # also: EllipsoidDomain, SimplexDomain
from pina.operator import grad, div, laplacian  # autodiff differential operators
from pina.equation import Equation, FixedValue  # residual wrapper / constant-value condition
from pina.model    import FeedForward         # the network (a plain torch.nn.Module)
from pina.solver   import PINN                # and variants — see the table below
```

### 3.1 `Problem` — *what to solve*

Subclass the appropriate base (`SpatialProblem`, `TimeDependentProblem`, `ParametricProblem`,
`InverseProblem`, or combinations). You declare:

- `output_variables` — names of the network outputs (e.g. `["u"]`).
- the domain(s) — `spatial_domain` / `temporal_domain` plus a `domains` dict naming each
  subdomain (interior and each boundary slice).
- `conditions` — the loss terms (next).

PINA also ships a **problem zoo** (`pina.problem.zoo`) with ready-made templates:
`Poisson2DSquareProblem`, `AllenCahnProblem`, `HelmholtzProblem`, `DiffusionReactionProblem`,
`AdvectionProblem`, `AcousticWaveProblem`, `SupervisedProblem`, etc. — useful for benchmarking a
solver against a known setup.

### 3.2 `Condition` — *the loss terms*

A `Condition` binds a **domain** to an **equation**. Two common forms:

- **PDE residual:** `Condition(domain="D", equation=Equation(residual_fn))`, where `residual_fn(input_, output_)`
  returns the residual that should be driven to 0. You build it with the differential operators:
  `grad`, `div`, `laplacian`. Outputs are `LabelTensor`s — pull a component with
  `output_.extract(["u"])` and differentiate with `grad(output_, input_, components=["u"], d=["x"])`.
- **Boundary/initial value:** `Condition(domain="x0", equation=FixedValue(1.0))` pins the output to a
  constant on that subdomain. For non-constant or operator BCs, wrap them in an `Equation`.

Other condition types exist for data-driven and mixed setups: `InputTargetCondition`,
`DomainEquationCondition`, `InputEquationCondition`, `DataCondition`.

### 3.3 Sampling — `discretise_domain`

```python
problem.discretise_domain(n=100, mode="grid", domains=["D", "x0"])
# mode ∈ {"grid", "random", "latin", "chebyshev", ...}; per-domain n via repeated calls
```

This materializes the collocation points each `Condition` is evaluated on.

### 3.4 `Solver` — *how to train it* (the variant catalogue)

All physics-informed solvers take `(problem, model, ...)` and expose `solver.model` as a plain
`torch.nn.Module`. Import from `pina.solver`:

| Solver | What it does | PINN-Lab use |
|---|---|---|
| `PINN` | vanilla physics-informed loss | baseline for every case |
| `GradientPINN` | adds the residual-gradient term ($\nabla \mathcal{N}[u]=0$) — gPINN | steep-gradient cases (`bench-burgers1d`, `mine-comminution-pbe`) |
| `CausalPINN` | weights each time slice by an exponential of earlier residuals — restores temporal causality | time-dependent / stiff (`bench-heat1d`, `poll-groundwater-rt`) |
| `SelfAdaptivePINN` | per-point trainable weights via minimax (gradient ascent on weights, descent on net) | self-adaptive study (`mine-sag-thermal`) |
| `RBAPINN` | residual-based attention: multiplicative per-point weight updated from running residuals | sharp-front cases |
| `CompetitivePINN` | adversarial/competitive (net vs. discriminator) formulation | comparison / robustness study |
| `DeepEnsemblePINN` | ensemble of PINNs for cheap epistemic UQ | UQ comparison vs. B-PINN |
| `SupervisedSolver`, `DeepEnsembleSupervisedSolver` | data-driven (no PDE) | hybrid supervised+physics |
| `ReducedOrderModelSolver`, `GAROM` | reduced-order modeling / generative ROM | ROM lane |

Neural-operator architectures (`DeepONet`, `FNO`, `AveragingNeuralOperator`) live under
`pina.model` and are trained through `SupervisedSolver` (data) or a physics-informed condition.

### 3.5 `Trainer` — Lightning, multi-GPU for free

`pina.Trainer` is a thin wrapper over `lightning.Trainer`, so every Lightning flag is available:

```python
trainer = Trainer(
    solver,
    max_epochs=1500,
    accelerator="gpu",   # "cpu" | "gpu" | "mps" | "auto"
    devices=2,           # number of GPUs (or a device list)
    strategy="ddp",      # distributed data parallel across the GPUs
    precision="16-mixed" # optional mixed precision
)
trainer.train()
```

`devices`/`strategy`/`precision` are passed straight through to Lightning — **multi-GPU PINN
training requires no PINA-specific code**, only these flags (and launching under DDP). This is the
main ergonomic win over hand-rolled loops. Adam→L-BFGS polishing is configured through the optimizer
passed to the solver (e.g. `TorchOptimizer(torch.optim.Adam, lr=1e-3)`), and a second L-BFGS phase
can be run as a follow-up `Trainer` fit.

---

## 4. Runnable minimal example

A complete, self-contained 1-D ODE: $u'(x) = u(x)$ on $[0,1]$ with $u(0)=1$ (analytic
solution $u(x)=e^{x}$). Trains a `PINN`, swaps in a variant in one line, then exports `solver.model`
to ONNX. **API verified against PINA 0.2.6.**

```python
import torch
from pina import Trainer, Condition
from pina.problem import SpatialProblem
from pina.domain import CartesianDomain
from pina.operator import grad
from pina.equation import Equation, FixedValue
from pina.model import FeedForward
from pina.solver import PINN            # swap for SelfAdaptivePINN / CausalPINN / RBAPINN / ...


# 1) PDE residual: u'(x) - u(x) = 0
def ode_residual(input_, output_):
    u_x = grad(output_, input_, components=["u"], d=["x"])
    u = output_.extract(["u"])
    return u_x - u


# 2) Problem: output var, domains, and the loss-term conditions
class SimpleODE(SpatialProblem):
    output_variables = ["u"]
    spatial_domain = CartesianDomain({"x": [0, 1]})
    domains = {
        "x0": CartesianDomain({"x": 0.0}),     # boundary point x = 0
        "D":  CartesianDomain({"x": [0, 1]}),  # interior
    }
    conditions = {
        "bc":   Condition(domain="x0", equation=FixedValue(1.0)),        # u(0) = 1
        "phys": Condition(domain="D",  equation=Equation(ode_residual)), # PDE residual
    }


problem = SimpleODE()

# 3) Sample collocation points on each domain
problem.discretise_domain(n=20,  mode="grid", domains=["x0"])
problem.discretise_domain(n=100, mode="grid", domains=["D"])

# 4) Model + solver + Lightning Trainer
model = FeedForward(input_dimensions=1, output_dimensions=1, layers=[32, 32])
solver = PINN(problem, model)
trainer = Trainer(solver, max_epochs=1500, accelerator="cpu")  # accelerator="gpu", devices=2, strategy="ddp" for multi-GPU
trainer.train()

# 5) Inference (sanity check vs analytic e^x)
solver.model.eval()
with torch.no_grad():
    xs = torch.linspace(0, 1, 5).reshape(-1, 1)
    pred = solver.model(xs)
    print("pred:", pred.flatten().tolist())
    print("true:", torch.exp(xs).flatten().tolist())
```

**Variant swap** — to run the self-adaptive variant on the *same* problem, change one line:

```python
from pina.solver import SelfAdaptivePINN
solver = SelfAdaptivePINN(problem, model)   # everything else identical
```

---

## 5. ONNX-export notes

PINA solvers wrap a plain `torch.nn.Module` at `solver.model`, so the export is the **standard
PyTorch path** — no PINA-specific exporter, identical to the contract used for DeepXDE
(`model.net`). Export the **bare model only**, never the solver/LightningModule.

```python
import torch

solver.model.eval()
d = 1  # input dimension = number of coordinates (here x; for (x,t) use d=2)
dummy = torch.zeros(1, d)

torch.onnx.export(
    solver.model, dummy, "ode.onnx",
    input_names=["x"], output_names=["u"],
    dynamic_axes={"x": {0: "batch"}, "u": {0: "batch"}},
    opset_version=17,
)
```

**Mandatory parity check before shipping** (PINN-Lab CI guard) — the exported ONNX must reproduce
the trained model on the same points:

```python
import numpy as np, onnxruntime as ort

xs = torch.linspace(0, 1, 64).reshape(-1, 1)
with torch.no_grad():
    torch_out = solver.model(xs).cpu().numpy()

sess = ort.InferenceSession("ode.onnx")
onnx_out = sess.run(["u"], {"x": xs.numpy().astype(np.float32)})[0]

assert np.allclose(torch_out, onnx_out, atol=1e-5), "ONNX parity FAILED"
print("ONNX parity OK")
```

**Gotchas specific to PINA:**

- **Export `solver.model`, not the solver.** The solver is a `LightningModule` with training state;
  only `.model` is the deployable net.
- **`LabelTensor` is a training convenience, not part of the graph.** `solver.model` consumes/returns
  ordinary tensors when called directly (as above), so the traced ONNX takes a plain `[N, d]`
  Float32 tensor and returns `[N, out]` — exactly what `onnxruntime-web` needs. Keep the **column
  order** of inputs identical to `output_variables`/coordinate order, and record it in the manifest.
- **Self-adaptive / RBA weights do not export — by design.** `SelfAdaptivePINN`/`RBAPINN` carry
  auxiliary per-point weight tensors *outside* `solver.model`; they shape training, not inference.
  The exported `solver.model` is the trained field approximator and is what the parity check
  validates.
- **Hard-constraint / feature transforms must be pure tensor ops** to be traced into the graph. If a
  case applies an output ansatz or input scaling, fold it into the `nn.Module` (or wrap the model)
  so it is captured — otherwise re-apply it in the live lane and the parity check will catch the
  mismatch.
- **Neural operators (DeepONet/FNO) take multiple/structured inputs.** Their ONNX export needs the
  right `dummy` signature (branch+trunk for DeepONet; gridded field for FNO) — verify parity the
  same way before shipping.

The verified path matches PINN-Lab's train→export→web contract: train in `.venv-pipeline` →
`torch.onnx.export(solver.model, ...)` → parity-check → `models/<case>.onnx` consumed by
`onnxruntime-web` (and the same evaluation baked to `data/artifacts/<case>` for the replay lane).

---

## 6. Role in PINN-Lab (which cases / lane)

- **Lane:** **offline / precompute** only, in `.venv-pipeline`. PINA + torch + PyG never enter the
  live/Pyodide lane — only the exported `.onnx` and the baked replay artifact ship to the browser.
- **Position in the engine roster:** **secondary engine**, complementary to the primary **DeepXDE**.
  DeepXDE handles most canonical/pollution cases; PINA is the engine of choice when a case is
  fundamentally a **variant study** or needs **Lightning multi-GPU**, and is the **MIT-licensed
  fallback** if a fully permissive dependency tree is ever required.
- **Primary case — `mine-sag-thermal` 🟠:** the **SA-PINN** study uses `SelfAdaptivePINN` for the
  hybrid supervised+physics setup (self-adaptive per-point weighting), per the coverage map
  (method 6, `methods/self-adaptive`). This case is `synthetic-illustrative` and must be labeled so.
- **Method exercises sourced here:** `SelfAdaptivePINN` (method 6), `CausalPINN` (method 2 cross-check
  against the DeepXDE causal recipe), `GradientPINN` (method 7, gPINN), and the `DeepEnsemblePINN`
  ensemble-UQ comparison against the B-PINN case (method 21). PINA's in-library variants make it the
  cleanest place to **A/B a single `Problem` across solvers** for the Methodology/Benchmark pages.
- **Operator lane:** PINA can host DeepONet/FNO, but PINN-Lab's operator cases
  (`bench-darcy-operator`) default to **neuraloperator**; use PINA operators only for a like-for-like
  variant comparison.
- **Cross-references:** `docs/frameworks/deepxde/` (primary engine, same ONNX contract),
  `docs/methods/self-adaptive.md`, `docs/architecture/train-export-onnx.md` (the shared
  train→export→parity→web bridge), `docs/cases/mine-sag-thermal.md`.

---

### Verification log

All facts dated **2026-06-21**, verified against primary sources:

- License **MIT** — confirmed against `LICENSE.rst` on `mathLab/PINA@master` (the auto-generated
  README badge briefly read Apache; the actual license file is MIT).
- Version **0.2.6**, Python **>=3.10**, deps `torch`/`lightning`/`torch_geometric`/`matplotlib` —
  PyPI JSON metadata for `pina-mathlab`.
- Module paths (`pina.solver`, `pina.problem`, `pina.operator`, `pina.domain`, `pina.equation`,
  `pina.model`) and the PINN example — current `mathLab/PINA` README code block.
- Solver/Problem/Condition class names — PINA Code Documentation
  (<https://mathlab.github.io/PINA/_rst/_code.html>).
