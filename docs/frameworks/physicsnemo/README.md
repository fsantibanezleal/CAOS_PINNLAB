# NVIDIA PhysicsNeMo (+ Sym) — the GPU / 3D secondary engine

> **Lane: GPU (NVIDIA-only).** This framework requires a CUDA-capable NVIDIA GPU for any
> non-trivial training and for the ONNX-deploy path. On a machine without an NVIDIA GPU it is
> *documentation-only*; the cases it owns (`bench-navier-cavity`, `mine-heap-leach-rt`) fall back to
> DeepXDE / CPU at reduced fidelity and must be labelled accordingly. See
> [`docs/guides/03_gpu-lane.md`](../../guides/03_gpu-lane.md).

PhysicsNeMo is PINN-Lab's **secondary precompute engine**, used specifically where DeepXDE's
declarative/CPU recipe runs out of road: large or geometry-heavy 3D domains, and cases that benefit
from a first-class, supported ONNX-export path. It is **not** the default — DeepXDE owns most cases
(see [`docs/frameworks/deepxde/`](../deepxde/README.md)). Read this page when a case is tagged
`engine: physicsnemo` in the cases registry.

---

## What & why

**What it is.** NVIDIA PhysicsNeMo (the framework formerly called **Modulus**) is an open-source,
PyTorch-based Physics-ML toolkit. It bundles several model families under one roof — physics-informed
neural networks (PINNs), Fourier/physics-informed neural operators (FNO / PINO / DeepONet), graph
neural networks, and diffusion models — together with the symbolic PDE layer **PhysicsNeMo-Sym**
(formerly **Modulus-Sym**), an industrial training loop, and a deployment module with
**first-class ONNX export**. Licensing is **Apache-2.0** (permissive — unlike DeepXDE's LGPL-2.1),
and the current release at the time of writing is **`nvidia-physicsnemo` 2.1.1 (2026-06-08)**,
Python `>=3.11,<3.14`. ([PyPI](https://pypi.org/project/nvidia-physicsnemo/),
[GitHub](https://github.com/NVIDIA/physicsnemo))

The distinctive PhysicsNeMo-Sym idea is **symbolic PDE authoring**: you write the governing equation
in [SymPy](https://www.sympy.org/) and the framework turns it into a residual node in an autodiff
computational graph. A PINN residual is the strong form of the PDE evaluated on the network output.
For example, the 1-D wave equation

$$ \frac{\partial^2 u}{\partial t^2} - c^2 \frac{\partial^2 u}{\partial x^2} = 0 $$

is declared symbolically as `u.diff(t, 2) - (c**2 * u.diff(x)).diff(x)`, and PhysicsNeMo assembles
the chain of automatic derivatives needed to compute that residual at every collocation point. The
training objective is the usual composite PINN loss

$$ \mathcal{L}(\theta) \;=\; \lambda_r \,\mathcal{L}_{\text{PDE}}(\theta) \;+\; \lambda_b \,\mathcal{L}_{\text{BC}}(\theta) \;+\; \lambda_i \,\mathcal{L}_{\text{IC}}(\theta) \;+\; \lambda_d \,\mathcal{L}_{\text{data}}(\theta), $$

where each term is a mean-squared residual over its sample set and the $\lambda$'s are per-constraint
weights (`lambda_weighting` in PhysicsNeMo). The PDE term

$$ \mathcal{L}_{\text{PDE}}(\theta) = \frac{1}{N_r}\sum_{i=1}^{N_r}\bigl\| \mathcal{N}\!\left[u_\theta\right](x_i) \bigr\|^2 $$

is exactly the symbolic residual $\mathcal{N}[\cdot]$ evaluated on interior points $x_i$.

**Why it earns a place in the stack** (the things DeepXDE does *not* give you cleanly):

- **Geometry-heavy and 3D domains via SDFs.** PhysicsNeMo represents geometry by a
  **signed distance function** $\phi(x)$ (negative inside, zero on the boundary, positive outside).
  It can build constructive-solid-geometry primitives or ingest a tessellated **STL** mesh, then
  sample interior and boundary collocation points and supply $\phi$ and its surface normals to the
  loss. This is the practical enabler for real 3D engineering geometry, where building a CSG/analytic
  boundary by hand (as DeepXDE largely expects) is painful. SDF-based sampling utilities ship in
  `physicsnemo.nn.functional` (`signed_distance_field`, `mesh_poisson_disk_sample`,
  `mesh_to_voxel_fraction`).
- **One toolbox for PINN *and* operator *and* GNN.** When a mining case needs a parametric surrogate
  (FNO/PINO) and a strong-form PINN in the same study, you stay in one library and one export path.
- **First-class ONNX deploy.** `physicsnemo.deploy.onnx` exports any `torch.nn.Module` /
  `physicsnemo.Module` to a portable ONNX artifact — the bridge to PINN-Lab's `onnxruntime-web`
  live lane. DeepXDE has no built-in exporter (you call `torch.onnx.export` on `model.net` yourself).
- **Industrial training loop.** Hydra-driven config, multi-GPU/multi-node, mixed precision, TensorBoard,
  checkpointing — the engineering scaffolding for the heavy cases.

**Honest scope (what it is *not*).** PhysicsNeMo does **not** make a PINN beat a good FEM/FVM solver on
a single well-posed forward solve; the PINN-Lab thesis on that (Krishnapriyan 2021 et al.) is unchanged.
It is a heavier, narrower dependency than DeepXDE — **NVIDIA-GPU-only**, a non-trivial CUDA install
matrix, and it carries name-churn from the Modulus→PhysicsNeMo rename (old code imports `modulus` /
`modulus.sym`; new code imports `physicsnemo` / `physicsnemo.sym`). Use it where its strengths
(3D/SDF, supported ONNX) actually pay for that weight — not as a default.

---

## Install (verified)

PhysicsNeMo expects **PyTorch installed first** for your platform, then the package with the right
CUDA extra. Verified against the official
[installation guide](https://docs.nvidia.com/physicsnemo/latest/getting-started/installation.html)
and [PyPI](https://pypi.org/project/nvidia-physicsnemo/) (June 2026).

```bash
# 0) isolated env (CAOS convention — never global)
python -m venv .venv-pipeline-gpu        # Python 3.11–3.13
source .venv-pipeline-gpu/bin/activate   # (Windows: .venv-pipeline-gpu\Scripts\activate)

# 1) PyTorch for your platform first (see https://pytorch.org/get-started/locally/)
#    then PhysicsNeMo with the CUDA-12 backend + neural-network extras + symbolic PDE layer:
pip install "nvidia-physicsnemo[cu12,nn-extras,sym]"
```

Key facts (verified):

- **CUDA selector.** `cu12` pins the CUDA-12 wheel set; `cu13` pins CUDA-13. With **neither** extra,
  PyTorch installs from PyPI with its default build. **Use `cu12`** for PINN-Lab: the
  `onnxruntime-gpu` runtime that `physicsnemo.deploy.onnx` checks against lags CUDA-13 support, so the
  ONNX round-trip is the safe lane on cu12.
- **`[sym]` extra** pulls in PhysicsNeMo-Sym — the symbolic SymPy PDE authoring, `instantiate_arch`,
  the constraint/`Solver`/`Domain` API, and the `PhysicsInformer` residual utility. Without it you get
  the model zoo and training utilities but **not** the symbolic PINN layer this page is about.
- **`[nn-extras]`** adds optimised NN kernels (the recommended baseline NN extra).
- **Other extras** available and combinable the same way:
  `utils-extras`, `mesh-extras`, `model-extras`, `datapipes-extras`, `gnns`, `perf`, `uq-extras`,
  `natten-cu12` / `natten-cu13`.
- **Container alternative** (pulls a fully provisioned CUDA + PhysicsNeMo image):
  ```bash
  docker pull nvcr.io/nvidia/physicsnemo/physicsnemo:<tag>
  docker run --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
             --runtime nvidia -it --rm nvcr.io/nvidia/physicsnemo/physicsnemo:<tag> bash
  ```

For the ONNX round-trip you additionally need the ONNX runtime; `physicsnemo.deploy.onnx` raises and
points you to `pip install onnxruntime onnxruntime_gpu` if it is missing:

```bash
pip install onnx onnxruntime onnxruntime-gpu
```

> **Smoke check (GPU lane only).** `python -c "import torch, physicsnemo, physicsnemo.sym; print(torch.cuda.is_available())"`
> must print `True` before you trust any timing/fidelity number from this engine.

---

## Core API / configure

Two ways to drive PhysicsNeMo-Sym. Pick per case.

### Path A — the declarative `Solver` (Hydra-configured)

The classic Modulus-Sym workflow: author the PDE symbolically, instantiate an architecture from config,
build `make_nodes()`, attach pointwise constraints to a `Domain`, and let `Solver` run the training
loop. Config (architecture size, optimiser, batch sizes, max steps) lives in a Hydra YAML, injected via
the `@physicsnemo.sym.main` decorator. Verified import paths
([inverse-problem tutorial](https://docs.nvidia.com/physicsnemo/25.11/physicsnemo-sym/user_guide/foundational/inverse_problem.html)):

```python
import physicsnemo.sym
from physicsnemo.sym.hydra import instantiate_arch, PhysicsNeMoConfig, to_absolute_path
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)
from physicsnemo.sym.eq.pde import PDE
from physicsnemo.sym.key import Key
from sympy import Symbol, Function
```

Authoring blocks:

- **Equation** — subclass `PDE`, declare SymPy symbols/functions, set `self.equations` to a dict of
  named residuals (each `== 0` at the solution).
- **Architecture** — `instantiate_arch(input_keys=[Key("x"), Key("t")], output_keys=[Key("u")], cfg=cfg.arch.fully_connected)`
  builds a `FullyConnectedArch` (configurable: `nr_layers`, `layer_size`, activation, weight-norm,
  adaptive activations, Fourier/periodic encodings) from the Hydra config — so you tune width/depth from
  YAML/CLI without touching code.
- **Nodes** — `nodes = eq.make_nodes() + [net.make_node(name="u_network")]` wires residuals and the
  network into one autodiff graph. For **inverse** problems, make the unknown coefficient a `Symbol`,
  give it its own small net, and `detach_names=[...]` the known quantities out of the residual's
  gradient path.
- **Constraints** — `PointwiseInteriorConstraint` (PDE residual = 0 on the interior),
  `PointwiseBoundaryConstraint` (BC/IC on the boundary), each with `lambda_weighting`; data terms via
  `PointwiseConstraint.from_numpy(...)`. Add them to a `Domain` with `domain.add_constraint(...)`.
- **Run** — `Solver(cfg, domain).solve()`.

### Path B — `PhysicsInformer` in a plain PyTorch loop (v2 idiom)

The modern PhysicsNeMo path drops the framework-owned loop. You write an ordinary PyTorch training loop
and use `physicsnemo.sym.eq.phy_informer.PhysicsInformer` to compute the PDE residual on your model
outputs and add it to the loss. Any `physicsnemo.Module` or plain `torch.nn.Module` plugs straight in —
no wrapper class, no dict-format conversion. Spatial gradients are computed by autodiff (ideal for
fully-differentiable point-cloud networks). Prefer this path when you want full control of the optimiser
schedule (e.g. Adam→L-BFGS, SOAP) or to interleave custom sampling/curriculum logic.
([PINN tutorials](https://docs.nvidia.com/physicsnemo/26.05/user-guide/pinns-tutorials/index.html))

---

## Runnable minimal example

A self-contained 1-D wave PINN via **Path B** (`PhysicsInformer` + a plain PyTorch loop), ending in the
**ONNX export** PINN-Lab depends on. Solves $u_{tt} - c^2 u_{xx} = 0$. Run on an NVIDIA GPU with
`nvidia-physicsnemo[cu12,nn-extras,sym]` + `onnxruntime-gpu` installed.

```python
import torch
from sympy import Symbol, Function

from physicsnemo.sym.eq.pde import PDE
from physicsnemo.sym.eq.phy_informer import PhysicsInformer
from physicsnemo.sym.models.fully_connected import FullyConnectedArch
from physicsnemo.sym.key import Key
from physicsnemo.deploy.onnx import export_to_onnx_stream  # first-class ONNX deploy

device = "cuda"


# 1) Author the PDE symbolically (the PhysicsNeMo-Sym idea).
class WaveEquation1D(PDE):
    def __init__(self, c: float = 1.0):
        x, t = Symbol("x"), Symbol("t")
        u = Function("u")(x, t)
        # residual == 0 at the solution
        self.equations = {"wave_equation": u.diff(t, 2) - (c ** 2 * u.diff(x)).diff(x)}


eq = WaveEquation1D(c=1.0)

# 2) Network: 2 inputs (x, t) -> 1 output (u).
net = FullyConnectedArch(
    input_keys=[Key("x"), Key("t")],
    output_keys=[Key("u")],
    nr_layers=4,
    layer_size=64,
).to(device)

# 3) Residual utility: builds the autodiff graph for the named residual.
phy = PhysicsInformer(
    required_outputs=["wave_equation"],
    equations=eq,
    grad_method="autodiff",
    device=device,
)

opt = torch.optim.Adam(net.parameters(), lr=1e-3)

# 4) Plain PyTorch training loop (interior residual + a tiny IC term sketch).
for step in range(20000):
    opt.zero_grad()
    # interior collocation points on (x, t) in [-1, 1] x [0, 1]
    x = (2 * torch.rand(4096, 1, device=device) - 1).requires_grad_(True)
    t = torch.rand(4096, 1, device=device).requires_grad_(True)
    out = net({"x": x, "t": t})                       # {"u": ...}
    res = phy.forward({"x": x, "t": t, "u": out["u"]})  # {"wave_equation": residual}
    loss = (res["wave_equation"] ** 2).mean()
    # ... add IC/BC mean-squared terms here (u(x,0)=u0, etc.) ...
    loss.backward()
    opt.step()

# 5) First-class ONNX export -> bytes -> file (PINN-Lab live-lane artifact).
net.eval()
sample = (torch.randn(1, 1, device=device), torch.randn(1, 1, device=device))
onnx_bytes = export_to_onnx_stream(net, sample)   # opset 15; moves model to CPU internally
with open("wave.onnx", "wb") as f:
    f.write(onnx_bytes)
```

> The `Solver`/Hydra route (Path A) is the alternative for the heavy declarative cases; it replaces the
> loop in steps 3–4 with `PointwiseInteriorConstraint` / `PointwiseBoundaryConstraint` on a `Domain` and
> a `Solver(cfg, domain).solve()` call, configured from a Hydra YAML.

---

## ONNX-export notes

This is why PhysicsNeMo is the **reference ONNX-export path** for PINN-Lab. The deploy helpers live in
`physicsnemo.deploy.onnx`
([source](https://docs.nvidia.com/physicsnemo/25.08/_modules/physicsnemo/deploy/onnx/utils.html)):

| function | signature (verified) | use |
|---|---|---|
| `export_to_onnx_stream` | `(model: nn.Module, invars: Tensor | tuple[Tensor,...], verbose=False) -> bytes` | Export model to an ONNX **byte stream** (write to disk yourself). **opset 15.** |
| `run_onnx_inference` | `(model: bytes | str, invars, device="cuda") -> tuple[Tensor]` | Run inference through an ORT session; outputs returned on CPU. |
| `get_ort_session` | `(model: bytes | str, device="cuda") -> ort.InferenceSession` | Build an ORT `InferenceSession` (CUDA default, CPU fallback). |
| `check_ort_install` | decorator | Guards the above; raises `ModuleNotFoundError` pointing to `pip install onnxruntime onnxruntime_gpu` if ORT is absent. |

Gotchas (all verified, and load-bearing for PINN-Lab's parity gate):

- **opset is 15.** `export_to_onnx_stream` exports at `opset_version=15`. If a case needs a newer opset
  op, export the bare `net` with `torch.onnx.export(..., opset_version=17)` directly — both produce a
  plain ONNX graph `onnxruntime-web` can load.
- **The model is moved to CPU for export and restored** to its original device afterwards. Exporting
  **while CUDA graphs are active will break** — export *after* training, with the net in `eval()`.
- **CUDA-12 lane only.** `onnxruntime-gpu` lags CUDA-13; keep the env on `cu12` so the export →
  ORT round-trip is supported.
- **Mandatory parity check.** After export, assert the ONNX output matches the trained net on the same
  points (PINN-Lab CI guard: ONNX-vs-`model.predict` parity). Any feature/output transform or hard-BC
  ansatz is only captured if it is a pure tensor op traced into the graph — Python-side post-processing
  is **not** exported and must be re-implemented in the web lane or folded into the net.
- **Export the bare net, not the training object.** Pass the `torch.nn.Module` / `physicsnemo.Module`,
  not a `Solver`/`Domain` wrapper — the live lane needs coordinates-in → field-out, nothing else.

The exported `wave.onnx` is exactly what `onnxruntime-web` loads in the App page: feed an `[N, d]`
Float32 coordinate tensor, get `[N, out]` field predictions, render. See
[`docs/architecture/train-export-onnx.md`](../../architecture/train-export-onnx.md) and
[`docs/guides/05_train-export-to-web.md`](../../guides/05_train-export-to-web.md).

---

## Role in PINN-Lab (which cases / lane)

**Lane:** GPU precompute (`requirements-precompute-gpu.txt` / `.venv-pipeline-gpu`). PhysicsNeMo is the
**secondary** offline engine; DeepXDE is primary and owns most cases. PhysicsNeMo is selected only where
its strengths apply.

| case | why PhysicsNeMo | method(s) | status |
|---|---|---|---|
| `bench-navier-cavity` | the hard steady-NS lid-driven-cavity benchmark; benefits from the industrial loop + the full Modulus-Sym recipe; co-owned with the jaxpi technique port | modified-MLP, NTK + grad-norm weighting | GPU lane |
| `mine-heap-leach-rt` | heap/in-situ-leach reactive transport in **3D / geometry-heavy** domains where SDF sampling and CSG/STL geometry are the enabler; co-owned with DeepXDE for the 1-D/2-D variants | reactive transport, domain decomposition (cPINN/XPINN) | GPU lane |

It is also the **reference ONNX-export path**: even DeepXDE-trained cases follow the same export-then-
parity-verify discipline documented here, and `physicsnemo.deploy.onnx` is the canonical helper when a
case is trained in PhysicsNeMo.

**When to choose PhysicsNeMo over DeepXDE** (decision rule for new cases):

- **Choose PhysicsNeMo** when the case is **3D and/or geometry-heavy** (real STL/CSG geometry, SDF
  sampling), needs **multi-GPU / large-scale** training, mixes **PINN + operator + GNN** in one study,
  or you want the **supported ONNX-deploy** module rather than hand-rolled `torch.onnx.export`. Also
  prefer it when a **permissive (Apache-2.0)** dependency matters.
- **Stay on DeepXDE** for the canonical 1-D/2-D benchmarks, fast CPU iteration, inverse-via-`dde.Variable`,
  and anything that does not need an NVIDIA GPU. DeepXDE's ~30-line declarative PDEs are lower-friction
  for the bulk of the 20-case catalogue.
- **On a non-NVIDIA machine**, PhysicsNeMo is documentation-only: `bench-navier-cavity` and
  `mine-heap-leach-rt` run on DeepXDE/CPU at reduced fidelity and are labelled
  `synthetic-illustrative` / reduced-fidelity on their Benchmark pages (honesty bar).

---

## References

- **Framework / docs.** NVIDIA PhysicsNeMo — [GitHub](https://github.com/NVIDIA/physicsnemo) ·
  [PyPI `nvidia-physicsnemo` 2.1.1](https://pypi.org/project/nvidia-physicsnemo/) ·
  [Installation](https://docs.nvidia.com/physicsnemo/latest/getting-started/installation.html) ·
  [PhysicsNeMo-Sym API](https://docs.nvidia.com/physicsnemo/latest/physicsnemo/api/physicsnemo.sym.html) ·
  [Inverse-problem tutorial](https://docs.nvidia.com/physicsnemo/25.11/physicsnemo-sym/user_guide/foundational/inverse_problem.html) ·
  [PINN tutorials (v2 `PhysicsInformer`)](https://docs.nvidia.com/physicsnemo/26.05/user-guide/pinns-tutorials/index.html) ·
  [`physicsnemo.deploy.onnx` source](https://docs.nvidia.com/physicsnemo/25.08/_modules/physicsnemo/deploy/onnx/utils.html).
  License: Apache-2.0.
- **PINN foundations.** M. Raissi, P. Perdikaris, G.E. Karniadakis, *Physics-informed neural networks*,
  J. Comput. Phys. 378 (2019) 686–707. [doi:10.1016/j.jcp.2019.05.045](https://doi.org/10.1016/j.jcp.2019.05.045).
- **Honest limits (PINN vs classical / failure modes).** A.S. Krishnapriyan et al., *Characterizing
  possible failure modes in physics-informed neural networks*, NeurIPS 2021.
  [arXiv:2109.01050](https://arxiv.org/abs/2109.01050).
- **Hard-constraint / distance-function ansatz (used in the SDF-geometry cases).** N. Sukumar,
  A. Srivastava, *Exact imposition of boundary conditions with distance functions in PINNs*, CMAME 389
  (2022). [doi:10.1016/j.cma.2021.114333](https://doi.org/10.1016/j.cma.2021.114333).
- **Operator branch (when staying in PhysicsNeMo for FNO/PINO).** Z. Li et al., *Fourier Neural
  Operator*, ICLR 2021, [arXiv:2010.08895](https://arxiv.org/abs/2010.08895); Z. Li et al.,
  *Physics-Informed Neural Operator (PINO)*, [arXiv:2111.03794](https://arxiv.org/abs/2111.03794).

> Cross-links: engine selection — [`docs/frameworks/deepxde/`](../deepxde/README.md) ·
> operator lane — [`docs/frameworks/neuraloperator/`](../neuraloperator/README.md) ·
> the gate / live-vs-precompute decision — [`docs/architecture/the-gate.md`](../../architecture/the-gate.md).
