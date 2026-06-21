# DeepXDE — the primary PINN-Lab engine

> Framework guide · PyTorch backend · the default offline precompute engine for most cases.
> Part of the PINN-Lab `docs/frameworks/` set. Companion pages: `physicsnemo/`, `neuraloperator/`, `jaxpi/`, `pina/`, `neuralpde-jl/`.

DeepXDE is the **primary engine** of PINN-Lab. Unless a case explicitly needs the GPU/geometry-heavy lane (PhysicsNeMo) or operator learning (neuraloperator), it is trained in DeepXDE on the PyTorch backend and exported to ONNX for the web lane. This page covers what it is, how to install and configure it deterministically, the core API surface PINN-Lab leans on, a runnable minimal example, RAR adaptive sampling, inverse problems via `dde.Variable`, and the train→ONNX→web bridge that the whole product hinges on.

---

## What & why

DeepXDE (Lu, Meng, Mao, Karniadakis) is a production-grade Python library for **scientific machine learning and physics-informed learning**. It lets you express a forward or inverse PDE problem declaratively — geometry, residual, boundary/initial conditions, sampler, network, optimizer — in roughly thirty lines, and trains a Physics-Informed Neural Network (PINN) against it. It is the most widely adopted, best-documented, and lowest-risk dependency in the PINN ecosystem (≈4.3k★, v1.15.0 as of late 2025), with a SIAM Review paper as the canonical citation (Lu et al., *SIAM Review* 63(1), 2021, [doi:10.1137/19M1274067](https://doi.org/10.1137/19M1274067)).

A PINN replaces a discretization by training a network $u_\theta(x)$ so that the PDE residual is driven to zero at a cloud of **collocation points** sampled inside the domain, while boundary/initial conditions are enforced as additional loss terms (or built in exactly via an output transform). For a generic PDE $\mathcal{N}[u](x)=0$ on $\Omega$ with conditions $\mathcal{B}[u]=g$ on $\partial\Omega$, the loss is

$$
\mathcal{L}(\theta) \;=\; \underbrace{\frac{1}{N_r}\sum_{i=1}^{N_r}\big|\mathcal{N}[u_\theta](x_i)\big|^2}_{\text{PDE residual}} \;+\; \sum_k \lambda_k\,\underbrace{\frac{1}{N_k}\sum_{j}\big|\mathcal{B}_k[u_\theta](x_j)-g_k(x_j)\big|^2}_{\text{BC/IC terms}} ,
$$

with all derivatives in $\mathcal{N}$ and $\mathcal{B}$ obtained by **automatic differentiation** of $u_\theta$ with respect to its inputs — never a finite-difference stencil. DeepXDE wraps this entirely: `dde.grad.jacobian` / `dde.grad.hessian` give the AD derivatives, `dde.data.PDE` / `dde.data.TimePDE` assemble the collocation sampler, and `Model.compile`/`Model.train` run the optimizer.

**Why it is PINN-Lab's default engine:**

- **Breadth of method coverage.** Native RAR/RAD adaptive sampling, five boundary-condition classes (`DirichletBC`, `NeumannBC`, `RobinBC`, `PeriodicBC`, `OperatorBC`), constructive solid geometry, hard-constraint output transforms, multi-scale Fourier feature networks, DeepONet, gPINN, fractional PINNs, and first-class **inverse problems** via `dde.Variable`. Most of the §4 method catalogue in the dossier maps onto a DeepXDE primitive.
- **Backend portability with a PyTorch escape hatch.** DeepXDE runs on TensorFlow 1.x/2.x, PyTorch, JAX and PaddlePaddle. We pin it to **PyTorch** because `model.net` is then a real `torch.nn.Module` that `torch.onnx.export` can serialize — the bridge to `onnxruntime-web`.
- **Declarative, auditable, deterministic.** A case is a small readable script; runs are reproducible with a fixed seed, which the precompute pipeline requires for trace stability.

**Honest limitations** (state these on the case Benchmark pages):

- **Not a replacement for FEM/FVM on a single well-posed forward solve.** For one forward problem a good classical solver is usually faster and more accurate (Krishnapriyan et al., NeurIPS 2021, [arXiv:2109.01050](https://arxiv.org/abs/2109.01050)). PINNs earn their place on **inverse problems, sparse/noisy data assimilation, parametric many-query surrogates, and mesh-impractical domains**.
- **Stiff / advection-dominated / high-frequency PDEs are hard** out of the box: the residual loss is ill-conditioned and the MLP has a spectral bias. These need the §4 recipe (causal training, Fourier features, curriculum, RAR, loss weighting) layered on top — DeepXDE supplies the hooks but not an automatic cure.
- **Uneven backend parity.** The JAX backend is the least complete (e.g. inverse problems are not supported on JAX). Stay on PyTorch.
- **`apply_output_transform` / `apply_feature_transform` are only ONNX-exportable if they are pure tensor ops** that trace into the graph — verify after export (see ONNX notes).
- **License is LGPL-2.1**, not MIT/Apache like several peers — see the license note below.

---

## Install (verified)

DeepXDE has **no backend pinned by default**: you install a backend first, then DeepXDE, then select the backend via the `DDE_BACKEND` environment variable. PINN-Lab fixes the backend to PyTorch.

```bash
# inside the heavy precompute venv (.venv-pipeline), never global
pip install torch            # the backend (CPU wheel shown; use the CUDA wheel on the GPU lane)
pip install deepxde          # latest stable (v1.15.0 as of 2025-12); requires Python >= 3.9
pip install onnx onnxruntime # for the ONNX export + parity check
```

**Select the PyTorch backend** — set the env var before importing (this is the supported mechanism; DeepXDE auto-detects a backend only if the var is unset, which is non-deterministic):

```bash
export DDE_BACKEND=pytorch          # Linux/macOS
```

```powershell
$env:DDE_BACKEND = "pytorch"        # Windows PowerShell (Felipe's shell)
```

Equivalently, set it in-process before the first `import deepxde`:

```python
import os
os.environ["DDE_BACKEND"] = "pytorch"
import deepxde as dde   # backend is now locked to PyTorch
```

A backend config also persists in `~/.deepxde/config.json`, but the **env var is authoritative and reproducible** — the pipeline always sets it explicitly so a case run is hermetic regardless of any prior `dde.backend.set_default_backend(...)` call.

Pin the exact versions in `requirements-precompute.txt` (`deepxde==1.15.0`, the matching `torch`, `onnx`, `onnxruntime`) so the trace is stable across machines. The GPU lane uses the same `deepxde` pin with the CUDA `torch` wheel in `requirements-precompute-gpu.txt`.

Sources: [DeepXDE on PyPI](https://pypi.org/project/DeepXDE/) · [Install and Setup](https://deepxde.readthedocs.io/en/latest/user/installation.html).

### License note (LGPL-2.1)

DeepXDE is licensed **LGPL-2.1** ([LICENSE](https://github.com/lululxvi/deepxde/blob/master/LICENSE)), more copyleft than the MIT/Apache peers (PINA, TorchPhysics, neuraloperator, PhysicsNeMo). As a **library dependency** that PINN-Lab imports but does not modify or statically link into a distributed binary, LGPL-2.1 does not encumber the product, and the **exported `.onnx` artifact is your own data**, not a derivative of DeepXDE's source — so the web lane is unencumbered. If a fully permissive dependency tree is ever required, the drop-in alternatives are **PINA (MIT)** or **TorchPhysics (Apache-2.0)**; PINN-Lab keeps DeepXDE as the default and documents those as escape hatches (dossier §8 open question 4).

---

## Core API / configure

The DeepXDE workflow, in the order PINN-Lab uses it:

**1. Geometry / time domain.** Spatial geometry, optionally crossed with a time domain.

```python
geom       = dde.geometry.Interval(-1, 1)            # 1D; also Rectangle, Disk, Polygon, Sphere, CSG unions/diffs
timedomain = dde.geometry.TimeDomain(0, 0.99)
geomtime   = dde.geometry.GeometryXTime(geom, timedomain)   # space-time domain for transient PDEs
```

**2. PDE residual** as a Python function `pde(x, y)` returning the residual(s). `x` is the input coordinate tensor (columns are the independent variables), `y = u_θ(x)` is the network output. Derivatives come from AD helpers:

- `dde.grad.jacobian(y, x, i, j)` — $\partial y_i / \partial x_j$ (first derivative; use for $\partial_t$, $\partial_x$, advection terms).
- `dde.grad.hessian(y, x, i, j)` — $\partial^2 y_i / \partial x_i \partial x_j$ (second derivative; use for diffusion/Laplacian terms).

The column index convention is "which input variable": for a `GeometryXTime` 1D problem, `j=0` is $x$ and `j=1` is $t$.

```python
def pde(x, y):
    dy_x  = dde.grad.jacobian(y, x, i=0, j=0)   # u_x
    dy_t  = dde.grad.jacobian(y, x, i=0, j=1)   # u_t
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)    # u_xx
    return dy_t + y * dy_x - 0.01 / np.pi * dy_xx   # viscous Burgers residual
```

**3. Boundary / initial conditions** via `dde.icbc.*`. Each takes the geometry, the prescribed value (a function of `x`), and a boolean selector picking the relevant boundary/initial points:

- `dde.icbc.DirichletBC(geom, func, on_boundary)` — $u = g$.
- `dde.icbc.NeumannBC(...)`, `dde.icbc.RobinBC(geom, func(X, y), on_boundary)`, `dde.icbc.PeriodicBC(...)`, `dde.icbc.OperatorBC(...)` for flux / mixed / periodic / general-operator conditions.
- `dde.icbc.IC(geom, func, on_initial)` — initial condition for transient problems.

```python
bc = dde.icbc.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
ic = dde.icbc.IC(geomtime, lambda x: -np.sin(np.pi * x[:, 0:1]), lambda _, on_initial: on_initial)
```

**4. Data object** — the collocation sampler that ties residual + conditions to point counts:

- `dde.data.PDE(geom, pde, ic_bcs, num_domain, num_boundary, solution=None, num_test=None)` — steady problems.
- `dde.data.TimePDE(geomtime, pde, ic_bcs, num_domain, num_boundary, num_initial, ...)` — transient problems (adds `num_initial`).

```python
data = dde.data.TimePDE(geomtime, pde, [bc, ic],
                        num_domain=2540, num_boundary=80, num_initial=160)
```

**5. Network.** `dde.nn.FNN(layer_sizes, activation, initializer)` is the standard MLP; `dde.nn.PFNN` for parallel sub-nets, `dde.nn.DeepONet`/`dde.nn.DeepONetCartesianProd` for operator learning, `dde.nn.MsFFN` for multi-scale Fourier features.

```python
net = dde.nn.FNN([2] + [20] * 3 + [1], "tanh", "Glorot normal")
```

**6. Transforms (optional but central to PINN-Lab).**

- `net.apply_feature_transform(fn)` — map raw inputs before the first layer, e.g. non-dimensionalization or a Fourier-feature embedding $\gamma(x)=[\cos(Bx),\sin(Bx)]$.
- `net.apply_output_transform(fn)` — postprocess the output, the mechanism for **hard-constraint (distance-function) ansätze**: build $\hat u(x)=g(x)+\phi(x)\,N_\theta(x)$ with $\phi$ vanishing on the boundary, so the BC holds *exactly* for any weights and the BC loss term is removed.

```python
# hard Dirichlet: u(±1)=0 enforced exactly via the factor (1 - x^2)
net.apply_output_transform(lambda x, y: (1 - x[:, 0:1] ** 2) * y)
```

**7. Compile → train (Adam → L-BFGS).** The reliable recipe is Adam for global exploration, then L-BFGS (full-batch) for high-accuracy local refinement on the stiff residual loss:

```python
model = dde.Model(data, net)
model.compile("adam", lr=1e-3, metrics=["l2 relative error"])
model.train(iterations=15000)
model.compile("L-BFGS")                 # second stage; "L-BFGS-B" also accepted
losshistory, train_state = model.train()
```

**8. Predict / read inverse coefficients.** `model.predict(X)` evaluates the field; `model.predict(X, operator=pde)` evaluates the residual (used by RAR and the parity/benchmark checks). For inverse problems the recovered coefficient is read from its `dde.Variable` (below).

Sources: [Burgers forward demo](https://deepxde.readthedocs.io/en/stable/demos/pinn_forward/burgers.html) · [Diffusion-resampling (RAR) demo](https://deepxde.readthedocs.io/en/stable/demos/pinn_forward/diffusion.1d.resample.html).

### RAR — residual-based adaptive sampling

RAR (Residual-based Adaptive Refinement) concentrates collocation points where the PDE is most violated — the high-leverage trick for sharp-front / shock solutions (Burgers, Allen-Cahn, settling fronts). Sample a large candidate pool, evaluate the residual, and add the worst points as **anchors** via `data.add_anchors`, retraining between additions:

```python
X = geomtime.random_points(100000)              # large candidate pool
err = 1.0
while err > 0.005:
    f = model.predict(X, operator=pde)          # PDE residual at every candidate
    err_eq = np.absolute(f)
    err = np.mean(err_eq)
    x_id = np.argmax(err_eq)                     # the single worst-violated point
    data.add_anchors(X[x_id])                    # inject it into the training set
    model.compile("adam", lr=1e-3)
    model.train(iterations=10000)
    model.compile("L-BFGS")
    model.train()
```

Add the top-$k$ points per round (greedy RAR-G) for faster convergence; the density-based RAD/RAR-D variant draws from $p(x)\propto \varepsilon(x)^k/\mathbb{E}[\varepsilon^k]+c$ ($k=c=1$ a robust default). Reference: Wu, Zhu, Tan, Kartha, Lu, *CMAME* 2023, [arXiv:2207.10289](https://arxiv.org/abs/2207.10289).

### Inverse problems via `dde.Variable`

Make an unknown PDE coefficient a trainable scalar, reference it inside `pde`, and pass it to `compile` via `external_trainable_variables` so the optimizer learns it jointly with the field from sparse observation data. The PDE acts as a physics prior that regularizes the under-determined inverse problem.

```python
C = dde.Variable(2.0)                            # unknown coefficient, initial guess 2.0

def pde(x, y):
    dy_t  = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return dy_t - C * dy_xx + ...                # C used directly in the residual

# observe_y = dde.icbc.PointSetBC(observe_X, observe_y_values)  # the sparse data term
model.compile("adam", lr=1e-3, metrics=["l2 relative error"],
              external_trainable_variables=C)
variable = dde.callbacks.VariableValue(C, period=1000)          # log C every 1000 iters
losshistory, train_state = model.train(iterations=50000, callbacks=[variable])
```

After training, read the recovered value from `C` (its `.value` / printed by the callback). This is the mechanism behind PINN-Lab's inverse cases — `mine-flotation-kinetics` (rate constant $k$), `poll-air-source-inv` (source term), `poll-groundwater-rt`, `ind-heat2d-inverse` (conductivity $k(x)$). Reference: Raissi, Perdikaris, Karniadakis, *JCP* 2019, [doi:10.1016/j.jcp.2019.05.045](https://doi.org/10.1016/j.jcp.2019.05.045).

Source: [Inverse diffusion demo](https://deepxde.readthedocs.io/en/latest/demos/pinn_inverse/diffusion.1d.inverse.html).

---

## Runnable minimal example

A complete 1D viscous Burgers case ($u_t + u\,u_x = \tfrac{0.01}{\pi}u_{xx}$ on $x\in[-1,1]$, $t\in[0,0.99]$, $u(x,0)=-\sin(\pi x)$, $u(\pm1,t)=0$) — train, evaluate, and export to ONNX. This is the `bench-burgers1d` skeleton (RAR-ready; the shock at $t\to1$ is the RAR showcase). Run with `DDE_BACKEND=pytorch`.

```python
import os
os.environ["DDE_BACKEND"] = "pytorch"     # lock the backend before importing deepxde
import deepxde as dde
import numpy as np
import torch

# 1. domain
geom       = dde.geometry.Interval(-1, 1)
timedomain = dde.geometry.TimeDomain(0, 0.99)
geomtime   = dde.geometry.GeometryXTime(geom, timedomain)

# 2. PDE residual (AD derivatives, no stencils)
def pde(x, y):
    dy_x  = dde.grad.jacobian(y, x, i=0, j=0)
    dy_t  = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return dy_t + y * dy_x - 0.01 / np.pi * dy_xx

# 3. conditions
bc = dde.icbc.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
ic = dde.icbc.IC(geomtime, lambda x: -np.sin(np.pi * x[:, 0:1]),
                 lambda _, on_initial: on_initial)

# 4. collocation sampler
data = dde.data.TimePDE(geomtime, pde, [bc, ic],
                        num_domain=2540, num_boundary=80, num_initial=160)

# 5. network
net = dde.nn.FNN([2] + [20] * 3 + [1], "tanh", "Glorot normal")

# 6. + 7. compile and train (Adam -> L-BFGS)
model = dde.Model(data, net)
model.compile("adam", lr=1e-3)
model.train(iterations=15000)
model.compile("L-BFGS")
losshistory, train_state = model.train()

# 8. export the RAW net to ONNX (the web bridge) and verify parity
model.net.eval()
dummy = torch.zeros(1, 2)                  # input dim d = 2 (x, t)
torch.onnx.export(
    model.net, dummy, "bench-burgers1d.onnx",
    input_names=["x"], output_names=["u"],
    dynamic_axes={"x": {0: "batch"}, "u": {0: "batch"}},
    opset_version=17,
)

# parity check: ONNX must match model.predict within tolerance
import onnxruntime as ort
X = geomtime.random_points(1000).astype(np.float32)
ref = model.predict(X)
sess = ort.InferenceSession("bench-burgers1d.onnx")
got = sess.run(["u"], {"x": X})[0]
assert np.max(np.abs(ref - got)) < 1e-4, "ONNX export diverged from model.predict"
print("ONNX parity OK")
```

> The numerical reference for Burgers is Raissi's `Burgers.npz` / `burgers_shock.mat` (MIT, Chebfun spectral solution) — a **validation anchor, not real-world data**. Label it as such on the Benchmark page (coverage map §A.1).

---

## ONNX-export notes

The product hinges on **exported PINNs running client-side**. The verified path and its pitfalls:

1. **Export the raw net, not the `dde.Model`.** `model.net` is the underlying `torch.nn.Module` (true only on the PyTorch backend). Do **not** use `model.save()` — it writes non-portable training checkpoints, not an inference graph.
2. **Call `model.net.eval()`** before export, supply a `dummy` of shape `(1, d)` with `d =` input dimension (e.g. 2 for $(x,t)$), and set `dynamic_axes` so the batch dimension is variable — the web lane feeds an `[N, d]` Float32 coordinate tensor and expects `[N, out]`. Use `opset_version=17`.
3. **Feature/output transforms are only captured if they trace into the graph.** `apply_feature_transform` / `apply_output_transform` and hard-BC ansätze are pure-tensor lambdas; `torch.onnx.export` traces the `nn.Module.forward`, which on the PyTorch backend includes these transforms — **but verify**, because anything using NumPy or non-traceable control flow silently drops from the graph.
4. **Parity check is mandatory** (shown in the example): assert the ONNX output matches `model.predict()` on the same points within tolerance (e.g. `1e-4`). The CI guard `ONNX-vs-model.predict parity` enforces this per case; a failing parity means a transform did not trace and the case must be fixed before shipping.
5. **Web side:** `onnxruntime-web` loads `<case>.onnx`; input = coordinates, output = the field. The net is a handful of dense+activation layers, so it runs in real time in the browser (WASM/WebGPU). The same evaluation is baked to `data/artifacts/<case>` + `manifests/<case>.json` for the replay fallback lane.

> PhysicsNeMo uses a different exporter (`physicsnemo.deploy.onnx`, cu12 build) — see `frameworks/physicsnemo/`. neuraloperator/PINA use the standard `torch.onnx.export` on a plain `nn.Module`, same as here.

---

## Role in PINN-Lab (which cases / lane)

**Lane:** the **offline precompute lane** (`requirements-precompute.txt`, `.venv-pipeline`). DeepXDE never runs in the live lane — the browser runs the exported ONNX via `onnxruntime-web`, not DeepXDE. On the GPU lane DeepXDE uses the same pins with a CUDA `torch` wheel.

**Cases (per the coverage map):** DeepXDE is the engine for the **majority of Group A/B/C cases** —

| group | DeepXDE cases |
|---|---|
| **A · canonical** | `bench-poisson2d` (anchor), `bench-heat1d`, `bench-burgers1d`, `bench-allencahn`, `bench-wave1d`; `bench-navier-cavity` shared with PhysicsNeMo on the GPU lane |
| **B · mining** | `mine-flotation-kinetics` (inverse $k$), `mine-heap-leach-rt` (shared w/ PhysicsNeMo), `mine-comminution-pbe` (integro-differential), `mine-thickener-settling` (RAR front), `mine-sag-thermal` (shared w/ PINA) |
| **C · pollution** | `poll-air-source-inv` (inverse, real OpenAQ/EPA data), `poll-ocean-transport`, `poll-groundwater-rt` (inverse, real USGS NWIS), `poll-soil-barrier`, `poll-tailings-seepage` (Richards), `poll-source-uq-bpinn` (shared w/ NeuralUQ) |
| **D · industrial** | `ind-heat2d-inverse` (inverse $k(x)$), `ind-helmholtz` |
| **control** | `ctrl-zero-source` (degenerate: $S=0$, IC $=0$ → field $\equiv 0$; the engine must not crash) |

The **operator cases** (`bench-darcy-operator` and parametric surrogates) go to neuraloperator; the **GPU/3D-geometry-heavy** cases go to PhysicsNeMo. Everything else is DeepXDE — which is why this is the most-used engine and the page to read first.

**Method primitives DeepXDE supplies to the §4 catalogue:** RAR/RAD adaptive sampling, hard-constraint output transforms, Fourier-feature networks (`MsFFN`), gPINN, Adam→L-BFGS, DeepONet, and native inverse via `dde.Variable`. Causal training, modified-MLP, PirateNets, NTK/grad-norm weighting and SOAP are layered on as manual recipes (technique source: `frameworks/jaxpi/`).

---

## Key references

- Lu, Meng, Mao, Karniadakis. **DeepXDE: A deep learning library for solving differential equations.** *SIAM Review* 63(1):208–228, 2021. [doi:10.1137/19M1274067](https://doi.org/10.1137/19M1274067).
- Raissi, Perdikaris, Karniadakis. **Physics-informed neural networks.** *JCP* 378:686–707, 2019. [doi:10.1016/j.jcp.2019.05.045](https://doi.org/10.1016/j.jcp.2019.05.045).
- Wu, Zhu, Tan, Kartha, Lu. **A comprehensive study of non-adaptive and residual-based adaptive sampling for PINNs.** *CMAME* 403, 2023. [arXiv:2207.10289](https://arxiv.org/abs/2207.10289).
- Krishnapriyan, Gholami, Zhe, Kirby, Mahoney. **Characterizing possible failure modes in PINNs.** NeurIPS 2021. [arXiv:2109.01050](https://arxiv.org/abs/2109.01050).
- Code & docs: [github.com/lululxvi/deepxde](https://github.com/lululxvi/deepxde) (LGPL-2.1) · [deepxde.readthedocs.io](https://deepxde.readthedocs.io/).

---

*Authored as PINN-Lab is built (ADR-0056 docs-as-you-version). Update this page when the DeepXDE pin or the ONNX export path changes.*
