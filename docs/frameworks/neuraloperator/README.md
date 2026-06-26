# neuraloperator — the operator-learning engine (FNO / TFNO / SFNO / GINO)

> Framework guide for PINN-Lab. Role: the **operator-learning** lane — parametric, many-query
> surrogates that map an *input function* (e.g. a permeability field $a(x)$) to a *solution field*
> $u(x)$ over a whole family of PDE instances, not one instance at a time. This is the engine behind
> the `bench-darcy-operator` case.

---

## What & why

A classical PINN (DeepXDE, PhysicsNeMo-Sym, jaxpi) learns **one** solution $u$ of **one** PDE
instance: change the boundary data, the source, or a coefficient field and you must retrain. An
**operator learner** instead approximates the solution *operator*

$$\mathcal{G}^{\dagger} : a \mapsto u, \qquad a \in \mathcal{A}, \; u \in \mathcal{U},$$

a map between two function spaces. Once trained on samples $\{(a_i, u_i)\}$ drawn from a distribution
over $\mathcal{A}$, a single forward pass returns the solution for a **new** input function $a$ — no
retraining, no inner optimization loop. This is exactly the regime the dossier reserves operators for:
**parametric / many-query surrogates**, where the same PDE family is solved thousands of times
(uncertainty quantification, inverse design, real-time control), and where a per-instance PINN or even
a classical solver would be too slow.

`neuraloperator` is the reference PyTorch library for this, authored by the group that introduced the
**Fourier Neural Operator (FNO)** (Li et al., ICLR 2021). It is now part of the PyTorch Ecosystem
(MIT license) and provides:

| Model | Import | When to use |
|---|---|---|
| **FNO** | `from neuralop.models import FNO` | The default. Regular grids, periodic-ish domains; the Darcy case. |
| **TFNO** | `from neuralop.models import TFNO` | Tensorized/factorized FNO — fewer parameters via low-rank (Tucker/CP) factorization of the spectral weights, better generalization on small data. |
| **SFNO** | `from neuralop.models import SFNO` | Spherical FNO — uses the spherical harmonic transform instead of the FFT; for data on the sphere (weather/climate). Needs `torch_harmonics` installed. |
| **GINO** | `from neuralop.models import GINO` | Geometry-Informed NO — a GNO encoder/decoder around an FNO latent grid; for **irregular geometries / point clouds / meshes** (3D CFD over arbitrary shapes). |

(Other exported models in 2.0.0: `UNO`, `UQNO`, `FNOGNO`, `CODANO`, `RNO`, `OTNO`, `LocalNO`.)

### The Fourier layer (the math)

FNO parameterizes the kernel integral operator in **Fourier space**. A single hidden layer maps
$v_t \mapsto v_{t+1}$ by

$$v_{t+1}(x) = \sigma\!\Big( W v_t(x) + \big(\mathcal{K} v_t\big)(x) \Big),$$

where $W$ is a pointwise (1×1 conv) linear skip term and $\mathcal{K}$ is the spectral convolution

$$\big(\mathcal{K} v_t\big)(x) = \mathcal{F}^{-1}\!\Big( R \cdot \mathcal{F}(v_t) \Big)(x).$$

Here $\mathcal{F}$ is the (discrete, real) Fourier transform, $R$ is a learnable complex weight tensor
applied **only to the lowest $k_{\max}$ modes** (higher modes are truncated to zero), and
$\mathcal{F}^{-1}$ is the inverse transform. Because the learnable parameters $R$ live on a fixed set
of **modes** rather than on grid points, the layer is **discretization-invariant**: the same weights
act on an input sampled on *any* grid resolution. That single property gives FNO its two headline
capabilities:

1. **Discretization invariance** — train at 16×16, evaluate at 128×128 (the operator is defined on the
   function, not the mesh).
2. **Zero-shot super-resolution** — predict a higher-resolution output than anything seen in training,
   in one forward pass, with no fine-tuning.

The number of retained modes $k_{\max}$ per axis is the `n_modes` hyperparameter — the single most
important architectural knob.

### PINO (physics-informed operator), in one line

You can train the same FNO with a **PDE-residual loss in addition to (or instead of) the data loss**,
and impose that residual at a *higher* resolution than the coarse training data. That is **PINO** (Li
et al., 2021/2024) and it is the recipe that lets `bench-darcy-operator` reach super-resolution
accuracy with far less data. PINO is a training strategy on top of an FNO, not a separate model class.

### Honest limitations (document these on the case page)

Operators are **bounded by their training distribution**, and FNO specifically has well-characterized
failure modes — call them out, do not hide them:

- **Fixed modes cannot synthesize unseen high frequencies.** The spectral layer truncates everything
  above $k_{\max}$; on inputs with energy above the trained band (or at resolutions the high-frequency
  content was never seen at), error concentrates near the Nyquist frequency of the training grid. Spectral
  bias toward low frequencies is intrinsic to the architecture (Qin et al., arXiv:2404.07200).
- **Out-of-distribution boundary conditions / parameters error badly.** A shift in BCs or in the
  coefficient distribution can inflate error by **more than an order of magnitude** — FNO overfits to
  the boundary/parameter regime it was trained on (Failure-modes study, arXiv:2601.11428).
- **Artificial energy dissipation on nonlinear systems.** Mode truncation accumulated across depth
  manifests as spurious dissipation and manipulated frequency content, and rollouts of chaotic systems
  diverge (arXiv:2601.11428; structural-dynamics study arXiv:2511.08753).

**Mandatory consequence for PINN-Lab:** every operator case must be validated **out-of-distribution
against a classical solver** (FEM/FVM/spectral), not just on held-out in-distribution test data. The
Benchmark page must show where the operator degrades.

---

## Install (verified)

`neuraloperator` 2.0.0 (released **2025-10-22**, MIT, Python ≥ 3.9, PyTorch backend). It belongs in
the **heavy precompute venv** (`.venv-pipeline`), never the live lane.

```bash
# in the precompute venv (PyTorch already installed for the chosen CUDA/CPU build)
pip install neuraloperator          # imports as `neuralop`
```

Editable/dev install (to track main, e.g. for PINO scripts under `scripts/`):

```bash
git clone https://github.com/neuraloperator/neuraloperator
pip install -e ./neuraloperator
```

Optional extras:

- **SFNO** needs the spherical-harmonic transform: `pip install torch-harmonics`
  (`SFNO`/`LocalNO` are guarded behind a try/except and only import if `torch_harmonics` is built).
- **GINO** uses neighbor-search kernels; for large 3D point clouds install the optional
  `open3d`/`torch-cluster`-style backends per the GINO docs (CPU fallback exists but is slow).
- `wandb` only if you set `wandb_log=True` in the `Trainer`.

Quick import sanity check:

```python
import neuralop
from neuralop.models import FNO, TFNO          # SFNO, GINO available too
from neuralop import Trainer, LpLoss, H1Loss
from neuralop.training import AdamW
from neuralop.data.datasets import load_darcy_flow_small
print(neuralop.__version__)                     # 2.0.0
```

---

## Core API / configure

### Model construction

```python
from neuralop.models import FNO

model = FNO(
    n_modes=(16, 16),            # retained Fourier modes per spatial axis (k_max); the key knob
    in_channels=1,               # # input function channels (e.g. permeability a(x))
    out_channels=1,              # # output channels (e.g. pressure u(x))
    hidden_channels=64,          # width of the lifted latent representation
    projection_channel_ratio=2,  # width of the final projection MLP, relative to hidden_channels
)
```

- `n_modes` is a tuple whose **length sets the spatial dimensionality** (`(16,16)` → 2D, `(16,16,16)`
  → 3D). More modes = more high-frequency capacity but more parameters and more overfitting risk.
- `in_channels` must include any positional/grid channels you concatenate; the built-in `DataProcessor`
  for the Darcy dataset already feeds a single permeability channel, so `in_channels=1` there.
- The model is a plain `torch.nn.Module`; `model(x)` takes `x` of shape `[batch, in_channels, *grid]`
  and returns `[batch, out_channels, *grid]`.

### Dataset

```python
from neuralop.data.datasets import load_darcy_flow_small

train_loader, test_loaders, data_processor = load_darcy_flow_small(
    n_train=1000,
    batch_size=64,
    n_tests=[100, 50],
    test_resolutions=[16, 32],   # <-- multi-resolution test sets enable zero-shot super-res eval
    test_batch_sizes=[32, 32],
)
```

`load_darcy_flow_small` returns a train loader, a **dict of test loaders keyed by resolution**, and a
`DataProcessor` that normalizes/positions inputs and must be `.to(device)`’d and passed into the
`Trainer`. The Darcy family it samples is the steady **second-order elliptic** problem

$$-\nabla\!\cdot\!\big(a(x)\,\nabla u(x)\big) = f(x), \quad x \in (0,1)^2, \qquad u|_{\partial\Omega}=0,$$

with $a(x)$ a piecewise-constant random permeability field (the input function) and $f$ fixed; $u$ is
the pressure field (the output). This is the canonical operator-learning benchmark and the
`bench-darcy-operator` data source.

### Losses, optimizer, Trainer

```python
from neuralop import LpLoss, H1Loss
from neuralop.training import AdamW
import torch

l2loss = LpLoss(d=2, p=2)        # relative L2 over the 2D field
h1loss = H1Loss(d=2)             # H1: penalizes value AND gradient — preferred for PDE fields
train_loss  = h1loss
eval_losses = {"h1": h1loss, "l2": l2loss}

optimizer = AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
```

`Trainer.__init__` and `Trainer.train` (verified against the 2.0.0 example) take:

```python
from neuralop import Trainer

trainer = Trainer(
    model=model,
    n_epochs=20,
    device=device,
    data_processor=data_processor,   # the processor returned by the dataset loader
    wandb_log=False,
    eval_interval=5,
    use_distributed=False,
    verbose=True,
)

trainer.train(
    train_loader=train_loader,
    test_loaders=test_loaders,       # dict {resolution: loader}
    optimizer=optimizer,
    scheduler=scheduler,
    regularizer=False,
    training_loss=train_loss,
    eval_losses=eval_losses,
)
```

---

## Runnable minimal example

End-to-end: load Darcy → build FNO → train with the `H1` loss → **demonstrate zero-shot
super-resolution** (train at 16×16, predict at 32×32). Adapted verbatim from the official
`examples/models/plot_FNO_darcy.py` (neuraloperator 2.0.0); runs on CPU in seconds.

```python
import torch
from neuralop.models import FNO
from neuralop import Trainer, LpLoss, H1Loss
from neuralop.training import AdamW
from neuralop.data.datasets import load_darcy_flow_small

device = "cpu"

# 1. Data: train at 16x16, hold out test sets at 16x16 AND 32x32
train_loader, test_loaders, data_processor = load_darcy_flow_small(
    n_train=1000, batch_size=64,
    n_tests=[100, 50], test_resolutions=[16, 32], test_batch_sizes=[32, 32],
)
data_processor = data_processor.to(device)

# 2. Model
model = FNO(n_modes=(16, 16), in_channels=1, out_channels=1,
            hidden_channels=32, projection_channel_ratio=2).to(device)

# 3. Optimizer / scheduler / losses
optimizer = AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
l2loss, h1loss = LpLoss(d=2, p=2), H1Loss(d=2)

# 4. Train
trainer = Trainer(model=model, n_epochs=20, device=device,
                  data_processor=data_processor, wandb_log=False,
                  eval_interval=5, use_distributed=False, verbose=True)
trainer.train(train_loader=train_loader, test_loaders=test_loaders,
              optimizer=optimizer, scheduler=scheduler, regularizer=False,
              training_loss=h1loss, eval_losses={"h1": h1loss, "l2": l2loss})

# 5. Zero-shot super-resolution: trained at 16x16, infer at 32x32 with the SAME weights
model.eval()
sample = test_loaders[32].dataset[0]
sample = data_processor.preprocess(sample, batched=False)
with torch.no_grad():
    pred_32 = model(sample["x"].unsqueeze(0))   # output is 32x32, never trained at this res
print("zero-shot super-res output shape:", pred_32.shape)
```

The discretization-invariance is the load-bearing point: `model` was only ever shown 16×16 data, yet
`model(x_32)` returns a coherent 32×32 field. Predictions at the unseen resolution are noisier (the
high-frequency content above the trained band was never observed — exactly the spectral-bias limitation
above), which is precisely the behavior the Benchmark page should quantify.

---

## ONNX-export notes

> **This is the sharp edge for the web lane — read before promising a live FNO case.**

The standard PINN-Lab contract exports the raw `nn.Module` and verifies parity:

```python
import torch
model.eval()
dummy = torch.randn(1, 1, 32, 32)        # [batch, in_channels, H, W]
torch.onnx.export(
    model, dummy, "darcy_fno.onnx",
    input_names=["a"], output_names=["u"],
    dynamic_axes={"a": {0: "batch", 2: "h", 3: "w"},
                  "u": {0: "batch", 2: "h", 3: "w"}},
    dynamo=True,                          # REQUIRED for FNO (see below)
    opset_version=18,
)
```

**Why FNO is harder to export than a dense PINN.** The spectral convolution is built on
`torch.fft.rfft` / `torch.fft.irfft` and **complex-valued weights** (`dtype=torch.cfloat`). The legacy
TorchScript tracer (`torch.onnx.export(..., dynamo=False)`) historically does **not** lower
`aten::fft_rfft` / complex ops to ONNX and fails with "unsupported operator" even at opset 17
(pytorch/pytorch #112382, #113444). Practical guidance, in order of preference:

1. **Use the dynamo-based exporter** (`dynamo=True`, opset ≥ 18), which has FFT/complex coverage the
   legacy path lacks. **Verify** it succeeds on your installed torch/onnx versions — FFT/complex export
   support is version-sensitive, so pin and test it in CI, do not assume.
2. **If export fails**, the FNO case stays **precompute-only**: bake the evaluated field to
   `data/artifacts/bench-darcy-operator` and serve it through the **replay lane**. This is an acceptable
   and honest outcome — the gate (`core/gate.py`) should mark the case `precompute` when the ONNX export
   or `onnxruntime-web` inference check fails. Do **not** ship a half-working live FNO.
3. **Alternative**: re-implement the spectral conv with FFT decomposed into real-valued matmuls (no
   complex dtype, no `fft_rfft` node) before export — more work; only if a live FNO is a hard requirement.

**Parity gate (mandatory).** Whichever path is taken, assert the ONNX runtime output matches the
PyTorch model on the same input within tolerance before shipping:

```python
import onnxruntime as ort, numpy as np
sess = ort.InferenceSession("darcy_fno.onnx")
onnx_out = sess.run(["u"], {"a": dummy.numpy()})[0]
torch_out = model(dummy).detach().numpy()
assert np.allclose(onnx_out, torch_out, atol=1e-4), "ONNX⇄torch parity FAILED"
```

**Size note.** An FNO is far larger than a few-layer dense PINN (spectral weights scale with
$k_{\max}^d \times \text{hidden}^2$). Check the `.onnx` byte size against the live-lane budget; large
operators are good replay-lane candidates even when export technically succeeds.

---

## Role in PINN-Lab (which cases / lane)

| Aspect | Mapping |
|---|---|
| **Lane** | **Offline / precompute** (`requirements-precompute.txt`, `.venv-pipeline`). Operator training is heavy and dataset-based; nothing from `neuralop` enters the live lane. |
| **Primary case** | **`bench-darcy-operator`** (canonical Group A) — exercises FNO + PINO, discretization-invariance, and zero-shot super-resolution. The one operator case kept in v1 to harden the operator lane (dossier Open-Question §7: keep one, defer the rest). |
| **Methods exercised** | `methods/fno.md` (#19) and `methods/pino.md` (#20) — both first land in `bench-darcy-operator`. |
| **Live vs replay** | Decided by the gate per case. FNO ONNX export is fragile (FFT/complex); if export + `onnxruntime-web` inference pass the gate, serve live, **else replay** the baked field. Default expectation for the first operator case is **replay-backed**, with live as a stretch goal. |
| **Validation obligation** | Operators are distribution-bounded — `bench-darcy-operator` **must** be validated **out-of-distribution** against a classical numerical Darcy reference (finite-difference), and the Benchmark page must surface the documented failure modes (unseen high freqs, OOD BC blow-up, energy dissipation). |
| **Future use** | Any *parametric* mining/pollution surrogate (a PDE family rather than a single instance) would also use this engine — deferred to v2 per the dossier; the Darcy case proves the lane first. |

**One-line stance:** use `neuraloperator` only where the problem is genuinely a *family* of PDE
instances to be solved many times; for single forward/inverse instances stay on DeepXDE /
PhysicsNeMo-Sym. Never present an operator as a PINN.

---

## References

- Li, Kovachki, Azizzadenesheli, Liu, Bhattacharya, Stuart, Anandkumar — *Fourier Neural Operator for
  Parametric Partial Differential Equations.* ICLR 2021. arXiv:2010.08895.
- Kovachki et al. — *Neural Operator: Learning Maps Between Function Spaces.* JMLR 2023. arXiv:2108.08481.
- Li et al. — *Physics-Informed Neural Operator for Learning PDEs* (PINO). ACM/IMS J. Data Science 2024.
  arXiv:2111.03794.
- Kossaifi, Kovachki, Li, et al. — *A Library for Learning Neural Operators* (the `neuraloperator`
  library paper). 2024/2025. arXiv:2412.10354.
- Failure modes: *Forcing and Diagnosing Failure Modes of Fourier Neural Operators Across Diverse PDE
  Families.* arXiv:2601.11428. · Qin et al., *Toward a Better Understanding of Fourier Neural Operators:
  Analysis and Improvement from a Spectral Perspective.* arXiv:2404.07200. · *FNOs for Structural
  Dynamics … Spectrogram Loss.* arXiv:2511.08753.
- ONNX/FFT export limitations: pytorch/pytorch issues #112382 (`aten::fft_rfft` to ONNX) and #113444
  (complex-valued operator miscompilation).
- Docs: <https://neuraloperator.github.io> · Code: <https://github.com/neuraloperator/neuraloperator>
  (MIT) · PyPI: <https://pypi.org/project/neuraloperator/> (2.0.0, 2025-10-22).

*Verified against neuraloperator 2.0.0 (Python ≥ 3.9, PyTorch, MIT). Code adapted from the official
`examples/models/plot_FNO_darcy.py`. Authored 2026-06-21.*
