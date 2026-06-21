# jaxpi — the technique donor (JAX / Equinox / Optax)

> **Role in PINN-Lab:** *technique source of truth, not a shipping engine.* jaxpi is where the
> modern "Expert's Guide" PINN training recipe lives in its canonical, paper-faithful form. We
> **read it, port the techniques** into our DeepXDE/PyTorch precompute lane, and never put it on the
> train→ONNX→web path (JAX→ONNX export is fragile, and — see below — its license forbids commercial
> redistribution). Every PINN-Lab case tagged with `modified-MLP`, `PirateNet`, `NTK weighting`,
> `causal weighting`, `random weight factorization`, or `SOAP` traces its method back to this page.

---

## 1. What & why

[`jaxpi`](https://github.com/PredictiveIntelligenceLab/jaxpi) is the open-source PINN library from the
**Predictive Intelligence Lab** (Paris Perdikaris' group, UPenn) — the same group that authored most of
the loss-weighting and architecture papers that constitute the current PINN state of the art. It is
built directly on the JAX scientific stack:

- **JAX** — `jit`, `grad`, `vmap`, forward/reverse-mode AD, and (critically for PINNs) *higher-order*
  derivatives of the network with respect to its inputs, plus `pmap` for multi-GPU data parallelism.
- **Flax-style modules** for the network definitions (the repo's `archs.py`), and **Optax** for the
  optimizer chain.
- **`ml_collections.ConfigDict`** for fully-declarative experiment configuration (every architecture,
  weighting scheme, and optimizer choice is a config field, not code).
- **Weights & Biases** for metric logging.

**Why it matters for us.** jaxpi is the *reference implementation* of the techniques in the dossier
§4 "SOTA methods catalogue". Where a paper says "we set each loss weight from the trace of the NTK",
jaxpi is the code that actually does it, validated on the paper's own benchmarks. It is the companion
code for, among others:

- **An Expert's Guide to Training Physics-Informed Neural Networks** — Wang, Sankaran, Wang & Perdikaris,
  2023 ([arXiv:2308.08468](https://arxiv.org/abs/2308.08468)). The paper that bundles Fourier features +
  modified-MLP + NTK/grad-norm weighting + causal training + random weight factorization into one recipe.
- **PirateNets: Physics-informed Deep Learning with Residual Adaptive Networks** — Wang, Li, Wang &
  Perdikaris, JMLR 2024 ([arXiv:2402.00326](https://arxiv.org/abs/2402.00326)). Released on the `pirate`
  branch in May 2024.
- **Gradient Alignment in Physics-informed Neural Networks: A Second-Order Optimization Perspective** —
  introduces **SOAP** (Shampoo-with-Adam-in-the-eigenbasis) for PINNs, achieving the first PINN solution
  of turbulent flow at Re ≈ 10⁴ ([arXiv:2502.00604](https://arxiv.org/abs/2502.00604), code on the same repo).
- The causal-training and NTK/grad-pathology papers
  ([arXiv:2203.07404](https://arxiv.org/abs/2203.07404),
  [arXiv:2007.14527](https://arxiv.org/abs/2007.14527),
  [arXiv:2001.04536](https://arxiv.org/abs/2001.04536)).

So we treat jaxpi as **canonical pseudocode that happens to run**: read it to get every constant and
update rule right, then re-implement the technique in our shipping lane.

### Why NOT ship on it (read this before depending on jaxpi)

1. **License is non-commercial and non-redistributable.** Despite the dossier's earlier note of
   "Apache-2.0", the actual [`LICENSE`](https://github.com/PredictiveIntelligenceLab/jaxpi/blob/main/LICENSE)
   is a **custom Penn academic license**: *"Penn Software PirateNet, Copyright (C) 2023 The Trustees of
   the University of Pennsylvania. … permission to use, copy, and modify the software … for **non-profit
   research purposes only**."* It also states: *"Recipient and Institution shall not distribute Software
   or Modifications to any third parties without the prior written approval of Penn"* and routes any
   commercial use through *"The Penn Center for Innovation."* For PINN-Lab — a public product that could
   be monetised — this means **jaxpi (and PirateNet code) cannot be vendored or shipped**; the ideas and
   equations are fine to re-implement, the code is not ours to redistribute. (The *technique* PirateNets,
   as a published method, is reproducible from the paper; what is restricted is *this codebase*.)
2. **JAX → ONNX is fragile.** Our web lane is `onnxruntime-web`. There is no first-class JAX→ONNX
   exporter; the path is `jax2tf` → TF SavedModel → tf2onnx, or `jaxonnxruntime`, both of which break on
   the exact ops PINNs lean on (higher-order AD residuals captured into the graph, custom Fourier-feature
   layers, weight-factorized dense layers). DeepXDE-on-PyTorch gives a clean `torch.onnx.export` instead.
3. **GPU-only, multi-GPU-first.** The codebase assumes CUDA GPUs (the README pins CUDA 12.4 / cuDNN 8.9)
   and is structured for `pmap` multi-GPU training with single-GPU evaluation. That is the right posture
   for a research lab, the wrong posture for a portable precompute pipeline that must also run on CPU.

**Net:** jaxpi is a **donor of techniques and constants**, exercised via one runnable example in this
repo and ported, method-by-method, into DeepXDE. See §6.

---

## 2. Install (verified)

Verified against the repository README (mid-2026). jaxpi is **not on PyPI** — install from source.

```bash
# 0. Use an isolated environment (CAOS rule: never global). A separate venv is recommended
#    because jaxpi pins an older JAX than our main pipeline.
python -m venv .venv-jaxpi
source .venv-jaxpi/bin/activate          # Windows: .venv-jaxpi\Scripts\activate

# 1. Up-to-date pip, then a CUDA-enabled JAX (GPU-only codebase)
pip3 install -U pip
pip3 install --upgrade jax jaxlib          # for CUDA 12: pip install -U "jax[cuda12]"

# 2. Clone and install jaxpi itself
git clone https://github.com/PredictiveIntelligenceLab/jaxpi.git
cd jaxpi
pip install .

# 3. Metric logging (examples log to Weights & Biases)
pip install wandb
wandb login            # or: export WANDB_MODE=offline   to skip the account
```

For **PirateNets** specifically, use the dedicated branch (the technique we port for `bench-allencahn`):

```bash
git clone -b pirate https://github.com/PredictiveIntelligenceLab/jaxpi.git jaxpi-pirate
```

**Verified environment (from the repo README):**

| Component | Pinned / tested |
|---|---|
| Python | 3.8+ |
| JAX / jaxlib | 0.4.36 |
| CUDA | 12.4 |
| cuDNN | 8.9 |
| Hardware | **NVIDIA GPU required** (multi-GPU training, single-GPU eval) |
| Logging | Weights & Biases |

> If you have no NVIDIA GPU: JAX will fall back to CPU, but the examples are tuned for GPU batch sizes
> (`batch_size_per_device: 4096`) and `pmap`; expect to drop batch size and step counts heavily. For
> PINN-Lab we do **not** depend on jaxpi at runtime, so this is only a concern when *studying* an example
> locally — and even then, reading the config + `models.py` is usually enough to port the technique.

---

## 3. Core API / configure

jaxpi's design is **declarative**: an `ml_collections.ConfigDict` describes the whole experiment, and a
per-example `train.py` / `eval.py` / `models.py` consumes it. Each example
(`examples/burgers/`, `examples/adv/`, `examples/ns_steady_cylinder/`, `examples/ks_chaotic/`, …)
contains:

```
examples/<pde>/
├─ main.py          # entry point: parses --config, dispatches to train/eval
├─ train.py         # training loop (model + evaluator init, step loop, W&B logging)
├─ eval.py          # evaluation / inference, error vs reference
├─ models.py        # the PDE residual + loss terms (subclass of jaxpi's base model)
├─ utils.py         # data loading, reference solution
└─ configs/
   ├─ default.py    # baseline hyperparameters
   └─ sota.py       # the full "Expert's Guide" stack turned on
```

### The config is where the techniques live

This is the load-bearing part for PINN-Lab: **every SOTA method is a field in the config.** Verified key
names and default values from `examples/burgers/configs/default.py`:

```python
# --- architecture (which technique knobs to port) ---
config.arch = ml_collections.ConfigDict()
config.arch.arch_name   = "Mlp"            # also: "ModifiedMlp", "PirateNet"
config.arch.num_layers  = 4
config.arch.hidden_dim  = 256
config.arch.out_dim     = 1
config.arch.activation  = "tanh"
config.arch.periodicity = {"period": (jnp.pi,), "axis": (1,), "trainable": (False,)}  # exact-periodic BC
config.arch.fourier_emb = {"embed_scale": 1, "embed_dim": 256}    # random Fourier features (sigma = embed_scale)
config.arch.reparam     = {"type": "weight_fact", "mean": 0.5, "stddev": 0.1}  # random weight factorization

# --- optimizer (Adam → annealed; SOAP available on the gradient-alignment branch) ---
config.optim = ml_collections.ConfigDict()
config.optim.optimizer     = "Adam"
config.optim.learning_rate = 1e-3
config.optim.decay_rate    = 0.9
config.optim.decay_steps   = 2000
config.optim.beta1, config.optim.beta2, config.optim.eps = 0.9, 0.999, 1e-8

# --- loss weighting (NTK vs grad-norm) + causal training ---
config.weighting = ml_collections.ConfigDict()
config.weighting.scheme             = "grad_norm"   # or "ntk"
config.weighting.use_causal         = True          # causal time-weighting on/off
config.weighting.causal_tol         = 1.0           # epsilon in the causal weight exp(-eps * cum residual)
config.weighting.momentum           = 0.9
config.weighting.update_every_steps = 1000

# --- training budget ---
config.training = ml_collections.ConfigDict()
config.training.max_steps             = 200_000
config.training.batch_size_per_device = 4096
```

Mapping each config knob to the dossier §4 method (this is the table to copy when porting):

| Config field | Technique (dossier §4) | Canonical reference |
|---|---|---|
| `arch.arch_name = "ModifiedMlp"` | **Modified-MLP** (U/V gating streams) | Wang, Teng & Perdikaris, SISC 2021 — [arXiv:2001.04536](https://arxiv.org/abs/2001.04536) |
| `arch.arch_name = "PirateNet"` | **PirateNets** (adaptive residual blocks + LS last-layer init) | [arXiv:2402.00326](https://arxiv.org/abs/2402.00326) |
| `arch.fourier_emb` | **Random Fourier features** (spectral-bias remedy) | Wang, Wang & Perdikaris, CMAME 2021 — [arXiv:2012.10047](https://arxiv.org/abs/2012.10047); Tancik et al. [arXiv:2006.10739](https://arxiv.org/abs/2006.10739) |
| `arch.reparam = {"type":"weight_fact",…}` | **Random weight factorization** (`w = exp(s)·v`) | Wang, Perdikaris et al. — [arXiv:2210.01274](https://arxiv.org/abs/2210.01274) |
| `weighting.scheme = "ntk"` | **NTK loss weighting** | Wang, Yu & Perdikaris, JCP 2022 — [arXiv:2007.14527](https://arxiv.org/abs/2007.14527) |
| `weighting.scheme = "grad_norm"` | **Gradient-pathology / grad-norm weighting** | Wang, Teng & Perdikaris, SISC 2021 — [arXiv:2001.04536](https://arxiv.org/abs/2001.04536) |
| `weighting.use_causal = True` | **Causal training** (respect temporal causality) | Wang, Sankaran & Perdikaris, CMAME 2024 — [arXiv:2203.07404](https://arxiv.org/abs/2203.07404) |
| `optim.optimizer = "SOAP"` (alignment branch) | **SOAP / gradient alignment** (2nd-order) | [arXiv:2502.00604](https://arxiv.org/abs/2502.00604) |

### CLI surface

```bash
cd examples/burgers
python3 main.py                               # run with configs/default.py
python3 main.py --config=configs/sota.py      # full Expert's-Guide stack
python3 main.py --config.mode=eval            # evaluation mode
python3 main.py --config.training.batch_size_per_device=2048    # override any nested field
CUDA_VISIBLE_DEVICES=0,1 python3 main.py       # multi-GPU data-parallel training
```

Any `config.<a>.<b>` field can be overridden on the command line — this is how the `default` vs `sota`
ablations are run.

---

## 4. Runnable minimal example

There is no PyPI "hello world"; the smallest *real* run is one of the shipped examples. The honest
minimal recipe (verified against the repo layout and the Burgers config above):

```bash
# 1. environment (see §2)
source .venv-jaxpi/bin/activate
export WANDB_MODE=offline          # no W&B account needed for a smoke run

# 2. clone + install
git clone https://github.com/PredictiveIntelligenceLab/jaxpi.git && cd jaxpi
pip install . wandb

# 3. run the 1D viscous Burgers example with the SOTA stack
cd examples/burgers
python3 main.py --config=configs/sota.py \
    --config.training.max_steps=20000          # shorten for a quick look

# 4. evaluate (single-GPU) — reads the checkpoint, reports relative-L2 vs the reference solution
python3 main.py --config=configs/sota.py --config.mode=eval
```

What you are looking at, mapped to our purposes: `models.py` defines the Burgers residual
$u_t + u\,u_x - \nu\,u_{xx} = 0$ and the loss terms; `configs/sota.py` turns on the modified-MLP +
Fourier features + weight factorization + NTK (or grad-norm) weighting + causal training; `train.py`
runs the weighted-residual loop with periodic weight updates. **This is exactly the recipe we want to
reproduce in DeepXDE for `bench-burgers1d` — read these three files, copy the constants, do not vendor
the code.**

A *conceptual* sketch of what the modified-MLP forward pass (the technique we port) computes — re-derived
from [arXiv:2001.04536](https://arxiv.org/abs/2001.04536), not copied from jaxpi:

$$
U = \phi(W_U x + b_U),\quad V = \phi(W_V x + b_V),\qquad
H^{(1)} = \phi(W^{(1)} x + b^{(1)}),
$$
$$
Z^{(l)} = \phi\!\big(W^{(l)} H^{(l)} + b^{(l)}\big),\qquad
H^{(l+1)} = \big(1 - Z^{(l)}\big)\odot U + Z^{(l)}\odot V .
$$

The two encoder streams $U,V$ are computed once and gate every hidden layer, which is what improves the
trainability of the high-order derivatives the PDE residual needs.

> We keep one runnable jaxpi example *in our docs only* (this page) to satisfy the coverage contract's
> "every framework has a runnable script". It does **not** enter `requirements-precompute.txt`.

---

## 5. ONNX-export notes (why this lane is closed)

**There is no supported jaxpi → ONNX → onnxruntime-web path, and we do not attempt one.** Concretely:

- **No native exporter.** JAX has no `torch.onnx.export` analogue. The only routes are:
  - `jax2tf` (export to a TF SavedModel) → `tf2onnx` → ONNX, or
  - `jaxonnxruntime` (experimental, partial op coverage).
- **The exact PINN ops break the conversion.** jaxpi's value comes from random Fourier embeddings,
  weight-factorized dense layers (`w = exp(s)·v`), exact-periodicity input transforms, and (during
  training) higher-order input AD for the residual. When converting an *inference* graph the custom
  layers are the problem: they survive `jax2tf` inconsistently across opsets, and any control flow or
  non-standard op silently drops or mis-converts. You cannot trust the artifact without a per-op parity
  check, which defeats the point.
- **License.** Even if conversion worked, the [Penn license](https://github.com/PredictiveIntelligenceLab/jaxpi/blob/main/LICENSE)
  forbids redistributing the software or modifications to third parties — and shipping a converted
  artifact derived from jaxpi code is a redistribution question we simply avoid by **re-implementing the
  technique** in DeepXDE/PyTorch and exporting *our own* net.

**The supported PINN-Lab export path is DeepXDE → PyTorch → ONNX** (see
`docs/architecture/train-export-onnx.md` and `docs/frameworks/deepxde/`):

```python
model.net.eval()
dummy = torch.zeros(1, d)                       # d = input dim (e.g. x, t)
torch.onnx.export(model.net, dummy, "case.onnx",
                  input_names=["x"], output_names=["u"],
                  dynamic_axes={"x": {0: "batch"}, "u": {0: "batch"}},
                  opset_version=17)
```

When a technique we ported from jaxpi adds a non-standard layer (Fourier features, weight
factorization), it is implemented as **pure traced tensor ops** on the PyTorch side so it lands in the
ONNX graph, and the export is gated by the mandatory **ONNX-vs-`model.predict` parity check** (CI guard,
dossier §7).

---

## 6. Role in PINN-Lab (which cases / lane)

**Lane:** *none of the three runtime lanes.* jaxpi is a **documentation + research lane** artifact — it
informs the precompute lane's algorithms but is never imported by `pinnlab/` and never appears in any
`requirements*.txt`. The shipping engine for these cases is **DeepXDE (PyTorch backend)**.

**Techniques we port from jaxpi, and the first case each lands in** (from the coverage map §B):

| Technique (donated by jaxpi) | First PINN-Lab case | Ported into | `docs/methods/` page |
|---|---|---|---|
| Modified-MLP | `bench-navier-cavity` | DeepXDE custom net | `methods/modified-mlp` |
| NTK loss weighting | `bench-navier-cavity` | DeepXDE loss weights | `methods/ntk-weighting` |
| Grad-norm weighting | `poll-air-source-inv` | DeepXDE loss weights | `methods/gradnorm-weighting` |
| Causal training | `bench-heat1d` | DeepXDE residual schedule | `methods/causal-training` |
| Random Fourier features | `bench-wave1d` | DeepXDE multi-scale Fourier net | `methods/fourier-features` |
| Random weight factorization | (backbone for cavity/AC nets) | DeepXDE dense reparam | `methods/modified-mlp` |
| PirateNets | `bench-allencahn` | DeepXDE residual-adaptive blocks (re-impl from paper) | `methods/piratenets` |
| SOAP / gradient alignment | optimizer study (all stiff cases) | DeepXDE/PINA optimizer | `methods/optimizers` |

**Hard rules for this framework in our repo:**

1. **Do not add jaxpi to any `requirements*.txt`.** It is GPU-only, JAX-pinned, and non-redistributable.
2. **Do not vendor jaxpi or PirateNet source.** Re-implement the *method* from its paper; cite the paper,
   not the file.
3. **Do not attempt JAX→ONNX for a shipping case.** The web lane is DeepXDE→PyTorch→ONNX only.
4. **When in doubt about a constant** (NTK update period, causal `tol`, Fourier `embed_scale`, weight-fact
   `mean`/`stddev`), the config values in §3 are the authoritative numbers to match — they are the same
   values the papers were validated with.

---

## "Expert's Guide" checklist

The dossier promised this page would carry the *which-technique-to-stack-and-why* checklist. This is the
recipe distilled from [arXiv:2308.08468](https://arxiv.org/abs/2308.08468), in the order to apply it
(stop early if the problem is easy — not every case needs every item):

1. **Non-dimensionalize the PDE.** Rescale space, time, and the field so coefficients are O(1). This alone
   fixes most ill-conditioning; do it before reaching for any trick.
2. **Pick the spectral-bias remedy for the solution's frequency content.**
   - Smooth/low-frequency → plain MLP is fine.
   - High-frequency / multi-scale → **random Fourier features** (tune `embed_scale` ≈ σ; too large
     over-fits noise, too small under-resolves) or **SIREN**.
3. **Use a strong backbone.** Default to **modified-MLP**; for deep nets (>6 effective layers) use
   **PirateNets** (adaptive residual gate α init 0 + physics-informed least-squares last-layer init).
4. **Add random weight factorization** (`w = exp(s)·v`, `mean=0.5, stddev=0.1`) — a cheap, almost-free
   conditioning win that stacks with everything above.
5. **Enforce exact constraints where you can.** Periodic BCs via the periodicity input transform; other
   Dirichlet BCs via a hard-constraint output ansatz — removes a loss term and its weight.
6. **Balance the loss terms automatically.** Choose **NTK weighting** (principled, more expensive) or
   **grad-norm weighting** (cheaper, noisier); update the weights every ~1000 steps, not every step.
7. **For any time-dependent PDE, turn on causal training** (`use_causal=True`, tune `causal_tol`/ε): later
   time slices are only weighted once earlier ones converge. Near-mandatory for stiff/chaotic problems;
   combine with curriculum / time-marching if a single window still fails.
8. **Optimize in two phases.** Adam (with the decay schedule) for global exploration → high-accuracy
   refinement. The 2025 SOTA upgrade is **SOAP** (Shampoo-eigenbasis Adam) for implicit gradient
   alignment — use it on the hard, multi-task-conflict cases (turbulence, lid-driven cavity).
9. **Add residual-adaptive sampling (RAR/RAD) if the solution has sharp gradients/shocks** — orthogonal to
   all of the above; cheap, high-leverage.
10. **Always validate against a reference** (analytic or FEM/FVM) and report relative-L2 honestly. A PINN
    that "looks right" but has 30% error is a failure; the benchmark table is the product.

In PINN-Lab terms: items 2–8 are *config knobs* in jaxpi and *ported components* in our DeepXDE pipeline;
item 9 is native to DeepXDE (`RAR`); item 10 is the `evaluate` stage + the Benchmark page. Read jaxpi to
get the order and the constants right; build and export with DeepXDE.

---

### References

- Wang, Sankaran, Wang & Perdikaris (2023). *An Expert's Guide to Training Physics-Informed Neural
  Networks.* [arXiv:2308.08468](https://arxiv.org/abs/2308.08468). (jaxpi companion code.)
- Wang, Li, Wang & Perdikaris (2024). *PirateNets: Physics-informed Deep Learning with Residual Adaptive
  Networks.* JMLR. [arXiv:2402.00326](https://arxiv.org/abs/2402.00326).
- *Gradient Alignment in Physics-informed Neural Networks: A Second-Order Optimization Perspective* (2025,
  SOAP). [arXiv:2502.00604](https://arxiv.org/abs/2502.00604).
- Wang, Sankaran & Perdikaris (2024). *Respecting causality for training physics-informed neural
  networks.* CMAME. [arXiv:2203.07404](https://arxiv.org/abs/2203.07404).
- Wang, Yu & Perdikaris (2022). *When and why PINNs fail to train: A neural tangent kernel perspective.*
  JCP. [arXiv:2007.14527](https://arxiv.org/abs/2007.14527).
- Wang, Teng & Perdikaris (2021). *Understanding and mitigating gradient flow pathologies in PINNs.* SISC.
  [arXiv:2001.04536](https://arxiv.org/abs/2001.04536). (Modified-MLP + grad-norm weighting.)
- Wang, Wang & Perdikaris (2021). *On the eigenvector bias of Fourier feature networks.* CMAME.
  [arXiv:2012.10047](https://arxiv.org/abs/2012.10047); Tancik et al. (2020), Fourier features,
  [arXiv:2006.10739](https://arxiv.org/abs/2006.10739).
- Random weight factorization: [arXiv:2210.01274](https://arxiv.org/abs/2210.01274).
- Repository: <https://github.com/PredictiveIntelligenceLab/jaxpi> ·
  PirateNet branch: <https://github.com/PredictiveIntelligenceLab/jaxpi/tree/pirate> ·
  License (Penn academic, non-commercial):
  <https://github.com/PredictiveIntelligenceLab/jaxpi/blob/main/LICENSE>

*Built on the JAX scientific stack (JAX + Flax-style modules + Optax + `ml_collections`). In PINN-Lab,
jaxpi is a technique donor only — its methods are ported to DeepXDE/PyTorch for the ONNX→onnxruntime-web
lane; jaxpi itself is never a dependency.*
