# Architectures (spectral-bias remedies & hard constraints)

> **Group:** `architectures` — the network-design layer of the PINN-Lab method catalogue.
> **Methods covered:** random / multi-scale Fourier features · modified-MLP · PirateNets · SIREN · hard-constraint output transforms.
> **PINN-Lab cases that exercise this group:** `bench-wave1d` (Fourier features + SIREN), `ind-helmholtz` (high-frequency Fourier features / SIREN), `bench-navier-cavity` (modified-MLP), `bench-allencahn` (PirateNets), `bench-poisson2d` / `bench-heat1d` / `bench-wave1d` (hard-constraint output transforms).

---

## What this group is

A plain tanh multilayer perceptron (MLP) is a *poor* default for solving partial differential equations (PDEs). Two structural pathologies dominate:

1. **Spectral bias** (a.k.a. the *F-principle*). A standard MLP learns the low-frequency content of a target function long before its high-frequency content; for a multi-scale or oscillatory PDE solution the high-frequency error never converges in practice. Through the Neural Tangent Kernel (NTK), the network's training dynamics are governed by an isotropic kernel whose eigenvalues decay fast with frequency, so high-frequency eigen-directions are learned orders of magnitude more slowly than low-frequency ones (Tancik et al. 2020; Wang, Wang & Perdikaris 2021).

2. **Derivative-trainability degradation with depth.** PINN losses depend on *derivatives* of the network (the PDE residual is built from $\partial_t u$, $\nabla^2 u$, …). With standard initialisation, those derivatives are badly scaled at init, deeper nets train *worse* not better, and the residual loss minimisation becomes unstable (Wang, Teng & Perdikaris 2021; Wang et al. 2024).

The `architectures` group collects the network-design fixes for both pathologies — input encodings (Fourier features), activation choices (SIREN), backbone topology (modified-MLP, PirateNets) — plus a separate, complementary tool: **hard-constraint output transforms**, which remove boundary/initial-condition loss terms entirely by building the constraint into the network's functional form. These are orthogonal to the *training* methods (loss weighting, causal training, adaptive sampling) documented in the other method groups, and they stack with them.

---

## 1. Random & multi-scale Fourier features

### What it is

Prepend a fixed (non-trainable) Fourier feature embedding to the network input so the MLP sees a high-dimensional bank of sinusoids instead of raw coordinates. This converts the network's effective NTK from a rapidly-decaying isotropic kernel into a *stationary* kernel with a **tunable bandwidth**, letting the MLP represent high frequencies that it otherwise could not learn (Tancik et al. 2020). Wang, Wang & Perdikaris (2021) brought this into the PINN setting and added the **multi-scale** variant: run several Fourier banks at different frequency scales $\sigma$ in parallel and concatenate, so a single network resolves a solution containing several length-scales at once. For space–time problems the spatial and temporal axes get *separate* scale sets ($\sigma_x$, $\sigma_t$) — the spatio-temporal multi-scale variant.

### Key equation

The random Fourier feature map for input $\mathbf{x}\in\mathbb{R}^d$ is

$$\gamma(\mathbf{x}) = \big[\cos(2\pi \mathbf{B}\mathbf{x}),\; \sin(2\pi \mathbf{B}\mathbf{x})\big],\qquad \mathbf{B}_{ij}\sim\mathcal{N}(0,\sigma^2),$$

with $\mathbf{B}\in\mathbb{R}^{m\times d}$ fixed at initialisation. The full network is $u_\theta(\mathbf{x}) = \mathrm{MLP}\big(\gamma(\mathbf{x})\big)$. The multi-scale network takes $k$ banks $\{\sigma_1,\dots,\sigma_k\}$, passes each $\gamma_{\sigma_i}(\mathbf{x})$ through the *same* MLP body, and linearly combines the outputs:

$$u_\theta(\mathbf{x}) = \mathbf{W}\,\Big[\,\mathrm{MLP}\big(\gamma_{\sigma_1}(\mathbf{x})\big);\dots;\mathrm{MLP}\big(\gamma_{\sigma_k}(\mathbf{x})\big)\,\Big] + \mathbf{b}.$$

The scale $\sigma$ is the single most sensitive hyperparameter: too small leaves the spectral bias in place, too large makes the target look like white noise and the network underfits / generalises poorly. Tune $\sigma$ per problem (it sets the bandwidth of the induced kernel), never inherit it blindly.

### Why / when to use it

Use Fourier features whenever the solution has **high-frequency or multi-scale structure**: oscillatory wave fields, high-wavenumber Helmholtz, boundary layers, anything where a plain MLP visibly smooths out the fine structure. It is cheap (a fixed linear layer) and high-leverage. Skip it for smooth, single-scale solutions where it only adds a hyperparameter to tune.

### Honest limitations

- $\sigma$ is brittle and problem-specific; a wrong scale is *worse* than no embedding.
- It targets spatial/temporal frequency, not the depth/derivative-trainability problem — combine with a good backbone for hard cases.
- The mapping $\mathbf{B}$ is fixed, so it cannot adapt if the true frequency content is unknown a priori; the multi-scale trick mitigates but does not eliminate this.

### References

- Tancik, M. et al. *Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains.* NeurIPS 2020. arXiv:[2006.10739](https://arxiv.org/abs/2006.10739).
- Wang, S., Wang, H. & Perdikaris, P. *On the eigenvector bias of Fourier feature networks: From regression to solving multi-scale PDEs with physics-informed neural networks.* CMAME 384:113938, 2021. arXiv:[2012.10047](https://arxiv.org/abs/2012.10047). DOI: [10.1016/j.cma.2021.113938](https://doi.org/10.1016/j.cma.2021.113938).

### Framework that implements it

**DeepXDE** ships this as `dde.nn.MsFFN` (multi-scale Fourier feature net) and `dde.nn.STMsFFN` (spatio-temporal, with separate `sigmas_x` / `sigmas_t`). It is also available in **jaxpi** (the technique source-of-truth) and **PhysicsNeMo** (`FourierNetArch` / modified-Fourier arch). Minimal DeepXDE usage:

```python
import deepxde as dde
# layer_sizes, activation, initializer, sigmas
net = dde.nn.MsFFN([2] + [100] * 3 + [1], "tanh", "Glorot uniform", sigmas=[1, 10])
# spatio-temporal variant: dde.nn.STMsFFN(..., sigmas_x=[1,10], sigmas_t=[1,10])
```

### PINN-Lab case that exercises it

`bench-wave1d` (1D wave equation $u_{tt}=c^2u_{xx}$, hyperbolic / oscillatory — the canonical Fourier-feature showcase) and `ind-helmholtz` (frequency-domain Helmholtz $\nabla^2 u + \kappa^2 u = f$, high-wavenumber). `poll-ocean-transport` also uses Fourier features per the dossier case table. The ONNX-export note matters here: the fixed `B` matrix and the $\cos/\sin$ map are pure tensor ops, so they trace cleanly into the exported graph for the `onnxruntime-web` live lane.

---

## 2. Modified-MLP

### What it is

A backbone topology, not an input encoding. Two extra encoder streams $\mathbf{U}$ and $\mathbf{V}$ are computed once from the input, then used to *gate* every hidden layer via a pointwise interpolation. This adds residual-like, multiplicative pathways through the network that markedly improve the trainability of the network's derivatives — which is exactly what a PINN loss stresses. It is the de-facto strong PINN backbone in the "Expert's Guide" stack and routinely the difference between convergence and stall on stiff/coupled systems.

### Key equation

Given input embedding $\mathbf{x}$ (often *after* a Fourier feature map), compute the two gating streams once:

$$\mathbf{U} = \phi(\mathbf{W}_U\mathbf{x}+\mathbf{b}_U),\qquad \mathbf{V} = \phi(\mathbf{W}_V\mathbf{x}+\mathbf{b}_V),$$

then every hidden layer $l$ is interpolated between them:

$$\mathbf{f}^{(l)} = \phi\big(\mathbf{W}^{(l)}\mathbf{H}^{(l)}+\mathbf{b}^{(l)}\big),\qquad \mathbf{H}^{(l+1)} = \big(1-\mathbf{f}^{(l)}\big)\odot \mathbf{U} + \mathbf{f}^{(l)}\odot \mathbf{V},$$

with $\mathbf{H}^{(1)}=\phi(\mathbf{W}^{(1)}\mathbf{x}+\mathbf{b}^{(1)})$, $\odot$ the Hadamard product and $\phi$ the activation. $\mathbf{U},\mathbf{V}$ are shared across all layers.

### Why / when to use it

Use it as the default backbone for any non-trivial PINN — especially stiff, coupled, or convection-dominated systems (steady Navier–Stokes, reaction–diffusion). It composes with Fourier features (encode first, then gate) and with NTK/grad-norm loss weighting.

### Honest limitations

- More parameters and compute per layer than a plain MLP (two extra full-width streams).
- It improves trainability, not expressivity per se; it does not by itself fix spectral bias (pair with Fourier features) nor deep-net degradation as aggressively as PirateNets.
- The original reference implementation is frozen TF1; the maintained, idiomatic version lives in **jaxpi**.

### References

- Wang, S., Teng, Y. & Perdikaris, P. *Understanding and mitigating gradient flow pathologies in physics-informed neural networks.* SIAM J. Sci. Comput. 43(5):A3055–A3081, 2021. arXiv:[2001.04536](https://arxiv.org/abs/2001.04536). DOI: [10.1137/20M1318043](https://doi.org/10.1137/20M1318043).

### Framework that implements it

**jaxpi** (`PredictiveIntelligenceLab/jaxpi`) is the canonical maintained implementation and the technique source-of-truth. In **DeepXDE** the closest first-class backbone is the plain/parallel FNN (`dde.nn.FNN`/`PFNN`); the modified-MLP is reproduced in the precompute pipeline by porting the jaxpi gating block. **PhysicsNeMo** offers a comparable gated architecture (`ModifiedFourierNetArch`).

### PINN-Lab case that exercises it

`bench-navier-cavity` (lid-driven cavity, steady Navier–Stokes $(\mathbf{u}\cdot\nabla)\mathbf{u} = -\nabla p/\rho + \nu\nabla^2\mathbf{u},\ \nabla\cdot\mathbf{u}=0$) — the hard NS benchmark, where the modified-MLP backbone is combined with NTK + grad-norm loss weighting per the coverage map.

---

## 3. PirateNets (residual adaptive networks)

### What it is

A deep-PINN architecture that fixes the counter-intuitive "deeper PINNs train worse" problem. Each residual block is multiplied by a **trainable gate** $\alpha$ initialised to **zero**, so at the start of training the network is *effectively shallow* (identity skips dominate) and **progressively deepens** as the $\alpha$'s grow — sidestepping the bad derivative-init of deep MLPs. PirateNets also support a **physics-informed initialisation** of the final linear layer by least-squares fitting available data / boundary information, giving a good starting iterate. The result is stable training of genuinely deep PINNs and state-of-the-art accuracy on stiff benchmarks (Allen–Cahn, Korteweg–de Vries, Gray–Scott, lid-driven cavity).

### Key equation

With a Fourier/embedding front-end $\Phi$, each adaptive residual block reads

$$\mathbf{x}^{(l+1)} = \alpha^{(l)}\, \mathbf{f}^{(l)}\!\big(\mathbf{x}^{(l)}\big) + \big(1-\alpha^{(l)}\big)\,\mathbf{x}^{(l)},\qquad \alpha^{(l)}\ \text{trainable},\ \ \alpha^{(l)}\!\big|_{t=0}=0,$$

where $\mathbf{f}^{(l)}$ is the (modified-MLP-style, gated) nonlinear block. At init $\alpha^{(l)}=0\Rightarrow \mathbf{x}^{(l+1)}=\mathbf{x}^{(l)}$, i.e. a linear/shallow map; training raises the $\alpha$'s to recruit depth only as needed. The last layer's weights are initialised by the least-squares solution $\mathbf{W}_{\text{out}} = \arg\min_\mathbf{W}\lVert \mathbf{W}\,\Phi(\mathbf{X}) - \mathbf{u}_{\text{data}}\rVert^2$.

### Why / when to use it

Reach for PirateNets when you genuinely need a **deep** network — stiff, sharp-interface, or high-complexity solutions where shallow nets plateau and naïve deep nets diverge. On easy/smooth problems a shallow modified-MLP is simpler and enough.

### Honest limitations

- More machinery (per-block gates, special init) than a plain backbone; overkill for easy cases.
- The physics-informed last-layer init needs data/boundary samples to fit against.
- Best-validated inside the JAX/jaxpi recipe; porting the full block faithfully (gates + init + Fourier front-end) is the work, and small deviations cost accuracy.

### References

- Wang, S., Li, B., Chen, Y. & Perdikaris, P. *PirateNets: Physics-informed Deep Learning with Residual Adaptive Networks.* Journal of Machine Learning Research 25(402):1–51, 2024. arXiv:[2402.00326](https://arxiv.org/abs/2402.00326).

### Framework that implements it

**jaxpi** (canonical reference implementation by the authors' lab) and **PINA** (`PirateNet` model class, MIT, PyTorch-Lightning). PINN-Lab ports it into the DeepXDE/PyTorch precompute lane (or runs it via PINA) for the relevant case.

### PINN-Lab case that exercises it

`bench-allencahn` (stiff Allen–Cahn $u_t = \varepsilon^2 u_{xx} + u - u^3$) — paired with causal / time-marching training; the canonical PirateNets stiffness showcase.

---

## 4. SIREN (sinusoidal activations)

### What it is

Replace every activation in the MLP with a **sine**, and use a frequency-scaled initialisation so the forward pass and all its derivatives stay well-conditioned. The defining property for PINNs: the derivative of a SIREN is itself a SIREN (since $\frac{d}{dx}\sin = \cos$ is a phase-shifted sine), so a SIREN represents **high-order derivatives faithfully** — exactly what a PDE residual needs. It is a competing spectral-bias remedy to Fourier features: instead of a fixed sinusoidal *input* map, the sinusoid lives in *every* layer and its frequencies are learned. Originally for implicit neural representations (images, SDFs, audio), it solves Poisson, Helmholtz, wave and Eikonal boundary-value problems directly.

### Key equation

A SIREN layer is

$$\mathbf{h}^{(l+1)} = \sin\!\big(\omega_0\,(\mathbf{W}^{(l)}\mathbf{h}^{(l)} + \mathbf{b}^{(l)})\big),$$

with the frequency factor $\omega_0\approx 30$ on the first layer. The **principled init** keeps the pre-activation distribution stable: draw weights uniformly

$$W_{ij}\sim\mathcal{U}\!\left(-\sqrt{\tfrac{6}{n}},\ \sqrt{\tfrac{6}{n}}\right)\quad\text{for hidden layers (fan-in }n\text{)},$$

with the first layer scaled by $\omega_0$. This init is what makes deep SIRENs trainable; using sine activations *without* it fails.

### Why / when to use it

Use SIREN for problems dominated by **high-order derivatives and oscillatory fields**: wave, Helmholtz, Eikonal, Poisson. It is a clean alternative to Fourier-feature + tanh when you want the high-frequency capacity to be *learned* rather than fixed by a chosen $\sigma$.

### Honest limitations

- Sensitive to $\omega_0$ and to the init scheme — get the init wrong and it does not train at all.
- The learned-frequency flexibility can overfit noisy data more readily than a fixed Fourier bank.
- Less standard than tanh in PINN frameworks; in DeepXDE it is supplied as a custom activation/layer rather than a turnkey net.

### References

- Sitzmann, V., Martel, J. N. P., Bergman, A. W., Lindell, D. B. & Wetzstein, G. *Implicit Neural Representations with Periodic Activation Functions.* NeurIPS 2020. arXiv:[2006.09661](https://arxiv.org/abs/2006.09661). Project: <https://www.vincentsitzmann.com/siren/>.

### Framework that implements it

Official reference: `vsitzmann/siren` (PyTorch, reusable `SineLayer`). In PINN-Lab it is used in the **DeepXDE** pipeline as a custom `sin` activation + the frequency-scaled initialiser on a standard `dde.nn.FNN`, then exported to ONNX (sine is a supported ONNX op, so it traces cleanly for the live lane).

### PINN-Lab case that exercises it

`bench-wave1d` (1D wave — SIREN vs. Fourier-feature comparison on the same hyperbolic problem) and `ind-helmholtz` (high-frequency Helmholtz). These are the two cases where the spectral-bias remedies are compared head-to-head.

---

## 5. Hard-constraint output transforms (distance-function ansatz)

### What it is

Instead of *penalising* boundary/initial-condition violations with an extra loss term (a "soft" constraint, which needs weighting and is only satisfied approximately), build the constraint **into the network's output** so it holds *exactly for any weights*. The trial solution is composed of a function $g$ that meets the boundary data and a smooth **approximate distance function** $\phi$ that vanishes on the boundary, multiplying the raw network $N_\theta$. Because the BC term disappears from the loss, so does its weight — removing an entire class of loss-balancing problems and guaranteeing the constraint is met to machine precision. Sukumar & Srivastava (2022) supply the general machinery for constructing $\phi$ on arbitrary geometry using R-functions (constructive solid geometry) and generalised barycentric coordinates; Lu et al. (hPINN) apply the same idea in DeepXDE.

### Key equation

The constrained ansatz for a Dirichlet problem $u|_{\partial\Omega}=g$ is

$$\hat{u}_\theta(\mathbf{x}) = g(\mathbf{x}) + \phi(\mathbf{x})\,N_\theta(\mathbf{x}),\qquad \phi(\mathbf{x})=0\ \ \forall\,\mathbf{x}\in\partial\Omega,\ \ \phi(\mathbf{x})>0\ \text{inside}.$$

Then $\hat{u}_\theta|_{\partial\Omega}=g$ holds **identically**, regardless of $\theta$, so only the PDE-residual loss remains. For a unit interval $[0,1]$ with $u(0)=a,\,u(1)=b$ one concrete choice is $g(x)=a(1-x)+bx$ and $\phi(x)=x(1-x)$. The same construction extends to time (initial conditions) and, via R-functions $\phi = \phi_1 \,\&_\alpha\, \phi_2$ (R-conjunction), to multi-piece and curved boundaries.

### Why / when to use it

Use hard constraints whenever the geometry/BCs are simple enough to write an exact $g$ and a vanishing $\phi$ — typically Dirichlet (and initial) conditions on intervals, boxes, and simple analytic shapes. It removes a loss term, removes its weight, and improves accuracy near the boundary. It is one of the highest-value, lowest-cost tricks for canonical benchmarks.

### Honest limitations

- **Curved / complex geometry is hard**: constructing a smooth, non-degenerate $\phi$ on a general domain is the whole subject of the Sukumar–Srivastava paper and is non-trivial.
- **Neumann/Robin conditions are trickier** than Dirichlet — the gradient BC is not as cleanly factorable into the ansatz.
- A poorly conditioned $\phi$ (vanishing too fast, or with kinks) can hurt optimisation near the boundary.
- **ONNX caveat (load-bearing for PINN-Lab):** the transform $g+\phi N$ is applied in Python on top of the net, so it is only present in the exported artifact if it is implemented as pure tensor ops traced into the graph. DeepXDE's `apply_output_transform` runs inside the forward pass, so exporting `model.net` captures it — but this **must be verified** with the ONNX-vs-`model.predict` parity check (dossier §3.2), otherwise the live/replay field silently violates the BC.

### References

- Sukumar, N. & Srivastava, A. *Exact imposition of boundary conditions with distance functions in physics-informed deep neural networks.* CMAME 389:114333, 2022. arXiv:[2104.08426](https://arxiv.org/abs/2104.08426). DOI: [10.1016/j.cma.2021.114333](https://doi.org/10.1016/j.cma.2021.114333).
- Lu, L., Pestourie, R., Yao, W., Wang, Z., Verdugo, F. & Johnson, S. G. *Physics-informed neural networks with hard constraints for inverse design (hPINN).* SIAM J. Sci. Comput. 43(6):B1105–B1132, 2021. DOI: [10.1137/21M1397908](https://doi.org/10.1137/21M1397908).

### Framework that implements it

**DeepXDE** via `net.apply_output_transform(transform)`, where `transform(x, y)` returns $g(x)+\phi(x)\,y$. **PhysicsNeMo** supports the same idea through output-transform hooks. Minimal DeepXDE usage:

```python
# enforce u(0)=0, u(1)=0 exactly on the interval [0,1]
net.apply_output_transform(lambda x, y: x * (1 - x) * y)
```

### PINN-Lab case that exercises it

`bench-poisson2d` (the anchor: 2D Poisson $\nabla^2 u = f$, $u|_{\partial\Omega}=g$ with the distance-function ansatz — already ✅ in the coverage map), and used again in `bench-heat1d` and `bench-wave1d` to impose initial/boundary conditions exactly. The mining/pollution cases `mine-thickener-settling` and `poll-tailings-seepage` (Richards equation) also use hard constraints per the coverage map.

---

## How this group maps into PINN-Lab

| Method | Key idea | Canonical ref | Framework (PINN-Lab) | First case |
|---|---|---|---|---|
| Random / multi-scale Fourier features | fixed $\cos/\sin$ input bank → tunable NTK bandwidth | [2006.10739](https://arxiv.org/abs/2006.10739), [2012.10047](https://arxiv.org/abs/2012.10047) | DeepXDE `MsFFN`/`STMsFFN`, jaxpi, PhysicsNeMo | `bench-wave1d`, `ind-helmholtz` |
| Modified-MLP | two encoder streams gate every layer | [2001.04536](https://arxiv.org/abs/2001.04536) | jaxpi (ported); PhysicsNeMo gated arch | `bench-navier-cavity` |
| PirateNets | zero-init adaptive residual gates + physics-informed last-layer init | [2402.00326](https://arxiv.org/abs/2402.00326) | jaxpi, PINA (`PirateNet`) | `bench-allencahn` |
| SIREN | $\sin$ activations + $\omega_0$-scaled init; derivatives stay SIRENs | [2006.09661](https://arxiv.org/abs/2006.09661) | DeepXDE custom activation; `vsitzmann/siren` | `bench-wave1d`, `ind-helmholtz` |
| Hard-constraint output transform | $\hat u = g + \phi\,N$, BC exact for any $\theta$ | [2104.08426](https://arxiv.org/abs/2104.08426), [10.1137/21M1397908](https://doi.org/10.1137/21M1397908) | DeepXDE `apply_output_transform`; PhysicsNeMo | `bench-poisson2d` (anchor) |

**Why these belong together.** Fourier features, SIREN, modified-MLP and PirateNets are four answers to the same two diseases — *spectral bias* (Fourier features, SIREN) and *derivative-trainability / depth degradation* (modified-MLP, PirateNets) — and they **stack**: the strongest PINN-Lab backbone for a hard case is a Fourier-feature front-end → modified-MLP or PirateNet body → SIREN or tanh activations, trained with the loss-weighting and causal methods from the other groups. Hard-constraint output transforms are the odd one out: they don't change capacity or conditioning, they remove a *loss term*, and they compose with every backbone above.

**ONNX / live-lane note.** All five are export-safe for the `onnxruntime-web` live lane provided they are pure tensor ops: the Fourier $\mathbf{B}$ matrix and $\cos/\sin$, the modified-MLP gating, the PirateNet gates, the SIREN $\sin$, and the `apply_output_transform` ansatz all trace into the ONNX graph. The **mandatory** gate is the ONNX-vs-`model.predict` parity check (dossier §3.2) — most critical for the hard-constraint transform, where a missed transform silently breaks the boundary condition in the browser.

---

*See also:* `methods/optimizers.md` (Adam→L-BFGS baseline used with every backbone here), `methods/ntk-weighting.md` and `methods/gradnorm-weighting.md` (loss weighting that pairs with modified-MLP), `methods/fourier-features.md` cross-reference, and the case pages `cases/bench-wave1d.md`, `cases/ind-helmholtz.md`, `cases/bench-allencahn.md`, `cases/bench-navier-cavity.md`, `cases/bench-poisson2d.md`.
