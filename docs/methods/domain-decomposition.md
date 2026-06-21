# Domain decomposition: cPINN / XPINN / FBPINNs

> **Group:** `domain-decomposition` (SOTA methods catalogue §E)
> **PINN-Lab method IDs:** #13 cPINN/XPINN · #14 FBPINNs
> **Exercised by:** `mine-heap-leach-rt` 🟠 (cPINN/XPINN) · `poll-soil-barrier` 🟢 (FBPINN)
> **Status:** documented; cases pending (see [coverage map](../cases/README.md))

## What it is

A single global PINN has to fit the PDE residual over the *entire* space-time domain with *one* network. As the domain grows or the solution becomes multi-scale, this fails for two compounding reasons: (i) the optimization problem becomes increasingly ill-conditioned, and (ii) the **spectral bias** of MLPs — their tendency to learn low frequencies first and high frequencies extremely slowly — caps the resolvable bandwidth of a fixed-width net. **Domain decomposition (DD)** attacks both by splitting the domain into smaller pieces, giving each piece its own (smaller, easier-to-train) network, and then *stitching the pieces back together* so the global solution is consistent. Each subnet only has to represent the solution locally, where it is smoother and lower-bandwidth, and the subnets can be trained in parallel. The methods in this group differ entirely in **how they stitch**:

- **cPINN / XPINN** — **non-overlapping** subdomains with **hard interface losses**. Continuity (of the solution, of fluxes, and of the residual) is added as extra penalty terms in the loss, evaluated on shared interface points. Conceptually closest to discontinuous-Galerkin / finite-volume DD.
- **FBPINNs** — **overlapping** subdomains with a **partition-of-unity (PoU) window**. The global solution is *constructed* as a sum of windowed subnets, so continuity holds **by construction** with no interface loss term at all. Conceptually closest to a finite-element basis expansion (hence "finite basis").

This trade — explicit interface penalties (cPINN/XPINN) vs. built-in continuity through overlap (FBPINNs) — is the central design axis of the group. For PINN-Lab the relevance is direct: the differentiator cases live on **large or multi-scale domains** — a heap-leach pile with metres-deep reactive fronts (`mine-heap-leach-rt`) and a layered soil/barrier profile with a thin low-permeability liner (`poll-soil-barrier`) — exactly the regime where a single global PINN is known to underperform.

---

## Method 13 — cPINN / XPINN

### Explanation

**cPINN** (conservative PINN, Jagtap, Kharazmi & Karniadakis 2020) was designed for **conservation laws**. It tiles the domain with *non-overlapping* subdomains $\Omega_q$, puts an independent network $u_q$ on each, and stitches neighbours along their shared interface $\Gamma_{pq}$ with two conditions: **(1) flux continuity** in strong form (the physically meaningful stitch for a conservation law — for hyperbolic problems the convective flux is matched, for viscous problems both convective and diffusive fluxes), and **(2) solution averaging**, which forces the two networks meeting at the interface to agree on the *average* of their predictions. **XPINN** (extended PINN, Jagtap & Karniadakis 2020) generalises cPINN to *arbitrary* nonlinear PDEs on *arbitrary* (including space-time) geometries by replacing the conservation-law-specific flux condition with a **generic residual-continuity** condition: the PDE residual itself is required to match across the interface, alongside the solution average. Because XPINN decomposes in space *and* time, it can assign more network capacity to regions of sharp dynamics. Both gain **parallelism** (subnets train concurrently) and **local representational capacity** (each subnet only fits a small, smoother patch).

### Key equation

Each subdomain $q$ minimises its own PDE-residual loss plus a shared **interface loss**. For XPINN, on the interface $\Gamma_{pq}$ between subdomains $p$ and $q$, the stitching terms enforce **solution-average agreement** and **residual continuity**:

$$
\mathcal{L}^{q}_{\text{interface}} \;=\;
\underbrace{\frac{w_u}{N_\Gamma}\sum_{i=1}^{N_\Gamma}\Big| u_q(x_i) - \langle u(x_i)\rangle \Big|^2}_{\text{solution average}}
\;+\;
\underbrace{\frac{w_r}{N_\Gamma}\sum_{i=1}^{N_\Gamma}\Big| \mathcal{R}_q(x_i) - \mathcal{R}_p(x_i) \Big|^2}_{\text{residual continuity}},
\qquad
\langle u(x_i)\rangle = \tfrac{1}{2}\big(u_q(x_i)+u_p(x_i)\big),
$$

where $\mathcal{R}_q(x) = \mathcal{N}[u_q](x) - f(x)$ is the PDE residual of subnet $q$, $\{x_i\}$ are the $N_\Gamma$ interface points, and $w_u, w_r$ weight the two stitching terms. For **cPINN** the residual-continuity term is replaced by **flux continuity** in strong form,

$$
\mathcal{L}^{q}_{\text{flux}} = \frac{w_f}{N_\Gamma}\sum_{i=1}^{N_\Gamma}\big| \mathbf{f}_q(x_i)\cdot\mathbf{n} - \mathbf{f}_p(x_i)\cdot\mathbf{n} \big|^2,
$$

with $\mathbf{f}$ the physical flux and $\mathbf{n}$ the interface normal. The global objective is $\sum_q \big(\mathcal{L}^q_{\text{PDE}} + \mathcal{L}^q_{\text{data/BC}} + \mathcal{L}^q_{\text{interface}}\big)$.

### Honest limitations

- The interface conditions are **soft penalties**, not hard constraints — continuity is only approximate, and the stitch can **lose accuracy** at interfaces, the classic weak point of XPINN. A poor interface-weight balance produces visible discontinuities at subdomain boundaries.
- XPINN needs **more data/collocation near interfaces** to make the residual-continuity term informative; over-fine partitioning starves each subnet of interior points and can *increase* total error (the well-known generalization-vs-decomposition trade studied by Hu et al., "When Do Extended PINNs Fail to Learn?", arXiv:2109.09444).
- Non-overlapping tiling means the stitching loss is the *only* coupling — there is no smooth blending, so the global solution's derivatives across interfaces are not guaranteed continuous unless explicitly penalised.

### Canonical references

- A. D. Jagtap, E. Kharazmi, G. E. Karniadakis. *Conservative physics-informed neural networks on discrete domains for conservation laws: Applications to forward and inverse problems.* **CMAME** 365:113028, 2020. DOI: [10.1016/j.cma.2020.113028](https://doi.org/10.1016/j.cma.2020.113028). Code: [github.com/AmeyaJagtap/Conservative_PINNs](https://github.com/AmeyaJagtap/Conservative_PINNs).
- A. D. Jagtap, G. E. Karniadakis. *Extended Physics-Informed Neural Networks (XPINNs): A Generalized Space-Time Domain Decomposition Based Deep Learning Framework for Nonlinear Partial Differential Equations.* **Communications in Computational Physics (CiCP)** 28(5):2002–2041, 2020. DOI: [10.4208/cicp.OA-2020-0164](https://doi.org/10.4208/cicp.OA-2020-0164). Code: [github.com/AmeyaJagtap/XPINNs](https://github.com/AmeyaJagtap/XPINNs).
- (failure-mode analysis) Z. Hu, A. D. Jagtap, G. E. Karniadakis, K. Kawaguchi. *When Do Extended Physics-Informed Neural Networks (XPINNs) Improve Generalization?* **SIAM J. Sci. Comput.** 44(5):A3158–A3182, 2022. arXiv: [2109.09444](https://arxiv.org/abs/2109.09444).

### Which framework implements it

There is **no first-class cPINN/XPINN in the core PINN-Lab engines** (DeepXDE, PhysicsNeMo) — both treat the global domain as one. The canonical implementations are the **Jagtap reference repos (TensorFlow 1)** linked above. In the PINN-Lab pipeline this method is therefore implemented as a **DeepXDE-on-PyTorch recipe**: instantiate one `dde.Model` per subdomain over a `dde.geometry` sub-region, sample interface points, and add the solution-average / flux / residual-continuity terms as custom loss components, training the subnets jointly. PhysicsNeMo can host the same pattern with multiple `FullyConnectedArch` nodes and `PointwiseInteriorConstraint`s on the interface manifolds. (PINA's `Condition` API can also express the interface penalties.)

### Which PINN-Lab case exercises it

**`mine-heap-leach-rt`** 🟠 (mining · reactive transport in a heap/in-situ leach pile). The pile is a large, layered domain with a moving reaction front (Darcy flow + anisotropic advection-diffusion-reaction, bimolecular $A+B\to C$). Tiling the pile by depth/lift and stitching with cPINN flux continuity matches the *physics* of a leaching column — the lixiviant flux must be conserved across each lift interface — making this the natural showcase for the conservation-aware stitch. Labelled `synthetic` (no open spatial-field leaching dataset; see [real-datasets](../cases/real-datasets.md)).

---

## Method 14 — FBPINNs (finite basis PINNs)

### Explanation

**FBPINNs** (Moseley, Markham & Nissen-Meyer 2023) take the finite-element idea literally: express the global solution as a **sum of subdomain networks multiplied by smooth, overlapping, compactly-supported window functions** that form a **partition of unity (PoU)** — at every point the windows sum to one. Because continuity is baked into this construction, **there is no interface loss term** — FBPINNs avoid XPINN's hard interface penalties entirely, which is their main practical advantage. Each subnet is additionally given its own input **normalisation** (so it sees $\mathcal{O}(1)$ coordinates locally) and the output is **unnormalised** back to physical scale, which together with the windowing is what lets FBPINNs resolve **high-frequency and multi-scale** solutions that defeat a single global PINN. The **multilevel** variant (Dolean, Heinlein, Mishra & Moseley 2024) stacks coarse-to-fine PoU levels — analogous to a multigrid hierarchy — and scales to **thousands of subdomains** and very high frequencies, addressing the fact that a single fine level alone still struggles to communicate information across the whole domain.

### Key equation

The global solution is a partition-of-unity-weighted sum of locally normalised subnetworks:

$$
\hat{u}(x) \;=\; C(x)\sum_{j=1}^{J} \, \omega_j(x)\,\big(u_{n,j}\circ \mathcal{N}_j\big)(x),
\qquad
\sum_{j=1}^{J}\omega_j(x) = 1 \;\; \forall x \in \Omega,
$$

where $\omega_j$ is the smooth, compactly-supported **window** for subdomain $j$ (the PoU; nonzero only on $\Omega_j$, and overlapping its neighbours), $u_{n,j}$ is subnet $j$, $\mathcal{N}_j$ is its per-subdomain input normalisation, and $C(x)$ is an optional global constraining operator that imposes boundary/initial conditions as a **hard constraint** (the distance-function ansatz). A common 1-D window built from a smooth bump and normalised over overlapping neighbours is

$$
\omega_j(x) = \frac{\phi_j(x)}{\sum_{k}\phi_k(x)},
\qquad
\phi_j(x) = \Big[1 + \cos\!\big(\pi\,\tfrac{x - x_j}{\sigma_j}\big)\Big]\,\mathbf{1}_{|x-x_j|<\sigma_j},
$$

i.e. a raised-cosine bump centred at $x_j$ with support half-width $\sigma_j$, renormalised so the $\omega_j$ sum to one. Because $\hat{u}$ is built this way, $C^k$ continuity across subdomain boundaries follows from the smoothness of the windows — **no interface penalty is needed**.

### Honest limitations

- The window centres, overlap width $\sigma_j$, and number of levels are **hyperparameters**; too little overlap reverts toward a hard partition, too much erases the locality benefit.
- The PoU construction works most naturally on **box/hyper-rectangular** decompositions; irregular geometries are harder to tile than with a generic XPINN interface.
- For a *single small smooth* problem FBPINNs add overhead with no benefit — the payoff is specifically **large-domain / multi-scale / high-frequency** problems (the regime the paper targets and benchmarks).
- It remains a **PINN**, not a faster forward solver — see the product-wide honesty note: for one well-posed forward solve a classical FEM/FVM solver is usually faster and more accurate ([what PINN-Lab is *not*](../README.md)).

### Canonical references

- B. Moseley, A. Markham, T. Nissen-Meyer. *Finite Basis Physics-Informed Neural Networks (FBPINNs): a scalable domain decomposition approach for solving differential equations.* **Advances in Computational Mathematics** 49(4):62, 2023. arXiv: [2107.07871](https://arxiv.org/abs/2107.07871) · DOI: [10.1007/s10444-023-10065-9](https://doi.org/10.1007/s10444-023-10065-9).
- V. Dolean, A. Heinlein, S. Mishra, B. Moseley. *Multilevel domain decomposition-based architectures for physics-informed neural networks.* **CMAME** 429:117116, 2024. DOI: [10.1016/j.cma.2024.117116](https://doi.org/10.1016/j.cma.2024.117116). arXiv: [2306.05486](https://arxiv.org/abs/2306.05486).

### Which framework implements it

**`benmoseley/FBPINNs`** — the reference library, **JAX** backend, **MIT** licensed, installed with `pip install -e .` from the cloned repo ([github.com/benmoseley/FBPINNs](https://github.com/benmoseley/FBPINNs)). This is the best-engineered decomposition library and the canonical FBPINN/multilevel-FBPINN implementation. **Export caveat for PINN-Lab:** like all JAX engines, the ONNX → `onnxruntime-web` path is fragile (`jax2tf`/`jaxonnxruntime`, see [frameworks/jaxpi](../frameworks/jaxpi/README.md) and the [train→export→ONNX contract](../architecture/train-export-onnx.md)). The pragmatic plan is to **mine FBPINNs for the windowing technique** and reimplement the PoU sum as a thin PyTorch wrapper over per-subdomain `dde.nn.FNN` subnets so the assembled $\hat{u}$ is a single exportable `torch.nn.Module` — keeping the web lane on the verified DeepXDE→PyTorch→ONNX path.

### Which PINN-Lab case exercises it

**`poll-soil-barrier`** 🟢 (pollution · contaminated-site / barrier-design, 1-D/2-D advection-diffusion through a layered soil profile with a thin low-permeability barrier). The thin liner is a **multi-scale** feature — a sharp concentration gradient across a few centimetres embedded in a metres-scale site — exactly what a PoU windowing scheme is built to resolve, and the coverage map pairs this case with **pre-training** for the warm start. Labelled `synthetic`/illustrative on its Benchmark page unless wired to a real head/contaminant sample.

---

## How the group maps into PINN-Lab

| Aspect | cPINN / XPINN (#13) | FBPINNs (#14) |
|---|---|---|
| Subdomains | non-overlapping | overlapping |
| Stitching | hard **interface loss** (solution avg + flux/residual continuity) | **partition-of-unity window** — no interface loss |
| Continuity | approximate (soft penalty) | by construction (smooth windows) |
| Canonical ref | Jagtap et al. CMAME 2020 / CiCP 2020 | Moseley et al. Adv. Comput. Math. 2023 |
| Reference impl | Jagtap repos (TF1) | `benmoseley/FBPINNs` (JAX, MIT) |
| PINN-Lab engine | DeepXDE/PyTorch recipe (multi-`Model` + custom interface loss) | PoU wrapper over PyTorch subnets (web-lane safe) |
| Exercised by | `mine-heap-leach-rt` 🟠 | `poll-soil-barrier` 🟢 |
| Why this case | large layered pile; flux must be conserved across lifts | thin barrier = multi-scale gradient in a large site |

**When to reach for this group (decision rule):** use domain decomposition when the case is **large-domain and/or multi-scale/high-frequency** and a single global PINN stalls or shows spectral-bias artefacts. Prefer **FBPINNs** when continuity matters and you want to avoid tuning interface weights (the default for PINN-Lab's smooth-but-multi-scale pollution/seepage cases); prefer **cPINN/XPINN** when the stitch should carry **physical meaning** — i.e. a conservation law where flux continuity across the interface *is* the right constraint (the heap-leach / reactive-transport case). Do **not** reach for it on small, smooth, single-scale benchmarks (Poisson, 1-D heat) — it only adds cost there.

---

### See also
- [methods/fbpinn](fbpinn.md) — deep-dive companion page for FBPINNs (method #14)
- [methods/hard-constraints](hard-constraints.md) — the $C(x)$ constraining operator FBPINNs reuse
- [methods/fourier-features](fourier-features.md) — the *other* multi-scale / spectral-bias remedy (architecture-side, not decomposition-side)
- [cases/mine-heap-leach-rt](../cases/mine-heap-leach-rt.md) · [cases/poll-soil-barrier](../cases/poll-soil-barrier.md)
- [frameworks/deepxde](../frameworks/deepxde/README.md) · [frameworks/jaxpi](../frameworks/jaxpi/README.md)
