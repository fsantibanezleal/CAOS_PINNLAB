# Variational and scalable PINNs — weak form & separable architectures

> **Group:** `variational-scalable` · **Methods covered:** hp-VPINNs (weak/variational form) · SPINN (separable PINNs)
> **Why this group exists.** The two families here attack two different walls that the *vanilla* collocation PINN
> hits: (1) the **regularity wall** — the strong-form residual demands second (or higher) derivatives everywhere,
> which is brutal on solutions with sharp gradients or low regularity; and (2) the **scaling wall** — the cost of a
> dense collocation grid grows as $N^d$ in $d$ dimensions, so a 4-D space-time problem at decent resolution is simply
> unaffordable point-wise. The **weak form** (hp-VPINNs) lowers the derivative order via integration by parts and
> localizes optimization through $h$/$p$-refinement; **separability** (SPINN) factorizes the network across axes so
> that a tensor-product grid of $>10^7$ collocation points fits on a single GPU. They are complementary and can be
> combined.

---

## What "variational" and "scalable" mean here

A standard PINN minimizes the **strong-form** PDE residual at a finite set of collocation points. For a PDE
$\mathcal{N}[u](x) = f(x)$ on a domain $\Omega$, with the network $u_\theta$ as the candidate solution, the interior
loss is

$$
\mathcal{L}_{\text{res}}(\theta) \;=\; \frac{1}{N}\sum_{i=1}^{N} \big\lvert\, \mathcal{N}[u_\theta](x_i) - f(x_i) \,\big\rvert^{2}.
$$

This formulation has two structural costs that the methods below remove:

1. **High derivative order.** $\mathcal{N}$ is applied *pointwise*, so for a second-order operator (Poisson, diffusion,
   Helmholtz) every collocation point needs $\partial^2 u_\theta/\partial x^2$ via automatic differentiation. Where the
   true solution is only piecewise-smooth or has steep layers, accurately matching a *second derivative* is far harder
   than matching the solution itself. **Variational / weak-form PINNs** integrate the residual against test functions
   and integrate by parts, shifting derivatives onto the (smooth, analytic) test functions and lowering the order the
   network must produce.

2. **Curse of dimensionality in the collocation set.** A tensor grid with $n$ points per axis is $n^d$ points; each is
   a full network forward+backward pass. **Separable PINNs** replace the single $d$-input MLP with $d$ small per-axis
   sub-networks and reconstruct the field by an outer product, so the *forward passes* scale as $d\cdot n$ while the
   *evaluated field* still covers all $n^d$ grid points — turning a multiplicative cost into an additive one.

---

## Method 1 — hp-VPINNs (variational / weak-form PINNs)

**What it is.** The hp-VPINN recasts the PDE in its **weak (variational) form**: instead of forcing the strong residual
to zero at points, it forces the residual to be orthogonal to a finite set of **test functions** (the Petrov–Galerkin
idea). The **trial space** is the neural network $u_\theta$, defined *globally* over the whole domain; the **test space**
is a set of **piecewise high-order polynomials** (Kharazmi et al. use Legendre polynomials per sub-element). Because the
weak form is obtained by multiplying by a test function and **integrating by parts**, the second-order operator becomes a
first-order one acting on $u_\theta$ and a derivative on the (analytically known) test function — so the network only has
to produce *lower-order* derivatives, which helps sharp-gradient / low-regularity solutions. The "hp" is borrowed from
finite elements: **h-refinement** splits the domain into more sub-elements (more local test functions), and
**p-refinement** raises the polynomial order of the test functions; together they give a *global approximation with a
local learning algorithm* — the variational residual decouples element-by-element, letting optimization localize where
the solution is hard (singularities, boundary layers). The price is that you must compute the variational integrals by
**numerical quadrature** (Gauss–Legendre), and you must choose the test basis and element partition — more setup and more
moving parts than a plain collocation PINN.

**Key equation.** For a problem $\mathcal{N}[u]=f$, multiply by a test function $v_k$, integrate over the domain, and
integrate by parts to move derivatives onto $v_k$. For the prototypical Poisson case $-u''(x)=f(x)$ on an element, the
**variational residual** the network must zero out for each test function $v_k$ is

$$
\mathcal{R}_k(\theta) \;=\; \int_{\Omega} u_\theta'(x)\, v_k'(x)\,\mathrm{d}x \;-\; \int_{\Omega} f(x)\, v_k(x)\,\mathrm{d}x \;=\; 0,
\qquad k = 1,\dots,K,
$$

where the boundary term from integration by parts vanishes because the test functions $v_k$ are chosen to be zero on the
element boundaries. Note the operator now needs only $u_\theta'$ (first order) rather than $u_\theta''$ (second order).
The loss is the sum of squared variational residuals over the test basis, evaluated by quadrature:

$$
\mathcal{L}_{\text{var}}(\theta) \;=\; \sum_{e}\sum_{k} \big\lvert \mathcal{R}_k^{(e)}(\theta) \big\rvert^{2},
\qquad
\mathcal{R}_k^{(e)}(\theta) \approx \sum_{q} w_q\, \big[\, u_\theta'(x_q)\,v_k'(x_q) - f(x_q)\,v_k(x_q) \,\big],
$$

with $\{x_q, w_q\}$ the Gauss–Legendre quadrature nodes/weights on element $e$ and $\{v_k\}$ the local (piecewise)
Legendre test functions. Boundary/initial conditions are still imposed as in standard PINNs (penalty or hard-constraint
ansatz).

**Canonical reference.** E. Kharazmi, Z. Zhang, G. E. Karniadakis, *"hp-VPINNs: Variational physics-informed neural
networks with domain decomposition,"* **Computer Methods in Applied Mechanics and Engineering (CMAME)**, Vol. 374,
113547, 2021 — DOI [10.1016/j.cma.2020.113547](https://doi.org/10.1016/j.cma.2020.113547); preprint
[arXiv:2003.05385](https://arxiv.org/abs/2003.05385). (Predecessor "VPINNs": Kharazmi, Zhang, Karniadakis, 2019,
[arXiv:1912.00873](https://arxiv.org/abs/1912.00873).)

**Framework / implementation.** Reference implementation: **`ehsankharazmi/hp-VPINNs`**
([github.com/ehsankharazmi/hp-VPINNs](https://github.com/ehsankharazmi/hp-VPINNs)), TensorFlow 1.x — used in PINN-Lab as
the *technique reference* for the weak-form lane. There is no first-class hp-VPINN in DeepXDE; in the precompute lane the
weak-form loss is assembled by hand (build the local Legendre test basis, evaluate $u_\theta'$ at Gauss–Legendre nodes
via `dde.grad.jacobian` / `torch.autograd`, sum the quadrature) on top of a DeepXDE network, or ported from the TF
reference. PINA also exposes variational-style residual hooks but does not ship hp-VPINN as a named solver.

**Why / when to use it.** Reach for the weak form when (a) the solution has **steep gradients, boundary layers, or low
regularity** that make the second-derivative strong residual badly conditioned; (b) the operator is **high-order** and you
want to integrate by parts down to first order; or (c) you want **localized refinement** — concentrate test functions
(h/p-refine) only where the solution is hard, instead of globally densifying collocation points. The exponential
$hp$-convergence story is strongest for problems with **localized features / singularities**.

**Honest limitations.** The weak form is **not free**: you pay for quadrature (accuracy is bounded by the quadrature
rule — under-integration silently corrupts the loss), you must design the test basis and element partition (extra
hyperparameters: number of elements, polynomial order), and the integration-by-parts manipulation must be re-derived per
PDE (harder for nonlinear or non-self-adjoint operators, and for general boundary terms). For *smooth, well-behaved*
forward problems a strong-form collocation PINN is usually simpler and just as good — the weak form earns its complexity
on sharp/low-regularity solutions. As with all PINNs, it does **not** beat a good classical $hp$-FEM solver on a single
well-posed forward solve; its value is the PINN package (mesh-free, inverse-friendly, differentiable artifact) *plus* the
weak form's robustness to low regularity.

**PINN-Lab case that exercises it (per the coverage map).** Method **#15 (`methods/hp-vpinn`)** is first exercised in
case **#20 `ind-helmholtz`** — the Helmholtz problem $\nabla^2 u + \kappa^2 u = f$ (frequency-domain), where the weak
form is paired with high-frequency Fourier features / SIREN. Helmholtz is a natural weak-form target: it is second-order
and oscillatory, so integration by parts to first order plus localized $p$-refinement of the test basis directly attacks
the regularity/conditioning pain.

---

## Method 2 — SPINN (separable physics-informed neural networks)

**What it is.** SPINN factorizes the network **per coordinate axis** to make a dense tensor-product collocation grid
affordable. Instead of one MLP $u_\theta(x_1,\dots,x_d)$ taking the full $d$-dimensional coordinate, SPINN uses **$d$
independent sub-networks** $\{f^{(j)}\}_{j=1}^{d}$, each taking a *single* scalar axis $x_j$ and emitting an
$r$-dimensional feature vector (one "body" per latent rank). The scalar field at a grid point is reconstructed by an
**outer product followed by a sum** over the $r$ rank-1 components — exactly a **CP / canonical tensor decomposition** of
the solution field on the grid. The decisive consequence: to evaluate the field on an $n^d$ tensor grid you only run each
1-D sub-net on its $n$ axis samples — **$d\cdot n$ forward passes instead of $n^d$**. Combined with **forward-mode
automatic differentiation** (cheap because each sub-net has a 1-D input), this lets SPINN train with **$>10^7$
collocation points on a single commodity GPU**, reporting roughly **62× wall-clock speedup and ~1,394× fewer FLOPs** at
matched collocation count, with *better* accuracy on multi-dimensional PDEs. The structural assumption is that the
solution is well approximated by a **low-rank** (rank-$r$) separable representation; the knob is $r$ — raise it for
solutions that are far from separable.

**Key equation.** With per-axis sub-networks $f^{(j)}: \mathbb{R} \to \mathbb{R}^{r}$ (the $k$-th output of
$f^{(j)}(x_j)$ written $f^{(j)}_k(x_j)$), SPINN represents the solution as a **rank-$r$ outer-product (CP)
decomposition**:

$$
u_\theta(x_1,\dots,x_d) \;=\; \sum_{k=1}^{r} \; \prod_{j=1}^{d} f^{(j)}_k(x_j).
$$

On a tensor-product grid $\{x_1^{(i_1)}\}\times\cdots\times\{x_d^{(i_d)}\}$ this is a sum of $r$ rank-1 tensors built from
only $d\cdot n$ sub-net evaluations, while covering all $n^d$ grid nodes — the source of the scaling win. The PDE residual
loss is the usual strong-form term evaluated on this grid (derivatives obtained efficiently by forward-mode AD over the
1-D inputs):

$$
\mathcal{L}_{\text{res}}(\theta) \;=\; \frac{1}{n^{d}} \sum_{i_1,\dots,i_d}
\Big\lvert\, \mathcal{N}\big[u_\theta\big]\big(x_1^{(i_1)},\dots,x_d^{(i_d)}\big) - f\big(\cdot\big) \,\Big\rvert^{2}.
$$

**Canonical reference.** J. Cho, S. Nam, H. Yang, S.-B. Yun, Y. Hong, E. Park, *"Separable Physics-Informed Neural
Networks,"* **Advances in Neural Information Processing Systems 36 (NeurIPS 2023), Spotlight.** Preprints:
[arXiv:2211.08761](https://arxiv.org/abs/2211.08761) (initial, *"Separable PINN: Mitigating the Curse of Dimensionality
in Physics-Informed Neural Networks"*) and the camera-ready
[arXiv:2306.15969](https://arxiv.org/abs/2306.15969). Project page:
[jwcho5576.github.io/spinn.github.io](https://jwcho5576.github.io/spinn.github.io/).

**Framework / implementation.** Reference implementation: **`stnamjef/SPINN`**
([github.com/stnamjef/SPINN](https://github.com/stnamjef/SPINN)), **JAX** (forward-mode AD is the enabling primitive).
A separable-PINN solver also ships in **`jinns`** (`SeparablePINN`, JAX/Equinox) — the preferred maintained path for
PINN-Lab's JAX experiments. In the PyTorch/DeepXDE precompute lane there is no native separable solver; SPINN is the
**JAX technique-donor** for the high-resolution lane, exercised through the `jaxpi`/`jinns` framework pages and ported
only where the per-axis factorization is worth the custom training loop. (Note: JAX→ONNX export is fragile, so SPINN-
trained cases that need the browser lane are re-evaluated to the compact replay artifact rather than shipped as live
ONNX — see `docs/architecture/train-export-onnx`.)

**Why / when to use it.** Use SPINN when the bottleneck is the **collocation grid size** in **3-D / 4-D space-time**
problems where a dense MLP collocation PINN is infeasible, and the solution is plausibly **low-rank separable**
(diffusion, many wave/transport problems, smooth high-dimensional fields). It is the natural backbone for the
**high-resolution variant** of cases in PINN-Lab where you want orders of magnitude more collocation points at fixed GPU
budget.

**Honest limitations.** SPINN's accuracy is **bounded by the rank $r$**: a genuinely non-separable / high-rank solution
(strong cross-axis coupling, turbulence-like features) needs a large $r$ and the advantage erodes — separability is an
*assumption*, not a free lunch, and you should validate it (sweep $r$, check the residual plateau). The outer-product
construction is most natural on **tensor-product (axis-aligned) grids and rectangular domains**; irregular geometry and
non-tensor sampling are awkward. The big speedups are reported on **forward** problems with smooth solutions; the
gains are smaller where you cannot use a structured grid. And as above, the canonical implementation is JAX — the
forward-mode-AD speedup does not transfer for free to PyTorch, and the JAX→ONNX path for the web lane is fragile.

**PINN-Lab case that exercises it (per the coverage map).** Method **#16 (`methods/separable-pinn`)** is exercised in the
**high-resolution variant of the heat / cavity cases** (coverage map: *"high-res variant (heat/cavity)"*) — i.e. a
SPINN-trained, $>10^6$-collocation-point rendering of **`bench-heat1d`** (and the 2-D `bench-navier-cavity` space-time
variant) used to demonstrate the scaling lane against the dense-MLP baseline. It is the group's answer to "how do we get
a high-resolution field without a $n^d$ collocation blow-up."

---

## When the weak form vs. separability helps — a quick decision guide

| Symptom in your case | Reach for | Why |
|---|---|---|
| Sharp gradients / boundary layers / low regularity; high-order operator | **hp-VPINN (weak form)** | Integration by parts lowers derivative order; h/p-refine localizes effort where the solution is hard |
| Need localized refinement (singularities) without globally densifying points | **hp-VPINN** | Variational residual decouples element-wise → local learning |
| 3-D / 4-D space-time, dense collocation grid infeasible, solution ~ low-rank | **SPINN (separable)** | Outer-product factorization turns $n^d$ field cost into $d\cdot n$ forward passes; $>10^7$ points on one GPU |
| Smooth, well-posed, low-dimensional forward solve | *neither* — a plain strong-form PINN (or a classical solver) is simpler | The extra machinery only earns its cost on regularity or scaling pain |
| Both low regularity **and** high dimension | **combine** (weak form on separable nets) | The two ideas are orthogonal: weak form for the regularity wall, separability for the scaling wall |

Both methods share the standard PINN honesty caveat (see `docs/README.md` and the Benchmark page): for a *single*
well-posed forward problem a tuned classical $hp$-FEM / spectral solver is typically faster and more accurate. The weak
form and separability widen the *range of problems a PINN can handle at all* (low-regularity, high-dimensional) and keep
the PINN advantages (mesh-free, inverse-ready, a single differentiable artifact) — they do not overturn that baseline.

---

### References

- Kharazmi, Zhang, Karniadakis (2021). *hp-VPINNs: Variational physics-informed neural networks with domain
  decomposition.* CMAME 374, 113547. DOI [10.1016/j.cma.2020.113547](https://doi.org/10.1016/j.cma.2020.113547) ·
  [arXiv:2003.05385](https://arxiv.org/abs/2003.05385). Code: [ehsankharazmi/hp-VPINNs](https://github.com/ehsankharazmi/hp-VPINNs).
- Kharazmi, Zhang, Karniadakis (2019). *Variational Physics-Informed Neural Networks (VPINNs).*
  [arXiv:1912.00873](https://arxiv.org/abs/1912.00873). (Predecessor — no domain decomposition.)
- Cho, Nam, Yang, Yun, Hong, Park (2023). *Separable Physics-Informed Neural Networks.* NeurIPS 2023 (Spotlight).
  [arXiv:2211.08761](https://arxiv.org/abs/2211.08761) / [arXiv:2306.15969](https://arxiv.org/abs/2306.15969). Code:
  [stnamjef/SPINN](https://github.com/stnamjef/SPINN) (JAX); separable solver also in
  [jinns](https://gitlab.com/mia_jinns/jinns) (`SeparablePINN`).

*Coverage-map anchors: method #15 hp-VPINN → case #20 `ind-helmholtz`; method #16 SPINN → high-res heat/cavity variant.*
