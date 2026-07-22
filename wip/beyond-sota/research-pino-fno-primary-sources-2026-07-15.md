# PINO and the FNO family — transcribed from the primary sources (2026-07-15)

Everything here was read from the paper itself (the PDF the owner supplied, and the arXiv pages), not from
memory. Equation numbers are the paper's own. Quotes are verbatim and marked as quotes. This dossier is the
source for the PINO implementation and for the docs written alongside it.

---

## 1. PINO — Physics-Informed Neural Operator

**Li, Zheng, Kovachki, Jin, Chen, Liu, Azizzadenesheli, Anandkumar.** *Physics-Informed Neural Operator for
Learning Partial Differential Equations.* arXiv:2111.03794v1, 9 November 2021. Journal version: ACM/IMS
Journal of Data Science 1(3), 2024, DOI [10.1145/3648506](https://doi.org/10.1145/3648506).

### 1.1 The problem it exists to solve

The paper frames exactly the two failure modes that matter here.

**PINN side** (§1, §2.2 "Challenges of PINN"), verbatim:

> "PINNs face several optimization issues: (1) the challenging optimization landscape from soft physics or
> PDE constraints, (2) the difficulty to propagate information from the initial or boundary conditions to
> unseen parts of the interior or to future times, and (3) the sensitivity to hyper-parameters selection. As
> a result, PINNs are still unable to compete with conventional solvers in most cases, and they often fail to
> converge on high-frequency or multi-scale PDEs."

and

> "as an iterative solver, PINNs have difficulty propagating information from the initial condition or
> boundary condition to unseen parts of the interior or to future times. For example, in challenging problems
> such as turbulence, PINNs are only able to solve the PDE on a relatively small domain, or otherwise,
> require extra observational data which is not always available in practice."

**This is the citable form of the critique we received.** The commenter's "long training times and gradient
explosions" maps onto (1) the optimization landscape of the soft constraint and (3) hyper-parameter
sensitivity; the paper's own answer to instability is quoted in §1.4 below.

**FNO / operator side** (§2.4 "Challenges of Operator learning"), verbatim:

> "the data challenges remain: (1) the need for training data, which assumes an existing solver or
> experimental setup, (2) the non-negligible generalization error, and (3) extrapolation to unseen
> conditions."

and

> "since FNO doesn't use any knowledge of the equation, it cannot get arbitrarily close to the ground truth by
> using the higher resolution as in conventional solvers, leaving a gap of generalization error."

That last sentence is the precise indictment of **our current** operator lane, which is data-driven only.

### 1.2 The formulation

Neural operator (Definition 1, eq. 8):

$$\mathcal{G}_\theta := \mathcal{Q} \circ (W_L + \mathcal{K}_L) \circ \cdots \circ \sigma(W_1 + \mathcal{K}_1) \circ \mathcal{P}$$

with $\mathcal{P}$ the pointwise lifting to a higher-dimensional channel space, $\mathcal{Q}$ the pointwise
projection back, $W_l$ pointwise linear maps and $\mathcal{K}_l$ integral kernel operators.

Fourier convolution operator (Definition 2, eq. 9) — the FNO layer:

$$(\mathcal{K} v_t)(x) = \mathcal{F}^{-1}\big(R \cdot (\mathcal{F} v_t)\big)(x)$$

where $R$ is learned (the truncated set of Fourier modes).

Losses:

- PDE loss, stationary (eq. 3): $\mathcal{L}_{\text{pde}}(a, u_\theta) = \|\mathcal{P}(u_\theta, a)\|^2_{L^2(D)} + \alpha \|u_\theta|_{\partial D} - g\|^2_{L^2(\partial D)}$
- PDE loss, dynamic (eq. 4): $\left\|\frac{du_\theta}{dt} - \mathcal{R}(u_\theta)\right\|^2_{L^2(T;D)} + \alpha\|u_\theta|_{\partial D} - g\|^2 + \beta\|u_\theta|_{t=0} - a\|^2_{L^2(D)}$
- data loss (eq. 5): $\mathcal{L}_{\text{data}}(u, \mathcal{G}_\theta(a)) = \|u - \mathcal{G}_\theta(a)\|^2_{\mathcal{U}}$
- operator data loss (eq. 6): $\mathcal{J}_{\text{data}}(\mathcal{G}_\theta) = \mathbb{E}_{a\sim\mu}[\mathcal{L}_{\text{data}}]$
- operator PDE loss (eq. 7): $\mathcal{J}_{\text{pde}}(\mathcal{G}_\theta) = \mathbb{E}_{a\sim\mu}[\mathcal{L}_{\text{pde}}(a, \mathcal{G}_\theta(a))]$

### 1.3 The two phases (§3) — this is the part to implement

**Phase 1, pre-train the solution operator.** Learn $\mathcal{G}_\theta$ using the data loss and/or the PDE
loss. The key move (Algorithm 1) is that the PDE loss needs **no labels**, so virtual instances can be sampled
endlessly:

> Algorithm 1: for i = 0,1,2...: compute $\mathcal{L}_{\text{data}}$ and $\mathcal{L}_{\text{pde}}$ on
> $(a_i, u_i)$, update $\mathcal{G}$; then for j = 1..K: sample $a'$ from $\mu$, compute
> $\mathcal{L}_{\text{pde}}$ on $a'$, update $\mathcal{G}$.

verbatim on the consequence:

> "One can sample virtual PDE instances by drawing additional initial conditions or coefficient conditions
> $a_j \sim \mu$ for training... In this sense, we have access to the unlimited dataset by sampling new $a_j$
> in each iteration."

**Phase 2, test-time optimization.** Use the pre-trained $\mathcal{G}_\theta(a)$ as the **ansatz** for the
queried instance and minimize the PDE loss on it, plus an **anchor loss** that keeps it near the pre-trained
operator:

$$\mathcal{L}_{op}\big(\mathcal{G}_{\theta_i}(a), \mathcal{G}_{\theta_0}(a)\big) := \big\|\mathcal{G}_{\theta_i}(a) - \mathcal{G}_{\theta_0}(a)\big\|^2_{\mathcal{U}}$$

updating with $\mathcal{L}_{\text{pde}} + \alpha \mathcal{L}_{op}$.

### 1.4 Why it is more stable (the answer to "gradient explosions")

Verbatim, §3.2 "Optimization landscape":

> "Using the operator as the ansatz has two major advantages: (1) PINN does point-wise optimization, while
> PINO does optimization in the space of functions. In the linear integral operation $\mathcal{K}$, the
> operator parameterizes the solution function as a sum of the basis function. Optimization of the set of
> coefficients and basis is easier than just optimizing a single function as in PINNs. (2) we can learn these
> basis functions in the operator learning phase which makes the later test-time optimization even easier. In
> PINO, we do not need to propagate the information from the initial condition and boundary condition to the
> interior. It just requires fine-tuning the solution function parameterized by the solution operator."

and on instability specifically, §3.2 "Trade-off" (2):

> "using a higher resolution and finer grid will reduce the truncation error. However, it may make the
> optimization unstable. Using hard constraints such as the anchor loss $\mathcal{L}_{op}$ relieves such a
> problem."

### 1.5 How the PDE residual is computed on a grid output (§3.3) — the crux

An operator outputs $u$ **on a grid**, so "just use autograd" is not automatic. The paper gives:

- **Numerical differentiation.** Finite difference or the Fourier gradient. Cost: *"given a n-points grid,
  finite difference requires $O(n)$ and Fourier method requires $O(n\log n)$."* Limits, verbatim: *"finite
  difference methods require a fine-resolution uniform grid; spectral methods require smoothness and uniform
  grids. Especially, these numerical errors on the gradient will be amplified on the output solution."*
- **Autograd.** Exact, but *"it is not straightforward to write out the solution function in the neural
  operator which directly outputs the numerical solution $u = \mathcal{G}_\theta(a)$ on a grid, especially for
  FNO which uses FFT. To apply autograd, we design a query function $\hat{u}$ that input $x$ and output
  $u(x)$"* — eq. (10): $u(x) = \mathcal{Q}(v_L(x)) = \mathcal{Q}\big((W_L v_{L-1})(x) + \mathcal{K}_L v_{L-1}(x)\big)$.
- The paper also states in its contributions: *"We develop an efficient method to compute the exact gradient
  for neural operators to incorporate the equation constraints."*

**Implementation decision for PINN-Lab:** our Darcy grid is uniform and the solution is smooth, so the
**spectral (Fourier) derivative** is the right first choice, with a finite-difference cross-check to prove the
residual is not an artefact of the differentiation scheme. Document the periodicity caveat (FFT assumes
periodicity; FC-PINO, Maust et al. arXiv:2211.15960, is the non-periodic fix).

### 1.6 The headline numbers (quote only these, with the source)

| Claim (verbatim) | Where |
|---|---|
| "PINO still outperforms PINN by 20x smaller error and 25x speedup on the chaotic Kolmogorov flow" — *even without any pre-training, using only PDE constraints for the given instance* | §1 contributions |
| "On average it has 7% smaller error on the transient and Kolmogorov flows [vs FNO], while matching the speedup of FNO (400x) compared to the GPU-based pseudo-spectral solver" | §1 contributions |
| "It can solve the 2d transient flow over an extremely long time period, where PINN and DeepONet fail to converge" | §1 contributions |
| "the pre-trained PINO model on the Navier Stokes equation can be easily transferred to different Reynolds numbers ranging from 100 to 500 using test-time optimization" | §1 contributions |
| "recovers the coefficient function in the Darcy flow which is 3000x faster than the conventional solvers using accelerated MCMC" | §1 contributions (inverse) |

**Honesty rule:** these are the *paper's* numbers on the *paper's* problems at the *paper's* scale. A modest
reproduction on our own Darcy grid may NOT quote them as its own result. Our case reports **our** measured
numbers and cites the paper's separately.

---

## 2. FNO — Fourier Neural Operator

**Li, Kovachki, Azizzadenesheli, Liu, Bhattacharya, Stuart, Anandkumar.** *Fourier Neural Operator for
Parametric Partial Differential Equations.* arXiv:2010.08895, submitted 18 October 2020, final 17 May 2021.
ICLR 2021.

Abstract, verbatim on the results:

> "The Fourier neural operator is the first ML-based method to successfully model turbulent flows with
> zero-shot super-resolution. It is up to three orders of magnitude faster compared to traditional PDE
> solvers. Additionally, it achieves superior accuracy compared to previous learning-based solvers under
> fixed resolution."

Experiments: Burgers' equation, Darcy flow, Navier-Stokes. This is the operator our `bench-darcy-operator`
already implements, on the Darcy family, data-driven.

---

## 3. The practitioner's reference for getting FNO right

**Duruisseaux, Kossaifi, Anandkumar.** *Fourier Neural Operators Explained: A Practical Perspective.*
arXiv:2512.01421, submitted 1 December 2025, revised 22 January 2026. 96 pages, 27 figures.

Why it matters here, verbatim from the abstract:

> "the practical use of FNOs is often hindered by an incomplete understanding among practitioners of their
> theoretical foundations, practical constraints, and implementation details, which can lead to their
> incorrect or unreliable application."

It is *"closely integrated with the NeuralOperator 2.0.0 library, offering modular state-of-the-art
implementations that faithfully reflect the theory."* Use it as the implementation authority for the spectral
parameterization, the component design, and the documented common mistakes. **Action:** read the relevant
sections before finalising our FNO/PINO layer, and record any correction it forces on our existing
`model/fno.py`.

---

## 4. What this dossier commits us to

1. Implement **PINO properly**: pre-training with data + PDE loss and virtual-instance sampling, then
   test-time optimization with the anchor loss. Not a token residual term.
2. Compute the residual **spectrally** on the uniform grid, with a finite-difference cross-check.
3. Report **our own** measured numbers: PINO vs data-only FNO vs a plain PINN on the *same* Darcy family and
   the same grid, including **training time** and a **data-reduction** curve (error vs number of labelled
   instances), which is the honest small-scale version of the paper's claim.
4. Fix the DeepONet/PINO overclaims in the docs (see the audit dossier).
5. State the caveat: FFT periodicity, and that a classical solver still wins a single well-posed forward solve.
