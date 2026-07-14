# ind-hidden-velocity — the current recovered from dye alone (the Hidden Fluid Mechanics mechanism)

The flagship of the estimation reframe (issue #48): the case that demonstrates, at CPU-lane scale, the mechanism of
**Hidden Fluid Mechanics** (Raissi, Yazdani & Karniadakis, *Science* 367(6481):1026-1030, 2020,
[doi:10.1126/science.aaw4741](https://doi.org/10.1126/science.aaw4741)): estimating a velocity field from
flow-visualization (passive scalar) data, "extracting quantitative information for which direct measurements may not
be possible". The network never sees a velocity datum, and no IC/BC on the dye is imposed — that absence is the
selling point the paper states explicitly ("agnostic to the geometry or the initial and boundary conditions").

## Problem

A steady incompressible cellular flow on the unit square advects and diffuses a passive dye:

$$ c_t + \mathbf{u}\cdot\nabla c = D\,\nabla^2 c, \qquad \nabla\cdot\mathbf{u} = 0, \qquad D = 0.02, $$

with the closed-form flow truth

$$ \mathbf{u}^* = A\big(\sin\pi x\cos\pi y,\ -\cos\pi x\sin\pi y\big), \qquad A = 1.5 $$

(divergence-free by construction; the walls are streamlines, so the advective normal flux vanishes there). A Gaussian
dye patch (center $(0.5, 0.2)$, $s_0^2 = 0.006$) is released at $t=0$ and sweeps roughly three quarters of the ring
around the vortex center by $t=1$.

**Observed:** ~640 sparse space-time samples of $c$ with Gaussian noise (0.5% of max), plus the physics.
**Held out:** another 160 samples never shown to the optimizer (out-of-sample dye validation).
**Unknown (the estimate):** the whole velocity field $(u, v)$ — two hidden fields that were never measured.

## The dye truth (a numerical reference with its checks)

There is no closed form for $c$ under a non-uniform flow, so the dye truth is a seeded explicit finite-difference
solve on a $129^2$ grid — with its stability VERIFIED, never assumed (the lesson from the diverged Navier FDM):

- **Central differencing is legitimate** because the cell Péclet number $A\,\Delta x/D \approx 0.59 < 2$; an upwind
  scheme would inject numerical diffusion of order $A\,\Delta x/2 \approx 0.006$, comparable to $D$ itself, and
  corrupt the reference.
- $\Delta t = 4\times10^{-4}$ satisfies both the advective and the diffusive CFL limits, asserted in code.
- No-flux walls via reflected ghost cells; **mass conservation is asserted** (< 1% drift) along with boundedness
  and finiteness. The solve fails loudly rather than baking garbage.

## Method

A single FNN `[3, 64, 64, 64, 64, 3]` maps $(x, y, t) \mapsto (u, v, c)$. The loss couples five terms
(`loss_weights = [1, 1, 1, 1, 60]`):

$$ \mathcal{L} = \big\|c_t + u\,c_x + v\,c_y - D\nabla^2 c\big\|^2 + \big\|u_x + v_y\big\|^2
   + \big\|u_t\big\|^2 + \big\|v_t\big\|^2
   + \lambda \sum_i \big(c_\theta(x_i, y_i, t_i) - c_i^{\text{obs}}\big)^2. $$

The transport residual is what couples the visible to the hidden: wherever the dye has a gradient, the
$\mathbf{u}\cdot\nabla c$ term forces the velocity to explain the patch's motion; the (soft) incompressibility
residual propagates that information along streamlines. $D$ is known, HFM-style. Incompressibility is soft rather
than a stream-function hard constraint because the exported ONNX must emit $(u, v, c)$ directly (a $\psi$-derivative
formulation does not export cleanly).

The $u_t = v_t = 0$ residuals encode the stated **steady-flow assumption**, and they are load-bearing: in the first
training run (without them) the net spent its freedom on a time-varying velocity that ~640 sparse dye samples cannot
pin, and the recovered current was 38-60% off even inside the swept region — measured, not guessed. Declaring the
current steady aggregates the dye information from ALL times into ONE field. This mirrors real practice: you assert
what you legitimately know (a quasi-steady current over the observation window) and let the data determine the rest.
Training: Adam $1.5\times10^4$ steps then L-BFGS, seed 42.

## Honest identifiability: the swept mask

Where dye never passed, $\nabla c \approx 0$ and transport does **not** constrain $\mathbf{u}$: the velocity is
unidentifiable there for ANY method — the same physics that made the heat2d-inverse recovery local to $|\nabla T|$.
The case bakes the **dye-swept mask** (where $\max_t |\nabla c_{\mathrm{FD}}|$ exceeds 5% of its max; ~67% of the
domain) into every trace and reports the error split:

- `speed_rel_rmse_swept` — relative speed RMSE **inside** the swept region (the recoverable claim),
- `speed_rel_rmse_dead` — the never-dyed dead zones (published as-is; expected large),
- the full-grid `l2_relative` on $u$ (the primary metric) honestly includes both.

## Result (measured, seed 42)

See the baked manifest for the shipped numbers (`data/derived/manifests/ind-hidden-velocity.json`): the recovered
current inside the swept region, the dead-zone error, the held-out dye RMSE (`dye_holdout_rmse`, from the 160
never-trained samples via the exported ONNX), and the recovered circulation center vs the true $(0.5, 0.5)$ on the
case's AnswerCard. The Diagnostics view overlays the recovered $u$ and $v$ mid-line profiles on the closed-form
truth.

## What the web shows

- **Field (HiddenFlowKit):** recovered speed $|\mathbf{u}|$ with the dye-sample dots (the only evidence), the true
  $|\mathbf{u}^*|$ on the same color scale, the pointwise error, the reconstructed dye at the variant's $t$, and the
  swept mask. Read the error map against the mask: the error lives where the dye never went.
- **Diagnostics:** mid-line $u$ and $v$ profiles, recovered vs closed form.
- **Live:** the trained ONNX sweeps $t$ in the browser.
- **AnswerCard:** "You can only see the dye. What is the current underneath?" with the recovered circulation center
  and the swept/dead error split.

## Scope and limits

Steady 2D incompressible flow, one passive scalar, known $D$, window $t\in[0,1]$. Synthetic by design: the
closed-form velocity truth is what makes the recovery exactly scoreable, which no real flow allows. Out of scope:
unsteady/turbulent flow, pressure (the velocity couples to transport here, not Navier-Stokes), reactive dye, unknown
$D$ (a natural extension: $D$ as a trainable variable). The dead-zone unidentifiability is physics, not a method
defect, and is shown rather than hidden.
