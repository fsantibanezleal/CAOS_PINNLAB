"""PINO — the Physics-Informed Neural Operator training scheme, on top of the FNO2d backbone.

Implements Li, Zheng, Kovachki, Jin, Chen, Liu, Azizzadenesheli & Anandkumar, *Physics-Informed Neural
Operator for Learning Partial Differential Equations*, arXiv:2111.03794 (ACM/IMS J. Data Science 1(3), 2024,
doi:10.1145/3648506). Transcribed from the paper; see `wip/beyond-sota/research-pino-fno-primary-sources-2026-07-15.md`.

PINO is NOT a new architecture: it is a training strategy on top of a neural operator. The operator is the
same FNO (paper Definitions 1-2, our `model/fno.py`). What PINO adds is the paper's Section 3:

  Phase 1  pre-train the solution operator with the data loss AND/OR the PDE loss (Algorithm 1). Because the
           PDE loss needs no labels, virtual instances a' ~ mu can be sampled endlessly: "we have access to
           the unlimited dataset by sampling new a_j in each iteration".
  Phase 2  test-time optimization: use the pre-trained G_theta(a) as the ANSATZ for the queried instance and
           minimise the PDE loss on it, plus the anchor (operator) loss
               L_op(G_theta_i(a), G_theta_0(a)) := || G_theta_i(a) - G_theta_0(a) ||^2
           which keeps the fine-tuned operator near the pre-trained one. The paper is explicit that this is
           what stabilises the fine grid: "using a higher resolution and finer grid will reduce the
           truncation error. However, it may make the optimization unstable. Using hard constraints such as
           the anchor loss L_op relieves such a problem."

WHY THE RESIDUAL IS AWKWARD HERE (paper Section 3.3). An operator outputs u on a GRID, so autograd in x is
not directly available: "it is not straightforward to write out the solution function in the neural operator
which directly outputs the numerical solution u = G_theta(a) on a grid, especially for FNO which uses FFT."
The paper offers numerical differentiation (finite difference O(n), Fourier O(n log n)) or a query-function
autograd. Their stated limits: "finite difference methods require a fine-resolution uniform grid; spectral
methods require smoothness and uniform grids. Especially, these numerical errors on the gradient will be
amplified on the output solution."

OUR CHOICE, and why. The Darcy benchmark is NON-PERIODIC (u = 0 on the boundary) and its coefficient field is
piecewise constant with sharp jumps. An FFT derivative assumes periodicity and smoothness, so it is the wrong
primary tool here (Gibbs ringing at the boundary and at the material interfaces); the periodic-domain fix is
Fourier continuation, FC-PINO (Maust et al., arXiv:2211.15960), which we do not implement. So the PRIMARY
residual is the finite-volume/finite-difference divergence form with harmonic-mean face conductivities, which
is the correct discretisation for a discontinuous coefficient. `spectral_laplacian` is provided as the
cross-check for smooth periodic problems and to make the trade-off visible rather than asserted.

HONESTY. The FD residual below uses the same 5-point harmonic-mean stencil as the reference solver in
`datasets/darcy.py`. That is deliberate (it is the correct discretisation of the same operator), but it means
"PINO matches the reference" is NOT a discretisation-accuracy result: driving this residual to zero IS solving
the reference's discrete system. What the physics term therefore buys, and what the case must measure, is
DATA EFFICIENCY: the error reached for a given number of labelled instances, including zero.

Offline engine only. NEVER import this in the live/Pyodide lane.
"""
from __future__ import annotations

import torch

# The a-priori output scale, used so that a genuinely DATA-FREE run (n_labels = 0) never touches a label.
# For -div(a grad u) = 1 on the unit square with u = 0 on the boundary and a in [3, 12], the 1-D analogue
# u_max = f L^2 / (8 a) gives ~1/(8*7.5) ~ 0.017. We take 0.02: a fixed property of the problem statement
# (f and the coefficient range), NOT a statistic fitted to solved instances.
U_SCALE = 0.02


def harmonic_faces(a: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Harmonic-mean face conductivities on the interior nodes, matching `datasets/darcy._solve_darcy`.

    The harmonic mean is the correct face average for a discontinuous coefficient: it is what makes the
    normal flux continuous across a material interface (an arithmetic mean over-conducts across a jump).
    `a` is [B,1,H,W] in PHYSICAL units. Returns (aE, aW, aN, aS), each [B,1,H-2,W-2].
    """
    aC = a[:, :, 1:-1, 1:-1]

    def hmean(x, y):
        return 2.0 * x * y / (x + y).clamp_min(1e-12)

    aE = hmean(aC, a[:, :, 1:-1, 2:])
    aW = hmean(aC, a[:, :, 1:-1, :-2])
    aN = hmean(aC, a[:, :, 2:, 1:-1])
    aS = hmean(aC, a[:, :, :-2, 1:-1])
    return aE, aW, aN, aS


def darcy_residual_fd(u: torch.Tensor, a: torch.Tensor, h: float, f: float = 1.0) -> torch.Tensor:
    """The strong-form Darcy residual R = -div(a grad u) - f on the INTERIOR, by finite volume.

    Discretisation (identical to the reference solver, harmonic-mean faces):

        [ (aE+aW+aN+aS) u_C - aE u_E - aW u_W - aN u_N - aS u_S ] / h^2  -  f

    `u`, `a` are [B,1,H,W] in PHYSICAL units; `h` is the grid spacing. Returns [B,1,H-2,W-2].
    """
    aE, aW, aN, aS = harmonic_faces(a)
    uC = u[:, :, 1:-1, 1:-1]
    uE = u[:, :, 1:-1, 2:]
    uW = u[:, :, 1:-1, :-2]
    uN = u[:, :, 2:, 1:-1]
    uS = u[:, :, :-2, 1:-1]
    div = ((aE + aW + aN + aS) * uC - aE * uE - aW * uW - aN * uN - aS * uS) / (h * h)
    return div - f


def spectral_laplacian(u: torch.Tensor, length: float = 1.0) -> torch.Tensor:
    """The Laplacian by FFT, for the SMOOTH + PERIODIC case only (paper 3.3, the O(n log n) route).

    Provided as the documented cross-check, not as the Darcy default: this assumes the field is periodic, and
    the Darcy problem is not (u = 0 on the boundary), so on Darcy it exhibits exactly the boundary ringing the
    paper warns about. Use it on periodic problems, or to demonstrate the trade-off honestly.
    """
    B, C, H, W = u.shape
    kx = torch.fft.fftfreq(H, d=length / H, device=u.device) * 2.0 * torch.pi
    ky = torch.fft.rfftfreq(W, d=length / W, device=u.device) * 2.0 * torch.pi
    k2 = (kx[:, None] ** 2 + ky[None, :] ** 2)[None, None]
    return torch.fft.irfft2(-(k2) * torch.fft.rfft2(u), s=(H, W))


def boundary_residual(u: torch.Tensor) -> torch.Tensor:
    """u = 0 on the boundary of the unit square: the four edges of the PHYSICAL field, flattened."""
    return torch.cat([
        u[:, :, 0, :].reshape(u.shape[0], -1),
        u[:, :, -1, :].reshape(u.shape[0], -1),
        u[:, :, :, 0].reshape(u.shape[0], -1),
        u[:, :, :, -1].reshape(u.shape[0], -1),
    ], dim=1)


def to_physical(u_net: torch.Tensor) -> torch.Tensor:
    """Network output -> physical u. A fixed affine scale (see U_SCALE): no label statistics involved, so a
    data-free run is genuinely data-free."""
    return U_SCALE * u_net


def pde_loss(u_net: torch.Tensor, a_phys: torch.Tensor, h: float, f: float = 1.0,
             bc_weight: float = 1.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The operator PDE loss of paper eq. (3)/(7): interior residual + the boundary term.

    Returns (total, interior_mse, boundary_mse). The residual is normalised by f so the term is
    dimensionless and comparable across problems.
    """
    u = to_physical(u_net)
    r = darcy_residual_fd(u, a_phys, h, f) / f
    interior = (r ** 2).mean()
    bnd = (boundary_residual(u) ** 2).mean() / (U_SCALE ** 2)
    return interior + bc_weight * bnd, interior.detach(), bnd.detach()


def relative_l2(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """Mean relative L2 over the batch (the field-standard operator metric)."""
    num = torch.norm((pred - true).reshape(len(pred), -1), dim=1)
    den = torch.norm(true.reshape(len(true), -1), dim=1).clamp_min(1e-12)
    return (num / den).mean()


def test_time_optimize(net, x: torch.Tensor, a_phys: torch.Tensor, h: float, *, steps: int = 100,
                       lr: float = 1e-4, anchor_weight: float = 1.0, f: float = 1.0):
    """Paper Section 3.2, phase 2: fine-tune the PRE-TRAINED operator on ONE queried instance.

    Minimises  L_pde + anchor_weight * L_op, where L_op = || G_theta_i(a) - G_theta_0(a) ||^2 is the anchor
    loss against the frozen pre-trained prediction. Returns (u_phys_after, history) where history records the
    interior residual and the relative change, so the case can SHOW the fine-tuning working rather than
    assert it.

    The net is deep-copied, so the pre-trained operator is left untouched.
    """
    import copy

    tuned = copy.deepcopy(net)
    tuned.train()
    with torch.no_grad():
        u0 = tuned(x).detach()                      # G_theta_0(a): the anchor
    opt = torch.optim.Adam(tuned.parameters(), lr=lr)
    history = []
    for s in range(steps):
        opt.zero_grad()
        u_net = tuned(x)
        lp, interior, bnd = pde_loss(u_net, a_phys, h, f)
        anchor = ((u_net - u0) ** 2).mean()
        (lp + anchor_weight * anchor).backward()
        torch.nn.utils.clip_grad_norm_(tuned.parameters(), 1.0)
        opt.step()
        if s % max(1, steps // 20) == 0 or s == steps - 1:
            history.append({"step": s, "interior": float(interior), "bnd": float(bnd),
                            "anchor": float(anchor.detach())})
    tuned.eval()
    with torch.no_grad():
        return to_physical(tuned(x)).detach(), history
