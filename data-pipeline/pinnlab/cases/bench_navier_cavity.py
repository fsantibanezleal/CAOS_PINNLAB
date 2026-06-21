"""Group A · canonical-benchmark — 2D lid-driven cavity, steady incompressible Navier-Stokes (3 outputs u,v,p).

Governing equations (Re = 1/nu = 100, nu = 0.01, rho = 1):
    x-momentum:  u u_x + v u_y = -(1/rho) p_x + nu (u_xx + u_yy)
    y-momentum:  u v_x + v v_y = -(1/rho) p_y + nu (v_xx + v_yy)
    continuity:  u_x + v_y = 0
on (0,1)^2. Lid (y=1): u = regularized profile 16 x^2 (1-x)^2 (-> 0 at the top corners to tame the singularity),
v = 0; other walls: no-slip u=v=0; pressure pinned at (0,0)=0 (gauge fix).

The hardest canonical case: 3 coupled outputs, 3 residuals, a pressure gauge, corner regularization, and loss
weighting (BCs + gauge up-weighted 10x). Validation anchor: the Ghia, Ghia & Shin (1982) Re=100 centerline
benchmark (digitized). On CPU this is reduced-fidelity (rel-L2 a few %); a GPU lane (PhysicsNeMo) tightens it.
"""
from __future__ import annotations

import numpy as np

from .base import CaseSpec, Variant

RHO = 1.0
NU = 0.01  # Re = U L / nu = 1/0.01 = 100

# Ghia et al. 1982, Re=100 — u along the vertical centerline x=0.5
GHIA_Y = np.array([0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531,
                   0.5000, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 0.9766, 1.0000])
GHIA_U = np.array([0.00000, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150, -0.15662, -0.21090,
                   -0.20581, -0.13641, 0.00332, 0.23151, 0.68717, 0.73722, 0.78871, 0.84123, 1.00000])
# Ghia et al. 1982, Re=100 — v along the horizontal centerline y=0.5
GHIA_X = np.array([0.0000, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563, 0.2266, 0.2344,
                   0.5000, 0.8047, 0.8594, 0.9063, 0.9453, 0.9531, 0.9609, 0.9688, 1.0000])
GHIA_V = np.array([0.00000, 0.09233, 0.10091, 0.10890, 0.12317, 0.16077, 0.17507, 0.17527,
                   0.05454, -0.24533, -0.22445, -0.16914, -0.10313, -0.08864, -0.07391, -0.05906, 0.00000])

CASE = CaseSpec(
    id="bench-navier-cavity",
    category="canonical-benchmark",
    title="2D lid-driven cavity — steady Navier-Stokes PINN (u, v, p)",
    governing_equations=(
        r"(\mathbf{u}\cdot\nabla)\mathbf{u} = -\tfrac1\rho\nabla p + \nu\nabla^2\mathbf{u},\ \nabla\cdot\mathbf{u}=0,"
        r"\ \text{Re}=100\ \text{on}\ (0,1)^2"
    ),
    method="multioutput-loss-weighting",
    engine="deepxde",
    real_or_synthetic="synthetic-illustrative",  # CPU lane = reduced fidelity; label honestly vs Ghia
    inputs=("x", "y"),
    outputs=("u", "v", "p"),
    domain={"x": (0.0, 1.0), "y": (0.0, 1.0)},
    grid={"x": 101, "y": 101},
    expected_band="primary vortex + corner eddies; rel-L2 vs Ghia centerlines ~12-22% on the CPU primitive-variable lane (the v-centerline is the hardest; a GPU PhysicsNeMo lane tightens it)",
    validation_anchor="benchmark-ghia",
    train={
        "lr": 1e-3,
        "adam": 20000,
        "lbfgs": True,
        "num_domain": 2601,
        "num_boundary": 400,
        "num_test": 10000,
        "loss_weights": [1, 1, 1, 10, 10, 10, 10, 10],  # [r1, r2, r3, bc_lid_u, bc_lid_v, bc_wall_u, bc_wall_v, bc_p]
    },
    notes="Primitive (u,v,p); regularized lid; pressure gauge pinned; BC+gauge up-weighted. Ghia 1982 centerline anchor.",
)


def variants() -> list[Variant]:
    # Single honest benchmark: the cavity has no closed form for any Re; only Re=100 is reliably reproducible on the
    # CPU primitive-variable lane (validated vs the Ghia 1982 centerlines). A Re sweep would fabricate un-anchorable
    # regimes (ADR-0016 §9.A), so this case ships one rigorous variant.
    return [Variant(
        "re100", "Re = 100 (Ghia benchmark)", "Re = 100 (benchmark de Ghia)", {},
        "Primary vortex + corner eddies; u,v on the centerlines validated vs Ghia 1982.",
        "Vórtice primario + remolinos de esquina; u,v en las líneas centrales validados vs Ghia 1982.",
    )]


def build(seed: int) -> dict:
    import deepxde as dde

    dde.config.set_random_seed(int(seed))
    geom = dde.geometry.Rectangle([0.0, 0.0], [1.0, 1.0])

    def pde(x, y):
        u = y[:, 0:1]
        v = y[:, 1:2]
        u_x = dde.grad.jacobian(y, x, i=0, j=0)
        u_y = dde.grad.jacobian(y, x, i=0, j=1)
        v_x = dde.grad.jacobian(y, x, i=1, j=0)
        v_y = dde.grad.jacobian(y, x, i=1, j=1)
        p_x = dde.grad.jacobian(y, x, i=2, j=0)
        p_y = dde.grad.jacobian(y, x, i=2, j=1)
        u_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
        u_yy = dde.grad.hessian(y, x, component=0, i=1, j=1)
        v_xx = dde.grad.hessian(y, x, component=1, i=0, j=0)
        v_yy = dde.grad.hessian(y, x, component=1, i=1, j=1)
        momentum_x = u * u_x + v * u_y + p_x / RHO - NU * (u_xx + u_yy)
        momentum_y = u * v_x + v * v_y + p_y / RHO - NU * (v_xx + v_yy)
        continuity = u_x + v_y
        return [momentum_x, momentum_y, continuity]

    def lid(x):
        return 16.0 * (x[:, 0:1] ** 2) * (1.0 - x[:, 0:1]) ** 2

    def boundary_lid(x, on_boundary):
        return on_boundary and np.isclose(x[1], 1.0)

    def boundary_wall(x, on_boundary):
        return on_boundary and not np.isclose(x[1], 1.0)

    bc_lid_u = dde.icbc.DirichletBC(geom, lid, boundary_lid, component=0)
    bc_lid_v = dde.icbc.DirichletBC(geom, lambda x: 0.0, boundary_lid, component=1)
    bc_wall_u = dde.icbc.DirichletBC(geom, lambda x: 0.0, boundary_wall, component=0)
    bc_wall_v = dde.icbc.DirichletBC(geom, lambda x: 0.0, boundary_wall, component=1)
    bc_p = dde.icbc.PointSetBC(np.array([[0.0, 0.0]]), np.array([[0.0]]), component=2)
    bcs = [bc_lid_u, bc_lid_v, bc_wall_u, bc_wall_v, bc_p]

    t = CASE.train
    data = dde.data.PDE(
        geom, pde, bcs,
        num_domain=t["num_domain"], num_boundary=t["num_boundary"], num_test=t["num_test"],
    )
    net = dde.nn.FNN([2] + [64] * 5 + [3], "tanh", "Glorot uniform")
    model = dde.Model(data, net)
    model.compile("adam", lr=t["lr"], loss_weights=t["loss_weights"])
    return {"model": model, "input_dim": 2}


def extra_metrics(sf) -> dict:
    """Relative-L2 of the PINN velocity vs the Ghia 1982 Re=100 centerlines (the standard cavity benchmark)."""
    x = np.asarray(sf.axes["x"], dtype=np.float64)
    y = np.asarray(sf.axes["y"], dtype=np.float64)
    u = sf.fields["u"]  # (nx, ny), indexed [ix, iy]
    v = sf.fields["v"]
    ix = int(np.argmin(np.abs(x - 0.5)))
    iy = int(np.argmin(np.abs(y - 0.5)))
    u_center = u[ix, :]                       # u(0.5, y)
    v_center = v[:, iy]                       # v(x, 0.5)
    u_at = np.interp(GHIA_Y, y, u_center)
    v_at = np.interp(GHIA_X, x, v_center)
    rel_u = float(np.linalg.norm(u_at - GHIA_U) / np.linalg.norm(GHIA_U))
    rel_v = float(np.linalg.norm(v_at - GHIA_V) / np.linalg.norm(GHIA_V))
    return {
        "ghia_u_rel_l2": round(rel_u, 4),
        "ghia_v_rel_l2": round(rel_v, 4),
        "l2_relative": round(0.5 * (rel_u + rel_v), 4),
    }
