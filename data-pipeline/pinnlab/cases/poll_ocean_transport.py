"""Group C · pollution-environmental — ocean/coastal pollutant transport (forward), 2D advection-diffusion + time.

Governing equation (passive scalar c carried by a prescribed divergence-free gyre v, eddy diffusion D):
    c_t + v.grad(c) = D (c_xx + c_yy) + f   on (0,1)^2 x (0,1],
    v = (-A sin(pi x) cos(pi y),  A cos(pi x) sin(pi y))  (incompressible, A=1),  D=0.01.
MMS solution (validation anchor): c*(x,y,t) = exp(-2 D pi^2 t) sin(pi x) sin(pi y); the source f = v.grad(c*) is
whatever the current does to c*, so c* is exact. real_or_synthetic = synthetic-illustrative (a physically-faithful
illustration of plastic/oil-spill spread, NOT fit to a real spill or a real ocean-current product).

Method — advection-diffusion PINN with SOFT IC/BC (Dirichlet = c* on the open boundary, IC = c* at t=0) so the
network genuinely learns the interior transport field; the relative-L2 reports the true PINN error. Pe = A L / D ~
100 (advection-dominated but converges without curriculum). First 2D-space + time applied case.
"""
from __future__ import annotations

import numpy as np

from .base import CaseSpec

A_CUR = 1.0
D_COEF = 0.01

CASE = CaseSpec(
    id="poll-ocean-transport",
    category="pollution-environmental",
    title="Ocean pollutant transport — 2D advection-diffusion PINN",
    governing_equations=(
        r"c_t + \mathbf{v}\cdot\nabla c = D\nabla^2 c + f,\ \nabla\cdot\mathbf{v}=0,\ D=0.01,"
        r"\ c^*=e^{-2D\pi^2 t}\sin(\pi x)\sin(\pi y)"
    ),
    method="advection-diffusion",
    engine="deepxde",
    real_or_synthetic="synthetic-illustrative",
    inputs=("x", "y", "t"),
    outputs=("c",),
    domain={"x": (0.0, 1.0), "y": (0.0, 1.0), "t": (0.0, 1.0)},
    grid={"x": 41, "y": 41, "t": 21},
    expected_band="a diffusing blob advected by the gyre; relative-L2 vs MMS analytic < 1e-2",
    validation_anchor="analytic",
    train={
        "lr": 1e-3,
        "adam": 18000,
        "lbfgs": True,
        "num_domain": 8000,
        "num_boundary": 400,
        "num_initial": 800,
        "num_test": 8000,
        "loss_weights": [1, 10, 50],  # [pde, bc, ic]
    },
    notes="Divergence-free gyre; MMS source f = v.grad(c*); soft Dirichlet BC (=c*) + IC (=c* at t=0); the net learns the interior.",
)


def analytic(xyt: np.ndarray) -> np.ndarray:
    xyt = np.asarray(xyt, dtype=np.float64)
    x = xyt[:, 0:1]
    y = xyt[:, 1:2]
    t = xyt[:, 2:3]
    return np.exp(-2.0 * D_COEF * np.pi ** 2 * t) * np.sin(np.pi * x) * np.sin(np.pi * y)


def build(seed: int) -> dict:
    import deepxde as dde
    import torch

    dde.config.set_random_seed(int(seed))
    geom = dde.geometry.Rectangle([0.0, 0.0], [1.0, 1.0])
    timedomain = dde.geometry.TimeDomain(0.0, 1.0)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    def mms_source(x):
        xx, yy, t = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        decay = torch.exp(-2.0 * D_COEF * np.pi ** 2 * t)
        cstar_x = np.pi * torch.cos(np.pi * xx) * torch.sin(np.pi * yy) * decay
        cstar_y = np.pi * torch.sin(np.pi * xx) * torch.cos(np.pi * yy) * decay
        vx = -A_CUR * torch.sin(np.pi * xx) * torch.cos(np.pi * yy)
        vy = A_CUR * torch.cos(np.pi * xx) * torch.sin(np.pi * yy)
        return vx * cstar_x + vy * cstar_y

    def pde(x, c):
        c_t = dde.grad.jacobian(c, x, i=0, j=2)
        c_x = dde.grad.jacobian(c, x, i=0, j=0)
        c_y = dde.grad.jacobian(c, x, i=0, j=1)
        c_xx = dde.grad.hessian(c, x, i=0, j=0)
        c_yy = dde.grad.hessian(c, x, i=1, j=1)
        vx = -A_CUR * torch.sin(np.pi * x[:, 0:1]) * torch.cos(np.pi * x[:, 1:2])
        vy = A_CUR * torch.cos(np.pi * x[:, 0:1]) * torch.sin(np.pi * x[:, 1:2])
        return c_t + vx * c_x + vy * c_y - D_COEF * (c_xx + c_yy) - mms_source(x)

    bc = dde.icbc.DirichletBC(geomtime, lambda X: analytic(X), lambda _, on_boundary: on_boundary)
    ic = dde.icbc.IC(geomtime, lambda X: analytic(X), lambda _, on_initial: on_initial)

    t = CASE.train
    data = dde.data.TimePDE(
        geomtime, pde, [bc, ic],
        num_domain=t["num_domain"], num_boundary=t["num_boundary"],
        num_initial=t["num_initial"], solution=analytic, num_test=t["num_test"],
    )
    net = dde.nn.FNN([3] + [64] * 4 + [1], "tanh", "Glorot normal")
    model = dde.Model(data, net)
    model.compile("adam", lr=t["lr"], loss_weights=t["loss_weights"], metrics=["l2 relative error"])
    return {"model": model, "input_dim": 3}
