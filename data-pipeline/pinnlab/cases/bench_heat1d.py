"""Group A · canonical-benchmark — 1D transient heat/diffusion, time-dependent HARD-CONSTRAINT PINN.

Governing equation:
    u_t = alpha * u_xx   on x in [0,1], t in [0,1], alpha = 1,
    IC  u(x,0) = sin(pi x),   Dirichlet  u(0,t) = u(1,t) = 0.
Manufactured exact solution (validation anchor):
    u*(x,t) = exp(-alpha * pi^2 * t) * sin(pi x).

Method — TIME-DEPENDENT PINN with HARD CONSTRAINTS: IC and BC are imposed *exactly* by the output transform
    u_hat(x,t) = t*x*(1-x)*N(x,t) + sin(pi x)
(at t=0 -> sin(pi x) = IC; at x=0,1 -> 0 = BC), so there is no IC/BC loss term. This is the first time-dependent
case; the causal-training concept is introduced on its docs page but is not load-bearing here (monotonic decay) —
it becomes essential on Allen-Cahn / Navier-Stokes.
"""
from __future__ import annotations

import numpy as np

from .base import CaseSpec

ALPHA = 1.0

CASE = CaseSpec(
    id="bench-heat1d",
    category="canonical-benchmark",
    title="1D transient heat/diffusion — time-dependent hard-constraint PINN",
    governing_equations=(
        r"\partial_t u = \alpha\,\partial_{xx} u\ \text{on}\ (0,1)\times(0,1],\ \alpha=1,\quad "
        r"u(x,0)=\sin(\pi x),\ u(0,t)=u(1,t)=0"
    ),
    method="time-dependent-hard-constraints",
    engine="deepxde",
    real_or_synthetic="synthetic",
    inputs=("x", "t"),
    outputs=("u",),
    domain={"x": (0.0, 1.0), "t": (0.0, 1.0)},
    grid={"x": 201, "t": 101},
    expected_band="exponential decay of a sine profile; relative-L2 vs analytic < 5e-3",
    validation_anchor="analytic",
    train={
        "layers": [2, 32, 32, 32, 1],
        "activation": "tanh",
        "lr": 1e-3,
        "adam": 15000,
        "lbfgs": True,
        "num_domain": 2540,
        "num_boundary": 80,
        "num_initial": 160,
        "num_test": 2540,
    },
    notes="IC+BC enforced exactly by an output transform; no IC/BC loss term.",
)


def analytic(xy: np.ndarray) -> np.ndarray:
    """u*(x,t) = exp(-alpha pi^2 t) sin(pi x), on [N,2] (x,t) -> [N,1]."""
    xy = np.asarray(xy, dtype=np.float64)
    x = xy[:, 0:1]
    t = xy[:, 1:2]
    return np.exp(-ALPHA * np.pi ** 2 * t) * np.sin(np.pi * x)


def build(seed: int) -> dict:
    import deepxde as dde
    import torch

    dde.config.set_random_seed(int(seed))
    geom = dde.geometry.Interval(0.0, 1.0)
    timedomain = dde.geometry.TimeDomain(0.0, 1.0)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    def pde(x, u):
        du_t = dde.grad.jacobian(u, x, i=0, j=1)
        du_xx = dde.grad.hessian(u, x, i=0, j=0)
        return du_t - ALPHA * du_xx

    t = CASE.train
    data = dde.data.TimePDE(
        geomtime, pde, [],
        num_domain=t["num_domain"], num_boundary=t["num_boundary"],
        num_initial=t["num_initial"], solution=analytic, num_test=t["num_test"],
    )
    net = dde.nn.FNN(t["layers"], t["activation"], "Glorot normal")

    def output_transform(x, u):
        xx = x[:, 0:1]
        tt = x[:, 1:2]
        return tt * xx * (1.0 - xx) * u + torch.sin(np.pi * xx)

    net.apply_output_transform(output_transform)
    model = dde.Model(data, net)
    model.compile("adam", lr=t["lr"], metrics=["l2 relative error"])
    return {"model": model, "input_dim": 2}
