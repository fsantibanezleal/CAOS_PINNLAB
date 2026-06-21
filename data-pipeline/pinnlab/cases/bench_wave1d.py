"""Group A · canonical-benchmark — 1D wave equation (hyperbolic), SIREN + HARD-CONSTRAINT PINN.

Governing equation:
    u_tt = c^2 u_xx   on x in [0,1], t in [0,1], c = 1,
    IC  u(x,0) = sin(pi x),  u_t(x,0) = 0,   BC  u(0,t) = u(1,t) = 0.
Manufactured exact solution (validation anchor):
    u*(x,t) = sin(pi x) cos(c pi t)   (a single-mode standing wave; half a period over t in [0,1]).

Method — SIREN (sin activation) as the spectral-bias remedy for the oscillatory hyperbolic solution, combined with
a HARD-CONSTRAINT output transform that satisfies BC + both ICs exactly:
    u_hat(x,t) = sin(pi x) + t^2 * x * (1-x) * N(x,t)
(t=0 -> sin(pi x) = IC#1; every term of d/dt carries a factor t -> u_t(x,0)=0 = IC#2; x(1-x) & sin(pi x) vanish at
x=0,1 -> BC). No IC/BC loss term, so no loss weighting to tune — the cleanest, most stable trainer for this case.
"""
from __future__ import annotations

import numpy as np

from .base import CaseSpec

C = 1.0

CASE = CaseSpec(
    id="bench-wave1d",
    category="canonical-benchmark",
    title="1D wave equation — SIREN + hard-constraint PINN",
    governing_equations=(
        r"u_{tt} = c^2 u_{xx}\ \text{on}\ (0,1)\times(0,1],\ c=1,\quad "
        r"u(x,0)=\sin(\pi x),\ u_t(x,0)=0,\ u(0,t)=u(1,t)=0"
    ),
    method="siren-hard-constraints",
    engine="deepxde",
    real_or_synthetic="synthetic",
    inputs=("x", "t"),
    outputs=("u",),
    domain={"x": (0.0, 1.0), "t": (0.0, 1.0)},
    grid={"x": 201, "t": 201},
    expected_band="standing wave sin(pi x) cos(pi t); relative-L2 vs analytic < 5e-3",
    validation_anchor="analytic",
    train={
        "layers": [2, 64, 64, 64, 1],
        "activation": "sin",
        "lr": 1e-3,
        "adam": 15000,
        "lbfgs": True,
        "num_domain": 2540,
        "num_boundary": 0,
        "num_initial": 160,
        "num_test": 4000,
    },
    notes="BC + both ICs enforced exactly by an output transform; SIREN sin activation for the oscillation.",
)


def analytic(xy: np.ndarray) -> np.ndarray:
    """u*(x,t) = sin(pi x) cos(c pi t), on [N,2] (x,t) -> [N,1]."""
    xy = np.asarray(xy, dtype=np.float64)
    x = xy[:, 0:1]
    t = xy[:, 1:2]
    return np.sin(np.pi * x) * np.cos(C * np.pi * t)


def build(seed: int) -> dict:
    import deepxde as dde
    import torch

    dde.config.set_random_seed(int(seed))
    geom = dde.geometry.Interval(0.0, 1.0)
    timedomain = dde.geometry.TimeDomain(0.0, 1.0)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    def pde(x, u):
        u_tt = dde.grad.hessian(u, x, i=1, j=1)
        u_xx = dde.grad.hessian(u, x, i=0, j=0)
        return u_tt - C ** 2 * u_xx

    t = CASE.train
    data = dde.data.TimePDE(
        geomtime, pde, [],
        num_domain=t["num_domain"], num_boundary=t["num_boundary"],
        num_initial=t["num_initial"], solution=analytic, num_test=t["num_test"],
    )
    net = dde.nn.FNN(t["layers"], t["activation"], "Glorot normal")

    def output_transform(x, u):
        xs = x[:, 0:1]
        ts = x[:, 1:2]
        return torch.sin(np.pi * xs) + ts ** 2 * xs * (1.0 - xs) * u

    net.apply_output_transform(output_transform)
    model = dde.Model(data, net)
    model.compile("adam", lr=t["lr"], metrics=["l2 relative error"])
    return {"model": model, "input_dim": 2}
