"""Group A · canonical-benchmark — 2D Poisson with homogeneous Dirichlet BC, solved by a HARD-CONSTRAINT PINN.

Governing equation:
    -Δu = f  on Ω = (0,1)²,   u = 0 on ∂Ω,   with f(x,y) = 2π² sin(πx) sin(πy).
Manufactured exact solution (the validation anchor):
    u*(x,y) = sin(πx) sin(πy).

Method exercised — HARD CONSTRAINTS (distance-function output transform, dossier §4 #12): the boundary condition is
satisfied *exactly* for any network weights via  û = x(1-x)·y(1-y)·N(x,y),  so there is NO boundary-loss term to
weight. This is the cleanest first case to harden the train → ONNX → web contract on, because the closed-form
solution gives an unambiguous relative-L2 check.
"""
from __future__ import annotations

import numpy as np

from .base import CaseSpec

CASE = CaseSpec(
    id="bench-poisson2d",
    category="canonical-benchmark",
    title="2D Poisson (Dirichlet) — hard-constraint PINN",
    governing_equations=(
        r"-\nabla^2 u = 2\pi^2 \sin(\pi x)\sin(\pi y)\ \text{on}\ (0,1)^2,\quad u|_{\partial\Omega}=0"
    ),
    method="hard-constraints",
    engine="deepxde",
    real_or_synthetic="synthetic",
    inputs=("x", "y"),
    outputs=("u",),
    domain={"x": (0.0, 1.0), "y": (0.0, 1.0)},
    grid={"x": 101, "y": 101},
    expected_band="smooth single bump, max u ≈ 1 at the centre; relative-L2 vs analytic < 1e-2",
    validation_anchor="analytic",
    train={
        "layers": [2, 64, 64, 64, 1],
        "activation": "tanh",
        "lr": 1e-3,
        "adam": 12000,
        "lbfgs": True,
        "num_domain": 2000,
        "num_boundary": 0,
        "num_test": 4000,
    },
    notes="Boundary enforced exactly by an output transform; no BC loss term.",
)


def analytic(xy: np.ndarray) -> np.ndarray:
    """u*(x,y) = sin(pi x) sin(pi y), on [N,2] -> [N,1]."""
    xy = np.asarray(xy, dtype=np.float64)
    return np.sin(np.pi * xy[:, 0:1]) * np.sin(np.pi * xy[:, 1:2])


def build(seed: int) -> dict:
    """Construct + Adam-compile the DeepXDE model. Heavy: lazy-imports deepxde/torch."""
    import deepxde as dde
    import torch

    dde.config.set_random_seed(int(seed))
    geom = dde.geometry.Rectangle([0.0, 0.0], [1.0, 1.0])

    def pde(x, u):
        u_xx = dde.grad.hessian(u, x, i=0, j=0)
        u_yy = dde.grad.hessian(u, x, i=1, j=1)
        f = 2.0 * (np.pi ** 2) * torch.sin(np.pi * x[:, 0:1]) * torch.sin(np.pi * x[:, 1:2])
        return -u_xx - u_yy - f

    t = CASE.train
    data = dde.data.PDE(
        geom, pde, [],
        num_domain=t["num_domain"], num_boundary=t["num_boundary"],
        solution=analytic, num_test=t["num_test"],
    )
    net = dde.nn.FNN(t["layers"], t["activation"], "Glorot uniform")
    # hard Dirichlet u=0 on the unit-square boundary, satisfied exactly for any weights:
    net.apply_output_transform(
        lambda x, u: (x[:, 0:1] * (1.0 - x[:, 0:1]) * x[:, 1:2] * (1.0 - x[:, 1:2])) * u
    )
    model = dde.Model(data, net)
    model.compile("adam", lr=t["lr"], metrics=["l2 relative error"])
    return {"model": model, "input_dim": 2}
