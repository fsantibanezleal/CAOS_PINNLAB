"""Control · degenerate zero-source case (the archetype's mandatory negative control).

    -nabla^2 u = 0  on (0,1)^2,  u = 0 on the boundary   =>   u == 0 everywhere.

The engine must run this without crashing and return a flat-zero field — a sanity check that the pipeline handles
the trivial/degenerate case (zero source, hard-zero boundary). The relative-L2 here is ||pred|| (the analytic truth
is identically zero), which should be tiny.
"""
from __future__ import annotations

import numpy as np

from .base import CaseSpec

CASE = CaseSpec(
    id="ctrl-zero-source",
    category="control",
    title="Degenerate control — zero source, zero field",
    governing_equations=r"-\nabla^2 u = 0\ \text{on}\ (0,1)^2,\ u|_{\partial\Omega}=0\ \Rightarrow\ u\equiv 0",
    method="hard-constraints",
    engine="deepxde",
    real_or_synthetic="synthetic",
    inputs=("x", "y"),
    outputs=("u",),
    domain={"x": (0.0, 1.0), "y": (0.0, 1.0)},
    grid={"x": 41, "y": 41},
    expected_band="field must stay identically 0 (engine must not crash); relative-L2 = ||pred|| ~ 0",
    validation_anchor="analytic",
    train={"layers": [2, 16, 16, 1], "activation": "tanh", "lr": 1e-3, "adam": 2000, "lbfgs": False},
    notes="Degenerate control per the archetype: zero source + hard-zero boundary -> zero field; verifies the pipeline handles the trivial case.",
)


def analytic(xy: np.ndarray) -> np.ndarray:
    return np.zeros((np.asarray(xy).shape[0], 1), dtype=np.float64)


def build(seed: int) -> dict:
    import deepxde as dde

    dde.config.set_random_seed(int(seed))
    geom = dde.geometry.Rectangle([0.0, 0.0], [1.0, 1.0])

    def pde(x, u):
        u_xx = dde.grad.hessian(u, x, i=0, j=0)
        u_yy = dde.grad.hessian(u, x, i=1, j=1)
        return -u_xx - u_yy  # zero source

    net = dde.nn.FNN(CASE.train["layers"], CASE.train["activation"], "Glorot uniform")
    net.apply_output_transform(
        lambda x, u: x[:, 0:1] * (1.0 - x[:, 0:1]) * x[:, 1:2] * (1.0 - x[:, 1:2]) * u
    )
    data = dde.data.PDE(geom, pde, [], num_domain=500, num_boundary=0, num_test=1000)
    model = dde.Model(data, net)
    model.compile("adam", lr=CASE.train["lr"])
    return {"model": model, "input_dim": 2}
