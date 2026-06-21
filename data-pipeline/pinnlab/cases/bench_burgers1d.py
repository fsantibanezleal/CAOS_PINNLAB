"""Group A · canonical-benchmark — 1D viscous Burgers, shock-front PINN with RESIDUAL-BASED ADAPTIVE REFINEMENT (RAR).

Governing equation:
    u_t + u u_x = nu u_xx,   nu = 0.01/pi,   x in [-1,1], t in [0,1],
    IC  u(x,0) = -sin(pi x),   BC  u(-1,t) = u(1,t) = 0.
The small viscosity makes a steep internal layer (quasi-shock) form near x=0 as t->1.

Method — RAR / RAR-G (residual-based adaptive refinement, Wu et al. CMAME 2023): start from uniform collocation
(soft BC/IC), then greedily ADD collocation where the residual is largest, so the sharp front gets resolved.
Validation anchor: the standard spectral reference `Burgers.npz` (Raissi 2019 / DeepXDE, MIT) — a NUMERICAL
reference, not real-world data (see wip/pinn-lab/real-datasets.md).
"""
from __future__ import annotations

import pathlib

import numpy as np

from .base import CaseSpec

_DATASET = pathlib.Path(__file__).resolve().parents[3] / "data" / "reference" / "burgers" / "Burgers.npz"

CASE = CaseSpec(
    id="bench-burgers1d",
    category="canonical-benchmark",
    title="1D viscous Burgers — RAR adaptive-sampling PINN (shock front)",
    governing_equations=(
        r"u_t + u u_x = \nu u_{xx},\ \nu=0.01/\pi,\ x\in[-1,1],\ t\in[0,1],\quad "
        r"u(x,0)=-\sin(\pi x),\ u(\pm 1,t)=0"
    ),
    method="rar-adaptive-sampling",
    engine="deepxde",
    real_or_synthetic="synthetic",  # validation anchor is a spectral numerical reference, NOT real-world data
    inputs=("x", "t"),
    outputs=("u",),
    domain={"x": (-1.0, 1.0), "t": (0.0, 1.0)},
    grid={"x": 256, "t": 100},  # nominal; eval_grid() below uses the exact Burgers.npz grid
    expected_band="smooth then a steep internal layer near x=0 as t->1; relative-L2 vs Burgers.npz < 5e-3",
    validation_anchor="dataset",
    train={
        "layers": [2, 20, 20, 20, 1],
        "lr": 1e-3,
        "adam": 10000,
        "lbfgs": True,
        "num_domain": 2500,
        "num_boundary": 100,
        "num_initial": 160,
        # RAR refinement (the method):
        "rar_rounds": 8,
        "rar_addk": 200,
        "rar_adam": 5000,
        "rar_pool": 50000,
    },
    notes="Soft BC/IC; RAR greedily adds top-k highest-residual collocation points to resolve the shock front.",
)


def _load():
    return np.load(_DATASET)


def _make_pde():
    import deepxde as dde

    nu = 0.01 / np.pi

    def pde(x, y):
        dy_x = dde.grad.jacobian(y, x, i=0, j=0)
        dy_t = dde.grad.jacobian(y, x, i=0, j=1)
        dy_xx = dde.grad.hessian(y, x, i=0, j=0)
        return dy_t + y * dy_x - nu * dy_xx

    return pde


def eval_grid():
    """Evaluate on the EXACT Burgers.npz grid (x:256 in [-1,1], t:100 in [0,1]) so the L2 aligns with the reference."""
    d = _load()
    x = d["x"].ravel().astype(np.float64)
    t = d["t"].ravel().astype(np.float64)
    xx, tt = np.meshgrid(x, t, indexing="ij")
    XY = np.stack([xx.ravel(), tt.ravel()], axis=1)
    return {"x": x, "t": t}, XY, (len(x), len(t))


def reference_on_grid() -> np.ndarray:
    """The spectral reference field usol, shape (256, 100) = (x, t) — aligned with eval_grid()."""
    return _load()["usol"].astype(np.float64)


def build(seed: int) -> dict:
    import deepxde as dde

    dde.config.set_random_seed(int(seed))
    geom = dde.geometry.Interval(-1.0, 1.0)
    timedomain = dde.geometry.TimeDomain(0.0, 0.99)  # upstream choice; reference grid still spans [0,1]
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)
    pde = _make_pde()

    bc = dde.icbc.DirichletBC(geomtime, lambda x: 0.0, lambda _, on_boundary: on_boundary)
    ic = dde.icbc.IC(geomtime, lambda x: -np.sin(np.pi * x[:, 0:1]), lambda _, on_initial: on_initial)

    t = CASE.train
    data = dde.data.TimePDE(
        geomtime, pde, [bc, ic],
        num_domain=t["num_domain"], num_boundary=t["num_boundary"], num_initial=t["num_initial"],
    )
    net = dde.nn.FNN(t["layers"], "tanh", "Glorot normal")
    model = dde.Model(data, net)
    model.compile("adam", lr=t["lr"])
    return {"model": model, "input_dim": 2}


def refine(model, case, seed: int) -> None:
    """RAR-G: greedily add the top-k highest-residual points and re-train, several rounds."""
    pde = _make_pde()
    geomtime = model.data.geom
    t = case.train
    for _ in range(int(t.get("rar_rounds", 8))):
        X = geomtime.random_points(int(t.get("rar_pool", 50000)))
        err = np.abs(np.asarray(model.predict(X, operator=pde)))[:, 0]
        idx = np.argsort(err)[-int(t.get("rar_addk", 200)):]
        model.data.add_anchors(X[idx])
        model.compile("adam", lr=t["lr"])
        model.train(iterations=int(t.get("rar_adam", 5000)), disregard_previous_best=True)
        model.compile("L-BFGS")
        model.train()
