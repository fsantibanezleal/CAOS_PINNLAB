"""Group A · canonical-benchmark — Allen-Cahn (stiff reaction-diffusion), HARD-CONSTRAINT + RAR PINN.

Governing equation:
    u_t = d u_xx + 5 (u - u^3),   d = 0.001,   x in [-1,1], t in [0,1],
    IC  u(x,0) = x^2 cos(pi x),   endpoint-matched (periodic-ish) BC.

The tiny diffusion vs the bistable reaction makes sharp, slowly-moving transition layers between u=+/-1 — a plain
soft PINN FAILS (collapses to a metastable state). Method — HARD CONSTRAINTS (IC baked into the output transform,
no IC/BC loss term) + RAR adaptive sampling to chase the moving interface. SOTA ceiling (PirateNets ~2e-5, jaxpi) is
cited, not claimed. Validation anchor: the spectral reference `Allen_Cahn.npz` (DeepXDE/Raissi, MIT) — numerical,
not real-world data.
"""
from __future__ import annotations

import pathlib

import numpy as np

from .base import CaseSpec

_DATASET = pathlib.Path(__file__).resolve().parents[3] / "data" / "reference" / "allencahn" / "Allen_Cahn.npz"
D_COEF = 0.001

CASE = CaseSpec(
    id="bench-allencahn",
    system_type="time-evol-1d",
    category="canonical-benchmark",
    title="Allen-Cahn (stiff reaction-diffusion) — hard-constraint + RAR PINN",
    governing_equations=(
        r"u_t = d\,u_{xx} + 5(u-u^3),\ d=0.001,\ x\in[-1,1],\ t\in[0,1],\quad u(x,0)=x^2\cos(\pi x)"
    ),
    method="hard-constraints-rar",
    engine="deepxde",
    real_or_synthetic="synthetic",
    inputs=("x", "t"),
    outputs=("u",),
    domain={"x": (-1.0, 1.0), "t": (0.0, 1.0)},
    grid={"x": 201, "t": 101},
    expected_band="metastable +/-1 plateaus with sharp transition layers; relative-L2 vs spectral ref < 1e-2",
    validation_anchor="dataset",
    train={
        "layers": [2, 64, 64, 64, 1],
        "lr": 1e-3,
        "adam": 20000,
        "lbfgs": True,
        "num_domain": 8000,
        "num_boundary": 400,
        "num_initial": 800,
        "rar_rounds": 4,
        "rar_addk": 600,
        "rar_adam": 5000,
        "rar_pool": 100000,
    },
    notes="Plain soft PINN fails; hard-constraint IC ansatz + RAR is the working DeepXDE recipe. PirateNets is the SOTA ceiling, not claimed.",
)


def _load():
    return np.load(_DATASET)


def _make_pde():
    import deepxde as dde

    def pde(x, y):
        dy_t = dde.grad.jacobian(y, x, i=0, j=1)
        dy_xx = dde.grad.hessian(y, x, i=0, j=0)
        return dy_t - D_COEF * dy_xx - 5.0 * (y - y ** 3)

    return pde


def eval_grid():
    d = _load()
    x = d["x"].ravel().astype(np.float64)  # 201, in [-1,1]
    t = d["t"].ravel().astype(np.float64)  # 101, in [0,1]
    xx, tt = np.meshgrid(x, t, indexing="ij")
    XY = np.stack([xx.ravel(), tt.ravel()], axis=1)
    return {"x": x, "t": t}, XY, (len(x), len(t))


def reference_on_grid() -> np.ndarray:
    # dataset u is (t,x)=(101,201); transpose to (x,t)=(201,101) to align with eval_grid()
    return _load()["u"].T.astype(np.float64)


def build(seed: int) -> dict:
    import deepxde as dde
    import torch

    dde.config.set_random_seed(int(seed))
    geom = dde.geometry.Interval(-1.0, 1.0)
    timedomain = dde.geometry.TimeDomain(0.0, 1.0)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)
    pde = _make_pde()

    t = CASE.train
    data = dde.data.TimePDE(
        geomtime, pde, [],
        num_domain=t["num_domain"], num_boundary=t["num_boundary"], num_initial=t["num_initial"],
    )
    net = dde.nn.FNN(t["layers"], "tanh", "Glorot normal")

    def output_transform(x, y):
        xs = x[:, 0:1]
        ts = x[:, 1:2]
        # u_hat = x^2 cos(pi x) [= IC at t=0]  +  t (1 - x^2) N  [vanishes at t=0 and endpoint-matches at x=+/-1]
        return xs ** 2 * torch.cos(np.pi * xs) + ts * (1.0 - xs ** 2) * y

    net.apply_output_transform(output_transform)
    model = dde.Model(data, net)
    model.compile("adam", lr=t["lr"])
    return {"model": model, "input_dim": 2}


def refine(model, case, seed: int) -> None:
    """RAR-G: greedily add top-k highest-residual points (chases the moving interface), several rounds."""
    pde = _make_pde()
    geomtime = model.data.geom
    t = case.train
    for _ in range(int(t.get("rar_rounds", 4))):
        X = geomtime.random_points(int(t.get("rar_pool", 100000)))
        err = np.abs(np.asarray(model.predict(X, operator=pde)))[:, 0]
        idx = np.argsort(err)[-int(t.get("rar_addk", 600)):]
        model.data.add_anchors(X[idx])
        model.compile("adam", lr=t["lr"])
        model.train(iterations=int(t.get("rar_adam", 5000)), disregard_previous_best=True)
    # one final L-BFGS polish after all anchors are added (was per-round — far cheaper, same accuracy on this case)
    model.compile("L-BFGS")
    model.train()
