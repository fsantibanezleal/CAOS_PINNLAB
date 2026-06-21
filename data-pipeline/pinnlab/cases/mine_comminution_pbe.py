"""Group B · mining-mineral-processing — comminution population balance (size-transport reduced model).

Grinding (SAG/ball milling) evolves the particle-size distribution n(s,t): breakage continuously shifts mass toward
smaller sizes. The full population-balance equation (PBE) is an integro-differential equation with selection +
breakage kernels; here we ship the REDUCED size-transport surrogate — a drift-diffusion in size space that captures
the net downward shift + spread without the breakage integral:
    n_t + G n_s = D n_ss + f   on s in [0,1] (normalized size), t in [0,1],
with constant grind drift G (toward smaller s) and size dispersion D. MMS anchor (n>0): n* = 1 + 0.5 e^{-t} sin(pi s);
the source f = L[n*] is closed-form. real_or_synthetic = synthetic-illustrative: a clean reduced model, NOT a fitted
mill PSD (no open SAG/ball-mill PSD dataset — real-datasets.md); the full breakage-kernel PBE is documented as the
complete model.
"""
from __future__ import annotations

import numpy as np

from .base import CaseSpec

G_DRIFT = 0.6
D_SIZE = 0.03

CASE = CaseSpec(
    id="mine-comminution-pbe",
    category="mining-mineral-processing",
    title="Comminution population balance — size-transport reduced PINN",
    governing_equations=(
        r"n_t + G\,n_s = D\,n_{ss} + f\ \text{on}\ s\in[0,1],\ t\in[0,1];\ "
        r"n^*=1+\tfrac12 e^{-t}\sin(\pi s)\ (\text{size-transport reduction of the PBE})"
    ),
    method="population-balance",
    engine="deepxde",
    real_or_synthetic="synthetic-illustrative",
    inputs=("s", "t"),
    outputs=("n",),
    domain={"s": (0.0, 1.0), "t": (0.0, 1.0)},
    grid={"s": 81, "t": 81},
    expected_band="size distribution drifting/spreading toward smaller sizes; relative-L2 vs MMS analytic < 1e-2",
    validation_anchor="analytic",
    train={"lr": 1e-3, "adam": 15000, "lbfgs": True, "num_domain": 3000, "num_boundary": 200, "num_initial": 200, "num_test": 6000, "loss_weights": [1, 10, 10]},
    notes="Reduced drift-diffusion in size space (proxy for the comminution PBE); soft Dirichlet/IC = n*; MMS source. Full breakage-kernel PBE is the complete model (docs).",
)


def _n_star_np(st: np.ndarray) -> np.ndarray:
    st = np.asarray(st, dtype=np.float64)
    s = st[:, 0:1]
    t = st[:, 1:2]
    return 1.0 + 0.5 * np.exp(-t) * np.sin(np.pi * s)


def analytic(st: np.ndarray) -> np.ndarray:
    return _n_star_np(st)


def build(seed: int) -> dict:
    import deepxde as dde
    import torch

    dde.config.set_random_seed(int(seed))
    geom = dde.geometry.Interval(0.0, 1.0)
    timedomain = dde.geometry.TimeDomain(0.0, 1.0)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    def mms_source(x):
        s = x[:, 0:1]
        t = x[:, 1:2]
        e = 0.5 * torch.exp(-t)
        n_t = -e * torch.sin(np.pi * s)
        n_s = e * np.pi * torch.cos(np.pi * s)
        n_ss = -e * np.pi ** 2 * torch.sin(np.pi * s)
        return n_t + G_DRIFT * n_s - D_SIZE * n_ss

    def pde(x, n):
        n_t = dde.grad.jacobian(n, x, i=0, j=1)
        n_s = dde.grad.jacobian(n, x, i=0, j=0)
        n_ss = dde.grad.hessian(n, x, i=0, j=0)
        return n_t + G_DRIFT * n_s - D_SIZE * n_ss - mms_source(x)

    bc = dde.icbc.DirichletBC(geomtime, _n_star_np, lambda _, on_boundary: on_boundary)
    ic = dde.icbc.IC(geomtime, _n_star_np, lambda _, on_initial: on_initial)

    t = CASE.train
    data = dde.data.TimePDE(
        geomtime, pde, [bc, ic],
        num_domain=t["num_domain"], num_boundary=t["num_boundary"],
        num_initial=t["num_initial"], solution=analytic, num_test=t["num_test"],
    )
    net = dde.nn.FNN([2] + [48] * 4 + [1], "tanh", "Glorot normal")
    model = dde.Model(data, net)
    model.compile("adam", lr=t["lr"], loss_weights=t["loss_weights"], metrics=["l2 relative error"])
    return {"model": model, "input_dim": 2}
