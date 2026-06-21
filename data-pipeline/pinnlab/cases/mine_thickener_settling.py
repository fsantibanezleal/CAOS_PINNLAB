"""Group B · mining-mineral-processing — thickener / tailings sedimentation (Bürger-Concha settling, forward MMS).

1D batch settling of a flocculated suspension: the solid volume fraction phi(z,t) obeys the strongly-degenerate
convection-diffusion (Kynch hindered settling + sediment consolidation) conservation law
    phi_t + d/dz f_bk(phi) = d/dz( D(phi) phi_z )
with Richardson-Zaki batch flux f_bk = v0 phi (1-phi/phi_max)^C and a degenerate diffusion D(phi) that switches on
only above the gel concentration phi_c (regularized by a tanh switch so the residual is C^1). A sharp settling front
descends from the clear supernatant and a consolidating bed rises — the canonical "applied" moving-front case.

real_or_synthetic = synthetic-illustrative: there is NO public measured (z,t,phi) thickener field (real-datasets.md);
the validation anchor is a Method-of-Manufactured-Solutions descending-front field with the source f = L[phi*] derived
analytically, exercising the genuine nonlinear f_bk + degenerate D. Parameter ranges from the Bürger-Concha literature.
"""
from __future__ import annotations

import numpy as np

from .base import CaseSpec

PHI_MAX = 0.66
PHI_C = 0.23
C_RZ = 5.0
V0 = -1.0
DELTA = 0.05
K_SIG = 6.0
EPS_REG = 1.0e-2
# manufactured descending-front parameters (scaled domain z,t in [0,1])
PHI_LO, PHI_HI = 0.05, 0.45
Z0, R_FRONT, W_FRONT = 0.9, 0.8, 0.06

CASE = CaseSpec(
    id="mine-thickener-settling",
    category="mining-mineral-processing",
    title="Thickener / tailings settling — Bürger-Concha degenerate conservation law",
    governing_equations=(
        r"\phi_t + \partial_z f_{bk}(\phi) = \partial_z(D(\phi)\phi_z),\ "
        r"f_{bk}=v_0\phi(1-\phi/\phi_{max})^C,\ D\ \text{degenerate above}\ \phi_c"
    ),
    method="nonlinear-settling",
    engine="deepxde",
    real_or_synthetic="synthetic-illustrative",
    inputs=("z", "t"),
    outputs=("phi",),
    domain={"z": (0.0, 1.0), "t": (0.0, 1.0)},
    grid={"z": 101, "t": 51},
    expected_band="descending settling front + rising consolidated bed; relative-L2 vs MMS analytic < 2e-2",
    validation_anchor="analytic",
    train={"lr": 1e-3, "adam": 18000, "lbfgs": True, "num_domain": 4000, "num_boundary": 200, "num_initial": 200, "num_test": 8000, "loss_weights": [1, 10, 10]},
    notes="Richardson-Zaki hindered settling + tanh-regularized degenerate diffusion; soft Dirichlet/IC = phi*; MMS source. RAR is the front-sharpening upgrade.",
)


def _phi_star_np(zt: np.ndarray) -> np.ndarray:
    zt = np.asarray(zt, dtype=np.float64)
    z = zt[:, 0:1]
    t = zt[:, 1:2]
    s = Z0 - R_FRONT * t
    return PHI_LO + (PHI_HI - PHI_LO) * 0.5 * (1.0 - np.tanh((z - s) / W_FRONT))


def analytic(zt: np.ndarray) -> np.ndarray:
    return _phi_star_np(zt)


def build(seed: int) -> dict:
    import deepxde as dde
    import torch

    dde.config.set_random_seed(int(seed))
    geom = dde.geometry.Interval(0.0, 1.0)
    timedomain = dde.geometry.TimeDomain(0.0, 1.0)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    def f_bk_prime(phi):
        s = torch.clamp(1.0 - phi / PHI_MAX, min=0.0)
        return V0 * (s ** C_RZ - (C_RZ / PHI_MAX) * phi * s ** (C_RZ - 1.0))

    def _switch(phi):
        return 0.5 * (1.0 + torch.tanh((phi - PHI_C) / EPS_REG))

    def D_of(phi):
        shape = torch.clamp(phi, min=1e-4) / PHI_C
        return DELTA * shape ** (K_SIG - 1.0) * _switch(phi)

    def D_prime(phi):
        p = torch.clamp(phi, min=1e-4)
        shape = (p / PHI_C) ** (K_SIG - 1.0)
        dshape = (K_SIG - 1.0) / PHI_C * (p / PHI_C) ** (K_SIG - 2.0)
        sw = _switch(phi)
        dsw = 0.5 * (1.0 - torch.tanh((phi - PHI_C) / EPS_REG) ** 2) / EPS_REG
        return DELTA * (dshape * sw + shape * dsw)

    def mms_source(x):
        # phi* and its analytic derivatives (descending tanh front)
        z = x[:, 0:1]
        t = x[:, 1:2]
        s = Z0 - R_FRONT * t
        u = (z - s) / W_FRONT
        th = torch.tanh(u)
        sech2 = 1.0 - th ** 2
        amp = (PHI_HI - PHI_LO)
        phi = PHI_LO + amp * 0.5 * (1.0 - th)
        phi_z = -amp / (2.0 * W_FRONT) * sech2
        phi_t = -amp * R_FRONT / (2.0 * W_FRONT) * sech2  # d/dt: du/dt = R/W
        phi_zz = amp / (W_FRONT ** 2) * sech2 * th
        return phi_t + f_bk_prime(phi) * phi_z - (D_prime(phi) * phi_z ** 2 + D_of(phi) * phi_zz)

    def pde(x, y):
        phi = y
        phi_z = dde.grad.jacobian(y, x, i=0, j=0)
        phi_t = dde.grad.jacobian(y, x, i=0, j=1)
        phi_zz = dde.grad.hessian(y, x, i=0, j=0)
        conv = f_bk_prime(phi) * phi_z
        diff = D_prime(phi) * phi_z ** 2 + D_of(phi) * phi_zz
        return phi_t + conv - diff - mms_source(x)

    bc = dde.icbc.DirichletBC(geomtime, _phi_star_np, lambda _, on_boundary: on_boundary)
    ic = dde.icbc.IC(geomtime, _phi_star_np, lambda _, on_initial: on_initial)

    t = CASE.train
    data = dde.data.TimePDE(
        geomtime, pde, [bc, ic],
        num_domain=t["num_domain"], num_boundary=t["num_boundary"],
        num_initial=t["num_initial"], solution=analytic, num_test=t["num_test"],
    )
    net = dde.nn.FNN([2] + [64] * 4 + [1], "tanh", "Glorot normal")
    model = dde.Model(data, net)
    model.compile("adam", lr=t["lr"], loss_weights=t["loss_weights"], metrics=["l2 relative error"])
    return {"model": model, "input_dim": 2}
