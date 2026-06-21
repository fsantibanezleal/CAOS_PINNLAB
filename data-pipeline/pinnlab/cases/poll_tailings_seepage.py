"""Group C · pollution-environmental (mining ∩ pollution) — tailings-dam unsaturated seepage (Richards equation).

1D vertical unsaturated seepage through a tailings deposit (z up, psi = pressure head < 0 unsaturated, t = time):
    C(psi) psi_t = d/dz[ K(psi) (psi_z + 1) ]  =>  C psi_t - K psi_zz - K'(psi)(psi_z^2 + psi_z) = f.
Constitutive closure — GARDNER exponential model (a standard Richards retention/conductivity model that keeps the
MMS source analytic; van Genuchten-Mualem is the fuller closure, documented in docs):
    K(psi) = Ks exp(alpha psi),   K'(psi) = alpha K,   theta(psi)=theta_r+(theta_s-theta_r)exp(alpha psi),
    C(psi) = dtheta/dpsi = (theta_s-theta_r) alpha exp(alpha psi).
MMS anchor: a smooth, strictly-unsaturated (psi<0) head profile drying in time, source f = L[psi*] derived in closed
form. real_or_synthetic = synthetic-illustrative: NO open unsaturated-zone psi(z,t) tailings dataset exists
(real-datasets.md); only the saturated-zone Darcy head inverse could use real USGS data (documented as the Tier-C
extension). Honest: the unsaturated lane is modeled, not measured.
"""
from __future__ import annotations

import numpy as np

from .base import CaseSpec

THETA_S, THETA_R = 0.43, 0.078
ALPHA = 2.0
KS = 0.25
A_PSI, PSI_TOP, TAU = 1.0, -0.1, 1.0  # head amplitude, top head, drying time-scale (scaled domain)

CASE = CaseSpec(
    id="poll-tailings-seepage",
    category="pollution-environmental",
    title="Tailings-dam unsaturated seepage — Richards equation (Gardner) PINN",
    governing_equations=(
        r"C(\psi)\psi_t = \partial_z[K(\psi)(\psi_z+1)],\ K=K_s e^{\alpha\psi},\ "
        r"\psi^*=-A+(A-\psi_{top})e^{-t/\tau}(1-z)"
    ),
    method="richards-seepage",
    engine="deepxde",
    real_or_synthetic="synthetic-illustrative",
    inputs=("z", "t"),
    outputs=("psi",),
    domain={"z": (0.0, 1.0), "t": (0.0, 1.0)},
    grid={"z": 101, "t": 51},
    expected_band="drying unsaturated head profile (psi<0); relative-L2 vs MMS analytic < 1e-2",
    validation_anchor="analytic",
    train={"lr": 1e-3, "adam": 18000, "lbfgs": True, "num_domain": 4000, "num_boundary": 200, "num_initial": 200, "num_test": 8000, "loss_weights": [1, 10, 10]},
    notes="Gardner exponential K(psi); MMS source closed-form; soft Dirichlet/IC = psi*. vG-Mualem + real USGS saturated Darcy inverse documented as extensions.",
)


def _psi_star_np(zt: np.ndarray) -> np.ndarray:
    zt = np.asarray(zt, dtype=np.float64)
    z = zt[:, 0:1]
    t = zt[:, 1:2]
    return -A_PSI + (A_PSI - PSI_TOP) * np.exp(-t / TAU) * (1.0 - z)


def analytic(zt: np.ndarray) -> np.ndarray:
    return _psi_star_np(zt)


def build(seed: int) -> dict:
    import deepxde as dde
    import torch

    dde.config.set_random_seed(int(seed))
    geom = dde.geometry.Interval(0.0, 1.0)
    timedomain = dde.geometry.TimeDomain(0.0, 1.0)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    def mms_source(x):
        z = x[:, 0:1]
        t = x[:, 1:2]
        decay = torch.exp(-t / TAU)
        psi = -A_PSI + (A_PSI - PSI_TOP) * decay * (1.0 - z)
        psi_t = -(A_PSI - PSI_TOP) / TAU * decay * (1.0 - z)
        psi_z = -(A_PSI - PSI_TOP) * decay
        # psi_zz = 0
        ek = torch.exp(ALPHA * psi)
        return ek * ((THETA_S - THETA_R) * ALPHA * psi_t - ALPHA * KS * (psi_z ** 2 + psi_z))

    def pde(x, y):
        psi = y
        psi_t = dde.grad.jacobian(y, x, i=0, j=1)
        psi_z = dde.grad.jacobian(y, x, i=0, j=0)
        psi_zz = dde.grad.hessian(y, x, i=0, j=0)
        ek = torch.exp(ALPHA * psi)
        cap = (THETA_S - THETA_R) * ALPHA * ek
        k = KS * ek
        kp = ALPHA * k
        return cap * psi_t - k * psi_zz - kp * (psi_z ** 2 + psi_z) - mms_source(x)

    bc = dde.icbc.DirichletBC(geomtime, _psi_star_np, lambda _, on_boundary: on_boundary)
    ic = dde.icbc.IC(geomtime, _psi_star_np, lambda _, on_initial: on_initial)

    t = CASE.train
    data = dde.data.TimePDE(
        geomtime, pde, [bc, ic],
        num_domain=t["num_domain"], num_boundary=t["num_boundary"],
        num_initial=t["num_initial"], solution=analytic, num_test=t["num_test"],
    )
    net = dde.nn.FNN([2] + [40] * 4 + [1], "tanh", "Glorot normal")
    model = dde.Model(data, net)
    model.compile("adam", lr=t["lr"], loss_weights=t["loss_weights"], metrics=["l2 relative error"])
    return {"model": model, "input_dim": 2}
