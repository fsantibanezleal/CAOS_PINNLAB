"""Group B · mining-mineral-processing — thickener / tailings sedimentation (Bürger-Concha settling, forward MMS),
PARAMETRIC in the front descent rate R.

1D batch settling of a flocculated suspension: the solid volume fraction phi(z,t) obeys the strongly-degenerate
convection-diffusion (Kynch hindered settling + sediment consolidation) conservation law
    phi_t + d/dz f_bk(phi) = d/dz( D(phi) phi_z )
with Richardson-Zaki batch flux f_bk = v0 phi (1-phi/phi_max)^C and a degenerate diffusion D(phi) that switches on
only above the gel concentration phi_c (regularized by a tanh switch so the residual is C^1). A sharp settling front
descends from the clear supernatant and a consolidating bed rises — the canonical "applied" moving-front case.

The validation anchor is a Method-of-Manufactured-Solutions descending tanh front whose source f = L[phi*] is derived
analytically through the GENUINE nonlinear f_bk + degenerate D, so phi* solves the modified PDE EXACTLY for every R:
    phi*(z,t;R) = PHI_LO + (PHI_HI - PHI_LO) * 0.5 * (1 - tanh((z - s)/W)),   s = Z0 - R t.
The physical knob is the front DESCENT RATE R (settling speed): a faster-settling suspension drops its mud-line
faster. Because the MMS source is recomputed analytically as a function of R, the anchor stays exact for every R, so
R is a NETWORK INPUT (the burgers/tailings parametric pattern): one trained net covers the whole settling-rate family
and the web `Live` tab sweeps R continuously via the shared ONNX.

real_or_synthetic = synthetic-illustrative: there is NO public measured (z,t,phi) thickener field (real-datasets.md);
the field is an exact MMS illustration through real Bürger-Concha physics, not calibrated to a plant.
"""
from __future__ import annotations

import numpy as np

from .base import CaseSpec, ParamSpec, Variant

PHI_MAX = 0.66
PHI_C = 0.23
C_RZ = 5.0
V0 = -1.0
DELTA = 0.05
K_SIG = 6.0
EPS_REG = 1.0e-2
# manufactured descending-front parameters (scaled domain z,t in [0,1])
PHI_LO, PHI_HI = 0.05, 0.45
Z0, W_FRONT = 0.9, 0.10
R_MIN, R_MAX = 0.30, 0.90          # front descent-rate (settling-speed) range (the swept network input)
R_DEF = 0.80                       # the legacy single-case value, now the slider default

CASE = CaseSpec(
    id="mine-thickener-settling",
    category="mining-mineral-processing",
    title="Thickener / tailings settling: Bürger-Concha degenerate conservation law (parametric descent rate)",
    governing_equations=(
        r"\phi_t + \partial_z f_{bk}(\phi) = \partial_z(D(\phi)\phi_z),\ "
        r"f_{bk}=v_0\phi(1-\phi/\phi_{max})^C,\ D\ \text{degenerate above}\ \phi_c"
    ),
    method="nonlinear-settling",
    engine="deepxde",
    real_or_synthetic="synthetic-illustrative",
    inputs=("z", "t", "R"),
    outputs=("phi",),
    domain={"z": (0.0, 1.0), "t": (0.0, 1.0), "R": (R_MIN, R_MAX)},
    grid={"z": 101, "t": 51},
    field_axes=("z", "t"),
    param_specs=(ParamSpec("R", "Front descent rate R", "Tasa de descenso del frente R", R_DEF, R_MIN, R_MAX, 0.02),),
    expected_band="descending settling front (faster for larger R) + rising consolidated bed; relative-L2 vs MMS analytic < 2e-2",
    validation_anchor="analytic",
    train={"lr": 1e-3, "adam": 24000, "lbfgs": True, "num_domain": 8000, "num_boundary": 800,
           "num_test": 9000, "loss_weights": [1, 10],
           "layers": [3, 64, 64, 64, 64, 1]},
    notes="Richardson-Zaki hindered settling + tanh-regularized degenerate diffusion; soft Dirichlet/IC = phi* on the "
          "whole (z,t,R) cube boundary; MMS source. Parametric in the front descent rate R (a network input). "
          "Adam->L-BFGS (RAR de-stabilised the stiff degenerate-diffusion residual, so it is dropped here).",
)


def _phi_star_np(ztR: np.ndarray) -> np.ndarray:
    ztR = np.asarray(ztR, dtype=np.float64)
    z = ztR[:, 0:1]
    t = ztR[:, 1:2]
    R = ztR[:, 2:3]
    s = Z0 - R * t
    return PHI_LO + (PHI_HI - PHI_LO) * 0.5 * (1.0 - np.tanh((z - s) / W_FRONT))


def analytic(ztR: np.ndarray) -> np.ndarray:
    return _phi_star_np(ztR)


def variants() -> list[Variant]:
    presets = [
        ("r030", 0.30, "Slow settle (R=0.30) — a gently descending mud-line; the bed barely forms.",
                        "Sedimentación lenta (R=0.30) — la interfase desciende suavemente; el lecho apenas se forma."),
        ("r042", 0.42, "R=0.42 — moderate descent.", "R=0.42 — descenso moderado."),
        ("r054", 0.54, "R=0.54 — clear settling, bed building.", "R=0.54 — sedimentación clara, el lecho crece."),
        ("r066", 0.66, "R=0.66 — fast front, well-defined bed.", "R=0.66 — frente rápido, lecho bien definido."),
        ("r078", 0.78, "R=0.78 — rapid settle (near the legacy case).", "R=0.78 — sedimentación rápida (cerca del caso original)."),
        ("r090", 0.90, "Fast settle (R=0.90) — the mud-line drops to the column bottom by t=1.",
                        "Sedimentación rápida (R=0.90) — la interfase llega al fondo de la columna en t=1."),
    ]
    return [Variant(vid, f"R={r:g}", f"R={r:g}", {"R": r}, en, es) for vid, r, en, es in presets]


def build(seed: int) -> dict:
    import deepxde as dde
    import torch

    dde.config.set_random_seed(int(seed))
    geom = dde.geometry.Hypercube([0.0, 0.0, R_MIN], [1.0, 1.0, R_MAX])

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
        z = x[:, 0:1]
        t = x[:, 1:2]
        R = x[:, 2:3]
        s = Z0 - R * t
        u = (z - s) / W_FRONT
        th = torch.tanh(u)
        sech2 = 1.0 - th ** 2
        amp = (PHI_HI - PHI_LO)
        phi = PHI_LO + amp * 0.5 * (1.0 - th)
        phi_z = -amp / (2.0 * W_FRONT) * sech2
        phi_t = -amp * R / (2.0 * W_FRONT) * sech2   # du/dt = R/W
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

    # phi* is exact on the WHOLE (z,t,R) cube boundary (incl. the t=0 IC face), so one soft Dirichlet anchor over all
    # boundary faces enforces both the spatial BCs and the initial condition (the case keeps its soft-constraint method).
    bc = dde.icbc.DirichletBC(geom, _phi_star_np, lambda _, on_boundary: on_boundary)

    t = CASE.train
    data = dde.data.PDE(
        geom, pde, [bc],
        num_domain=t["num_domain"], num_boundary=t["num_boundary"],
        solution=analytic, num_test=t["num_test"],
    )
    net = dde.nn.FNN(t["layers"], "tanh", "Glorot normal")
    model = dde.Model(data, net)
    model.compile("adam", lr=t["lr"], loss_weights=t["loss_weights"], metrics=["l2 relative error"])
    return {"model": model, "input_dim": 3}
