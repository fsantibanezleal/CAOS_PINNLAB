"""Group B · mining-mineral-processing — heap / in-situ leaching reactive transport (forward, MMS).

Saturated porous medium, downward Darcy percolation of lixiviant; two aqueous reactants advect-disperse and react
bimolecularly A+B->C at rate kf*cA*cB:
    cA_t + v.grad(cA) = D (cA_xx + cA_zz) - kf cA cB + fA
    cB_t + v.grad(cB) = D (cB_xx + cB_zz) - kf cA cB + fB
on (x,z,t) in (0,1)^2 x (0,1], v=(0, vz=1) (downward), D=0.05, kf=1. MMS anchor (offset +1.5 keeps c>0):
    cA* = e^{-t} sin(pi x) cos(pi z) + 1.5,   cB* = e^{-t/2} cos(pi x) sin(pi z) + 1.5
with the source f = L[c*] derived analytically below.

real_or_synthetic = synthetic-illustrative: Chilean-Cu/REE-RELEVANT (parameter ranges from the heap-bioleach
literature) but NOT fitted to any column-test or plant dataset; the real model uses a shrinking-core sink + dual
porosity + variable v — the single bimolecular kf*cA*cB is a deliberate, well-posed teaching simplification.
Method precedent: Reactive-Transport PIML for critical minerals (arXiv:2506.15960).
"""
from __future__ import annotations

import numpy as np

from .base import CaseSpec, ParamSpec, Variant

PI = np.pi
VZ = 1.0
D_COEF = 0.05
KF = 1.0

CASE = CaseSpec(
    id="mine-heap-leach-rt",
    category="mining-mineral-processing",
    title="Heap-leach reactive transport — advection-diffusion-reaction PINN (2 species, time-scrubber)",
    governing_equations=(
        r"\partial_t c_i + \mathbf{v}\cdot\nabla c_i = D\nabla^2 c_i - k_f c_A c_B + f_i,\ "
        r"\mathbf{v}=(0,1),\ D=0.05,\ k_f=1\ \text{on}\ (0,1)^2\times(0,1]"
    ),
    method="advection-diffusion-reaction",
    engine="deepxde",
    real_or_synthetic="synthetic-illustrative",
    inputs=("x", "z", "t"),
    outputs=("cA", "cB"),
    domain={"x": (0.0, 1.0), "z": (0.0, 1.0), "t": (0.0, 1.0)},
    grid={"x": 41, "z": 41},
    field_axes=("x", "z"),
    param_specs=(ParamSpec("t", "Time t", "Tiempo t", 0.5, 0.0, 1.0, 0.02),),
    expected_band="downward-advecting reacting fronts; relative-L2 vs MMS analytic < 2e-2 per species",
    validation_anchor="analytic",
    train={
        "lr": 1e-3,
        "adam": 20000,
        "lbfgs": True,
        "num_domain": 4000,
        "num_boundary": 400,
        "num_initial": 400,
        "num_test": 8000,
        "loss_weights": [1, 1, 10, 10, 10, 10],  # [eqA, eqB, bcA, bcB, icA, icB]
    },
    notes="Bimolecular reaction + downward advection; soft Dirichlet/IC = c*; MMS source. cA is the primary output; cB error in extra_metrics.",
)


def _cA_np(x: np.ndarray) -> np.ndarray:
    xx, zz, tt = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return np.exp(-tt) * np.sin(PI * xx) * np.cos(PI * zz) + 1.5


def _cB_np(x: np.ndarray) -> np.ndarray:
    xx, zz, tt = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return np.exp(-0.5 * tt) * np.cos(PI * xx) * np.sin(PI * zz) + 1.5


_STATE: dict[str, float] = {}  # analytic() stashes the current snapshot t for extra_metrics (per-variant, sequential)


def analytic(xzt: np.ndarray) -> np.ndarray:
    """Primary output cA* (cB* checked in extra_metrics at the same snapshot t)."""
    xzt = np.asarray(xzt, dtype=np.float64)
    _STATE["t"] = float(xzt[0, 2])  # the variant's snapshot time (grid is param-filled in inputs order x,z,t)
    return _cA_np(xzt)


def variants() -> list[Variant]:
    presets = [
        ("t00", 0.0, "Lixiviant front entering (t=0) — both species at the manufactured initial state, sharpest contrast.",
                     "Frente de lixiviante entrando (t=0) — ambas especies en el estado inicial manufacturado, máximo contraste."),
        ("t02", 0.2, "t=0.2 — fronts advecting downward; cA has decayed faster than cB.",
                     "t=0.2 — frentes advectando hacia abajo; cA decae más rápido que cB."),
        ("t04", 0.4, "t=0.4 — clear split in the two species' amplitudes (e^{-t} vs e^{-t/2}).",
                     "t=0.4 — separación clara entre las amplitudes de las especies (e^{-t} vs e^{-t/2})."),
        ("t06", 0.6, "t=0.6 — cA noticeably weaker; reaction sink most uneven across the bed.",
                     "t=0.6 — cA notablemente más débil; el sumidero de reacción más desigual en el lecho."),
        ("t08", 0.8, "t=0.8 — both relaxing toward the +1.5 baseline; cA nearly flat.",
                     "t=0.8 — ambas relajando hacia la línea base +1.5; cA casi plana."),
        ("t10", 1.0, "t=1.0 — late percolation; cA almost uniform, cB still modulated.",
                     "t=1.0 — percolación tardía; cA casi uniforme, cB aún modulada."),
    ]
    return [Variant(vid, f"t={tv:g}", f"t={tv:g}", {"t": tv}, en, es) for vid, tv, en, es in presets]


def extra_metrics(sf) -> dict:
    from ..model.analytic import l2_relative

    t_snap = float(_STATE.get("t", 0.5))
    x = np.asarray(sf.axes["x"], dtype=np.float64)
    z = np.asarray(sf.axes["z"], dtype=np.float64)
    xx, zz = np.meshgrid(x, z, indexing="ij")
    tt = np.full(xx.size, t_snap)
    XZT = np.stack([xx.ravel(), zz.ravel(), tt], axis=1)
    cB_truth = _cB_np(XZT)
    cB_pred = sf.fields["cB"].reshape(-1, 1)
    return {"cB_l2_relative": round(l2_relative(cB_pred, cB_truth), 6)}


def build(seed: int) -> dict:
    import deepxde as dde
    import torch

    dde.config.set_random_seed(int(seed))
    geom = dde.geometry.Rectangle([0.0, 0.0], [1.0, 1.0])
    timedomain = dde.geometry.TimeDomain(0.0, 1.0)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    def fA(x):
        xx, zz, tt = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        eA = torch.exp(-tt)
        cA = eA * torch.sin(PI * xx) * torch.cos(PI * zz) + 1.5
        cB = torch.exp(-0.5 * tt) * torch.cos(PI * xx) * torch.sin(PI * zz) + 1.5
        dt = -eA * torch.sin(PI * xx) * torch.cos(PI * zz)
        adv = VZ * (-PI * eA * torch.sin(PI * xx) * torch.sin(PI * zz))  # vz * dcA_z
        lap = -2.0 * PI * PI * eA * torch.sin(PI * xx) * torch.cos(PI * zz)  # (d_xx + d_zz) cA-mode
        return dt + adv - D_COEF * lap + KF * cA * cB

    def fB(x):
        xx, zz, tt = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        eB = torch.exp(-0.5 * tt)
        cA = torch.exp(-tt) * torch.sin(PI * xx) * torch.cos(PI * zz) + 1.5
        cB = eB * torch.cos(PI * xx) * torch.sin(PI * zz) + 1.5
        dt = -0.5 * eB * torch.cos(PI * xx) * torch.sin(PI * zz)
        adv = VZ * (PI * eB * torch.cos(PI * xx) * torch.cos(PI * zz))  # vz * dcB_z
        lap = -2.0 * PI * PI * eB * torch.cos(PI * xx) * torch.sin(PI * zz)
        return dt + adv - D_COEF * lap + KF * cA * cB

    def pde(x, y):
        cA, cB = y[:, 0:1], y[:, 1:2]
        dcA_t = dde.grad.jacobian(y, x, i=0, j=2)
        dcB_t = dde.grad.jacobian(y, x, i=1, j=2)
        dcA_z = dde.grad.jacobian(y, x, i=0, j=1)
        dcB_z = dde.grad.jacobian(y, x, i=1, j=1)
        dcA_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
        dcA_zz = dde.grad.hessian(y, x, component=0, i=1, j=1)
        dcB_xx = dde.grad.hessian(y, x, component=1, i=0, j=0)
        dcB_zz = dde.grad.hessian(y, x, component=1, i=1, j=1)
        rxn = KF * cA * cB
        eq_A = dcA_t + VZ * dcA_z - D_COEF * (dcA_xx + dcA_zz) + rxn - fA(x)
        eq_B = dcB_t + VZ * dcB_z - D_COEF * (dcB_xx + dcB_zz) + rxn - fB(x)
        return [eq_A, eq_B]

    bcA = dde.icbc.DirichletBC(geomtime, _cA_np, lambda _, on_b: on_b, component=0)
    bcB = dde.icbc.DirichletBC(geomtime, _cB_np, lambda _, on_b: on_b, component=1)
    icA = dde.icbc.IC(geomtime, _cA_np, lambda _, on_i: on_i, component=0)
    icB = dde.icbc.IC(geomtime, _cB_np, lambda _, on_i: on_i, component=1)

    t = CASE.train
    data = dde.data.TimePDE(
        geomtime, pde, [bcA, bcB, icA, icB],
        num_domain=t["num_domain"], num_boundary=t["num_boundary"],
        num_initial=t["num_initial"], num_test=t["num_test"],
    )
    net = dde.nn.FNN([3] + [40] * 4 + [2], "tanh", "Glorot normal")
    model = dde.Model(data, net)
    model.compile("adam", lr=t["lr"], loss_weights=t["loss_weights"])
    return {"model": model, "input_dim": 3}
