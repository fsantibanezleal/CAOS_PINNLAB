"""Group C · pollution-environmental (mining ∩ pollution) — tailings-dam unsaturated seepage (Richards equation),
PARAMETRIC in the Gardner sorptive number alpha.

1D vertical unsaturated seepage through a tailings deposit (z up, psi = pressure head < 0 unsaturated, t = time):
    C(psi) psi_t = d/dz[ K(psi) (psi_z + 1) ]  =>  C psi_t - K psi_zz - K'(psi)(psi_z^2 + psi_z) = 0   (source-free).
Constitutive closure — GARDNER exponential model:
    K(psi) = Ks exp(alpha psi),   K'(psi) = alpha K,   theta(psi)=theta_r+(theta_s-theta_r)exp(alpha psi),
    C(psi) = dtheta/dpsi = (theta_s-theta_r) alpha exp(alpha psi).

The KIRCHHOFF transform m = exp(alpha psi) linearises the nonlinear operator EXACTLY into a constant-coefficient
advection-diffusion in m, which admits a clean separable exact solution whose psi-field genuinely depends on alpha
(unlike a manufactured source, which would absorb all alpha-dependence and make the Live slider cosmetic):
    psi*(z,t;alpha) = (1/alpha) ln( M0 + A exp(-lam(alpha) t) exp(-kappa z) ),
    lam(alpha) = (Ks/(theta_s-theta_r)) * kappa * (alpha-kappa)/alpha   (dispersion relation -> exact, source-free).
With M0+A<1 and M0>0 the argument m stays in (0,1) so psi<0 strictly (always unsaturated — the physical invariant).
alpha is a NETWORK INPUT: one trained net covers the whole sorptive-number family; the web `Live` tab sweeps alpha
(suction depth roughly doubles across the range). real_or_synthetic = synthetic-illustrative: the Gardner closure is
standard, the field is an exact illustration, NOT calibrated to a real deposit (no open unsaturated psi(z,t) dataset).
"""
from __future__ import annotations

import numpy as np

from .base import CaseSpec, ParamSpec, Variant

THETA_S, THETA_R = 0.43, 0.078
KS = 0.25
KAPPA, M0, A_AMP = 0.9, 0.30, 0.62   # m = M0 + A_AMP e^{-lam t} e^{-KAPPA z}; M0+A_AMP<1 => psi<0 strictly
ALPHA_MIN, ALPHA_MAX = 1.0, 2.5      # Gardner sorptive number (swept network input)


def _lam(alpha):
    # dispersion: (theta_s-theta_r) m_t = (Ks/alpha) m_zz + Ks m_z  =>  lam = (Ks/dth) kappa (alpha-kappa)/alpha
    return (KS / (THETA_S - THETA_R)) * KAPPA * (alpha - KAPPA) / alpha


CASE = CaseSpec(
    id="poll-tailings-seepage",
    system_type="time-evol-1d",
    category="pollution-environmental",
    title="Tailings-dam unsaturated seepage — Richards (Gardner), parametric sorptive number α",
    governing_equations=(
        r"C(\psi)\psi_t=\partial_z[K(\psi)(\psi_z+1)],\ K=K_s e^{\alpha\psi},\ "
        r"\psi^*=\tfrac1\alpha\ln\!\big(M_0+A\,e^{-\lambda(\alpha)t}e^{-\kappa z}\big),\ "
        r"\lambda=\tfrac{K_s}{\theta_s-\theta_r}\tfrac{\kappa(\alpha-\kappa)}{\alpha}"
    ),
    method="richards-seepage",
    engine="deepxde",
    real_or_synthetic="synthetic-illustrative",
    inputs=("z", "t", "alpha"),
    outputs=("psi",),
    domain={"z": (0.0, 1.0), "t": (0.0, 1.0), "alpha": (ALPHA_MIN, ALPHA_MAX)},
    grid={"z": 101, "t": 51},
    field_axes=("z", "t"),
    param_specs=(ParamSpec("alpha", "Sorptive number α", "Número sortivo α", 1.6, ALPHA_MIN, ALPHA_MAX, 0.05),),
    expected_band="drying unsaturated head profile (psi<0); deeper suction at small alpha; relative-L2 vs exact < 1e-2",
    validation_anchor="analytic",
    train={"lr": 1e-3, "adam": 18000, "lbfgs": True, "num_domain": 6000, "num_boundary": 300,
           "num_initial": 300, "num_test": 8000, "loss_weights": [1, 10, 10]},
    notes="Gardner exponential K(psi); Kirchhoff transform m=e^{alpha psi} linearises the operator => exact SOURCE-FREE family; alpha (sorptive number) is the swept network input; soft Dirichlet/IC = psi*.",
)


def _psi_star_np(zta: np.ndarray) -> np.ndarray:
    zta = np.asarray(zta, dtype=np.float64)
    z, t, alpha = zta[:, 0:1], zta[:, 1:2], zta[:, 2:3]
    lam = (KS / (THETA_S - THETA_R)) * KAPPA * (alpha - KAPPA) / alpha
    m = M0 + A_AMP * np.exp(-lam * t) * np.exp(-KAPPA * z)
    return np.log(m) / alpha


def analytic(zta: np.ndarray) -> np.ndarray:
    return _psi_star_np(zta)


def variants() -> list[Variant]:
    presets = [
        ("a100", 1.0, "Broad pore-size (α=1.0) — deepest, most stratified suction.", "Poros amplios (α=1.0) — succión más profunda y estratificada."),
        ("a130", 1.3, "α=1.3 — strong suction gradient.", "α=1.3 — fuerte gradiente de succión."),
        ("a160", 1.6, "α=1.6 — moderate Gardner sorptivity.", "α=1.6 — sortividad Gardner moderada."),
        ("a190", 1.9, "α=1.9 — shallower profile, faster drying.", "α=1.9 — perfil menos profundo, secado más rápido."),
        ("a220", 2.2, "α=2.2 — weak suction, near-saturated top.", "α=2.2 — succión débil, tope casi saturado."),
        ("a250", 2.5, "Fine pore-size (α=2.5) — shallowest suction.", "Poros finos (α=2.5) — succión más somera."),
    ]
    return [Variant(vid, f"α={a:g}", f"α={a:g}", {"alpha": a}, en, es) for vid, a, en, es in presets]


def build(seed: int) -> dict:
    import deepxde as dde

    dde.config.set_random_seed(int(seed))
    geom = dde.geometry.Hypercube([0.0, 0.0, ALPHA_MIN], [1.0, 1.0, ALPHA_MAX])

    def pde(x, y):
        psi = y
        psi_t = dde.grad.jacobian(y, x, i=0, j=1)
        psi_z = dde.grad.jacobian(y, x, i=0, j=0)
        psi_zz = dde.grad.hessian(y, x, i=0, j=0)
        alpha = x[:, 2:3]
        ek = (alpha * psi).exp()
        cap = (THETA_S - THETA_R) * alpha * ek
        k = KS * ek
        kp = alpha * k
        return cap * psi_t - k * psi_zz - kp * (psi_z ** 2 + psi_z)   # source-free (exact Kirchhoff family)

    def on_zbc(X, on_boundary):
        return on_boundary and (np.isclose(X[0], 0.0) or np.isclose(X[0], 1.0))

    def on_ic(X, on_boundary):
        return on_boundary and np.isclose(X[1], 0.0)

    bc = dde.icbc.DirichletBC(geom, _psi_star_np, on_zbc)
    ic = dde.icbc.DirichletBC(geom, _psi_star_np, on_ic)

    t = CASE.train
    data = dde.data.PDE(
        geom, pde, [bc, ic],
        num_domain=t["num_domain"], num_boundary=t["num_boundary"],
        solution=analytic, num_test=t["num_test"],
    )
    net = dde.nn.FNN([3] + [48] * 4 + [1], "tanh", "Glorot normal")
    model = dde.Model(data, net)
    model.compile("adam", lr=t["lr"], loss_weights=t["loss_weights"], metrics=["l2 relative error"])
    return {"model": model, "input_dim": 3}
