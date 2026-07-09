"""Group B · mining-mineral-processing — comminution population balance (size-transport reduced model), PARAMETRIC
in the grind rate g.

Grinding (SAG/ball milling) evolves the particle-size distribution n(s,t): fragmentation continuously shifts mass
toward smaller sizes. The full population-balance equation (PBE) is an integro-differential equation with selection +
breakage kernels; here we ship the REDUCED size-transport surrogate — a drift-diffusion in size space whose drift IS
the net downward shift (the Fokker-Planck reduction of the breakage operator) and whose diffusion D is the
fragmentation spread:
    n_t + (-g) n_s = D n_ss   on s in [0,1] (normalized size, 1=coarse), t in [0,1],
with the GRIND RATE g (downward drift toward smaller s) as a NETWORK INPUT and size dispersion D constant. EXACT
anchor (the advected-diffused Gaussian / 1D Green's function, valid for ANY g):
    n* = sqrt(s0sq/(s0sq+2Dt)) * exp( -(s - (s0 - g t))^2 / (2 (s0sq + 2 D t)) ).
A narrow coarse feed (centered at s0) drifts down in size with g and spreads by dispersion D, its peak decaying as
mass is conserved. real_or_synthetic = synthetic-illustrative: a clean reduced model, NOT a fitted mill PSD (no open
SAG/ball-mill PSD dataset with a grind-rate axis — real-datasets.md); the full breakage-kernel PBE is documented as
the complete model.
"""
from __future__ import annotations

import numpy as np

from .base import CaseSpec, ParamSpec, Variant

G_MIN, G_MAX = 0.0, 0.6
D_SIZE = 0.012
S0_CENTER = 0.8          # coarse-feed center (large size)
SIG0_SQ = 0.01           # initial size variance (sigma0 ~ 0.10)

CASE = CaseSpec(
    id="mine-comminution-pbe",
    category="mining-mineral-processing",
    title="Comminution population balance: size-transport reduced PINN (parametric grind rate)",
    governing_equations=(
        r"n_t + (-g)\,n_s = D\,n_{ss}\ \text{on}\ s\in[0,1],\ t\in[0,1];\ "
        r"n^*=\sqrt{\tfrac{\sigma_0^2}{\sigma_0^2+2Dt}}\,e^{-(s-(s_0-g t))^2/(2(\sigma_0^2+2Dt))}\ "
        r"(\text{size-transport reduction of the PBE})"
    ),
    method="population-balance",
    engine="deepxde",
    real_or_synthetic="synthetic-illustrative",
    inputs=("s", "t", "g"),
    outputs=("n",),
    domain={"s": (0.0, 1.0), "t": (0.0, 1.0), "g": (G_MIN, G_MAX)},
    grid={"s": 81, "t": 81},
    field_axes=("s", "t"),
    param_specs=(ParamSpec("g", "Grind rate g", "Tasa de molienda g", 0.3, G_MIN, G_MAX, 0.02),),
    expected_band="a size distribution drifting toward smaller sizes (faster for larger g) and spreading; relative-L2 vs exact <=2.1e-2 (the high-grind g=0.6 corner is advection-leaning, Pe~50)",
    validation_anchor="analytic",
    train={"lr": 1e-3, "adam": 15000, "lbfgs": True, "num_domain": 5000, "num_test": 8000},
    notes="Parametric in the grind rate g (continuous input); reduced drift-diffusion in size space (proxy for the comminution PBE); exact advected-diffused Gaussian anchor; IC baked by the output transform. Full breakage-kernel PBE is the complete model (docs).",
)


def analytic(stg: np.ndarray) -> np.ndarray:
    """n*(s,t;g): advected-diffused Gaussian (1D Green's function), on [N,3] (s,t,g) -> [N,1]."""
    stg = np.asarray(stg, dtype=np.float64)
    s, t, g = stg[:, 0:1], stg[:, 1:2], stg[:, 2:3]
    var = SIG0_SQ + 2.0 * D_SIZE * t
    amp = np.sqrt(SIG0_SQ / var)
    return amp * np.exp(-((s - (S0_CENTER - g * t)) ** 2) / (2.0 * var))


def variants() -> list[Variant]:
    presets = [
        ("g00", 0.00, "No grinding (g=0) — the feed only spreads in place; no downward shift.", "Sin molienda (g=0) — la alimentación solo se ensancha; sin desplazamiento."),
        ("g12", 0.12, "g=0.12 — gentle grinding, a slow shift toward finer sizes.", "g=0.12 — molienda suave, desplazamiento lento hacia finos."),
        ("g24", 0.24, "g=0.24 — moderate grinding.", "g=0.24 — molienda moderada."),
        ("g36", 0.36, "g=0.36 — the bulk is clearly shifting down in size.", "g=0.36 — el grueso baja claramente en tamaño."),
        ("g48", 0.48, "g=0.48 — hard grinding, most mass now in the fines.", "g=0.48 — molienda intensa, casi toda la masa en finos."),
        ("g60", 0.60, "Hard grind (g=0.6) — the distribution is driven far toward the finest sizes.", "Molienda intensa (g=0.6) — la distribución llega muy abajo, a los tamaños más finos."),
    ]
    return [Variant(vid, f"g={gv:g}", f"g={gv:g}", {"g": gv}, en, es) for vid, gv, en, es in presets]


def build(seed: int) -> dict:
    import deepxde as dde
    import torch

    dde.config.set_random_seed(int(seed))
    geom = dde.geometry.Hypercube([0.0, 0.0, G_MIN], [1.0, 1.0, G_MAX])   # (s, t, g)

    def pde(x, n):
        n_t = dde.grad.jacobian(n, x, i=0, j=1)
        n_s = dde.grad.jacobian(n, x, i=0, j=0)
        n_ss = dde.grad.hessian(n, x, i=0, j=0)
        g = x[:, 2:3]
        return n_t - g * n_s - D_SIZE * n_ss            # n_t + (-g) n_s = D n_ss

    t = CASE.train
    data = dde.data.PDE(
        geom, pde, [],
        num_domain=t["num_domain"], num_boundary=0, solution=analytic, num_test=t["num_test"],
    )
    net = dde.nn.FNN([3] + [48] * 4 + [1], "tanh", "Glorot normal")

    def output_transform(x, n):
        s = x[:, 0:1]
        tt = x[:, 1:2]
        # IC baseline g0(s) = n*(s,0;g) = exp(-(s-s0)^2/(2 sig0^2))  (g-independent at t=0); t-factor -> IC exact.
        g0 = torch.exp(-((s - S0_CENTER) ** 2) / (2.0 * SIG0_SQ))
        return g0 + tt * n
    net.apply_output_transform(output_transform)

    model = dde.Model(data, net)
    model.compile("adam", lr=t["lr"], metrics=["l2 relative error"])
    return {"model": model, "input_dim": 3}
