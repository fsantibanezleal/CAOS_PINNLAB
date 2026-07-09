"""Group A · canonical-benchmark — 2D Poisson with homogeneous Dirichlet BC, solved by a HARD-CONSTRAINT PINN,
PARAMETRIC in the source mode k (the reference case for the variant + Live-slider pattern).

Governing equation:
    -Delta u = f(x,y;k)  on Omega = (0,1)^2,   u = 0 on the boundary.
The forcing is the manufactured source of the closed-form, boundary-vanishing solution (valid for ANY continuous k):
    u*(x,y;k) = g(x;k) g(y;k),   g(t;k) = t(1-t) sin(k pi t),   f = -Delta u*  (closed form).
Because g(0)=g(1)=0, u* satisfies the homogeneous Dirichlet BC for every k, so k can be a CONTINUOUS network input.

Method exercised — HARD CONSTRAINTS (distance-function output transform): the boundary condition is satisfied
*exactly* for any weights via  u_hat = x(1-x) y(1-y) N(x,y,k),  so there is NO boundary-loss term. ONE trained net
covers the whole mode family k in [1,3]; the App offers preset mode chips and the `Live` tab sweeps k continuously,
re-evaluating the field in the browser via the shared ONNX. Relative-L2 vs the closed form is the validation anchor.
"""
from __future__ import annotations

import numpy as np

from .base import CaseSpec, ParamSpec, Variant

K_MIN, K_MAX = 1.0, 3.0

CASE = CaseSpec(
    id="bench-poisson2d",
    category="canonical-benchmark",
    title="2D Poisson (Dirichlet), parametric source mode: hard-constraint PINN",
    governing_equations=(
        r"-\nabla^2 u = f(x,y;k)\ \text{on}\ (0,1)^2,\ u|_{\partial\Omega}=0;\quad "
        r"u^*=g(x)g(y),\ g(t)=t(1-t)\sin(k\pi t)"
    ),
    method="hard-constraints",
    engine="deepxde",
    real_or_synthetic="synthetic",
    inputs=("x", "y", "k"),
    outputs=("u",),
    domain={"x": (0.0, 1.0), "y": (0.0, 1.0), "k": (K_MIN, K_MAX)},
    grid={"x": 101, "y": 101},
    field_axes=("x", "y"),
    param_specs=(ParamSpec("k", "Source mode k", "Modo de fuente k", 1.0, K_MIN, K_MAX, 0.05),),
    expected_band="boundary-vanishing field with k×k internal lobes; sharper/more oscillatory as k grows; relative-L2 vs analytic < 2e-2",
    validation_anchor="analytic",
    train={
        "layers": [3, 96, 96, 96, 96, 1],
        "activation": "tanh",
        "lr": 1e-3,
        "adam": 16000,
        "lbfgs": True,
        "num_domain": 6000,
        "num_boundary": 0,
        "num_test": 6000,
    },
    notes="Parametric in k (continuous network input); boundary enforced exactly by an output transform; MMS forcing.",
)


def _g_np(t: np.ndarray, k: np.ndarray) -> np.ndarray:
    return (t - t * t) * np.sin(k * np.pi * t)


def analytic(xyk: np.ndarray) -> np.ndarray:
    """u*(x,y;k) = g(x;k) g(y;k), on [N,3] -> [N,1]."""
    xyk = np.asarray(xyk, dtype=np.float64)
    x, y, k = xyk[:, 0:1], xyk[:, 1:2], xyk[:, 2:3]
    return _g_np(x, k) * _g_np(y, k)


def variants() -> list[Variant]:
    presets = [
        ("k1", 1.0, "Fundamental mode (k=1) — a single boundary-vanishing bump.", "Modo fundamental (k=1) — un único lóbulo que se anula en el borde."),
        ("k15", 1.5, "k=1.5 — the lobes begin to subdivide.", "k=1.5 — los lóbulos empiezan a subdividirse."),
        ("k2", 2.0, "k=2 — a 2×2 lobe pattern.", "k=2 — patrón de 2×2 lóbulos."),
        ("k225", 2.25, "k=2.25 — off-integer mode (still boundary-vanishing).", "k=2.25 — modo no entero (sigue anulándose en el borde)."),
        ("k25", 2.5, "k=2.5 — finer oscillation, harder for a smooth net.", "k=2.5 — oscilación más fina, más difícil para una red suave."),
        ("k3", 3.0, "k=3 — a 3×3 lobe pattern; the spectral-bias stress test.", "k=3 — patrón de 3×3 lóbulos; el test de sesgo espectral."),
    ]
    return [Variant(vid, f"Mode k={k:g}", f"Modo k={k:g}", {"k": k}, en, es) for vid, k, en, es in presets]


def build(seed: int) -> dict:
    """Construct + Adam-compile the parametric DeepXDE model over (x, y, k). Heavy: lazy-imports deepxde/torch."""
    import deepxde as dde
    import torch

    dde.config.set_random_seed(int(seed))
    geom = dde.geometry.Hypercube([0.0, 0.0, K_MIN], [1.0, 1.0, K_MAX])

    def mms_f(x):
        t = x[:, 0:1]
        s = x[:, 1:2]
        k = x[:, 2:3]
        kp = k * np.pi

        def g(u):
            return (u - u * u) * torch.sin(kp * u)

        def gpp(u):
            su = torch.sin(kp * u)
            cu = torch.cos(kp * u)
            return -2.0 * su + 2.0 * (1.0 - 2.0 * u) * kp * cu - (u - u * u) * (kp ** 2) * su

        # f = -Delta u* = -(g''(x) g(y) + g(x) g''(y))
        return -(gpp(t) * g(s) + g(t) * gpp(s))

    def pde(x, u):
        u_xx = dde.grad.hessian(u, x, i=0, j=0)
        u_yy = dde.grad.hessian(u, x, i=1, j=1)
        return -u_xx - u_yy - mms_f(x)

    t = CASE.train
    data = dde.data.PDE(
        geom, pde, [],
        num_domain=t["num_domain"], num_boundary=t["num_boundary"],
        solution=analytic, num_test=t["num_test"],
    )
    net = dde.nn.FNN(t["layers"], t["activation"], "Glorot uniform")
    # hard Dirichlet u=0 on the unit-square boundary (k does not affect the BC), satisfied exactly for any weights:
    net.apply_output_transform(
        lambda x, u: (x[:, 0:1] * (1.0 - x[:, 0:1]) * x[:, 1:2] * (1.0 - x[:, 1:2])) * u
    )
    model = dde.Model(data, net)
    model.compile("adam", lr=t["lr"], metrics=["l2 relative error"])
    return {"model": model, "input_dim": 3}
