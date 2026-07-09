"""Group A · canonical-benchmark — 1D wave equation (hyperbolic), SIREN + HARD-CONSTRAINT PINN, PARAMETRIC in the
wave speed c.

Governing equation:
    u_tt = c^2 u_xx   on x in [0,1], t in [0,1],
    IC  u(x,0) = sin(pi x),  u_t(x,0) = 0,   BC  u(0,t) = u(1,t) = 0.
Manufactured exact solution (validation anchor), valid for ANY c:
    u*(x,t;c) = sin(pi x) cos(c pi t)   (a single-mode standing wave oscillating at frequency c).

Method — SIREN (sin activation) as the spectral-bias remedy for the oscillatory hyperbolic solution, combined with a
HARD-CONSTRAINT output transform that satisfies BC + both ICs exactly:
    u_hat(x,t) = sin(pi x) + t^2 * x * (1-x) * N(x,t,c)
(t=0 -> sin(pi x) = IC#1; every d/dt term carries a factor t -> u_t(x,0)=0 = IC#2; x(1-x) & sin(pi x) vanish at
x=0,1 -> BC). The wave speed c is a network INPUT: ONE trained net covers the whole speed family; the web `Live` tab
sweeps c continuously — watch the standing wave oscillate faster (large c) or slower (small c) via the shared ONNX.
"""
from __future__ import annotations

import numpy as np

from .base import CaseSpec, ParamSpec, Variant

C_MIN, C_MAX = 0.5, 2.0

CASE = CaseSpec(
    id="bench-wave1d",
    category="canonical-benchmark",
    title="1D wave equation, parametric speed: SIREN + hard-constraint PINN",
    governing_equations=(
        r"u_{tt} = c^2 u_{xx}\ \text{on}\ (0,1)\times(0,1],\ "
        r"u(x,0)=\sin(\pi x),\ u_t(x,0)=0,\ u(0,t)=u(1,t)=0;\ u^*=\sin(\pi x)\cos(c\pi t)"
    ),
    method="siren-hard-constraints",
    engine="deepxde",
    real_or_synthetic="synthetic",
    inputs=("x", "t", "c"),
    outputs=("u",),
    domain={"x": (0.0, 1.0), "t": (0.0, 1.0), "c": (C_MIN, C_MAX)},
    grid={"x": 161, "t": 161},
    field_axes=("x", "t"),
    param_specs=(ParamSpec("c", "Wave speed c", "Velocidad de onda c", 1.0, C_MIN, C_MAX, 0.02),),
    expected_band="standing wave sin(pi x) cos(c pi t), oscillating faster for larger c; relative-L2 vs analytic < 3e-2",
    validation_anchor="analytic",
    train={
        "layers": [3, 96, 96, 96, 96, 96, 1],
        "activation": "sin",
        "lr": 8e-4,
        "adam": 22000,
        "lbfgs": True,
        "num_domain": 9000,
        "num_test": 8000,
    },
    notes="Parametric in c (continuous network input); BC + both ICs enforced exactly by an output transform; SIREN.",
)


def analytic(xtc: np.ndarray) -> np.ndarray:
    """u*(x,t;c) = sin(pi x) cos(c pi t), on [N,3] (x,t,c) -> [N,1]."""
    xtc = np.asarray(xtc, dtype=np.float64)
    x, t, c = xtc[:, 0:1], xtc[:, 1:2], xtc[:, 2:3]
    return np.sin(np.pi * x) * np.cos(c * np.pi * t)


def variants() -> list[Variant]:
    presets = [
        ("c05", 0.5, "Slow wave (c=0.5) — only a quarter period over the window.", "Onda lenta (c=0.5) — solo un cuarto de período en la ventana."),
        ("c075", 0.75, "c=0.75 — three-eighths of a period.", "c=0.75 — tres octavos de período."),
        ("c10", 1.0, "c=1 — half a period; the profile flips sign by t=1.", "c=1 — medio período; el perfil cambia de signo en t=1."),
        ("c125", 1.25, "c=1.25 — past the half period.", "c=1.25 — pasado el medio período."),
        ("c15", 1.5, "c=1.5 — three-quarters of a period.", "c=1.5 — tres cuartos de período."),
        ("c20", 2.0, "Fast wave (c=2) — a full period; the standing wave returns to its start.", "Onda rápida (c=2) — un período completo; la onda estacionaria vuelve al inicio."),
    ]
    return [Variant(vid, f"c={c:g}", f"c={c:g}", {"c": c}, en, es) for vid, c, en, es in presets]


def build(seed: int) -> dict:
    import deepxde as dde
    import torch

    dde.config.set_random_seed(int(seed))
    geom = dde.geometry.Hypercube([0.0, 0.0, C_MIN], [1.0, 1.0, C_MAX])

    def pde(x, u):
        u_tt = dde.grad.hessian(u, x, i=1, j=1)
        u_xx = dde.grad.hessian(u, x, i=0, j=0)
        c = x[:, 2:3]
        return u_tt - c ** 2 * u_xx

    t = CASE.train
    data = dde.data.PDE(
        geom, pde, [],
        num_domain=t["num_domain"], num_boundary=0, solution=analytic, num_test=t["num_test"],
    )
    net = dde.nn.FNN(t["layers"], t["activation"], "Glorot normal")

    def output_transform(x, u):
        xs = x[:, 0:1]
        ts = x[:, 1:2]
        return torch.sin(np.pi * xs) + ts ** 2 * xs * (1.0 - xs) * u

    net.apply_output_transform(output_transform)
    model = dde.Model(data, net)
    model.compile("adam", lr=t["lr"], metrics=["l2 relative error"])
    return {"model": model, "input_dim": 3}
