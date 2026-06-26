"""Group A · canonical-benchmark — 1D transient heat/diffusion, time-dependent HARD-CONSTRAINT PINN, PARAMETRIC in
the thermal diffusivity alpha.

Governing equation:
    u_t = alpha * u_xx   on x in [0,1], t in [0,1],
    IC  u(x,0) = sin(pi x),   Dirichlet  u(0,t) = u(1,t) = 0.
Manufactured exact solution (validation anchor), valid for ANY alpha:
    u*(x,t;alpha) = exp(-alpha * pi^2 * t) * sin(pi x).

Method — TIME-DEPENDENT PINN with HARD CONSTRAINTS: IC and BC are imposed *exactly* by the output transform
    u_hat(x,t) = t*x*(1-x)*N(x,t,alpha) + sin(pi x)   (at t=0 -> sin(pi x) = IC; at x=0,1 -> 0 = BC),
so there is no IC/BC loss term. The diffusivity `alpha` is a network INPUT: ONE trained net covers the whole
diffusivity family, and the web `Live` tab sweeps alpha continuously — watch the sine profile decay faster (large
alpha) or slower (small alpha) via the shared ONNX, with no retraining.
"""
from __future__ import annotations

import numpy as np

from .base import CaseSpec, ParamSpec, Variant

A_MIN, A_MAX = 0.1, 1.0

CASE = CaseSpec(
    id="bench-heat1d",
    system_type="time-evol-1d",
    category="canonical-benchmark",
    title="1D transient heat/diffusion, parametric diffusivity — time-dependent hard-constraint PINN",
    governing_equations=(
        r"\partial_t u = \alpha\,\partial_{xx} u\ \text{on}\ (0,1)\times(0,1],\ "
        r"u(x,0)=\sin(\pi x),\ u(0,t)=u(1,t)=0;\ u^*=e^{-\alpha\pi^2 t}\sin(\pi x)"
    ),
    method="time-dependent-hard-constraints",
    engine="deepxde",
    real_or_synthetic="synthetic",
    inputs=("x", "t", "alpha"),
    outputs=("u",),
    domain={"x": (0.0, 1.0), "t": (0.0, 1.0), "alpha": (A_MIN, A_MAX)},
    grid={"x": 161, "t": 101},
    field_axes=("x", "t"),
    param_specs=(ParamSpec("alpha", "Diffusivity α", "Difusividad α", 0.5, A_MIN, A_MAX, 0.02),),
    expected_band="exponential decay of a sine profile, faster for larger α; relative-L2 vs analytic < 1e-2",
    validation_anchor="analytic",
    train={
        "layers": [3, 48, 48, 48, 48, 1],
        "activation": "tanh",
        "lr": 1e-3,
        "adam": 15000,
        "lbfgs": True,
        "num_domain": 5000,
        "num_test": 5000,
    },
    notes="Parametric in α (continuous network input); IC+BC enforced exactly by an output transform; no IC/BC loss.",
)


def analytic(xta: np.ndarray) -> np.ndarray:
    """u*(x,t;alpha) = exp(-alpha pi^2 t) sin(pi x), on [N,3] (x,t,alpha) -> [N,1]."""
    xta = np.asarray(xta, dtype=np.float64)
    x, t, a = xta[:, 0:1], xta[:, 1:2], xta[:, 2:3]
    return np.exp(-a * np.pi ** 2 * t) * np.sin(np.pi * x)


def variants() -> list[Variant]:
    presets = [
        ("a01", 0.1, "Slow diffusion (α=0.1) — the profile barely decays over the window.", "Difusión lenta (α=0.1) — el perfil apenas decae en la ventana."),
        ("a02", 0.2, "α=0.2 — gentle decay.", "α=0.2 — decaimiento suave."),
        ("a04", 0.4, "α=0.4 — moderate decay.", "α=0.4 — decaimiento moderado."),
        ("a06", 0.6, "α=0.6 — the sine fades to ~30% by t=1.", "α=0.6 — la sinusoide cae a ~30% en t=1."),
        ("a08", 0.8, "α=0.8 — fast decay.", "α=0.8 — decaimiento rápido."),
        ("a10", 1.0, "Fast diffusion (α=1.0) — the profile collapses to near zero by t=1.", "Difusión rápida (α=1.0) — el perfil colapsa a casi cero en t=1."),
    ]
    return [Variant(vid, f"α={a:g}", f"α={a:g}", {"alpha": a}, en, es) for vid, a, en, es in presets]


def build(seed: int) -> dict:
    import deepxde as dde
    import torch

    dde.config.set_random_seed(int(seed))
    geom = dde.geometry.Hypercube([0.0, 0.0, A_MIN], [1.0, 1.0, A_MAX])

    def pde(x, u):
        du_t = dde.grad.jacobian(u, x, i=0, j=1)
        du_xx = dde.grad.hessian(u, x, i=0, j=0)
        alpha = x[:, 2:3]
        return du_t - alpha * du_xx

    t = CASE.train
    data = dde.data.PDE(
        geom, pde, [],
        num_domain=t["num_domain"], num_boundary=0, solution=analytic, num_test=t["num_test"],
    )
    net = dde.nn.FNN(t["layers"], t["activation"], "Glorot normal")

    def output_transform(x, u):
        xx = x[:, 0:1]
        tt = x[:, 1:2]
        return tt * xx * (1.0 - xx) * u + torch.sin(np.pi * xx)

    net.apply_output_transform(output_transform)
    model = dde.Model(data, net)
    model.compile("adam", lr=t["lr"], metrics=["l2 relative error"])
    return {"model": model, "input_dim": 3}
