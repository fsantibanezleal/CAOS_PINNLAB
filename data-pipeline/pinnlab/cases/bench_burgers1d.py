"""Group A · canonical-benchmark — 1D viscous Burgers, PARAMETRIC in the viscosity nu, HARD-CONSTRAINT PINN with
RESIDUAL-BASED ADAPTIVE REFINEMENT (RAR) for the sharp front.

Governing equation:
    u_t + u u_x = nu u_xx,   x in [-1,1], t in [0,1].
Exact TRAVELING-SHOCK family (validation anchor, closed form for ANY nu) — Whitham, *Linear and Nonlinear Waves*:
    u*(x,t;nu) = s - (Delta/2) tanh( k (x - x0 - s t) ),   k = Delta/(4 nu),
with left/right states u_L = s + Delta/2, u_R = s - Delta/2. We use u_L=1, u_R=0 (=> Delta=1, s=1/2) and an initial
front position x0=-0.4, so a front of width ~4*nu translates to the right at speed s and stays interior to [-1,1].
The viscosity nu sets the front THICKNESS: small nu => a razor-sharp internal layer (quasi-shock), large nu => a
diffuse ramp. nu is a NETWORK INPUT, so ONE trained net covers the whole viscosity family and the web `Live` tab
sweeps nu continuously — watch the shock sharpen (small nu) or smear (large nu) via the shared ONNX.

Method — HARD CONSTRAINTS (IC + both Dirichlet BCs satisfied exactly by an output transform, no IC/BC loss) PLUS
RAR / RAR-G (residual-based adaptive refinement, Wu et al. CMAME 2023): after the base fit, greedily ADD collocation
points where the PDE residual is largest, so the moving front is resolved without a globally dense grid.
"""
from __future__ import annotations

import numpy as np

from .base import CaseSpec, ParamSpec, Variant

NU_MIN, NU_MAX = 0.02, 0.08
DELTA, SPEED, X0 = 1.0, 0.5, -0.4  # u_L=1, u_R=0 => Delta=1, s=1/2; initial front at x0


def _k(nu):
    return DELTA / (4.0 * nu)


CASE = CaseSpec(
    id="bench-burgers1d",
    category="canonical-benchmark",
    title="1D viscous Burgers, parametric viscosity — hard-constraint PINN + RAR (traveling shock)",
    governing_equations=(
        r"u_t + u\,u_x = \nu\,u_{xx}\ \text{on}\ (-1,1)\times(0,1],\ "
        r"u^*=s-\tfrac{\Delta}{2}\tanh\!\big(\tfrac{\Delta}{4\nu}(x-x_0-st)\big),\ \Delta=1,\ s=\tfrac12"
    ),
    method="hard-constraints-rar",
    engine="deepxde",
    real_or_synthetic="synthetic",
    inputs=("x", "t", "nu"),
    outputs=("u",),
    domain={"x": (-1.0, 1.0), "t": (0.0, 1.0), "nu": (NU_MIN, NU_MAX)},
    grid={"x": 241, "t": 121},
    field_axes=("x", "t"),
    param_specs=(ParamSpec("nu", "Viscosity ν", "Viscosidad ν", 0.04, NU_MIN, NU_MAX, 0.002),),
    expected_band="a tanh front of width ~4ν translating right at speed 1/2; relative-L2 vs analytic < 2e-2",
    validation_anchor="analytic",
    train={
        "layers": [3, 96, 96, 96, 96, 96, 1],
        "activation": "tanh",
        "lr": 8e-4,
        "adam": 18000,
        "lbfgs": True,
        "num_domain": 9000,
        "num_test": 8000,
        # RAR refinement (the method) — add high-residual points near the moving front:
        "rar_rounds": 5,
        "rar_addk": 400,
        "rar_adam": 3000,
        "rar_pool": 60000,
    },
    notes="Parametric in ν (continuous input); IC + both Dirichlet BCs exact via an output transform; RAR resolves the front.",
)


def analytic(xtn: np.ndarray) -> np.ndarray:
    """u*(x,t;nu) = s - (Delta/2) tanh(k(nu) (x - x0 - s t)), on [N,3] (x,t,nu) -> [N,1]."""
    xtn = np.asarray(xtn, dtype=np.float64)
    x, t, nu = xtn[:, 0:1], xtn[:, 1:2], xtn[:, 2:3]
    return SPEED - 0.5 * DELTA * np.tanh(_k(nu) * (x - X0 - SPEED * t))


def variants() -> list[Variant]:
    presets = [
        ("nu02", 0.02, "Sharp shock (ν=0.02) — a razor-thin internal layer (width ~0.08).", "Shock agudo (ν=0.02) — capa interna finísima (ancho ~0.08)."),
        ("nu03", 0.03, "ν=0.03 — still a steep front.", "ν=0.03 — front aún empinado."),
        ("nu04", 0.04, "ν=0.04 — moderate front thickness.", "ν=0.04 — grosor de front moderado."),
        ("nu05", 0.05, "ν=0.05 — a visibly smoother ramp.", "ν=0.05 — rampa visiblemente más suave."),
        ("nu06", 0.06, "ν=0.06 — diffuse front.", "ν=0.06 — front difuso."),
        ("nu08", 0.08, "Diffuse shock (ν=0.08) — a broad ramp (width ~0.32).", "Shock difuso (ν=0.08) — rampa ancha (ancho ~0.32)."),
    ]
    return [Variant(vid, f"ν={nu:g}", f"ν={nu:g}", {"nu": nu}, en, es) for vid, nu, en, es in presets]


def _make_pde():
    import deepxde as dde

    def pde(x, u):
        u_t = dde.grad.jacobian(u, x, i=0, j=1)
        u_x = dde.grad.jacobian(u, x, i=0, j=0)
        u_xx = dde.grad.hessian(u, x, i=0, j=0)
        nu = x[:, 2:3]
        return u_t + u * u_x - nu * u_xx

    return pde


def build(seed: int) -> dict:
    import deepxde as dde
    import torch

    dde.config.set_random_seed(int(seed))
    geom = dde.geometry.Hypercube([-1.0, 0.0, NU_MIN], [1.0, 1.0, NU_MAX])
    pde = _make_pde()

    t = CASE.train
    data = dde.data.PDE(
        geom, pde, [],
        num_domain=t["num_domain"], num_boundary=0, solution=analytic, num_test=t["num_test"],
    )
    net = dde.nn.FNN(t["layers"], t["activation"], "Glorot normal")

    def output_transform(x, u):
        xs = x[:, 0:1]
        ts = x[:, 1:2]
        nus = x[:, 2:3]
        k = DELTA / (4.0 * nus)
        # IC baseline g(x;nu) = u*(x,0;nu); it already matches the (time-constant, saturated) Dirichlet BCs at x=+-1.
        g = SPEED - 0.5 * DELTA * torch.tanh(k * (xs - X0))
        # t*(1-x^2) vanishes at t=0 (=> IC exact) and at x=+-1 (=> BCs exact); the net learns only the interior evolution.
        return g + ts * (1.0 - xs ** 2) * u

    net.apply_output_transform(output_transform)
    model = dde.Model(data, net)
    model.compile("adam", lr=t["lr"], metrics=["l2 relative error"])
    return {"model": model, "input_dim": 3}


def refine(model, case, seed: int) -> None:
    """RAR-G: greedily add the top-k highest-residual points (concentrated on the moving front) and re-train."""
    pde = _make_pde()
    geom = model.data.geom
    t = case.train
    for _ in range(int(t.get("rar_rounds", 5))):
        X = geom.random_points(int(t.get("rar_pool", 60000)))
        err = np.abs(np.asarray(model.predict(X, operator=pde)))[:, 0]
        idx = np.argsort(err)[-int(t.get("rar_addk", 400)):]
        model.data.add_anchors(X[idx])
        model.compile("adam", lr=t["lr"], metrics=["l2 relative error"])
        model.train(iterations=int(t.get("rar_adam", 3000)), disregard_previous_best=True)
    if t.get("lbfgs", True):
        model.compile("L-BFGS")
        model.train()
