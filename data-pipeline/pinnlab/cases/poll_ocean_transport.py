"""Group C · pollution-environmental — ocean/coastal pollutant transport (forward), 2D advection-diffusion, baked as
a TIME-SCRUBBER workbench: field_axes=(x,y), the swept parameter is TIME t, so the web `Live` tab scrubs t and the
shared ONNX replays the spill drifting and spreading.

Governing equation (passive scalar c carried by a uniform coastal current v, eddy diffusion D, no source):
    c_t + v.grad(c) = D (c_xx + c_yy)   on (0,1)^2 x (0,1],  v = (0.45, 0.35),  D = 0.01.
EXACT solution (validation anchor, the advected-diffused Gaussian / 2D Green's function — NOT a manufactured source):
    c*(x,y,t) = (s0^2 / s2) * exp( -((x - x0 - vx t)^2 + (y - y0 - vy t)^2) / (2 s2) ),   s2 = s0^2 + 2 D t.
A Gaussian pollutant patch released at (x0,y0) drifts with the current (center moves at v) and spreads by eddy
diffusion (variance grows as 2 D t), its peak decaying as mass is conserved. real_or_synthetic = synthetic-illustrative
(a physically-faithful illustration of a plastic/oil patch, NOT fit to a real spill or a real ocean-current product).

Method — advection-diffusion PINN with SOFT IC/BC (Dirichlet = c* on the boundary, IC = c* at t=0), so the network
genuinely learns the interior transport field and the relative-L2 reports the true PINN error. Pe = |v| L / D ~ 45.
"""
from __future__ import annotations

import numpy as np

from .base import CaseSpec, ParamSpec, Variant

VX, VY = 0.45, 0.35
D_COEF = 0.01
X0, Y0 = 0.25, 0.30
S0_SQ = 0.008  # initial Gaussian variance (sigma0 ~ 0.089)


CASE = CaseSpec(
    id="poll-ocean-transport",
    category="pollution-environmental",
    title="Ocean pollutant transport: 2D advection-diffusion PINN (time-scrubber)",
    governing_equations=(
        r"c_t + \mathbf{v}\cdot\nabla c = D\nabla^2 c,\ \mathbf{v}=(0.45,0.35),\ D=0.01,\ "
        r"c^*=\tfrac{s_0^2}{s_0^2+2Dt}\,e^{-\|\mathbf{x}-\mathbf{x}_0-\mathbf{v}t\|^2/(2(s_0^2+2Dt))}"
    ),
    method="advection-diffusion",
    engine="deepxde",
    real_or_synthetic="synthetic-illustrative",
    inputs=("x", "y", "t"),
    outputs=("c",),
    domain={"x": (0.0, 1.0), "y": (0.0, 1.0), "t": (0.0, 1.0)},
    grid={"x": 81, "y": 81},
    field_axes=("x", "y"),
    param_specs=(ParamSpec("t", "Time t", "Tiempo t", 0.5, 0.0, 1.0, 0.02),),
    expected_band="a Gaussian patch drifting with the current and spreading by diffusion; relative-L2 vs exact < 1e-2",
    validation_anchor="analytic",
    train={
        "lr": 1e-3,
        "adam": 18000,
        "lbfgs": True,
        "num_domain": 9000,
        "num_boundary": 400,
        "num_initial": 800,
        "num_test": 8000,
        "loss_weights": [1, 10, 50],  # [pde, bc, ic]
    },
    notes="Uniform current; exact advected-diffused Gaussian (no MMS source); soft Dirichlet BC (=c*) + IC (=c* at t=0); time is the swept parameter.",
)


def analytic(xyt: np.ndarray) -> np.ndarray:
    """c*(x,y,t): advected-diffused Gaussian, on [N,3] (x,y,t) -> [N,1]."""
    xyt = np.asarray(xyt, dtype=np.float64)
    x, y, t = xyt[:, 0:1], xyt[:, 1:2], xyt[:, 2:3]
    s2 = S0_SQ + 2.0 * D_COEF * t
    r2 = (x - X0 - VX * t) ** 2 + (y - Y0 - VY * t) ** 2
    return (S0_SQ / s2) * np.exp(-r2 / (2.0 * s2))


def variants() -> list[Variant]:
    presets = [
        ("t00", 0.0, "Release (t=0) — the patch at its source, compact and intense.", "Vertido (t=0) — el parche en su origen, compacto e intenso."),
        ("t02", 0.2, "t=0.2 — drifting with the current, starting to spread.", "t=0.2 — derivando con la corriente, empezando a esparcirse."),
        ("t04", 0.4, "t=0.4 — clearly advected and broader.", "t=0.4 — claramente advectado y más ancho."),
        ("t06", 0.6, "t=0.6 — past the domain center, diluting.", "t=0.6 — pasado el centro del dominio, diluyéndose."),
        ("t08", 0.8, "t=0.8 — broad and faint, nearing the far corner.", "t=0.8 — ancho y tenue, cerca de la esquina lejana."),
        ("t10", 1.0, "t=1.0 — maximally spread and diluted.", "t=1.0 — máxima dispersión y dilución."),
    ]
    return [Variant(vid, f"t={tv:g}", f"t={tv:g}", {"t": tv}, en, es) for vid, tv, en, es in presets]


def build(seed: int) -> dict:
    import deepxde as dde

    dde.config.set_random_seed(int(seed))
    geom = dde.geometry.Rectangle([0.0, 0.0], [1.0, 1.0])
    timedomain = dde.geometry.TimeDomain(0.0, 1.0)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    def pde(x, c):
        c_t = dde.grad.jacobian(c, x, i=0, j=2)
        c_x = dde.grad.jacobian(c, x, i=0, j=0)
        c_y = dde.grad.jacobian(c, x, i=0, j=1)
        c_xx = dde.grad.hessian(c, x, i=0, j=0)
        c_yy = dde.grad.hessian(c, x, i=1, j=1)
        return c_t + VX * c_x + VY * c_y - D_COEF * (c_xx + c_yy)

    bc = dde.icbc.DirichletBC(geomtime, lambda X: analytic(X), lambda _, on_boundary: on_boundary)
    ic = dde.icbc.IC(geomtime, lambda X: analytic(X), lambda _, on_initial: on_initial)

    t = CASE.train
    data = dde.data.TimePDE(
        geomtime, pde, [bc, ic],
        num_domain=t["num_domain"], num_boundary=t["num_boundary"],
        num_initial=t["num_initial"], solution=analytic, num_test=t["num_test"],
    )
    net = dde.nn.FNN([3] + [64] * 4 + [1], "tanh", "Glorot normal")
    model = dde.Model(data, net)
    model.compile("adam", lr=t["lr"], loss_weights=t["loss_weights"], metrics=["l2 relative error"])
    return {"model": model, "input_dim": 3}
