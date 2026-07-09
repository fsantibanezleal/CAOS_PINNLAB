"""Control · manufactured-solution (MMS) verification of the Poisson operator, PARAMETRIC in the source amplitude a,
HARD-CONSTRAINT PINN (zero Dirichlet boundary baked exactly).

Governing equation:
    -nabla^2 u = f(x,y;a)  on (0,1)^2,   u|_{boundary} = 0.
Manufactured exact solution (validation anchor, closed form for ANY a) — the method of manufactured solutions:
    u*(x,y;a) = a * g(x,y),   g(x,y) = sin(pi x) sin(pi y) + 1/2 sin(2 pi x) sin(2 pi y),
which vanishes on the whole boundary for every a. Substituting into -nabla^2 gives the imposed source
    f(x,y;a) = a * ( 2 pi^2 sin(pi x) sin(pi y) + 4 pi^2 sin(2 pi x) sin(2 pi y) ).
At a=0 this is the archetype's mandatory degenerate negative control (f == 0 => u == 0): the engine must run without
crashing and return a flat-zero field, with relative-L2 = ||pred|| ~ 0. The amplitude a is a NETWORK INPUT, so ONE
trained net covers the whole family and the web `Live` tab sweeps a continuously via the shared ONNX — watch the field
fade to flat zero as a -> 0 (the degenerate control as the limit of the family).

Method — HARD CONSTRAINTS: the zero Dirichlet boundary is satisfied exactly by the output transform
x(1-x)y(1-y) * N, so there is no boundary loss term; the relative-L2 vs the manufactured u* is the true error.
"""
from __future__ import annotations

import numpy as np

from .base import CaseSpec, ParamSpec, Variant

A_MIN, A_MAX = 0.0, 1.0
PI = np.pi


def _g(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fixed two-mode manufactured shape (vanishes on the boundary)."""
    return np.sin(PI * x) * np.sin(PI * y) + 0.5 * np.sin(2.0 * PI * x) * np.sin(2.0 * PI * y)


CASE = CaseSpec(
    id="ctrl-zero-source",
    category="control",
    title="Manufactured-solution control: parametric source amplitude (Poisson, hard-zero boundary)",
    governing_equations=(
        r"-\nabla^2 u = f(x,y;a)\ \text{on}\ (0,1)^2,\ u|_{\partial\Omega}=0,\ "
        r"u^*=a\big(\sin\pi x\sin\pi y+\tfrac12\sin 2\pi x\sin 2\pi y\big)"
    ),
    method="hard-constraints",
    engine="deepxde",
    real_or_synthetic="synthetic",
    inputs=("x", "y", "a"),
    outputs=("u",),
    domain={"x": (0.0, 1.0), "y": (0.0, 1.0), "a": (A_MIN, A_MAX)},
    grid={"x": 41, "y": 41},
    field_axes=("x", "y"),
    param_specs=(ParamSpec("a", "Source amplitude a", "Amplitud de fuente a", 1.0, A_MIN, A_MAX, 0.05),),
    expected_band="a=0 recovers the degenerate control (flat zero); a=1 a two-mode field; relative-L2 vs manufactured u* < 1e-2",
    validation_anchor="analytic",
    train={
        "layers": [3, 48, 48, 48, 48, 1],
        "activation": "tanh",
        "lr": 1e-3,
        "adam": 12000,
        "lbfgs": True,
        "num_domain": 4000,
        "num_test": 4000,
    },
    notes="MMS verification family containing the archetype's degenerate zero control at a=0; amplitude a is a network input; hard-zero boundary via output transform; relative-L2 vs manufactured u* is the true error.",
)


def analytic(xya: np.ndarray) -> np.ndarray:
    """u*(x,y;a) = a * g(x,y), on [N,3] (x,y,a) -> [N,1]."""
    xya = np.asarray(xya, dtype=np.float64)
    x, y, a = xya[:, 0:1], xya[:, 1:2], xya[:, 2:3]
    return a * _g(x, y)


def variants() -> list[Variant]:
    presets = [
        ("a00", 0.0, "Degenerate control (a=0) — zero source, identically-zero field.", "Control degenerado (a=0) — fuente cero, campo idénticamente cero."),
        ("a02", 0.2, "a=0.2 — the manufactured field switching on, faint.", "a=0.2 — el campo manufacturado encendiéndose, tenue."),
        ("a04", 0.4, "a=0.4 — two-mode structure clearly visible.", "a=0.4 — estructura de dos modos claramente visible."),
        ("a06", 0.6, "a=0.6 — stronger lobes.", "a=0.6 — lóbulos más marcados."),
        ("a08", 0.8, "a=0.8 — near full amplitude.", "a=0.8 — casi amplitud plena."),
        ("a10", 1.0, "Full amplitude (a=1) — dominant fundamental lobe with a finer second-mode ripple.", "Amplitud plena (a=1) — lóbulo fundamental dominante con ondulación más fina del segundo modo."),
    ]
    return [Variant(vid, f"a={av:g}", f"a={av:g}", {"a": av}, en, es) for vid, av, en, es in presets]


def build(seed: int) -> dict:
    import deepxde as dde
    import torch

    dde.config.set_random_seed(int(seed))
    geom = dde.geometry.Hypercube([0.0, 0.0, A_MIN], [1.0, 1.0, A_MAX])

    def pde(x, u):
        u_xx = dde.grad.hessian(u, x, i=0, j=0)
        u_yy = dde.grad.hessian(u, x, i=1, j=1)
        xs, ys, a = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        f = a * (
            2.0 * np.pi ** 2 * torch.sin(np.pi * xs) * torch.sin(np.pi * ys)
            + 4.0 * np.pi ** 2 * torch.sin(2.0 * np.pi * xs) * torch.sin(2.0 * np.pi * ys)
        )
        return -u_xx - u_yy - f  # -lap u = f

    t = CASE.train
    net = dde.nn.FNN(t["layers"], t["activation"], "Glorot uniform")
    net.apply_output_transform(
        lambda x, u: x[:, 0:1] * (1.0 - x[:, 0:1]) * x[:, 1:2] * (1.0 - x[:, 1:2]) * u
    )
    data = dde.data.PDE(
        geom, pde, [],
        num_domain=t["num_domain"], num_boundary=0, solution=analytic, num_test=t["num_test"],
    )
    model = dde.Model(data, net)
    model.compile("adam", lr=t["lr"], metrics=["l2 relative error"])
    return {"model": model, "input_dim": 3}
