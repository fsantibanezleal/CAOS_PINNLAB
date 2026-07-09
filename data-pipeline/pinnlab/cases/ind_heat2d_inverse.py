"""Group D · industrial-fluids-heat — 2D INVERSE heat conduction: recover the conductivity field k(x,y) from sparse
interior temperature sensors.

    div(k(x,y) grad T) = q(x,y)  on (0,1)^2,  T=0 on the boundary.
Given the source q and ~100 sparse noisy temperature samples T(x_i,y_i), infer the WHOLE conductivity field k(x,y).
The unknown k is the SECOND network output (a field, not a dde.Variable scalar); the PDE prior fills the gaps where
there are no sensors. This is the canonical sparse-data field-inverse problem where PINNs beat classical FEM/FVM.

real_or_synthetic = synthetic (MMS): no open 2D thermal-field inverse dataset exists (real-datasets.md §6). The
manufactured triple (T*, k*, q) is the ground truth:
    T* = sin(pi x) sin(pi y),   k* = 1 + 0.5 sin(pi x) sin(pi y),   q = div(k* grad T*)  (SymPy-derived).
Method — INVERSE field PINN (PFNN 2-output, product-rule residual, PointSetBC observations, softplus(k) positivity).
Primary score = relative-L2 of the recovered k vs k* (T error in extra_metrics).
"""
from __future__ import annotations

import numpy as np

from .base import CaseSpec, Variant

N_SENSORS = 100
NOISE = 0.01


def k_true(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    return 1.0 + 0.5 * np.sin(np.pi * X[:, 0:1]) * np.sin(np.pi * X[:, 1:2])


def t_true(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    return np.sin(np.pi * X[:, 0:1]) * np.sin(np.pi * X[:, 1:2])


# the sparse sensors are shared by build() (the PINN data) and extra_metrics (baked into the trace for the App).
_OBS_XY: np.ndarray | None = None
_OBS_T: np.ndarray | None = None


def _sensors(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    ob_xy = rng.uniform(0.05, 0.95, size=(N_SENSORS, 2))
    ob_T = t_true(ob_xy) + NOISE * rng.standard_normal((N_SENSORS, 1))
    return ob_xy, ob_T


CASE = CaseSpec(
    id="ind-heat2d-inverse",
    system_type="inverse-assim",
    view_kit="InverseOverlayKit",
    category="industrial-fluids-heat",
    title="2D inverse heat conduction: recover conductivity k(x,y) from sparse sensors",
    governing_equations=(
        r"\nabla\!\cdot(k(x,y)\nabla T)=q\ \text{on}\ (0,1)^2,\ T|_{\partial\Omega}=0;\ "
        r"\text{recover}\ k\ \text{from sparse }T\text{ obs};\ k^*=1+\tfrac12\sin\pi x\sin\pi y"
    ),
    method="inverse-field",
    engine="deepxde",
    real_or_synthetic="synthetic",
    inputs=("x", "y"),
    outputs=("k", "T"),  # k (the inverse target) is the PRIMARY output -> l2_relative scores k vs k*
    domain={"x": (0.0, 1.0), "y": (0.0, 1.0)},
    grid={"x": 81, "y": 81},
    field_axes=("x", "y"),  # the baked heatmap is the recovered k(x,y); single-variant inverse (no parameter axis)
    expected_band="recovered conductivity field k(x,y); relative-L2 vs k* ~1e-2-5e-2 (loosest where |grad T| is small, near the boundary)",
    validation_anchor="analytic",
    train={"lr": 1e-3, "adam": 20000, "lbfgs": True, "num_domain": 2000, "num_boundary": 200, "num_test": 4000, "loss_weights": [1, 100]},
    notes="Inverse field via the 2nd PFNN output; product-rule div(k grad T); softplus(k)>0; 100 sparse noisy T sensors. PINNs-are-for-inverse.",
)


def analytic(xy: np.ndarray) -> np.ndarray:
    """Primary output is k -> score the recovered conductivity against k*."""
    return k_true(xy)


def variants() -> list[Variant]:
    # Single honest benchmark: k* is a fixed MMS field and there is no network-input knob to sweep (the data-side
    # knobs — noise / sensor count — would each need a fresh inverse solve). One variant, never a fabricated sweep.
    return [Variant(
        "default", "Recovered k(x,y)", "k(x,y) recuperado", {},
        "Conductivity field recovered from ~100 sparse noisy T sensors; relative-L2 vs the exact k*.",
        "Campo de conductividad recuperado desde ~100 sensores T dispersos y ruidosos; L2-relativo vs el k* exacto.",
    )]


def extra_metrics(sf) -> dict:
    from ..model.analytic import l2_relative, linspace_grid

    _coords, XY, shape = linspace_grid(CASE.domain, CASE.grid)
    t_pred = sf.fields["T"].reshape(-1, 1)
    # bake the truth field + the sparse sensors so the InverseOverlayKit can overlay observations on the recovered
    # field and compare k vs k* (the inverse "answer" + its evidence).
    sf.fields["k_true"] = k_true(XY).reshape(shape)
    if _OBS_XY is not None:
        sf.inverse = {
            "param": "k",
            "observe_output": "T",
            "observations": [
                [round(float(x), 4), round(float(y), 4), round(float(tv), 4)]
                for (x, y), tv in zip(_OBS_XY, _OBS_T.reshape(-1))
            ],
        }
    return {"T_l2_relative": round(l2_relative(t_pred, t_true(XY)), 6)}


def build(seed: int) -> dict:
    import deepxde as dde
    import torch

    dde.config.set_random_seed(int(seed))
    geom = dde.geometry.Rectangle([0.0, 0.0], [1.0, 1.0])

    def q_source(x):
        px = np.pi * x[:, 0:1]
        py = np.pi * x[:, 1:2]
        s = torch.sin(px) * torch.sin(py)
        return (np.pi ** 2 / 2.0) * (-4.0 * s ** 2 + torch.sin(px) ** 2 + torch.sin(py) ** 2 - 4.0 * s)

    def pde(x, y):
        k = y[:, 0:1]
        # T = y[:, 1:2]
        T_x = dde.grad.jacobian(y, x, i=1, j=0)
        T_y = dde.grad.jacobian(y, x, i=1, j=1)
        T_xx = dde.grad.hessian(y, x, component=1, i=0, j=0)
        T_yy = dde.grad.hessian(y, x, component=1, i=1, j=1)
        k_x = dde.grad.jacobian(y, x, i=0, j=0)
        k_y = dde.grad.jacobian(y, x, i=0, j=1)
        div_kgradT = k * (T_xx + T_yy) + k_x * T_x + k_y * T_y
        return div_kgradT - q_source(x)

    # sparse noisy temperature sensors (synthetic measurements) — shared with extra_metrics via module globals
    global _OBS_XY, _OBS_T
    ob_xy, ob_T = _sensors(seed)
    _OBS_XY, _OBS_T = ob_xy, ob_T
    observe_T = dde.icbc.PointSetBC(ob_xy, ob_T, component=1)  # data on T (component 1)

    net = dde.nn.PFNN([2, [40, 40], [40, 40], [40, 40], 2], "tanh", "Glorot uniform")

    def output_transform(x, y):
        xx = x[:, 0:1]
        yy = x[:, 1:2]
        k = torch.nn.functional.softplus(y[:, 0:1]) + 1e-3        # k > 0
        t = xx * (1.0 - xx) * yy * (1.0 - yy) * y[:, 1:2]          # T = 0 on the boundary, exactly
        return torch.cat([k, t], dim=1)

    net.apply_output_transform(output_transform)

    t = CASE.train
    data = dde.data.PDE(
        geom, pde, [observe_T],
        num_domain=t["num_domain"], num_boundary=t["num_boundary"], anchors=ob_xy, num_test=t["num_test"],
    )
    model = dde.Model(data, net)
    model.compile("adam", lr=t["lr"], loss_weights=t["loss_weights"])
    return {"model": model, "input_dim": 2}
