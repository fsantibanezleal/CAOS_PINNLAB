"""Group B · mining-mineral-processing — froth-flotation kinetics (parametric first-order PINN).

Batch froth flotation follows first-order kinetics: the floatable mineral concentration C decays as
    dC/dt = -k C,   C(t=0) = 1,   recovery R(t) = 1 - C(t) = 1 - exp(-k t),
where k is the flotation rate constant (1/min). Instead of one fixed k, this case learns the WHOLE FAMILY of
solutions over a range of rate constants by taking k as a second network input — a parametric PINN. The output is the
2D field C(k, t): one trained network gives the concentration (and recovery) for any rate constant in [0.5, 5]
without retraining, exactly the lumped first-order model used to compare flotation circuits.

real_or_synthetic = synthetic-illustrative: the lumped first-order kinetics is the standard flotation model, but the
field is a clean analytic illustration, NOT fitted to a plant/lab assay (the Kaggle iron-ore flotation set is a 0-D
process time-series without a rate-constant axis — real-datasets.md). Validation anchor: the exact C* = exp(-k t).
"""
from __future__ import annotations

import numpy as np

from .base import CaseSpec, Variant

K_MIN, K_MAX = 0.5, 5.0

CASE = CaseSpec(
    id="mine-flotation-kinetics",
    category="mining-mineral-processing",
    title="Froth-flotation kinetics — parametric first-order PINN C(k,t)",
    governing_equations=(
        r"\partial_t C = -k\,C,\ C(k,0)=1\ \Rightarrow\ C^*=e^{-k t},\ R=1-C;\ k\in[0.5,5]\ (\text{rate constant})"
    ),
    method="parametric-kinetics",
    engine="deepxde",
    real_or_synthetic="synthetic-illustrative",
    inputs=("k", "t"),
    outputs=("C",),
    domain={"k": (K_MIN, K_MAX), "t": (0.0, 1.0)},
    grid={"k": 81, "t": 81},
    field_axes=("k", "t"),
    expected_band="exponential decay of floatable concentration; faster for larger k; relative-L2 vs analytic < 5e-3",
    validation_anchor="analytic",
    train={"lr": 1e-3, "adam": 10000, "lbfgs": True, "num_domain": 2000, "num_boundary": 0, "num_initial": 200, "num_test": 4000},
    notes="Parametric PINN: k is a 2nd input so one net covers the whole rate-constant family. Hard IC C(k,0)=1. R=1-C derived in the App.",
)


def analytic(kt: np.ndarray) -> np.ndarray:
    kt = np.asarray(kt, dtype=np.float64)
    k = kt[:, 0:1]
    t = kt[:, 1:2]
    return np.exp(-k * t)


def variants() -> list[Variant]:
    # The rate constant k is a FIELD AXIS, so one heatmap shows the whole family at once — a single, honest variant.
    return [Variant(
        "family", "Full k-family", "Familia completa de k", {},
        "The whole rate-constant family C(k,t) in one map: recovery R=1−C fills in faster for larger k.",
        "Toda la familia de constantes de tasa C(k,t) en un mapa: la recuperación R=1−C sube más rápido a mayor k.",
    )]


def build(seed: int) -> dict:
    import deepxde as dde

    dde.config.set_random_seed(int(seed))
    geom = dde.geometry.Interval(K_MIN, K_MAX)          # the "spatial" axis is the rate constant k
    timedomain = dde.geometry.TimeDomain(0.0, 1.0)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    def pde(x, C):
        dC_t = dde.grad.jacobian(C, x, i=0, j=1)         # dC/dt  (input 1 = t)
        k = x[:, 0:1]                                    # input 0 = k (parameter)
        return dC_t + k * C

    net = dde.nn.FNN([2] + [32] * 3 + [1], "tanh", "Glorot normal")
    # hard IC: C(k, 0) = 1 exactly (t factor vanishes the net contribution at t=0)
    net.apply_output_transform(lambda x, y: 1.0 + x[:, 1:2] * y)

    t = CASE.train
    data = dde.data.TimePDE(
        geomtime, pde, [],
        num_domain=t["num_domain"], num_boundary=t["num_boundary"],
        num_initial=t["num_initial"], solution=analytic, num_test=t["num_test"],
    )
    model = dde.Model(data, net)
    model.compile("adam", lr=t["lr"], metrics=["l2 relative error"])
    return {"model": model, "input_dim": 2}
