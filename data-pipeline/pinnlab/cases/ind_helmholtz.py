"""Group D · industrial-fluids-heat — 2D Helmholtz (frequency domain), FOURIER-FEATURE PINN (spectral-bias showcase).

Governing equation (DeepXDE sign convention):
    -u_xx - u_yy - k0^2 u - f = 0  on (0,1)^2,  u=0 on the boundary,  k0 = 2 pi n (n=3),
    f = k0^2 sin(k0 x) sin(k0 y)  chosen so the MMS solution u* = sin(k0 x) sin(k0 y) is exact.

Method — random FOURIER-FEATURE input embedding (Tancik 2020; Wang-Wang-Perdikaris multi-scale 2021): the spatial
frequency makes a plain tanh MLP suffer spectral bias; a frozen Gaussian Fourier map injects the frequencies into
layer 1. SOFT Dirichlet BC with loss weighting (the robust recipe for oscillatory solutions — a multiplicative hard
constraint fights the oscillation near the boundary). The Fourier map is a pure-tensor apply_feature_transform that
traces into ONNX. n=3 (k0=6pi) is high enough to need the Fourier map, low enough to converge on CPU.
"""
from __future__ import annotations

import numpy as np

from .base import CaseSpec

N_WAVES = 3
K0 = 2.0 * np.pi * N_WAVES
N_FEATURES = 64
SIGMAS = (1.0, float(N_WAVES))

CASE = CaseSpec(
    id="ind-helmholtz",
    category="industrial-fluids-heat",
    title="2D Helmholtz (high-wavenumber) — Fourier-feature PINN",
    governing_equations=(
        r"\nabla^2 u + k_0^2 u = -f,\ k_0=2\pi n,\ n=3,\ \text{on}\ (0,1)^2,\ u|_{\partial\Omega}=0,\ "
        r"u^*=\sin(k_0 x)\sin(k_0 y)"
    ),
    method="fourier-features",
    engine="deepxde",
    real_or_synthetic="synthetic",
    inputs=("x", "y"),
    outputs=("u",),
    domain={"x": (0.0, 1.0), "y": (0.0, 1.0)},
    grid={"x": 121, "y": 121},
    expected_band="high-frequency standing pattern; relative-L2 vs MMS analytic ~1e-1 on the CPU lane (Fourier features lift the spectral-bias plateau; GPU + frequency annealing tighten it further)",
    validation_anchor="analytic",
    train={"lr": 1e-3, "adam": 25000, "lbfgs": True, "num_test": 4000, "loss_weights": [1, 100]},
    notes="Fourier-feature embedding (frozen Gaussian B, seeded) + soft Dirichlet BC with 100x weight; the feature map traces into ONNX.",
)


def analytic(xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float64)
    return np.sin(K0 * xy[:, 0:1]) * np.sin(K0 * xy[:, 1:2])


def build(seed: int) -> dict:
    import deepxde as dde
    import torch

    dde.config.set_random_seed(int(seed))
    geom = dde.geometry.Rectangle([0.0, 0.0], [1.0, 1.0])

    def pde(x, y):
        dy_xx = dde.grad.hessian(y, x, i=0, j=0)
        dy_yy = dde.grad.hessian(y, x, i=1, j=1)
        f = K0 ** 2 * torch.sin(K0 * x[:, 0:1]) * torch.sin(K0 * x[:, 1:2])
        return -dy_xx - dy_yy - K0 ** 2 * y - f

    torch.manual_seed(0)  # frozen Fourier matrix B -> identical for train, parity, ONNX export
    b_list = [torch.randn(2, N_FEATURES) * s for s in SIGMAS]

    def fourier_features(x):
        feats = []
        for b in b_list:
            proj = 2.0 * np.pi * (x @ b.to(x.dtype))
            feats.append(torch.sin(proj))
            feats.append(torch.cos(proj))
        return torch.cat(feats, dim=1)

    in_dim = 2 * N_FEATURES * len(SIGMAS)
    net = dde.nn.FNN([in_dim] + [128] * 4 + [1], "tanh", "Glorot uniform")
    net.apply_feature_transform(fourier_features)

    bc = dde.icbc.DirichletBC(geom, lambda x: 0.0, lambda _, on_boundary: on_boundary)

    nx = int(12 * N_WAVES)  # ~12 collocation points per wavelength per axis
    t = CASE.train
    data = dde.data.PDE(
        geom, pde, [bc],
        num_domain=nx * nx, num_boundary=4 * nx, solution=analytic, num_test=t["num_test"],
    )
    model = dde.Model(data, net)
    model.compile("adam", lr=t["lr"], loss_weights=t["loss_weights"], metrics=["l2 relative error"])
    return {"model": model, "input_dim": 2}
