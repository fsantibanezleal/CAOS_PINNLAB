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

from .base import CaseSpec, Variant

N_WAVES = 3
K0 = 2.0 * np.pi * N_WAVES
N_FEATURES = 64
SIGMAS = (1.0, float(N_WAVES))

CASE = CaseSpec(
    id="ind-helmholtz",
    category="industrial-fluids-heat",
    title="2D Helmholtz (high-wavenumber): Fourier-feature PINN",
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


def variants() -> list[Variant]:
    # Single honest benchmark at a fixed high wavenumber: parametric-in-n is rejected (one frozen Fourier map can't
    # span a wavenumber band on the CPU lane without fabricating regimes). n=3 is the showcase of the method itself.
    return [Variant(
        "n3", "n=3 (k0=6π)", "n=3 (k0=6π)", {},
        "Fixed high-wavenumber standing wave; Fourier features defeat the spectral-bias plateau.",
        "Onda estacionaria de número de onda alto fijo; las características de Fourier vencen la meseta de sesgo espectral.",
    )]


def _helmholtz_model(seed: int, *, fourier: bool, n_waves: int = N_WAVES):
    """Build a Helmholtz PINN. `fourier=True` is the ADAPTED lane (random Fourier-feature input embedding); with
    `fourier=False` it is the NAIVE lane — the SAME architecture on raw (x,y), so its spectral-bias failure at high
    wavenumber is real, not staged. `n_waves` lets the diagnostics sweep the wavenumber."""
    import deepxde as dde
    import torch

    dde.config.set_random_seed(int(seed))
    k0 = 2.0 * np.pi * n_waves
    geom = dde.geometry.Rectangle([0.0, 0.0], [1.0, 1.0])

    def pde(x, y):
        dy_xx = dde.grad.hessian(y, x, i=0, j=0)
        dy_yy = dde.grad.hessian(y, x, i=1, j=1)
        f = k0 ** 2 * torch.sin(k0 * x[:, 0:1]) * torch.sin(k0 * x[:, 1:2])
        return -dy_xx - dy_yy - k0 ** 2 * y - f

    def sol(xy):
        xy = np.asarray(xy, dtype=np.float64)
        return np.sin(k0 * xy[:, 0:1]) * np.sin(k0 * xy[:, 1:2])

    if fourier:
        torch.manual_seed(0)  # frozen Fourier matrix B -> identical for train, parity, ONNX export
        b_list = [torch.randn(2, N_FEATURES) * s for s in (1.0, float(n_waves))]

        def fourier_features(x):
            feats = []
            for b in b_list:
                proj = 2.0 * np.pi * (x @ b.to(x.dtype))
                feats.append(torch.sin(proj))
                feats.append(torch.cos(proj))
            return torch.cat(feats, dim=1)

        in_dim = 2 * N_FEATURES * 2
        net = dde.nn.FNN([in_dim] + [128] * 4 + [1], "tanh", "Glorot uniform")
        net.apply_feature_transform(fourier_features)
    else:
        # NAIVE: a plain tanh MLP directly on (x,y) — no frequency injection -> spectral bias.
        net = dde.nn.FNN([2] + [128] * 4 + [1], "tanh", "Glorot uniform")

    bc = dde.icbc.DirichletBC(geom, lambda x: 0.0, lambda _, on_boundary: on_boundary)
    nx = int(12 * n_waves)  # ~12 collocation points per wavelength per axis
    t = CASE.train
    data = dde.data.PDE(
        geom, pde, [bc],
        num_domain=nx * nx, num_boundary=4 * nx, solution=sol, num_test=t["num_test"],
    )
    model = dde.Model(data, net)
    model.compile("adam", lr=t["lr"], loss_weights=t["loss_weights"], metrics=["l2 relative error"])
    return model


def build(seed: int) -> dict:
    """The ADAPTED lane (Fourier features) — the case's shipped method."""
    return {"model": _helmholtz_model(seed, fourier=True), "input_dim": 2}


def build_naive(seed: int) -> dict:
    """The NAIVE lane (plain tanh MLP) — run so the spectral-bias failure is visible next to the standard."""
    return {"model": _helmholtz_model(seed, fourier=False), "input_dim": 2}


def standard_field(coords) -> np.ndarray:
    """The STANDARD PDE solution as a classical 5-point finite-difference Helmholtz solve on the case grid:
    -u_xx - u_yy - k0^2 u = f, u=0 on the boundary. A real classical solver (with its own dispersion error at high
    k0), the ground-truth lane every PINN is compared against."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    x = np.asarray(coords["x"], dtype=np.float64)
    y = np.asarray(coords["y"], dtype=np.float64)
    nx, ny = len(x), len(y)
    hx = float(x[1] - x[0])
    hy = float(y[1] - y[0])
    X, Y = np.meshgrid(x, y, indexing="ij")
    f = (K0 ** 2) * np.sin(K0 * X) * np.sin(K0 * Y)
    ix = np.arange(1, nx - 1)
    iy = np.arange(1, ny - 1)
    ni, nj = len(ix), len(iy)
    N = ni * nj

    def idx(i, j):
        return i * nj + j

    rows, cols, vals = [], [], []
    b = np.zeros(N)
    for a, i in enumerate(ix):
        for bb, j in enumerate(iy):
            r = idx(a, bb)
            rows.append(r); cols.append(r); vals.append(2.0 / hx ** 2 + 2.0 / hy ** 2 - K0 ** 2)
            if a > 0:
                rows.append(r); cols.append(idx(a - 1, bb)); vals.append(-1.0 / hx ** 2)
            if a < ni - 1:
                rows.append(r); cols.append(idx(a + 1, bb)); vals.append(-1.0 / hx ** 2)
            if bb > 0:
                rows.append(r); cols.append(idx(a, bb - 1)); vals.append(-1.0 / hy ** 2)
            if bb < nj - 1:
                rows.append(r); cols.append(idx(a, bb + 1)); vals.append(-1.0 / hy ** 2)
            b[r] = f[i, j]
    A = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))
    u_int = spla.spsolve(A, b)
    u = np.zeros((nx, ny))
    u[np.ix_(ix, iy)] = u_int.reshape(ni, nj)
    return u
