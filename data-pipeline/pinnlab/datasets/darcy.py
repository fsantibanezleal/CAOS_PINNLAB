"""Generate the Darcy-flow operator-learning dataset (the Li et al. FNO benchmark), in-build + seeded.

Steady Darcy flow on the unit square:  -div(a(x) grad u(x)) = f,  u|_boundary = 0,  with f == 1.
The coefficient field a(x) is a TWO-VALUE (piecewise-constant) thresholded Gaussian random field (a low-pass-filtered
white-noise field thresholded at 0 -> {3, 12}), giving sharp material interfaces — exactly the canonical benchmark.
u is obtained from a reference finite-difference solve (5-point scheme, harmonic-mean face conductivities, sparse
direct solve). Pure NumPy/SciPy, deterministic given the seed; no vendored file — the pairs are regenerated on demand.
"""
from __future__ import annotations

import numpy as np


def _make_coeff(rng: np.random.Generator, n: int, sigma: float = 3.0, lo: float = 3.0, hi: float = 12.0) -> np.ndarray:
    from scipy.ndimage import gaussian_filter

    g = gaussian_filter(rng.standard_normal((n, n)), sigma=sigma)
    return np.where(g > 0.0, hi, lo)


def _solve_darcy(a: np.ndarray) -> np.ndarray:
    """Solve -div(a grad u) = 1, u=0 on the boundary, on the n x n grid; harmonic-mean faces, sparse direct."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    n = a.shape[0]
    h = 1.0 / (n - 1)
    N = n * n
    def idx(i, j):
        return i * n + j
    A = sp.lil_matrix((N, N))
    b = np.zeros(N)
    for i in range(n):
        for j in range(n):
            k = idx(i, j)
            if i in (0, n - 1) or j in (0, n - 1):
                A[k, k] = 1.0
                continue
            aC = a[i, j]
            aE = 2 * aC * a[i, j + 1] / (aC + a[i, j + 1])
            aW = 2 * aC * a[i, j - 1] / (aC + a[i, j - 1])
            aN = 2 * aC * a[i + 1, j] / (aC + a[i + 1, j])
            aS = 2 * aC * a[i - 1, j] / (aC + a[i - 1, j])
            A[k, k] = (aE + aW + aN + aS) / h ** 2
            b[k] = 1.0
            A[k, idx(i, j + 1)] = -aE / h ** 2
            A[k, idx(i, j - 1)] = -aW / h ** 2
            A[k, idx(i + 1, j)] = -aN / h ** 2
            A[k, idx(i - 1, j)] = -aS / h ** 2
    u = spla.spsolve(A.tocsr(), b)
    return u.reshape(n, n)


def make_dataset(seed: int, n_grid: int, n_train: int, n_test: int):
    """Return (Xtr, Ytr, Xte, Yte, stats) as float32 torch tensors. X = [B,3,H,W] (a_norm, x-grid, y-grid),
    Y = [B,1,H,W] (u_norm). stats holds the a/u normalization constants (for de-normalizing the baked fields)."""
    import torch

    rng = np.random.default_rng(seed)
    ntot = n_train + n_test
    a_list = [_make_coeff(rng, n_grid) for _ in range(ntot)]
    u_list = [_solve_darcy(a) for a in a_list]
    a_arr = np.asarray(a_list)[:, None]          # [N,1,H,W]
    u_arr = np.asarray(u_list)[:, None]
    a_mu, a_sd = float(a_arr.mean()), float(a_arr.std())
    u_mu, u_sd = float(u_arr.mean()), float(u_arr.std())
    a_n = (a_arr - a_mu) / a_sd
    u_n = (u_arr - u_mu) / u_sd
    gx, gy = np.meshgrid(np.linspace(0, 1, n_grid), np.linspace(0, 1, n_grid), indexing="ij")
    grid = np.stack([gx, gy])[None].repeat(ntot, 0)
    X = np.concatenate([a_n, grid], axis=1).astype(np.float32)
    Y = u_n.astype(np.float32)
    stats = {"a_mu": a_mu, "a_sd": a_sd, "u_mu": u_mu, "u_sd": u_sd}
    def t(z):
        return torch.as_tensor(z)
    return t(X[:n_train]), t(Y[:n_train]), t(X[n_train:]), t(Y[n_train:]), stats, (a_arr, u_arr)
