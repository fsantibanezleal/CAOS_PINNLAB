"""Pure-Python (numpy-only) numeric helpers shared by the offline stages, the live lane and the API — Pyodide-safe.

The per-case closed-form reference solutions live in each `cases/<case>.py` (`analytic()`); this module holds the
domain-agnostic helpers: evaluation grids, relative-L2 error vs the reference, and max-abs error. NEVER import
torch/deepxde here — this is the light core that may run in more than one lane.
"""
from __future__ import annotations

import numpy as np


def linspace_grid(domain: dict[str, tuple[float, float]], res: dict[str, int]):
    """Build per-axis coordinate vectors + the flattened [N, d] query matrix (C-order over the named axes).

    Returns (coords1d: {axis -> 1D np.ndarray}, XY: [N, d] np.ndarray, shape: tuple[int, ...]).
    The flatten order matches `np.meshgrid(..., indexing="ij")` so a field reshaped to `shape` then `ravel()`ed
    aligns row-for-row with XY (used by infer/evaluate).
    """
    names = list(domain.keys())
    coords = {n: np.linspace(domain[n][0], domain[n][1], res[n]) for n in names}
    mesh = np.meshgrid(*[coords[n] for n in names], indexing="ij")
    XY = np.stack([m.ravel() for m in mesh], axis=1).astype(np.float64)
    shape = tuple(res[n] for n in names)
    return coords, XY, shape


def l2_relative(pred: np.ndarray, truth: np.ndarray) -> float:
    """Relative L2 error ||pred - truth|| / ||truth||. If truth ≡ 0 (degenerate control), return ||pred||."""
    p = np.asarray(pred, dtype=np.float64).ravel()
    t = np.asarray(truth, dtype=np.float64).ravel()
    denom = float(np.linalg.norm(t))
    if denom == 0.0:
        return float(np.linalg.norm(p))
    return float(np.linalg.norm(p - t) / denom)


def max_abs_error(pred: np.ndarray, truth: np.ndarray) -> float:
    p = np.asarray(pred, dtype=np.float64).ravel()
    t = np.asarray(truth, dtype=np.float64).ravel()
    return float(np.max(np.abs(p - t)))
