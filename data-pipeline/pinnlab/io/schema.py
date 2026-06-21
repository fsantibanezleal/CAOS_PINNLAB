"""Typed objects passed between pipeline stages — the inter-stage contract.

Pure dataclasses + numpy ONLY (Pyodide-safe): never import torch/deepxde here. The heavy PINN engine lives in
`stages/train.py` and each case's `build()`; the light metadata (CaseSpec) lives in `cases/`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ObservationRow:
    """One validated observation datum for an INVERSE case (the bring-your-own-data path).

    Forward cases need no external data (the PDE + BC/IC fully determine the solution); inverse cases assimilate
    sparse, possibly noisy measurements of the field to recover unknown coefficients. `coords` follows the case's
    input-axis order (e.g. (x, y) or (x, t)).
    """

    case_id: str
    coords: tuple[float, ...]
    value: float
    weight: float = 1.0


@dataclass
class SolutionField:
    """The engine output for one case (infer stage): the solution sampled on an evaluation grid + scalars.

    - `dims`   : ordered axis names of the meshgrid (e.g. ("y", "x") so `fields[name]` indexes [iy, ix]).
    - `axes`   : input-axis name -> 1D coordinate list (e.g. {"x": [...], "y": [...]}).
    - `fields` : output name -> ndarray on the meshgrid (shape = tuple(len(axes[d]) for d in dims)).
    - `scalars`: case-level metrics (e.g. {"l2_relative": ..., "max_abs_residual": ...}).
    - `inverse`: recovered coefficients for inverse cases (name -> {"value", "truth", "rel_err"}).
    """

    case_id: str
    dims: tuple[str, ...]
    axes: dict[str, list[float]]
    fields: dict[str, np.ndarray]
    scalars: dict[str, Any] = field(default_factory=dict)
    inverse: dict[str, Any] = field(default_factory=dict)
