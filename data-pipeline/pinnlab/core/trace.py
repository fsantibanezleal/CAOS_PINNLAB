"""The compact TRACE = the web-replay artifact (a decimated solution field + summary). Part of CONTRACT 2: its
shape is mirrored by frontend/src/lib/contract.types.ts, so a drift fails the web build. Schema id is versioned.

A PINN trace bakes the trained solution sampled on a coarse grid (the field the SPA renders for first paint and as
the replay fallback). The live lane re-evaluates the exported ONNX at arbitrary resolution on top of this.
"""
from __future__ import annotations

import numpy as np

from ..io.schema import SolutionField

TRACE_SCHEMA = "pinnlab.field/v1"
MAX_AXIS = 81  # decimate each axis so the committed artifact stays small (replay, not the raw high-res field)


def _decimate_index(n: int, cap: int = MAX_AXIS) -> list[int]:
    if n <= cap:
        return list(range(n))
    return [round(i * (n - 1) / (cap - 1)) for i in range(cap)]


def build_trace(sf: SolutionField, *, value_round: int = 5) -> dict:
    """Decimate a SolutionField to a compact, JSON-serializable replay artifact. A 1-D trace (an ODE/trajectory
    case: dims = ("t",)) keeps more samples than a 2-D heatmap — 81 points is too coarse for a smooth animated
    trajectory, whereas an 81x81 field is already 6561 cells."""
    cap = 601 if len(sf.dims) <= 1 else MAX_AXIS
    idx = {name: _decimate_index(len(coords), cap) for name, coords in sf.axes.items()}
    axes_out = {name: [round(float(sf.axes[name][i]), 5) for i in ix] for name, ix in idx.items()}

    # gather the decimation indices along the field's dims order
    dim_idx = [idx[d] for d in sf.dims]
    fields_out: dict[str, list] = {}
    for name, arr in sf.fields.items():
        sub = arr
        for axis, ii in enumerate(dim_idx):
            sub = np.take(sub, ii, axis=axis)
        fields_out[name] = np.round(sub.astype(float), value_round).tolist()

    return {
        "schema": TRACE_SCHEMA,
        "case_id": sf.case_id,
        "dims": list(sf.dims),
        "axes": axes_out,
        "fields": fields_out,
        "summary": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in sf.scalars.items()},
        "inverse": sf.inverse,
    }
