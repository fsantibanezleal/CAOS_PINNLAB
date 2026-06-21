"""Stage 4 — infer: evaluate the trained PINN over the case's eval grid -> a SolutionField (raw, undecimated; the
trace stage decimates it for the compact web artifact). A case may define `eval_grid()` (e.g. to match a reference
dataset's grid exactly); otherwise the uniform linspace grid from the CaseSpec is used. Multi-output cases (e.g.
the cavity (u,v,p)) reshape each output column onto the grid."""
from __future__ import annotations

import numpy as np

from ..io.schema import SolutionField
from ..model.analytic import linspace_grid
from ..registry import case_module, get_case


def run(case_id: str, model) -> SolutionField:
    case = get_case(case_id)
    mod = case_module(case_id)
    if getattr(mod, "eval_grid", None) is not None:
        coords, XY, shape = mod.eval_grid()
    else:
        coords, XY, shape = linspace_grid(case.domain, case.grid)
    pred = np.asarray(model.predict(XY), dtype=np.float64)
    if pred.ndim == 1:
        pred = pred[:, None]
    fields = {name: pred[:, k].reshape(shape) for k, name in enumerate(case.outputs)}
    return SolutionField(
        case_id=case_id,
        dims=tuple(case.inputs),
        axes={a: [float(v) for v in coords[a]] for a in case.inputs},
        fields=fields,
        scalars={},
    )
