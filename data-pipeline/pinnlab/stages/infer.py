"""Stage 4 — infer: evaluate the trained PINN over the case's FIELD grid -> a SolutionField (raw, undecimated; the
trace stage decimates it for the compact web artifact), at a given parameter regime (variant). For a PARAMETRIC case
the network input includes the parameter axes; `param_grid` fills them with the variant's constant values, so the
baked field is the 2-D heatmap over the field axes at that regime (the web `Live` tab sweeps the parameter via the
shared ONNX). A case may define `eval_grid()` (e.g. to match a reference dataset's grid) for the non-parametric path.
Multi-output cases (e.g. the cavity (u,v,p)) reshape each output column onto the grid."""
from __future__ import annotations

import numpy as np

from ..io.schema import SolutionField
from ..model.analytic import param_grid
from ..registry import case_module, get_case


def run(case_id: str, model, params: dict | None = None) -> SolutionField:
    case = get_case(case_id)
    mod = case_module(case_id)
    if (not case.param_specs) and getattr(mod, "eval_grid", None) is not None:
        coords, XY, shape = mod.eval_grid()
    else:
        coords, XY, shape = param_grid(case, params)
    pred = np.asarray(model.predict(XY), dtype=np.float64)
    if pred.ndim == 1:
        pred = pred[:, None]
    fields = {name: pred[:, k].reshape(shape) for k, name in enumerate(case.outputs)}
    return SolutionField(
        case_id=case_id,
        dims=tuple(case.axes),
        axes={a: [float(v) for v in coords[a]] for a in case.axes},
        fields=fields,
        scalars={},
    )
