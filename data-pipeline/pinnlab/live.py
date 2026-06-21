"""LIVE-lane note (PINN-Lab): the interactive live lane is **onnxruntime-web evaluating the exported `.onnx` in the
BROWSER** (see frontend/src/engine/) — NOT Pyodide running DeepXDE. This module provides only a Pyodide-safe ANALYTIC
fallback: for cases with a closed-form reference it renders the exact field without any engine — used as a teaching
overlay ("show the exact solution") and an offline sanity check. It imports ONLY the numpy core (no torch/deepxde)."""
from __future__ import annotations

from .core.trace import build_trace
from .io.schema import SolutionField
from .model.analytic import linspace_grid
from .registry import case_module, get_case


def analytic_trace(case_id: str) -> dict:
    """Render a case's closed-form reference field as a trace (Pyodide-safe). Raises if the case has no closed form."""
    case = get_case(case_id)
    mod = case_module(case_id)
    analytic = getattr(mod, "analytic", None)
    if analytic is None:
        raise ValueError(f"{case_id} has no closed-form analytic solution")
    coords, XY, shape = linspace_grid(case.domain, case.grid)
    truth = analytic(XY)
    if truth is None:
        raise ValueError(f"{case_id} analytic() returned None")
    primary = case.outputs[0]
    sf = SolutionField(
        case_id=case_id,
        dims=tuple(case.inputs),
        axes={a: [float(v) for v in coords[a]] for a in case.inputs},
        fields={primary: truth.reshape(shape)},
        scalars={"source": "analytic"},
    )
    return build_trace(sf)
