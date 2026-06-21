"""Stage 5 — evaluate (the TEST stage): relative-L2 of the PINN field vs the analytic/FEM reference on the eval
grid, plus the ONNX-vs-model parity (the train->web contract check). Leakage-safe: the reference is the closed-form
solution (or an external FEM result), never PINN output."""
from __future__ import annotations

from ..model.analytic import l2_relative, linspace_grid, max_abs_error
from ..registry import case_module, get_case


def run(case_id: str, sf, onnx_info: dict) -> dict:
    case = get_case(case_id)
    mod = case_module(case_id)
    metrics: dict = {
        "onnx_parity_max_abs": round(float(onnx_info.get("parity_max_abs", 0.0)), 9),
        "validation_anchor": case.validation_anchor,
    }
    analytic = getattr(mod, "analytic", None)
    if case.validation_anchor == "analytic" and analytic is not None:
        _coords, XY, _shape = linspace_grid(case.domain, case.grid)
        truth = analytic(XY)
        if truth is not None:
            primary = case.outputs[0]
            pred = sf.fields[primary].reshape(-1, 1)
            metrics["l2_relative"] = round(l2_relative(pred, truth), 6)
            metrics["max_abs_error"] = round(max_abs_error(pred, truth), 6)
    sf.scalars.update({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
    return metrics
