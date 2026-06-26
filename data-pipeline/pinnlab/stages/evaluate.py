"""Stage 5 — evaluate (the TEST stage): relative-L2 of the PINN field vs the validation anchor (analytic / reference
dataset / numerical), plus the ONNX-vs-model parity (the train->web contract check). Evaluated AT the variant's parameter
regime (for a parametric case the analytic anchor is computed on the same param-filled grid as the baked field). A
case may additionally expose `extra_metrics(sf)` for a bespoke anchor (e.g. Ghia cavity, held-out real data).
Leakage-safe: the reference is the closed-form/dataset/numerical truth, never PINN output."""
from __future__ import annotations

import numpy as np

from ..model.analytic import l2_relative, max_abs_error, param_grid
from ..registry import case_module, get_case


def _eval_grid(case, mod, params):
    if (not case.param_specs) and getattr(mod, "eval_grid", None) is not None:
        return mod.eval_grid()
    return param_grid(case, params)


def _truth_and_pred(case, mod, sf, params):
    primary = case.outputs[0]
    pred = sf.fields[primary].reshape(-1, 1)
    if case.validation_anchor == "analytic" and getattr(mod, "analytic", None) is not None:
        _coords, XY, _shape = _eval_grid(case, mod, params)
        truth = mod.analytic(XY)
        return (truth.reshape(-1, 1) if truth is not None else None), pred
    if case.validation_anchor in ("dataset", "fem-ref") and getattr(mod, "reference_on_grid", None) is not None:
        truth = np.asarray(mod.reference_on_grid(), dtype=np.float64).reshape(-1, 1)
        return truth, pred
    return None, pred


def run(case_id: str, sf, onnx_info: dict, params: dict | None = None) -> dict:
    case = get_case(case_id)
    mod = case_module(case_id)
    metrics: dict = {
        "onnx_parity_max_abs": round(float(onnx_info.get("parity_max_abs", 0.0)), 9),
        "validation_anchor": case.validation_anchor,
    }
    truth, pred = _truth_and_pred(case, mod, sf, params or {})
    if truth is not None:
        metrics["l2_relative"] = round(l2_relative(pred, truth), 6)
        metrics["max_abs_error"] = round(max_abs_error(pred, truth), 6)
    if getattr(mod, "extra_metrics", None) is not None:
        metrics.update(mod.extra_metrics(sf))
    sf.scalars.update({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
    return metrics
