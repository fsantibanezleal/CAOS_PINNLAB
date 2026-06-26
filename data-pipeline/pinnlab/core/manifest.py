"""CONTRACT 2 — artifact (pipeline -> web). The manifest is the authoritative, versioned record of a baked PINN
case: its governing equation, the SOTA method + engine, the tunable parameters, the exported ONNX pointer + parity,
the lane/gate verdict, and a FAMILY OF VARIANTS (parameter regimes) — each with its own compact replay-artifact
pointer + per-regime metrics. The web loads ONLY manifests + artifacts; frontend/src/lib/contract.ts mirrors this
schema so a drift fails the web build. A flat index.json inventories every case (ADR-0057 default).

Schema v2 (SimLab manifest/v2 analogue): the per-case record carries `variants: [...]` (≥1) instead of a single
field, so the App offers a chip selector + side-by-side comparison and (for parametric cases) a `Live` slider.
"""
from __future__ import annotations

from typing import Any

from .. import __version__
from .trace import TRACE_SCHEMA

MANIFEST_SCHEMA = "pinnlab.manifest/v2"
INDEX_SCHEMA = "pinnlab.index/v1"


def build_case_manifest(
    *,
    case: Any,
    seed: int,
    onnx: dict,
    gate: dict,
    variants: list[dict],
    run_ms: float,
    flags: list[dict] | None = None,
    inverse: dict | None = None,
) -> dict:
    """`case` is a CaseSpec; `onnx` = {path, bytes, parity_max_abs, opset, input_dim}; `variants` = the per-regime
    entries (id, labels, notes, params, trace {path,bytes}, metrics) assembled by the pipeline."""
    return {
        "schema": MANIFEST_SCHEMA,
        "case_id": case.id,
        "category": case.category,
        "title": case.title,
        "governing_equations": case.governing_equations,
        "method": case.method,
        "real_or_synthetic": case.real_or_synthetic,
        "expected_band": case.expected_band,
        "validation_anchor": case.validation_anchor,
        "inputs": list(case.inputs),
        "outputs": list(case.outputs),
        "field_axes": list(case.axes),
        "system_type": case.system_type,   # ADR-0063: kind-of-system axis (Experiments still group by `category`)
        "view_kit": case.kit,              # which web kit renders Field/Live (default "HeatmapKit")
        "param_specs": [
            {
                "key": p.key, "label_en": p.label_en, "label_es": p.label_es,
                "default": p.default, "min": p.min, "max": p.max, "step": p.step, "unit": p.unit,
            }
            for p in case.param_specs
        ],
        "engine": {"package": "pinnlab", "version": __version__, "framework": case.engine},
        "seed": seed,
        "trace_schema": TRACE_SCHEMA,
        "onnx": onnx,
        "lane": gate["lane"],
        "gate": gate,
        "flags": flags or [],
        "inverse": inverse or {},
        "run_ms": round(run_ms, 2),
        "variants": variants,
    }


def build_index(entries: list[dict]) -> dict:
    """entries: [{case_id, category, title, manifest_path}] -> the flat authoritative inventory."""
    return {
        "schema": INDEX_SCHEMA,
        "engine_version": __version__,
        "n_cases": len(entries),
        "categories": sorted({e["category"] for e in entries}),
        "cases": sorted(entries, key=lambda e: e["case_id"]),
    }
