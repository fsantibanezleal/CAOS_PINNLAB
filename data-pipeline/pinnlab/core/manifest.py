"""CONTRACT 2 — artifact (pipeline -> web). The manifest is the authoritative, versioned record of a baked PINN
case: its governing equation, the SOTA method + engine that solved it, the compact replay-artifact pointer + byte
size, the exported ONNX pointer + the ONNX-vs-model parity check, the lane/gate verdict, CONTRACT-1 flags (inverse
cases), and the evaluation metrics (relative-L2 vs the analytic/FEM reference). The web loads ONLY manifests +
artifacts; frontend/src/lib/contract.types.ts mirrors this schema so a drift fails the web build. A flat index.json
inventories every case (ADR-0057 default).
"""
from __future__ import annotations

from typing import Any

from .. import __version__
from .trace import TRACE_SCHEMA

MANIFEST_SCHEMA = "pinnlab.manifest/v1"
INDEX_SCHEMA = "pinnlab.index/v1"


def build_case_manifest(
    *,
    case: Any,
    seed: int,
    run_ms: float,
    artifact_rel: str,
    trace_bytes: int,
    gate: dict,
    onnx: dict,
    metrics: dict,
    flags: list[dict] | None = None,
    inverse: dict | None = None,
) -> dict:
    """`case` is a CaseSpec; `onnx` = {path, bytes, parity_max_abs, opset, input_dim}; `metrics` from evaluate."""
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
        "engine": {
            "package": "pinnlab",
            "version": __version__,
            "framework": case.engine,            # deepxde | physicsnemo | neuraloperator
        },
        "seed": seed,
        "artifact": {"path": artifact_rel, "format": "json", "trace_schema": TRACE_SCHEMA, "bytes": trace_bytes},
        "onnx": onnx,
        "lane": gate["lane"],
        "gate": gate,
        "flags": flags or [],
        "inverse": inverse or {},
        "metrics": metrics,
        "run_ms": round(run_ms, 2),
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
