"""Stage 6 — export (CONTRACT 2): write the compact replay artifact (the decimated field) + the case manifest. The
manifest records the lane/gate verdict (ONNX size + ort-web infer time + artifact bytes), the ONNX pointer + parity,
the CONTRACT-1 flags (inverse cases) and the evaluation metrics."""
from __future__ import annotations

from pathlib import Path

from ..core.gate import classify_lane
from ..core.manifest import build_case_manifest
from ..core.trace import build_trace
from ..io.formats import write_json


def run(
    *,
    case,
    sf,
    onnx_info: dict,
    metrics: dict,
    seed: int,
    run_ms: float,
    derived_dir: str,
    manifests_dir: str,
    flags: list[dict] | None = None,
    inverse: dict | None = None,
) -> dict:
    trace = build_trace(sf)
    artifact_rel = f"{case.id}/field.json"
    trace_bytes = write_json(Path(derived_dir) / artifact_rel, trace)
    gate = classify_lane(
        onnx_bytes=int(onnx_info["onnx_bytes"]),
        infer_ms=float(onnx_info.get("infer_ms", 0.0)),
        trace_bytes=trace_bytes,
    )
    onnx_meta = {
        "path": Path(onnx_info["onnx_path"]).name,
        "bytes": int(onnx_info["onnx_bytes"]),
        "parity_max_abs": round(float(onnx_info["parity_max_abs"]), 9),
        "opset": int(onnx_info.get("opset", 18)),
        "input_dim": int(onnx_info["input_dim"]),
    }
    manifest = build_case_manifest(
        case=case, seed=seed, run_ms=run_ms, artifact_rel=artifact_rel, trace_bytes=trace_bytes,
        gate=gate, onnx=onnx_meta, metrics=metrics, flags=flags, inverse=inverse,
    )
    write_json(Path(manifests_dir) / f"{case.id}.json", manifest)
    return manifest
