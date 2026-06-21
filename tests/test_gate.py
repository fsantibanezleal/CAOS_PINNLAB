"""The measured lane GATE (ADR-0054, PINN-Lab variant): a case is LIVE iff its ONNX is small, its ort-web-proxy
inference is fast, and its replay artifact is small — otherwise PRECOMPUTE. Pure logic; no training required."""
from pinnlab.core.gate import INFER_MS_GATE, ONNX_BYTES_GATE, TRACE_BYTES_GATE, classify_lane


def test_small_fast_case_is_live():
    g = classify_lane(onnx_bytes=20_000, infer_ms=3.0, trace_bytes=50_000)
    assert g["lane"] == "live" and not g["reasons"]


def test_oversized_onnx_falls_back_to_precompute():
    g = classify_lane(onnx_bytes=ONNX_BYTES_GATE + 1, infer_ms=3.0, trace_bytes=50_000)
    assert g["lane"] == "precompute" and any("onnx_bytes" in r for r in g["reasons"])


def test_slow_or_heavy_artifact_falls_back():
    g = classify_lane(onnx_bytes=20_000, infer_ms=INFER_MS_GATE + 1, trace_bytes=TRACE_BYTES_GATE + 1)
    assert g["lane"] == "precompute"
    assert any("infer_ms" in r for r in g["reasons"]) and any("trace_bytes" in r for r in g["reasons"])
