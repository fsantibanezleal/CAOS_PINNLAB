"""CONTRACT 2 (artifact) tests: the manifest is a complete, self-consistent record of a baked PINN case — it points
to a real artifact with the recorded byte size, carries the governing equation + method + engine, records the ONNX
pointer + parity, and the lane verdict agrees with the gate."""
from pinnlab import pipeline

PILOT = "bench-poisson2d"


def test_manifest_is_complete_and_consistent():
    m = pipeline.precompute(PILOT, seed=7, quick=True)
    assert m["schema"].startswith("pinnlab.manifest/")
    # provenance every numeric claim needs
    for key in ("governing_equations", "method", "real_or_synthetic", "validation_anchor"):
        assert m[key], f"manifest missing {key}"
    assert m["engine"]["framework"] == "deepxde"
    # ONNX bridge recorded (poisson2d is parametric: inputs (x,y,k) -> input_dim 3)
    assert m["onnx"]["opset"] == 18 and m["onnx"]["input_dim"] >= 2
    assert 0.0 <= m["onnx"]["parity_max_abs"] < 1e-4
    # manifest/v2: at least one baked variant, each with a field trace
    assert m["variants"] and all(v["trace"]["path"] for v in m["variants"])
    # lane verdict consistent with the measured gate
    assert m["lane"] == m["gate"]["lane"]
    assert m["gate"]["onnx_bytes"] == m["onnx"]["bytes"]
