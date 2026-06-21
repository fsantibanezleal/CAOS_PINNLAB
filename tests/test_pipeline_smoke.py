"""Pipeline smoke (quick path): the canonical pilot case runs through all six stages, exports a faithful ONNX
(parity vs model.predict), classifies a lane from measurements, and writes a consistent artifact + manifest. The
`quick` path uses few iterations — it checks the train->ONNX->web plumbing + contracts, NOT convergence accuracy
(accuracy is asserted separately by the offline precompute + the Benchmark page)."""
import json

from pinnlab import pipeline, registry

PILOT = "bench-poisson2d"


def test_pilot_runs_and_exports_faithful_onnx():
    m = pipeline.precompute(PILOT, seed=7, quick=True)
    # the exported ONNX must match the trained net (the train->web bridge is faithful regardless of convergence)
    assert m["onnx"]["parity_max_abs"] < 1e-4, f"ONNX parity too large: {m['onnx']['parity_max_abs']}"
    assert m["onnx"]["bytes"] > 0 and m["onnx"]["input_dim"] == 2
    # a tiny MLP -> the gate must measure a LIVE lane
    assert m["lane"] in ("live", "precompute") and m["gate"]["lane"] == m["lane"]
    assert "l2_relative" in m["metrics"]  # evaluated vs the analytic reference


def test_artifact_matches_manifest():
    m = pipeline.precompute(PILOT, seed=7, quick=True)
    artifact = pipeline.DERIVED / m["artifact"]["path"]
    assert artifact.exists(), "manifest points to a non-existent artifact"
    assert artifact.stat().st_size == m["artifact"]["bytes"], "manifest byte size drifted from the artifact"
    trace = json.loads(artifact.read_text(encoding="utf-8"))
    assert trace["schema"].startswith("pinnlab.field/")
    assert set(trace["fields"]) == {"u"} and trace["dims"] == ["x", "y"]


def test_run_all_writes_index():
    entries = pipeline.run_all(seed=7, quick=True)  # smoke plumbing only; real artifacts are baked by scripts/precompute
    assert len(entries) == len(registry.list_cases()) >= 1
    idx = json.loads((pipeline.MANIFESTS / "index.json").read_text(encoding="utf-8"))
    assert idx["n_cases"] == len(entries)
    assert idx["schema"].startswith("pinnlab.index/")
