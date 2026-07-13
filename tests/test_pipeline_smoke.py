"""Pipeline smoke (quick path): the canonical pilot case runs through all six stages, exports a faithful ONNX
(parity vs model.predict), classifies a lane from measurements, and writes a consistent artifact + manifest. The
`quick` path uses few iterations — it checks the train->ONNX->web plumbing + contracts, NOT convergence accuracy
(accuracy is asserted separately by the offline precompute + the Benchmark page).

SANDBOXED suite-wide by the autouse fixture in conftest.py (hard rule: tests must NEVER write the canonical
artifacts) — every pipeline write target (DERIVED / MANIFESTS / MODELS) is redirected to a per-test tmp dir."""
import json

from pinnlab import pipeline, registry

PILOT = "bench-poisson2d"


def test_pilot_runs_and_exports_faithful_onnx():
    m = pipeline.precompute(PILOT, seed=7, quick=True)
    # the exported ONNX must match the trained net (the train->web bridge is faithful regardless of convergence)
    assert m["onnx"]["parity_max_abs"] < 1e-4, f"ONNX parity too large: {m['onnx']['parity_max_abs']}"
    assert m["onnx"]["bytes"] > 0 and m["onnx"]["input_dim"] >= 2  # poisson2d is parametric (x,y,k) -> 3
    # a tiny MLP -> the gate must measure a LIVE lane
    assert m["lane"] in ("live", "precompute") and m["gate"]["lane"] == m["lane"]
    # manifest/v2: each variant carries its own metrics (relative-L2 vs the analytic reference)
    assert m["variants"] and all("l2_relative" in v["metrics"] for v in m["variants"])


def test_artifact_matches_manifest():
    m = pipeline.precompute(PILOT, seed=7, quick=True)
    # manifest/v2: one field trace per variant (parameter regime); check the first variant's trace.
    v0 = m["variants"][0]
    artifact = pipeline.DERIVED / v0["trace"]["path"]
    assert artifact.exists(), "manifest points to a non-existent artifact"
    assert artifact.stat().st_size == v0["trace"]["bytes"], "manifest byte size drifted from the artifact"
    trace = json.loads(artifact.read_text(encoding="utf-8"))
    assert trace["schema"].startswith("pinnlab.field/")
    assert set(trace["fields"]) == {"u"} and trace["dims"] == ["x", "y"]


def test_run_all_writes_index():
    entries = pipeline.run_all(seed=7, quick=True)  # smoke plumbing only; real artifacts are baked by scripts/precompute
    assert len(entries) == len(registry.list_cases()) >= 1
    idx = json.loads((pipeline.MANIFESTS / "index.json").read_text(encoding="utf-8"))
    assert idx["n_cases"] == len(entries)
    assert idx["schema"].startswith("pinnlab.index/")
