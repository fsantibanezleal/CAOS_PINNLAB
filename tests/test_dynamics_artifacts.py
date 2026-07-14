"""Contract tests for the method-ladder + dynamics artifacts (issues #25/#36/#38).

READ-ONLY by design (tests must never write canonical artifacts): they validate that every baked
comparison/training/frames/diagnostics file referenced by a manifest exists, carries the required keys, and is
internally consistent (frame shapes match the axes; L2 arrays align with the checkpoints; manifest summaries match
the trace summaries). A drift here means the web app would render wrong or missing dynamics.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DERIVED = REPO / "data" / "derived"
MANIFESTS = DERIVED / "manifests"


def _manifests():
    for p in sorted(MANIFESTS.glob("*.json")):
        if p.name == "index.json":
            continue
        yield json.loads(p.read_text(encoding="utf-8"))


ALL = list(_manifests())
WITH_COMPARISON = [m for m in ALL if m.get("comparison")]
WITH_TRAINING = [m for m in ALL if m.get("training")]
WITH_EVOLUTION = [m for m in ALL if m.get("evolution")]
WITH_DIAGNOSTICS = [m for m in ALL if m.get("diagnostics")]


def test_coverage_floor():
    """The deep pass promised broad coverage - a regression that drops baked blocks should fail loudly."""
    assert len(WITH_COMPARISON) >= 15, f"comparisons dropped: {len(WITH_COMPARISON)}"
    assert len(WITH_TRAINING) >= 2, "training-dynamics cases dropped"
    assert len(WITH_EVOLUTION) >= 2, "evolution-frame cases dropped"
    assert len(WITH_DIAGNOSTICS) >= 3, "diagnostics cases dropped"


@pytest.mark.parametrize("m", WITH_COMPARISON, ids=lambda m: m["case_id"])
def test_comparison_contract(m):
    c = m["comparison"]
    trace_path = DERIVED / c["trace"]
    assert trace_path.exists(), f"missing {c['trace']}"
    t = json.loads(trace_path.read_text(encoding="utf-8"))
    assert t.get("schema", "").startswith("pinnlab.compare/")
    dims = t["dims"]
    axes = t["axes"]
    n0, n1 = len(axes[dims[0]]), len(axes[dims[1]])
    lane_keys = {l["key"] for l in c["lanes"]}
    assert "standard" in lane_keys, "every comparison must carry the standard lane"
    for l in c["lanes"]:
        assert l["role"] in {"reference", "exact", "baseline", "fix", "data"}
        fld = t["fields"].get(l["key"])
        assert fld is not None, f"lane field {l['key']} missing in trace"
        assert len(fld) == n0 and len(fld[0]) == n1, f"{l['key']} shape != axes"
        if l.get("err"):
            e = t["fields"].get(l["err"])
            assert e is not None and len(e) == n0 and len(e[0]) == n1
    # the manifest summary (used by the Benchmark ladder columns) must MATCH the trace summary
    ms = c.get("summary") or {}
    ts = t.get("summary") or {}
    for k, v in ms.items():
        assert ts.get(k) == v, f"manifest summary {k} drifted from trace"


@pytest.mark.parametrize("m", WITH_TRAINING, ids=lambda m: m["case_id"])
def test_training_contract(m):
    p = DERIVED / m["training"]["path"]
    assert p.exists()
    t = json.loads(p.read_text(encoding="utf-8"))
    assert t.get("schema", "").startswith("pinnlab.training/")
    cps = t["checkpoints"]
    assert len(cps) >= 5 and cps == sorted(cps)
    ax = list(t["axes"].values())
    n0, n1 = len(ax[0]), len(ax[1])
    assert {"naive", "adapted"} <= set(t["lanes"].keys())
    for k, lane in t["lanes"].items():
        assert len(lane["frames"]) == len(cps), f"{k}: frames != checkpoints"
        assert len(lane["l2"]) == len(cps), f"{k}: l2 != checkpoints"
        for fr in lane["frames"]:
            assert len(fr) == n0 and len(fr[0]) == n1, f"{k}: frame shape != axes"
        assert lane["label_en"] and lane["label_es"]


@pytest.mark.parametrize("m", WITH_EVOLUTION, ids=lambda m: m["case_id"])
def test_evolution_contract(m):
    p = DERIVED / m["evolution"]["path"]
    assert p.exists()
    t = json.loads(p.read_text(encoding="utf-8"))
    assert t.get("schema", "").startswith("pinnlab.evolution/")
    nT = len(t["t"])
    assert nT >= 8, "an evolution bake should be a smooth sequence, not a couple of stills"
    fa = m["field_axes"]
    n0, n1 = len(t["axes"][fa[0]]), len(t["axes"][fa[1]])
    for out in m["outputs"]:
        frames = t["frames"].get(out)
        assert frames is not None, f"output {out} missing"
        assert len(frames) == nT
        assert len(frames[0]) == n0 and len(frames[0][0]) == n1
    assert m.get("view_kit") == "SpatioTemporalKit", "an evolution bake implies the animated 2-D kit"


@pytest.mark.parametrize("m", WITH_DIAGNOSTICS, ids=lambda m: m["case_id"])
def test_diagnostics_contract(m):
    p = DERIVED / m["diagnostics"]["path"]
    assert p.exists()
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d.get("schema", "").startswith("pinnlab.diagnostics/")
    has_content = any(k in d for k in ("wavenumber_sweep", "radial_spectrum", "line_comparisons"))
    assert has_content, "diagnostics file carries no renderable content"
    if "wavenumber_sweep" in d:
        ws = d["wavenumber_sweep"]
        assert len(ws["n"]) == len(ws["naive"]) == len(ws["adapted"])
    for lc in d.get("line_comparisons", []):
        for s in lc["series"]:
            assert len(s["x"]) == len(s["y"]), f"series {s['label']} x/y mismatch"


# ---- THE ESTIMATE block (issue #46): every case answers an engineering question with computed numbers ----

def test_estimate_coverage_all_cases():
    """The estimation reframe covers the WHOLE catalogue: a case without its question/answer is a regression."""
    missing = [m["case_id"] for m in ALL if not m.get("estimate")]
    assert not missing, f"cases without an estimate block: {missing}"


@pytest.mark.parametrize("m", ALL, ids=lambda m: m["case_id"])
def test_estimate_contract(m):
    e = m.get("estimate")
    assert e, "estimate block missing"
    for k in ("question_en", "question_es", "why_en", "why_es"):
        assert isinstance(e.get(k), str) and e[k].strip(), f"{k} missing/empty"
    assert e["question_en"].strip().endswith("?") or "(" in e["question_en"], "the question should read as a question"
    items = e.get("items")
    assert isinstance(items, list) and len(items) >= 1, "at least one answer item"
    variant_ids = {v["id"] for v in m["variants"]}
    for it in items:
        assert isinstance(it.get("label_en"), str) and it["label_en"].strip()
        assert isinstance(it.get("label_es"), str) and it["label_es"].strip()
        has_value = isinstance(it.get("value"), str) and it["value"].strip()
        has_values = isinstance(it.get("values"), dict) and len(it["values"]) > 0
        assert has_value != has_values, "an item carries value XOR values"
        if has_values:
            unknown = set(it["values"]) - variant_ids
            assert not unknown, f"values keyed by unknown variant ids: {unknown}"
            assert all(isinstance(v, str) and v.strip() for v in it["values"].values())


def test_index_carries_questions():
    """build_estimates.py patches index.json so the case list + story can lead with the question."""
    idx = json.loads((MANIFESTS / "index.json").read_text(encoding="utf-8"))
    missing = [c["case_id"] for c in idx["cases"] if not (c.get("question_en") and c.get("question_es"))]
    assert not missing, f"index entries without questions: {missing}"
