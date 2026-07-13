"""Make pinnlab importable whether or not `pip install -e .` has run (belt-and-suspenders for CI/local).

Plus the SUITE-WIDE write sandbox (hard rule: tests must NEVER write the canonical artifacts). Any test that runs
the pipeline (precompute / run_all, quick or not) gets pipeline.DERIVED / MANIFESTS / MODELS redirected to a
per-test tmp dir. A quick-mode smoke once clobbered committed bakes and its manifest rewrite silently DROPPED the
comparison/training/evolution blocks the web app renders. Never remove this fixture; it is autouse on purpose."""
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "data-pipeline"))


@pytest.fixture(autouse=True)
def _sandbox_pipeline_writes(tmp_path, monkeypatch):
    from pinnlab import pipeline

    derived = tmp_path / "derived"
    manifests = derived / "manifests"
    models = tmp_path / "models"
    manifests.mkdir(parents=True)
    models.mkdir()
    monkeypatch.setattr(pipeline, "DERIVED", derived)
    monkeypatch.setattr(pipeline, "MANIFESTS", manifests)
    monkeypatch.setattr(pipeline, "MODELS", models)
