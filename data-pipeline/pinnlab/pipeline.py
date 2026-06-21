"""The offline pipeline orchestrator + CLI (ADR-0057, PINN-specialized). For each case it runs the six named stages
— preprocess (validate/ingest) -> feature_extraction (sampling plan) -> train (DeepXDE -> ONNX, parity) ->
infer (grid field) -> evaluate (relative-L2 vs analytic) -> export (CONTRACT 2 artifact + manifest) — and writes a
flat index.json. Deterministic given (case, seed).

    python -m pinnlab.pipeline                      # all cases
    python -m pinnlab.pipeline bench-poisson2d --seed 7
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from . import registry
from .core.manifest import build_index
from .io.formats import write_json
from .stages import evaluate, export, feature_extraction, infer, preprocess, train

# data-pipeline/pinnlab/pipeline.py -> parents[2] = repo root (works under `pip install -e .` too)
REPO_ROOT = Path(__file__).resolve().parents[2]
DERIVED = REPO_ROOT / "data" / "derived"
MANIFESTS = DERIVED / "manifests"
MODELS = REPO_ROOT / "models"

STAGES = ("preprocess", "feature_extraction", "train", "infer", "evaluate", "export")


def precompute(case_id: str, seed: int = 42, *, quick: bool = False) -> dict:
    case = registry.get_case(case_id)
    t0 = time.perf_counter()
    pre = preprocess.run(case_id)
    if not pre["well_posed"]:
        raise ValueError(f"{case_id} not well-posed: {pre['flags']}")
    plan = feature_extraction.run(case_id)
    built = train.run(case_id, seed=seed, models_dir=str(MODELS), sampling=plan, quick=quick)
    sf = infer.run(case_id, built["model"])
    metrics = evaluate.run(case_id, sf, built)
    run_ms = (time.perf_counter() - t0) * 1000.0
    return export.run(
        case=case, sf=sf, onnx_info=built, metrics=metrics, seed=seed, run_ms=run_ms,
        derived_dir=str(DERIVED), manifests_dir=str(MANIFESTS), flags=pre.get("flags"),
    )


def run_all(seed: int = 42, *, quick: bool = False) -> list[dict]:
    entries = []
    for c in registry.list_cases():
        precompute(c.id, seed=seed, quick=quick)
        entries.append({"case_id": c.id, "category": c.category, "title": c.title,
                        "manifest_path": f"manifests/{c.id}.json"})
    write_json(MANIFESTS / "index.json", build_index(entries))
    return entries


def main() -> None:
    ap = argparse.ArgumentParser(prog="pinnlab.pipeline")
    ap.add_argument("case", nargs="?", default="all", help="a case id, or 'all'")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.case == "all":
        entries = run_all(args.seed)
        print(f"precomputed {len(entries)} cases -> {DERIVED}")
        for e in entries:
            print(f"  {e['case_id']:24s} [{e['category']}]")
        print(f"index -> {MANIFESTS / 'index.json'}")
    else:
        m = precompute(args.case, args.seed)
        print(
            f"precomputed {args.case}: lane={m['lane']} "
            f"l2={m['metrics'].get('l2_relative')} parity={m['metrics'].get('onnx_parity_max_abs')} "
            f"onnx={m['onnx']['bytes']}B infer={m['gate']['infer_ms']}ms -> {DERIVED / m['artifact']['path']}"
        )


if __name__ == "__main__":
    main()
