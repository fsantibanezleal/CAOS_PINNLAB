"""The offline pipeline orchestrator + CLI (ADR-0057, PINN-specialized). For each case it trains the PINN ONCE
(DeepXDE/FNO -> ONNX, parity), then bakes ONE compact field trace PER VARIANT (parameter regime) by evaluating the
trained net at that regime, and writes a manifest/v2 listing every variant with its metrics + a flat index.json.

A PARAMETRIC case (the parameter is a network input) ships one ONNX shared by all variants; the web replays each
baked regime and the `Live` tab sweeps the parameter via the shared ONNX. A non-parametric case ships a single
`default` variant. Deterministic given (case, seed).

    python -m pinnlab.pipeline                      # all cases
    python -m pinnlab.pipeline bench-poisson2d --seed 7
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from . import registry
from .core.gate import classify_lane
from .core.manifest import build_case_manifest, build_index
from .core.trace import build_trace
from .io.formats import write_json
from .stages import evaluate, feature_extraction, infer, preprocess, train

REPO_ROOT = Path(__file__).resolve().parents[2]
DERIVED = REPO_ROOT / "data" / "derived"
MANIFESTS = DERIVED / "manifests"
MODELS = REPO_ROOT / "models"


def precompute(case_id: str, seed: int = 42, *, quick: bool = False) -> dict:
    case = registry.get_case(case_id)
    t0 = time.perf_counter()

    pre = preprocess.run(case_id)
    if not pre["well_posed"]:
        raise ValueError(f"{case_id} not well-posed: {pre['flags']}")
    plan = feature_extraction.run(case_id)

    # train ONCE -> the (possibly parametric) trained net + its exported ONNX
    built = train.run(case_id, seed=seed, models_dir=str(MODELS), sampling=plan, quick=quick)

    # bake ONE field trace per variant (parameter regime); collect per-variant metrics
    variants = registry.case_variants(case_id)
    entries: list[dict] = []
    max_trace_bytes = 0
    for var in variants:
        sf = infer.run(case_id, built["model"], params=var.params)
        metrics = evaluate.run(case_id, sf, built, params=var.params)
        trace = build_trace(sf)
        rel = f"{case_id}/{var.id}.json"
        tb = write_json(DERIVED / rel, trace)
        max_trace_bytes = max(max_trace_bytes, tb)
        entries.append({
            "id": var.id,
            "label_en": var.label_en,
            "label_es": var.label_es,
            "note_en": var.note_en,
            "note_es": var.note_es,
            "params": var.params,
            "trace": {"path": rel, "bytes": tb},
            "metrics": metrics,
        })

    gate = classify_lane(
        onnx_bytes=int(built["onnx_bytes"]),
        infer_ms=float(built.get("infer_ms", 0.0)),
        trace_bytes=max_trace_bytes,
        web_drivable=bool(built.get("web_drivable", True)),
    )
    onnx_meta = {
        "path": Path(built["onnx_path"]).name,
        "bytes": int(built["onnx_bytes"]),
        "parity_max_abs": round(float(built["parity_max_abs"]), 9),
        "opset": int(built.get("opset", 18)),
        "input_dim": int(built["input_dim"]),
    }
    manifest = build_case_manifest(
        case=case, seed=seed, onnx=onnx_meta, gate=gate, variants=entries,
        run_ms=(time.perf_counter() - t0) * 1000.0, flags=pre.get("flags"),
    )
    write_json(MANIFESTS / f"{case.id}.json", manifest)
    return manifest


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
    ap.add_argument("--quick", action="store_true", help="few iters, no L-BFGS/refine — CI smoke only, NOT for real artifacts")
    args = ap.parse_args()
    if args.case == "all":
        entries = run_all(args.seed, quick=args.quick)
        print(f"precomputed {len(entries)} cases -> {DERIVED}")
        for e in entries:
            print(f"  {e['case_id']:24s} [{e['category']}]")
        print(f"index -> {MANIFESTS / 'index.json'}")
    else:
        m = precompute(args.case, args.seed, quick=args.quick)
        vs = m["variants"]
        l2s = [v["metrics"].get("l2_relative") for v in vs]
        print(
            f"precomputed {args.case}: lane={m['lane']} variants={len(vs)} "
            f"l2={[round(x, 5) if isinstance(x, (int, float)) else x for x in l2s]} "
            f"parity={m['onnx']['parity_max_abs']} onnx={m['onnx']['bytes']}B infer={m['gate']['infer_ms']}ms"
        )


if __name__ == "__main__":
    main()
