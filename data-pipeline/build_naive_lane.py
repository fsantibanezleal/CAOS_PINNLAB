"""Add a REAL naive-PINN lane to a case that already has a standard-vs-adapted comparison, with FAST training.

Usage: python build_naive_lane.py <case_id> [adam_iters]

The case must expose `build_naive(seed)` (the plain PINN without the fix). This trains it fast (no L-BFGS - the naive
lane's behaviour is visible in a few thousand iterations; the point is the honest contrast, not squeezing it), then:
  - evaluates it on the SAME grid as the existing comparison.json,
  - computes err_naive = |naive - standard| and the real naive_vs_std L2,
  - inserts the naive lane into the comparison trace + manifest (after 'standard'),
  - exports <case>-naive.onnx.
Nothing invented: the naive field is a real (short) training run of the plain network.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from pinnlab.io.formats import strip_onnx_metadata
from pinnlab.model.analytic import l2_relative
from pinnlab.registry import case_module, get_case

REPO = Path(__file__).resolve().parents[1]
DERIVED = REPO / "data" / "derived"
MODELS = REPO / "models"


def rnd(a):
    return np.round(np.asarray(a, dtype=float), 5).tolist()


def main():
    cid = sys.argv[1]
    adam = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    case = get_case(cid)
    mod = case_module(cid)
    if not hasattr(mod, "build_naive"):
        raise SystemExit(f"{cid} has no build_naive()")

    man_path = DERIVED / "manifests" / f"{cid}.json"
    man = json.loads(man_path.read_text(encoding="utf-8"))
    cmp = man.get("comparison")
    if not cmp:
        raise SystemExit(f"{cid} has no comparison yet - run build_standard_comparisons.py first")
    cmp_path = DERIVED / cmp["trace"]
    tr = json.loads(cmp_path.read_text(encoding="utf-8"))
    if "standard" not in tr["fields"]:
        raise SystemExit(f"{cid} comparison has no 'standard' field")

    t0 = time.perf_counter()
    print(f"[{cid}] train naive ({adam} adam) ...", flush=True)
    built = mod.build_naive(42)
    model = built["model"]
    model.train(iterations=adam, display_every=max(1, adam // 3))

    # eval the naive net on the SAME (decimated) grid the comparison was baked on, at variant[0]'s parameters
    dims = list(tr["dims"])
    axes = {a: np.asarray(tr["axes"][a], dtype=float) for a in dims}
    mesh = np.meshgrid(*[axes[a] for a in dims], indexing="ij")
    field_cols = {a: m.ravel() for a, m in zip(dims, mesh)}
    shape = tuple(len(axes[a]) for a in dims)
    params = (man["variants"][0].get("params") or {})
    n = int(next(iter(field_cols.values())).size)
    XY = np.empty((n, len(case.inputs)), dtype=np.float64)
    for j, ax in enumerate(case.inputs):
        XY[:, j] = field_cols[ax] if ax in field_cols else float(params.get(ax, 0.0))
    naive = np.asarray(model.predict(XY), dtype=np.float64)[:, 0].reshape(shape)

    standard = np.asarray(tr["fields"]["standard"], dtype=float)
    err_naive = np.abs(naive - standard)
    rel = round(l2_relative(naive, standard), 5)
    print(f"[{cid}] naive_vs_std L2 = {rel*100:.1f}%  (adapted = {tr['summary'].get('adapted_vs_std', '?')})", flush=True)

    tr["fields"]["naive"] = rnd(naive)
    tr["fields"]["err_naive"] = rnd(err_naive)
    tr["summary"]["naive_vs_std"] = rel
    cmp_path.write_text(json.dumps(tr, ensure_ascii=False), encoding="utf-8")

    # export the naive ONNX
    MODELS.mkdir(parents=True, exist_ok=True)
    net = model.net
    net.eval()
    onnx_name = f"{cid}-naive.onnx"
    d = int(built["input_dim"])
    torch.onnx.export(
        net, (torch.zeros(1, d, dtype=torch.float32),), str(MODELS / onnx_name),
        input_names=["coords"], output_names=["u"],
        dynamic_axes={"coords": {0: "n"}, "u": {0: "n"}}, opset_version=18, dynamo=True,
        verbose=False, external_data=False,
    )
    strip_onnx_metadata(MODELS / onnx_name)

    # insert the naive lane into the manifest comparison (after 'standard'), if not already present
    lanes = cmp["lanes"]
    if not any(l["key"] == "naive" for l in lanes):
        ins = next((i for i, l in enumerate(lanes) if l["role"] == "reference"), 0) + 1
        lanes.insert(ins, {
            "key": "naive", "label_en": "naive PINN", "label_es": "PINN ingenua",
            "role": "baseline", "err": "err_naive",
        })
    cmp["onnx_naive"] = onnx_name
    # refresh the note with both real numbers
    a = tr["summary"].get("adapted_vs_std")
    cmp["note_en"] = f"The naive plain PINN (no {case.method}) vs the {case.method} fix, both against the classical standard solution. Naive L2 = {rel*100:.1f}%, adapted L2 = {a*100:.1f}%." if a is not None else cmp.get("note_en", "")
    cmp["note_es"] = f"La PINN ingenua (sin {case.method}) frente a la corrección {case.method}, ambas contra la solución estándar clásica. L2 ingenua = {rel*100:.1f}%, adaptada = {a*100:.1f}%." if a is not None else cmp.get("note_es", "")
    man_path.write_text(json.dumps(man, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"[{cid}] DONE in {(time.perf_counter()-t0)/60:.1f} min - naive lane added.", flush=True)


if __name__ == "__main__":
    main()
