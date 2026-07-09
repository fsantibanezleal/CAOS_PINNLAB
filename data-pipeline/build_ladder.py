"""Generalized ladder for a single-output case at a CHOSEN regime (default: the hardest variant).

Usage: python build_ladder.py <case_id> [variant_id] [adam]

Trains ONLY the naive lane (fast); reuses the already-exported adapted ONNX (no wasteful retrain) evaluated at the
chosen regime; standard = the case's analytic reference at that regime. Bakes standard | naive | adapted + error
maps and patches the manifest. For a parametric spectral-bias case, choose the HARD regime (e.g. wave c=2, poisson
k=3) so the naive lane's failure is visible; a naive lane at an easy regime would be honest but undramatic.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from pinnlab.io.formats import strip_onnx_metadata
from pinnlab.model.analytic import l2_relative
from pinnlab.registry import case_module, get_case

REPO = Path(__file__).resolve().parents[1]
DERIVED = REPO / "data" / "derived"
MODELS = REPO / "models"
CAP = 81


def dec_idx(n, cap=CAP):
    return list(range(n)) if n <= cap else [round(i * (n - 1) / (cap - 1)) for i in range(cap)]


def rnd(a):
    return np.round(np.asarray(a, dtype=float), 5).tolist()


def main():
    cid = sys.argv[1]
    vid = sys.argv[2] if len(sys.argv) > 2 else None
    adam = int(sys.argv[3]) if len(sys.argv) > 3 else 3500
    case = get_case(cid)
    mod = case_module(cid)
    man_path = DERIVED / "manifests" / f"{cid}.json"
    man = json.loads(man_path.read_text(encoding="utf-8"))
    # pick the regime: explicit vid, else the LAST variant (usually the hardest parameter)
    variants = man["variants"]
    var = next((v for v in variants if v["id"] == vid), variants[-1])
    params = var.get("params") or {}
    print(f"[{cid}] regime = {var['id']} params={params}", flush=True)

    # decimated field grid over case.axes, at this regime
    axes = case.axes
    coords = {a: np.linspace(case.domain[a][0], case.domain[a][1], case.grid[a]) for a in axes}
    idx = {a: dec_idx(len(coords[a])) for a in axes}
    coords = {a: coords[a][idx[a]] for a in axes}
    mesh = np.meshgrid(*[coords[a] for a in axes], indexing="ij")
    field_cols = {a: m.ravel() for a, m in zip(axes, mesh)}
    shape = tuple(len(coords[a]) for a in axes)
    n = int(next(iter(field_cols.values())).size)
    XY = np.empty((n, len(case.inputs)), dtype=np.float64)
    for j, ax in enumerate(case.inputs):
        XY[:, j] = field_cols[ax] if ax in field_cols else float(params.get(ax, 0.0))

    t0 = time.perf_counter()
    print("[1/3] standard (analytic) + train ADAPTED fresh (fair) ...", flush=True)
    standard = np.asarray(mod.analytic(XY), dtype=np.float64).reshape(shape)
    # Train BOTH lanes fresh with the SAME budget so the contrast is the METHOD, not the training. (Reusing the
    # pre-trained adapted ONNX would be unfair - different budgets - and can invert the result.)
    ba = mod.build(42)
    ma = ba["model"]
    ma.train(iterations=adam, display_every=max(1, adam // 3))
    adapted = np.asarray(ma.predict(XY), dtype=np.float64)[:, 0].reshape(shape)

    print(f"[2/3] train NAIVE fresh ({adam} adam) ...", flush=True)
    built = mod.build_naive(42)
    model = built["model"]
    model.train(iterations=adam, display_every=max(1, adam // 3))
    naive = np.asarray(model.predict(XY), dtype=np.float64)[:, 0].reshape(shape)

    err_naive = np.abs(naive - standard)
    err_adapted = np.abs(adapted - standard)
    l2 = {"naive_vs_std": round(l2_relative(naive, standard), 5), "adapted_vs_std": round(l2_relative(adapted, standard), 5)}
    print("   L2:", l2, flush=True)

    print("[3/3] bake + export naive ONNX + patch manifest ...", flush=True)
    trace = {
        "schema": "pinnlab.compare/v1", "case_id": cid, "dims": list(axes),
        "axes": {a: [round(float(v), 5) for v in coords[a]] for a in axes},
        "fields": {"standard": rnd(standard), "adapted": rnd(adapted), "naive": rnd(naive),
                   "err_adapted": rnd(err_adapted), "err_naive": rnd(err_naive)},
        "summary": {**l2, "regime": var["id"]},
    }
    (DERIVED / cid).mkdir(parents=True, exist_ok=True)
    (DERIVED / cid / "comparison.json").write_text(json.dumps(trace, ensure_ascii=False), encoding="utf-8")

    net = model.net
    net.eval()
    torch.onnx.export(net, (torch.zeros(1, int(built["input_dim"]), dtype=torch.float32),), str(MODELS / f"{cid}-naive.onnx"),
                      input_names=["coords"], output_names=["u"], dynamic_axes={"coords": {0: "n"}, "u": {0: "n"}},
                      opset_version=18, dynamo=True, verbose=False, external_data=False)
    strip_onnx_metadata(MODELS / f"{cid}-naive.onnx")

    regime_lbl = ", ".join(f"{k}={v:g}" for k, v in params.items()) or var["id"]
    man["comparison"] = {
        "trace": f"{cid}/comparison.json",
        "lanes": [
            {"key": "standard", "label_en": "standard (analytic)", "label_es": "estándar (analítica)", "role": "reference"},
            {"key": "naive", "label_en": "naive PINN", "label_es": "PINN ingenua", "role": "baseline", "err": "err_naive"},
            {"key": "adapted", "label_en": f"{case.method} PINN", "label_es": f"PINN {case.method}", "role": "fix", "err": "err_adapted"},
        ],
        "onnx_naive": f"{cid}-naive.onnx",
        "note_en": f"At the hard regime ({regime_lbl}): the naive PINN reaches L2 = {l2['naive_vs_std']*100:.0f}% vs the standard while the {case.method} fix reaches {l2['adapted_vs_std']*100:.1f}%.",
        "note_es": f"En el régimen difícil ({regime_lbl}): la PINN ingenua llega a L2 = {l2['naive_vs_std']*100:.0f}% vs el estándar mientras que la corrección {case.method} llega a {l2['adapted_vs_std']*100:.1f}%.",
    }
    man_path.write_text(json.dumps(man, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"[{cid}] DONE in {(time.perf_counter()-t0)/60:.1f} min. L2={l2}", flush=True)


if __name__ == "__main__":
    main()
