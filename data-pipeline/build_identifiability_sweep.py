"""IDENTIFIABILITY SWEEP (issue #44): the computed answer to 'how much information does a PINN need?'.

Trains the heat2d-inverse net at n = 10, 25, 50, 100 sensors at ONE fixed fast budget (n=0 reuses the already-baked
pure-physics naive lane, which used the same budget), evaluates the recovered k(x,y) against the exact k*, and bakes
a Diagnostics line: recovered-k relative-L2 vs number of sensors. All real training runs; consistent budget so the
curve is internally fair.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from pinnlab.cases import ind_heat2d_inverse as C

REPO = Path(__file__).resolve().parents[1]
DERIVED = REPO / "data" / "derived"
CID = "ind-heat2d-inverse"
ADAM = 3500
GRID = 61
NS = [10, 25, 50, 100]


def main():
    t0 = time.perf_counter()
    x = np.linspace(0, 1, GRID)
    X, Y = np.meshgrid(x, x, indexing="ij")
    XY = np.stack([X.ravel(), Y.ravel()], axis=1).astype(np.float64)
    k_exact = C.k_true(XY).reshape(GRID, GRID)

    man_path = DERIVED / "manifests" / f"{CID}.json"
    man = json.loads(man_path.read_text(encoding="utf-8"))
    n0_l2 = float((man.get("comparison", {}).get("summary", {}) or {}).get("naive_vs_std", 3.56416))

    ns = [0] + NS
    l2s = [round(n0_l2, 5)]
    for n in NS:
        print(f"[n={n}] training ({ADAM} adam) ...", flush=True)
        model = C.build_n_sensors(42, n)["model"]
        model.train(iterations=ADAM, display_every=ADAM)
        k_pred = np.asarray(model.predict(XY), dtype=np.float64)[:, 0].reshape(GRID, GRID)
        rel = float(np.linalg.norm(k_pred - k_exact) / (np.linalg.norm(k_exact) or 1.0))
        l2s.append(round(rel, 5))
        print(f"[n={n}] recovered-k L2 vs k* = {rel*100:.1f}%", flush=True)

    diag = {
        "schema": "pinnlab.diagnostics/v1", "case_id": CID,
        "line_comparisons": [{
            "title_en": "IDENTIFIABILITY: recovered-k error vs number of sensors (same training budget)",
            "title_es": "IDENTIFICABILIDAD: error de k recuperada vs numero de sensores (mismo presupuesto)",
            "xLabel": "number of T sensors", "yLabel": "k L2 vs k*", "yLog": True,
            "series": [
                {"label": "recovered k (fast budget)", "color": "var(--accent)", "x": [float(v) for v in ns], "y": l2s},
                {"label": "per-run points", "color": "var(--accent)", "scatter": True, "x": [float(v) for v in ns], "y": l2s},
            ],
        }],
        "rmse": {f"n{n}": v for n, v in zip(ns, l2s)},
    }
    (DERIVED / CID).mkdir(exist_ok=True)
    (DERIVED / CID / "diagnostics.json").write_text(json.dumps(diag, ensure_ascii=False), encoding="utf-8")
    man["diagnostics"] = {"path": f"{CID}/diagnostics.json"}
    man_path.write_text(json.dumps(man, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"DONE in {(time.perf_counter()-t0)/60:.1f} min: n={ns} L2={l2s}", flush=True)


if __name__ == "__main__":
    main()
