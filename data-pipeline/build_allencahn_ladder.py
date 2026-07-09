"""allencahn ladder: the spectral-reference STANDARD vs the naive soft PINN (collapses) vs the hard-constraint+RAR
adapted PINN. All real. The standard is the vendored spectral solution `Allen_Cahn.npz`; the naive is a short real
training run of the plain soft PINN; the adapted is the already-baked field.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from pinnlab.cases import bench_allencahn as A
from pinnlab.io.formats import strip_onnx_metadata
from pinnlab.model.analytic import l2_relative

REPO = Path(__file__).resolve().parents[1]
DERIVED = REPO / "data" / "derived"
MODELS = REPO / "models"
CID = "bench-allencahn"
ADAM = 6000


def rnd(a):
    return np.round(np.asarray(a, dtype=float), 5).tolist()


def main():
    t0 = time.perf_counter()
    man_path = DERIVED / "manifests" / f"{CID}.json"
    man = json.loads(man_path.read_text(encoding="utf-8"))
    var = man["variants"][0]
    tr_path = DERIVED / var["trace"]["path"]
    tr = json.loads(tr_path.read_text(encoding="utf-8"))
    ax_x = np.asarray(tr["axes"]["x"], dtype=float)
    ax_t = np.asarray(tr["axes"]["t"], dtype=float)
    adapted = np.asarray(tr["fields"]["u"], dtype=float)

    print("[1/3] standard = spectral reference subsampled to the trace grid ...", flush=True)
    ref = A.reference_on_grid()  # (201,101) over (x,t)
    grids, _XY, _shape = A.eval_grid()
    rx, rt = grids["x"], grids["t"]
    xi = [int(np.argmin(np.abs(rx - v))) for v in ax_x]
    ti = [int(np.argmin(np.abs(rt - v))) for v in ax_t]
    standard = ref[np.ix_(xi, ti)]

    print(f"[2/3] train NAIVE soft PINN ({ADAM} adam) ...", flush=True)
    built = A.build_naive(42)
    model = built["model"]
    model.train(iterations=ADAM, display_every=ADAM // 3)
    Xx, Tt = np.meshgrid(ax_x, ax_t, indexing="ij")
    XYq = np.stack([Xx.ravel(), Tt.ravel()], axis=1)
    naive = np.asarray(model.predict(XYq), dtype=np.float64)[:, 0].reshape(len(ax_x), len(ax_t))

    err_adapted = np.abs(adapted - standard)
    err_naive = np.abs(naive - standard)
    l2 = {
        "naive_vs_std": round(l2_relative(naive, standard), 5),
        "adapted_vs_std": round(l2_relative(adapted, standard), 5),
    }
    print("   L2:", l2, flush=True)

    print("[3/3] bake comparison + export naive ONNX + patch manifest ...", flush=True)
    cmp_trace = {
        "schema": "pinnlab.compare/v1", "case_id": CID,
        "dims": ["x", "t"], "axes": {"x": [round(float(v), 5) for v in ax_x], "t": [round(float(v), 5) for v in ax_t]},
        "fields": {"standard": rnd(standard), "adapted": rnd(adapted), "naive": rnd(naive),
                   "err_adapted": rnd(err_adapted), "err_naive": rnd(err_naive)},
        "summary": l2,
    }
    (DERIVED / CID).mkdir(parents=True, exist_ok=True)
    (DERIVED / CID / "comparison.json").write_text(json.dumps(cmp_trace, ensure_ascii=False), encoding="utf-8")

    net = model.net
    net.eval()
    torch.onnx.export(
        net, (torch.zeros(1, 2, dtype=torch.float32),), str(MODELS / f"{CID}-naive.onnx"),
        input_names=["coords"], output_names=["u"], dynamic_axes={"coords": {0: "n"}, "u": {0: "n"}},
        opset_version=18, dynamo=True, verbose=False, external_data=False,
    )
    strip_onnx_metadata(MODELS / f"{CID}-naive.onnx")

    man["comparison"] = {
        "trace": f"{CID}/comparison.json",
        "lanes": [
            {"key": "standard", "label_en": "standard (spectral)", "label_es": "estándar (espectral)", "role": "reference"},
            {"key": "naive", "label_en": "naive soft PINN", "label_es": "PINN suave ingenua", "role": "baseline", "err": "err_naive"},
            {"key": "adapted", "label_en": "hard-constraint + RAR PINN", "label_es": "PINN restricciones duras + RAR", "role": "fix", "err": "err_adapted"},
        ],
        "onnx_naive": f"{CID}-naive.onnx",
        "note_en": f"The naive SOFT PINN collapses to a metastable state and smears the sharp +/-1 transition layers (L2 = {l2['naive_vs_std']*100:.0f}% vs the spectral standard); the hard-constraint + RAR fix tracks them (L2 = {l2['adapted_vs_std']*100:.1f}%).",
        "note_es": f"La PINN SUAVE ingenua colapsa a un estado metaestable y difumina las capas de transición +/-1 (L2 = {l2['naive_vs_std']*100:.0f}% vs el estándar espectral); la corrección de restricciones duras + RAR las sigue (L2 = {l2['adapted_vs_std']*100:.1f}%).",
    }
    man_path.write_text(json.dumps(man, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"DONE in {(time.perf_counter()-t0)/60:.1f} min. L2={l2}", flush=True)


if __name__ == "__main__":
    main()
