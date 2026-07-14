"""ind-hidden-velocity validation + index registration (issue #48).

1) HELD-OUT dye validation: 160 of the 800 seeded dye samples were never shown to the optimizer. The exported
   ONNX is evaluated at those space-time points and the RMSE vs the (noisy) held-out measurements is written
   into every variant trace summary + the manifest as `dye_holdout_rmse`: a genuine out-of-sample check that
   the reconstructed dye (the only observed quantity) generalizes.
2) A diagnostics block: recovered u and v PROFILES along the mid-lines vs the closed-form truth (line
   comparisons the DiagnosticsKit renders), the direct visual read of "the hidden current was recovered".
3) Registers the case in manifests/index.json (single-case precompute does not touch the index; run_all would
   rewrite ALL manifests and strip the enrichment blocks, so the index entry is patched here instead).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pinnlab.cases import ind_hidden_velocity as HV

REPO = Path(__file__).resolve().parents[1]
DERIVED = REPO / "data" / "derived"
MANIFESTS = DERIVED / "manifests"
MODELS = REPO / "models"
CID = "ind-hidden-velocity"
SEED = 42


def main():
    man = json.loads((MANIFESTS / f"{CID}.json").read_text(encoding="utf-8"))

    # ---- 1) held-out dye RMSE via the exported ONNX ----
    import onnxruntime as ort

    sess = ort.InferenceSession(str(MODELS / man["onnx"]["path"]), providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    obs = HV.dye_observations(SEED)
    pred = sess.run(None, {iname: obs["hold_xyz"].astype(np.float32)})[0]  # [N, 3] = (u, v, c)
    c_pred = pred[:, 2:3].astype(np.float64)
    rmse = float(np.sqrt(np.mean((c_pred - obs["hold_c"]) ** 2)))
    c_scale = float(np.abs(obs["hold_c"]).max())
    print(f"held-out dye RMSE: {rmse:.5f} (max |c_obs| {c_scale:.3f}, rel {rmse / c_scale * 100:.2f}%)", flush=True)

    # ---- 2) diagnostics: recovered current profiles vs the closed form on the mid-lines ----
    n = 81
    xs = np.linspace(0.0, 1.0, n)
    mid = 0.5
    # u along y at x=0.5 (u* = A sin(pi/2) cos(pi y) = A cos(pi y)) and v along x at y=0.5 (v* = -A cos(pi x))
    pts_u = np.column_stack([np.full(n, mid), xs, np.full(n, 0.5)]).astype(np.float32)
    pts_v = np.column_stack([xs, np.full(n, mid), np.full(n, 0.5)]).astype(np.float32)
    u_rec = sess.run(None, {iname: pts_u})[0][:, 0].astype(np.float64)
    v_rec = sess.run(None, {iname: pts_v})[0][:, 1].astype(np.float64)
    u_star = HV.u_true(np.column_stack([np.full(n, mid), xs])).reshape(-1)
    v_star = HV.v_true(np.column_stack([xs, np.full(n, mid)])).reshape(-1)
    lc = [
        {
            "title_en": "Recovered u along x=0.5 vs the closed-form truth (never measured, estimated from dye)",
            "title_es": "u recuperada a lo largo de x=0.5 vs la verdad cerrada (nunca medida, estimada desde el tinte)",
            "xLabel": "y", "yLabel": "u",
            "series": [
                {"label": "u* (closed form)", "color": "var(--muted)", "x": [round(float(v), 4) for v in xs], "y": [round(float(v), 4) for v in u_star]},
                {"label": "u recovered from dye", "color": "var(--good)", "x": [round(float(v), 4) for v in xs], "y": [round(float(v), 4) for v in u_rec]},
            ],
        },
        {
            "title_en": "Recovered v along y=0.5 vs the closed-form truth",
            "title_es": "v recuperada a lo largo de y=0.5 vs la verdad cerrada",
            "xLabel": "x", "yLabel": "v",
            "series": [
                {"label": "v* (closed form)", "color": "var(--muted)", "x": [round(float(v), 4) for v in xs], "y": [round(float(v), 4) for v in v_star]},
                {"label": "v recovered from dye", "color": "var(--accent)", "x": [round(float(v), 4) for v in xs], "y": [round(float(v), 4) for v in v_rec]},
            ],
        },
    ]
    diag = {"schema": "pinnlab.diagnostics/v1", "case_id": CID, "line_comparisons": lc, "rmse": {"dye_holdout": round(rmse, 6)}}
    (DERIVED / CID).mkdir(exist_ok=True)
    (DERIVED / CID / "diagnostics.json").write_text(json.dumps(diag, ensure_ascii=False), encoding="utf-8")
    man["diagnostics"] = {"path": f"{CID}/diagnostics.json"}

    # patch the holdout RMSE into the manifest variants + each trace summary
    for var in man["variants"]:
        var["metrics"]["dye_holdout_rmse"] = round(rmse, 6)
        tp = DERIVED / var["trace"]["path"]
        tr = json.loads(tp.read_text(encoding="utf-8"))
        tr.setdefault("summary", {})["dye_holdout_rmse"] = round(rmse, 6)
        tp.write_text(json.dumps(tr, ensure_ascii=False), encoding="utf-8")
        var["trace"]["bytes"] = tp.stat().st_size
    (MANIFESTS / f"{CID}.json").write_text(json.dumps(man, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    # ---- 3) index registration (append once, ordered after ind-heat2d-inverse; n_cases updated) ----
    idx = json.loads((MANIFESTS / "index.json").read_text(encoding="utf-8"))
    if not any(c["case_id"] == CID for c in idx["cases"]):
        entry = {
            "case_id": CID, "category": man["category"], "title": man["title"],
            "manifest_path": f"manifests/{CID}.json",
            "system_type": man.get("system_type"), "view_kit": man.get("view_kit"),
            "method": man.get("method"), "real_or_synthetic": man.get("real_or_synthetic"),
        }
        pos = next((i for i, c in enumerate(idx["cases"]) if c["case_id"] == "ind-heat2d-inverse"), len(idx["cases"]) - 1)
        idx["cases"].insert(pos + 1, entry)
        idx["n_cases"] = len(idx["cases"])
        (MANIFESTS / "index.json").write_text(json.dumps(idx, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"index.json: {CID} registered, n_cases={idx['n_cases']}")
    else:
        print("index.json: already registered")


if __name__ == "__main__":
    main()
