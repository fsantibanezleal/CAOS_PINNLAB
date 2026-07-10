"""navier-cavity validation: the canonical Ghia-Ghia-Shin (1982) Re=100 centerline comparison.

No finicky field solver: the PINN velocity along the two cavity centerlines is compared to Ghia's tabulated
benchmark points (the standard way lid-driven cavity solutions are validated). Baked as a `diagnostics` block that
the DiagnosticsKit renders (Ghia = points, PINN = line). Real numbers from the baked PINN field.
"""
from __future__ import annotations

import json
import io
from pathlib import Path

import numpy as np

DERIVED = Path(__file__).resolve().parents[1] / "data" / "derived"
CID = "bench-navier-cavity"

# Ghia et al. (1982), Re=100. u along the vertical centerline x=0.5 (y, u); v along y=0.5 (x, v).
GHIA_U_Y = [0.0, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531, 0.5, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 0.9766, 1.0]
GHIA_U = [0.0, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150, -0.15662, -0.21090, -0.20581, -0.13641, 0.00332, 0.23151, 0.68717, 0.73722, 0.78871, 0.84123, 1.0]
GHIA_V_X = [0.0, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563, 0.2266, 0.2344, 0.5, 0.8047, 0.8594, 0.9063, 0.9453, 0.9531, 0.9609, 0.9688, 1.0]
GHIA_V = [0.0, 0.09233, 0.10091, 0.10890, 0.12317, 0.16077, 0.17507, 0.17527, 0.05454, -0.24533, -0.22445, -0.16914, -0.10313, -0.08864, -0.07391, -0.05906, 0.0]


def col_at(field, axis_vals, target):
    """The field column/row interpolated to `target` along the given axis (linear between the two nearest lines)."""
    a = np.asarray(axis_vals, float)
    i = int(np.clip(np.searchsorted(a, target) - 1, 0, len(a) - 2))
    t = (target - a[i]) / (a[i + 1] - a[i] or 1)
    return (1 - t) * field[i] + t * field[i + 1]


def main():
    man = json.loads((DERIVED / "manifests" / f"{CID}.json").read_text(encoding="utf-8"))
    var = man["variants"][0]
    tr = json.loads((DERIVED / var["trace"]["path"]).read_text(encoding="utf-8"))
    ax = np.asarray(tr["axes"]["x"], float)
    ay = np.asarray(tr["axes"]["y"], float)
    u = np.asarray(tr["fields"]["u"], float)  # [ix][iy]
    v = np.asarray(tr["fields"]["v"], float)

    # u along the vertical centerline x=0.5 -> u(y); v along y=0.5 -> v(x)
    u_center = col_at(u, ax, 0.5)          # length ny, over ay
    v_center = np.array([col_at(v[i], ay, 0.5) for i in range(len(ax))])  # length nx, over ax

    # errors vs Ghia (interpolate the PINN curve onto Ghia's points)
    rmse_u = float(np.sqrt(np.mean((np.interp(GHIA_U_Y, ay, u_center) - GHIA_U) ** 2)))
    rmse_v = float(np.sqrt(np.mean((np.interp(GHIA_V_X, ax, v_center) - GHIA_V) ** 2)))
    print(f"navier centerline RMSE vs Ghia: u={rmse_u:.4f}  v={rmse_v:.4f}", flush=True)

    diag = {
        "schema": "pinnlab.diagnostics/v1", "case_id": CID,
        "line_comparisons": [
            {"title_en": "u along the vertical centerline (x=0.5) vs Ghia 1982",
             "title_es": "u en la linea central vertical (x=0.5) vs Ghia 1982",
             "xLabel": "y", "yLabel": "u",
             "series": [
                 {"label": "Ghia 1982", "color": "var(--muted)", "scatter": True, "x": [round(float(y), 4) for y in GHIA_U_Y], "y": [round(float(uu), 4) for uu in GHIA_U]},
                 {"label": "PINN", "color": "var(--good)", "x": [round(float(y), 4) for y in ay], "y": [round(float(uu), 4) for uu in u_center]},
             ]},
            {"title_en": "v along the horizontal centerline (y=0.5) vs Ghia 1982",
             "title_es": "v en la linea central horizontal (y=0.5) vs Ghia 1982",
             "xLabel": "x", "yLabel": "v",
             "series": [
                 {"label": "Ghia 1982", "color": "var(--muted)", "scatter": True, "x": [round(float(x), 4) for x in GHIA_V_X], "y": [round(float(vv), 4) for vv in GHIA_V]},
                 {"label": "PINN", "color": "var(--good)", "x": [round(float(x), 4) for x in ax], "y": [round(float(vv), 4) for vv in v_center]},
             ]},
        ],
        "rmse": {"u_centerline": round(rmse_u, 4), "v_centerline": round(rmse_v, 4)},
    }
    (DERIVED / CID).mkdir(exist_ok=True)
    (DERIVED / CID / "diagnostics.json").write_text(json.dumps(diag, ensure_ascii=False), encoding="utf-8")
    man["diagnostics"] = {"path": f"{CID}/diagnostics.json"}
    (DERIVED / "manifests" / f"{CID}.json").write_text(json.dumps(man, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("navier diagnostics baked (Ghia centerline validation).", flush=True)


if __name__ == "__main__":
    main()
