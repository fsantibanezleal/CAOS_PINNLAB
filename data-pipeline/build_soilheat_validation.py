"""soil-heat-real validation: the PINN's reconstructed temperature vs the REAL measured USCRN sensors at the
HELD-OUT depths (10/20/50 cm) - out-of-sample, in degrees C. This is the honest real-data validation: those interior
sensors were NEVER shown to the optimizer, so the fit is a genuine out-of-sample test of the recovered diffusivity.

Baked as a `diagnostics` block (line comparisons: measured points vs PINN curve) that the DiagnosticsKit renders.
All real: the measured series is the USCRN data; the PINN series is the baked field interpolated to those depths.
"""
from __future__ import annotations

import json
import io
from pathlib import Path

import numpy as np

from pinnlab.cases import env_soil_heat_real as S

DERIVED = Path(__file__).resolve().parents[1] / "data" / "derived"
CID = "env-soil-heat-real"


def main():
    man = json.loads((DERIVED / "manifests" / f"{CID}.json").read_text(encoding="utf-8"))
    var = man["variants"][0]
    tr = json.loads((DERIVED / var["trace"]["path"]).read_text(encoding="utf-8"))
    z_axis = np.asarray(tr["axes"]["z"], float)   # normalized depth in [0,1]
    t_axis = np.asarray(tr["axes"]["t"], float)   # normalized time in [0,1]
    field = np.asarray(tr["fields"]["T"], float)  # [iz][it], NORMALIZED T

    days = np.asarray(S._D["t_grid_days_from_2019_01_01"], float)
    t_days_axis = t_axis * days[-1]  # trace normalized t -> days
    lc = []
    rmse = {}
    colors = {10: "var(--accent)", 20: "var(--good)", 50: "var(--accent-2)"}
    for d in S._HOLDOUT:  # [10,20,50]
        zeta = S._ZETA[d]
        pinn_norm = np.array([np.interp(zeta, z_axis, field[:, j]) for j in range(field.shape[1])])
        pinn_c = pinn_norm * S._SCALE + S._MEAN                    # denormalize to degrees C
        meas_c = np.asarray(S._D["temp_c"][str(d)], float)         # REAL measured, degrees C, on the full day grid
        meas_on_axis = np.interp(t_days_axis, days, meas_c)        # subsample the real series to the trace time axis
        r = float(np.sqrt(np.mean((pinn_c - meas_on_axis) ** 2)))
        rmse[f"{d}cm"] = round(r, 3)
        lc.append({
            "title_en": f"{d} cm (HELD OUT): PINN reconstruction vs REAL measured temperature - RMSE {r:.2f} degC",
            "title_es": f"{d} cm (FUERA DE MUESTRA): reconstruccion PINN vs temperatura REAL medida - RMSE {r:.2f} degC",
            "xLabel": "days from 2019-01-01", "yLabel": "T (degC)",
            "series": [
                {"label": f"measured {d} cm (real)", "color": "var(--muted)", "scatter": True,
                 "x": [round(float(x), 1) for x in t_days_axis], "y": [round(float(y), 3) for y in meas_on_axis]},
                {"label": "PINN (out-of-sample)", "color": colors.get(d, "var(--good)"),
                 "x": [round(float(x), 1) for x in t_days_axis], "y": [round(float(y), 3) for y in pinn_c]},
            ],
        })
    print("held-out RMSE (degC):", rmse, flush=True)

    diag = {"schema": "pinnlab.diagnostics/v1", "case_id": CID, "line_comparisons": lc, "rmse": rmse}
    (DERIVED / CID).mkdir(exist_ok=True)
    (DERIVED / CID / "diagnostics.json").write_text(json.dumps(diag, ensure_ascii=False), encoding="utf-8")
    man["diagnostics"] = {"path": f"{CID}/diagnostics.json"}
    (DERIVED / "manifests" / f"{CID}.json").write_text(json.dumps(man, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("soil-heat-real diagnostics baked (held-out real-sensor validation).", flush=True)


if __name__ == "__main__":
    main()
