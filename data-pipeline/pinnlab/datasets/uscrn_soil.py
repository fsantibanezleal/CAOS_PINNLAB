"""Vendor REAL soil-temperature observations from NOAA's USCRN open archive for `env-soil-heat-real`.

The U.S. Climate Reference Network reports daily-mean soil temperature at five standard depths (5, 10, 20, 50, 100 cm)
at research-grade stations. Subsurface heat conduction is a genuinely diffusive 1D system — the surface temperature
signal propagates downward damped + phase-lagged exactly as the heat equation predicts — so this is the ideal REAL
dataset for a PINN inverse that recovers thermal diffusivity and validates it out-of-sample against held-out depths.

Station IL_Champaign_9_SW (central Illinois) over 2019–2021 has complete soil-temperature records (0% missing in 2021,
near-complete prior years). The shallow (5 cm) and deep (100 cm) sensors are the Dirichlet boundaries; the 10/20/50 cm
sensors are held out to validate the recovered diffusivity against real interior temperatures.

Run once to (re)generate `data/reference/uscrn/soil_temp_il_champaign.json`; the case + CI read the vendored file
offline. This IS the ingestion data-contract in action for a real dataset.
"""
from __future__ import annotations

import json
import urllib.request
from datetime import date
from pathlib import Path

import numpy as np

BASE = "https://www.ncei.noaa.gov/pub/data/uscrn/products/daily01/{year}/CRND0103-{year}-{station}.txt"
STATION = "IL_Champaign_9_SW"
YEARS = [2019, 2020, 2021]
DEPTHS_CM = [5, 10, 20, 50, 100]
COLS = {5: 23, 10: 24, 20: 25, 50: 26, 100: 27}  # SOIL_TEMP_{d}_DAILY column indices (USCRN headers.txt)
DATE_COL = 1
MISSING = {"-9999.0", "-99999", "-9999", "-99.000"}
N_T = 81  # field time grid (<= MAX_AXIS); ~13.5-day spacing over 3 yr


def _fetch_year(year: int) -> tuple[list[int], dict[int, list[float]]]:
    url = BASE.format(year=year, station=STATION)
    req = urllib.request.Request(url, headers={"User-Agent": "pinnlab/0.3 (research)"})
    txt = urllib.request.urlopen(req, timeout=60).read().decode()
    epoch = date(2019, 1, 1).toordinal()
    days: list[int] = []
    cols: dict[int, list[float]] = {d: [] for d in DEPTHS_CM}
    for line in txt.splitlines():
        r = line.split()
        if not r:
            continue
        ds = r[DATE_COL]
        day = date(int(ds[:4]), int(ds[4:6]), int(ds[6:8])).toordinal() - epoch
        days.append(day)
        for d in DEPTHS_CM:
            v = r[COLS[d]]
            cols[d].append(np.nan if v in MISSING else float(v))
    return days, cols


def _interp_nan(y: np.ndarray) -> np.ndarray:
    y = y.copy()
    idx = np.arange(len(y))
    good = ~np.isnan(y)
    y[~good] = np.interp(idx[~good], idx[good], y[good])
    return y


def fetch_and_vendor(out_path: str | Path) -> dict:
    all_days: list[int] = []
    raw: dict[int, list[float]] = {d: [] for d in DEPTHS_CM}
    for yr in YEARS:
        days, cols = _fetch_year(yr)
        all_days += days
        for d in DEPTHS_CM:
            raw[d] += cols[d]
    days = np.array(all_days, dtype=np.float64)
    order = np.argsort(days)
    days = days[order]
    series = {d: _interp_nan(np.array(raw[d], dtype=np.float64)[order]) for d in DEPTHS_CM}

    t_days = np.linspace(days.min(), days.max(), N_T)
    resamp = {d: np.interp(t_days, days, series[d]) for d in DEPTHS_CM}
    allv = np.concatenate(list(resamp.values()))
    mean, scale = float(allv.mean()), float(allv.std())

    span_s = float((days.max() - days.min()) * 86400.0)  # seconds spanned (for physical alpha conversion)
    payload = {
        "schema": "pinnlab.dataset.uscrn/v1",
        "source": "NOAA USCRN daily01 — SOIL_TEMP_{5,10,20,50,100}_DAILY (deg C)",
        "url_template": BASE,
        "fetched": str(date.today()),
        "station": STATION,
        "years": YEARS,
        "depths_cm": DEPTHS_CM,
        "boundary_depths_cm": [5, 100],
        "holdout_depths_cm": [10, 20, 50],
        "t_grid_days_from_2019_01_01": [round(float(x), 2) for x in t_days],
        "span_seconds": span_s,
        "L_meters": (DEPTHS_CM[-1] - DEPTHS_CM[0]) / 100.0,  # 0.95 m column
        "temp_c": {str(d): [round(float(x), 3) for x in resamp[d]] for d in DEPTHS_CM},
        "normalization": {"mean_c": round(mean, 4), "scale_c": round(scale, 4)},
        "note": "Subsurface heat conduction is diffusive: T(z,t) obeys T_t = alpha T_zz. 5 cm + 100 cm are Dirichlet "
                "boundaries; 10/20/50 cm are held out to validate the recovered thermal diffusivity out-of-sample.",
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


if __name__ == "__main__":
    here = Path(__file__).resolve().parents[3]
    p = here / "data" / "reference" / "uscrn" / "soil_temp_il_champaign.json"
    info = fetch_and_vendor(p)
    print("vendored", p)
    print("station:", info["station"], "years:", info["years"], "n_t:", len(info["t_grid_days_from_2019_01_01"]))
    print("norm:", info["normalization"], "L_m:", info["L_meters"], "span_s:", info["span_seconds"])
