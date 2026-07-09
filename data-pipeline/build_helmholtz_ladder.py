"""Phase-1 reference: compute the REAL Helmholtz method ladder and bake it for the web.

Lanes (all REAL, no invented numbers):
  standard : classical 5-point FDM Helmholtz solve (ground truth, with its own dispersion error)
  analytic : the MMS closed form sin(k0 x) sin(k0 y)
  naive    : plain tanh MLP PINN (spectral bias -> fails at high k0)
  adapted  : Fourier-feature PINN (the fix)
Diagnostics:
  wavenumber sweep n=1..5 : naive vs adapted relative-L2 (the naive lane collapses as n grows)
  radial spectral energy  : |FFT(field)| vs |k| for standard/naive/adapted (the plateau naive cannot cross)

Writes:
  data/derived/ind-helmholtz/ladder-n3.json   (comparison trace: all fields on the 81x81 grid + per-lane L2)
  data/derived/ind-helmholtz/diagnostics.json (the sweep + spectral energy)
  models/ind-helmholtz-naive.onnx             (naive lane, for the Live toggle)
  patches data/derived/manifests/ind-helmholtz.json with `comparison` + `diagnostics` blocks
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

import deepxde as dde  # noqa: F401 (sets backend)
import torch

from pinnlab.cases import ind_helmholtz as H
from pinnlab.io.formats import write_json, strip_onnx_metadata

REPO = Path(__file__).resolve().parents[1]
DERIVED = REPO / "data" / "derived"
MODELS = REPO / "models"
GRID = 81
ADAM_MAIN = 12000
ADAM_SWEEP = 3500


def train_lane(fourier: bool, n_waves: int, adam: int, lbfgs: bool, seed: int = 42):
    model = H._helmholtz_model(seed, fourier=fourier, n_waves=n_waves)
    model.train(iterations=adam, display_every=max(1, adam // 4))
    if lbfgs:
        model.compile("L-BFGS", loss_weights=H.CASE.train["loss_weights"])
        model.train()
    return model


def field_on_grid(model, coords):
    X, Y = np.meshgrid(coords["x"], coords["y"], indexing="ij")
    XY = np.stack([X.ravel(), Y.ravel()], axis=1).astype(np.float64)
    u = np.asarray(model.predict(XY), dtype=np.float64).reshape(GRID, GRID)
    return u


def rel_l2(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) or 1.0))


def radial_spectrum(field):
    """Azimuthally-averaged |FFT| vs radial wavenumber — the spectral-energy profile."""
    F = np.abs(np.fft.fftshift(np.fft.fft2(field)))
    n = field.shape[0]
    cy = cx = n // 2
    yy, xx = np.mgrid[0:n, 0:n]
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2).astype(int)
    nbins = n // 2
    prof = np.array([F[r == k].mean() if np.any(r == k) else 0.0 for k in range(nbins)])
    return prof


def main():
    t0 = time.perf_counter()
    coords = {"x": np.linspace(0.0, 1.0, GRID), "y": np.linspace(0.0, 1.0, GRID)}
    X, Y = np.meshgrid(coords["x"], coords["y"], indexing="ij")

    print("[1/5] standard (FDM) + analytic ...", flush=True)
    u_std = H.standard_field(coords)
    u_analytic = (np.sin(H.K0 * X) * np.sin(H.K0 * Y))

    print("[2/5] train NAIVE (plain tanh) ...", flush=True)
    m_naive = train_lane(fourier=False, n_waves=H.N_WAVES, adam=ADAM_MAIN, lbfgs=True)
    u_naive = field_on_grid(m_naive, coords)

    print("[3/5] train ADAPTED (Fourier) ...", flush=True)
    m_adapt = train_lane(fourier=True, n_waves=H.N_WAVES, adam=ADAM_MAIN, lbfgs=True)
    u_adapt = field_on_grid(m_adapt, coords)

    # errors vs the STANDARD classical solution (the ground truth we compare against)
    err_naive = np.abs(u_naive - u_std)
    err_adapt = np.abs(u_adapt - u_std)
    l2 = {
        "naive_vs_std": round(rel_l2(u_naive, u_std), 5),
        "adapted_vs_std": round(rel_l2(u_adapt, u_std), 5),
        "naive_vs_analytic": round(rel_l2(u_naive, u_analytic), 5),
        "adapted_vs_analytic": round(rel_l2(u_adapt, u_analytic), 5),
        "std_vs_analytic": round(rel_l2(u_std, u_analytic), 5),
    }
    print("   L2:", l2, flush=True)

    print("[4/5] wavenumber sweep n=1..5 (naive vs adapted) ...", flush=True)
    sweep = {"n": [], "naive": [], "adapted": []}
    for n in (1, 2, 3, 4, 5):
        cg = {"x": np.linspace(0, 1, GRID), "y": np.linspace(0, 1, GRID)}
        k0 = 2 * np.pi * n
        Xn, Yn = np.meshgrid(cg["x"], cg["y"], indexing="ij")
        exact = np.sin(k0 * Xn) * np.sin(k0 * Yn)
        mn = train_lane(fourier=False, n_waves=n, adam=ADAM_SWEEP, lbfgs=False)
        ma = train_lane(fourier=True, n_waves=n, adam=ADAM_SWEEP, lbfgs=False)
        ln = rel_l2(field_on_grid(mn, cg), exact)
        la = rel_l2(field_on_grid(ma, cg), exact)
        sweep["n"].append(n); sweep["naive"].append(round(ln, 4)); sweep["adapted"].append(round(la, 4))
        print(f"   n={n}: naive L2={ln:.3f}  adapted L2={la:.3f}", flush=True)

    spec = {
        "k": list(range(GRID // 2)),
        "standard": [round(float(v), 4) for v in radial_spectrum(u_std)],
        "naive": [round(float(v), 4) for v in radial_spectrum(u_naive)],
        "adapted": [round(float(v), 4) for v in radial_spectrum(u_adapt)],
    }

    print("[5/5] export naive ONNX + bake trace + diagnostics + patch manifest ...", flush=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    net = m_naive.net
    net.eval()
    onnx_path = MODELS / "ind-helmholtz-naive.onnx"
    torch.onnx.export(
        net, (torch.zeros(1, 2, dtype=torch.float32),), str(onnx_path),
        input_names=["coords"], output_names=["u"],
        dynamic_axes={"coords": {0: "n"}, "u": {0: "n"}}, opset_version=18, dynamo=True,
        verbose=False, external_data=False,
    )
    strip_onnx_metadata(onnx_path)

    def rnd(a):
        return np.round(a.astype(float), 5).tolist()

    trace = {
        "schema": "pinnlab.compare/v1",
        "case_id": "ind-helmholtz",
        "dims": ["x", "y"],
        "axes": {"x": [round(float(v), 5) for v in coords["x"]], "y": [round(float(v), 5) for v in coords["y"]]},
        "fields": {
            "standard": rnd(u_std),
            "analytic": rnd(u_analytic),
            "naive": rnd(u_naive),
            "adapted": rnd(u_adapt),
            "err_naive": rnd(err_naive),
            "err_adapted": rnd(err_adapt),
        },
        "summary": l2,
    }
    write_json(DERIVED / "ind-helmholtz" / "ladder-n3.json", trace)
    write_json(DERIVED / "ind-helmholtz" / "diagnostics.json", {
        "schema": "pinnlab.diagnostics/v1",
        "case_id": "ind-helmholtz",
        "wavenumber_sweep": sweep,
        "radial_spectrum": spec,
        "l2": l2,
    })

    man_path = DERIVED / "manifests" / "ind-helmholtz.json"
    man = json.loads(man_path.read_text(encoding="utf-8"))
    man["comparison"] = {
        "trace": "ind-helmholtz/ladder-n3.json",
        "lanes": [
            {"key": "standard", "label_en": "standard (FDM)", "label_es": "estándar (FDM)", "role": "reference"},
            {"key": "naive", "label_en": "naive PINN", "label_es": "PINN ingenua", "role": "baseline", "err": "err_naive"},
            {"key": "adapted", "label_en": "Fourier-feature PINN", "label_es": "PINN Fourier", "role": "fix", "err": "err_adapted"},
            {"key": "analytic", "label_en": "analytic (MMS)", "label_es": "analítica (MMS)", "role": "exact"},
        ],
        "onnx_naive": "ind-helmholtz-naive.onnx",
        "note_en": "The naive plain-tanh PINN suffers spectral bias and cannot resolve the high-wavenumber pattern; the Fourier-feature embedding fixes it. Both are compared to the classical FDM standard solution.",
        "note_es": "La PINN ingenua (tanh) sufre sesgo espectral y no resuelve el patrón de alto número de onda; las características de Fourier lo corrigen. Ambas se comparan con la solución estándar FDM.",
    }
    man["diagnostics"] = {"path": "ind-helmholtz/diagnostics.json"}
    write_json(man_path, man)

    print(f"DONE in {(time.perf_counter()-t0)/60:.1f} min. L2={l2}", flush=True)


if __name__ == "__main__":
    main()
