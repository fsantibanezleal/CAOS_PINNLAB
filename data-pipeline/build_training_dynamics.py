"""TRAINING DYNAMICS: bake "watch it learn" frame sequences - the field at training checkpoints, naive vs adapted.

The deepest PINN insight is HOW each lane converges (or fails to): the naive Helmholtz PINN stays a low-frequency
blur across all of training (spectral bias is a training pathology, not a capacity limit), while the Fourier lane
snaps onto the pattern; the naive soft Allen-Cahn PINN collapses toward a metastable state while the hard-constraint
lane sharpens the layers. This script trains each lane in CHUNKS and evaluates the field at CHECKPOINTS, baking an
animation the web can replay. All real fields from real training; L2 vs the exact solution recorded per checkpoint.

Writes data/derived/<case>/training.json and patches the manifest with {"training": {"path": ...}}.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DERIVED = REPO / "data" / "derived"
GRID = 61  # keep the multi-frame artifact compact
CHECKPOINTS = [0, 250, 500, 1000, 2000, 4000, 8000, 12000]


def rnd(a):
    return np.round(np.asarray(a, dtype=float), 4).tolist()


def train_with_checkpoints(model, predict_grid, exact, checkpoints):
    """Train in chunks; at each checkpoint eval the field + L2 vs exact. Returns (frames, l2s)."""
    frames, l2s = [], []
    prev = 0
    for cp in checkpoints:
        if cp > prev:
            model.train(iterations=cp - prev, display_every=max(1, (cp - prev)))
            prev = cp
        f = predict_grid(model)
        frames.append(rnd(f))
        l2s.append(round(float(np.linalg.norm(f - exact) / (np.linalg.norm(exact) or 1.0)), 5))
        print(f"    checkpoint {cp}: L2={l2s[-1]:.4f}", flush=True)
    return frames, l2s


def bake_helmholtz():
    from pinnlab.cases import ind_helmholtz as H

    print("[helmholtz] training dynamics (naive vs Fourier) ...", flush=True)
    x = np.linspace(0, 1, GRID)
    X, Y = np.meshgrid(x, x, indexing="ij")
    XY = np.stack([X.ravel(), Y.ravel()], axis=1).astype(np.float64)
    exact = (np.sin(H.K0 * X) * np.sin(H.K0 * Y))

    def predict(model):
        return np.asarray(model.predict(XY), dtype=np.float64)[:, 0].reshape(GRID, GRID)

    out = {"schema": "pinnlab.training/v1", "case_id": "ind-helmholtz",
           "checkpoints": CHECKPOINTS, "axes": {"x": rnd(x), "y": rnd(x)}, "lanes": {}}
    for key, fourier in (("naive", False), ("adapted", True)):
        print(f"  [{key}] ...", flush=True)
        m = H._helmholtz_model(42, fourier=fourier)
        frames, l2s = train_with_checkpoints(m, predict, exact, CHECKPOINTS)
        out["lanes"][key] = {"frames": frames, "l2": l2s,
                             "label_en": "naive tanh PINN" if key == "naive" else "Fourier-feature PINN",
                             "label_es": "PINN tanh ingenua" if key == "naive" else "PINN Fourier"}
    _write("ind-helmholtz", out)


def bake_allencahn():
    from pinnlab.cases import bench_allencahn as A

    print("[allencahn] training dynamics (naive soft vs hard-constraint) ...", flush=True)
    cps = [0, 250, 500, 1000, 2000, 4000, 6000]
    x = np.linspace(-1, 1, GRID)
    t = np.linspace(0, 1, GRID)
    Xx, Tt = np.meshgrid(x, t, indexing="ij")
    XY = np.stack([Xx.ravel(), Tt.ravel()], axis=1).astype(np.float64)
    # exact = the spectral reference subsampled onto this grid
    grids, _xy, _shape = A.eval_grid()
    ref = A.reference_on_grid()
    xi = [int(np.argmin(np.abs(grids["x"] - v))) for v in x]
    ti = [int(np.argmin(np.abs(grids["t"] - v))) for v in t]
    exact = ref[np.ix_(xi, ti)]

    def predict(model):
        return np.asarray(model.predict(XY), dtype=np.float64)[:, 0].reshape(GRID, GRID)

    out = {"schema": "pinnlab.training/v1", "case_id": "bench-allencahn",
           "checkpoints": cps, "axes": {"x": rnd(x), "t": rnd(t)}, "lanes": {}}
    for key, builder in (("naive", A.build_naive), ("adapted", A.build)):
        print(f"  [{key}] ...", flush=True)
        m = builder(42)["model"]
        frames, l2s = train_with_checkpoints(m, predict, exact, cps)
        out["lanes"][key] = {"frames": frames, "l2": l2s,
                             "label_en": "naive soft PINN" if key == "naive" else "hard-constraint PINN",
                             "label_es": "PINN suave ingenua" if key == "naive" else "PINN restricciones duras"}
    _write("bench-allencahn", out)


def _write(cid, out):
    (DERIVED / cid).mkdir(exist_ok=True)
    p = DERIVED / cid / "training.json"
    p.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    man_path = DERIVED / "manifests" / f"{cid}.json"
    man = json.loads(man_path.read_text(encoding="utf-8"))
    man["training"] = {"path": f"{cid}/training.json"}
    man_path.write_text(json.dumps(man, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"[{cid}] training.json baked ({p.stat().st_size/1024:.0f} KB)", flush=True)


if __name__ == "__main__":
    t0 = time.perf_counter()
    bake_helmholtz()
    bake_allencahn()
    print(f"ALL DONE in {(time.perf_counter()-t0)/60:.1f} min", flush=True)
