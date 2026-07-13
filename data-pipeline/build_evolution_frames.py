"""Bake SMOOTH 2-D time-frame sequences for the transport cases (issue #36): offline ONNX evals of the trained
net at N time steps -> an animated Field view (SpatioTemporalKit). No training - the net is already trained; this
just evaluates it on a time grid so the web can PLAY the evolution instead of showing one static snapshot.

Cases: poll-ocean-transport (drifting/spreading pollutant patch), mine-heap-leach-rt (advecting reacting fronts).
Patches each manifest with {"evolution": {"path": ...}, "view_kit": "SpatioTemporalKit"} + the index.json badge.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

REPO = Path(__file__).resolve().parents[1]
DERIVED = REPO / "data" / "derived"
MODELS = REPO / "models"
GRID = 61

CASES = [
    ("poll-ocean-transport", 24),
    ("mine-heap-leach-rt", 16),
]


def rnd(a):
    return np.round(np.asarray(a, dtype=float), 4).tolist()


def bake(cid: str, n_frames: int):
    man_path = DERIVED / "manifests" / f"{cid}.json"
    man = json.loads(man_path.read_text(encoding="utf-8"))
    fa = man["field_axes"]
    inputs = man["inputs"]
    outputs = man["outputs"]
    tspec = next((p for p in man.get("param_specs", []) if p["key"] not in fa), None)
    if tspec is None:
        print(f"[{cid}] SKIP: no non-axis parameter to sweep"); return
    tkey = tspec["key"]
    tvals = np.linspace(float(tspec["min"]), float(tspec["max"]), n_frames)

    # the spatial grid (from the case domain recorded in the manifest? domain not in manifest -> use the baked trace axes)
    tr = json.loads((DERIVED / man["variants"][0]["trace"]["path"]).read_text(encoding="utf-8"))
    ax0_full = np.asarray(tr["axes"][fa[0]], float)
    ax1_full = np.asarray(tr["axes"][fa[1]], float)
    ax0 = np.linspace(ax0_full[0], ax0_full[-1], GRID)
    ax1 = np.linspace(ax1_full[0], ax1_full[-1], GRID)
    X0, X1 = np.meshgrid(ax0, ax1, indexing="ij")
    n = GRID * GRID

    sess = ort.InferenceSession(str(MODELS / man["onnx"]["path"]), providers=["CPUExecutionProvider"])
    frames: dict[str, list] = {o: [] for o in outputs}
    for tv in tvals:
        XY = np.empty((n, len(inputs)), dtype=np.float32)
        for j, ax in enumerate(inputs):
            XY[:, j] = X0.ravel() if ax == fa[0] else X1.ravel() if ax == fa[1] else float(tv)
        out = sess.run(["u"], {"coords": XY})[0]
        out = np.asarray(out, dtype=np.float64).reshape(n, -1)
        for k, o in enumerate(outputs):
            frames[o].append(rnd(out[:, k].reshape(GRID, GRID)))

    payload = {
        "schema": "pinnlab.evolution/v1", "case_id": cid,
        "t": [round(float(v), 5) for v in tvals],
        "axes": {fa[0]: rnd(ax0), fa[1]: rnd(ax1)},
        "frames": frames,
    }
    (DERIVED / cid).mkdir(exist_ok=True)
    p = DERIVED / cid / "frames.json"
    p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    man["evolution"] = {"path": f"{cid}/frames.json"}
    man["view_kit"] = "SpatioTemporalKit"
    man_path.write_text(json.dumps(man, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"[{cid}] {n_frames} frames x {len(outputs)} outputs baked ({p.stat().st_size/1024:.0f} KB); view_kit=SpatioTemporalKit")


def patch_index():
    idx_path = DERIVED / "manifests" / "index.json"
    idx = json.loads(idx_path.read_text(encoding="utf-8"))
    for c in idx.get("cases", []):
        if c["case_id"] in {cid for cid, _ in CASES}:
            c["view_kit"] = "SpatioTemporalKit"
    idx_path.write_text(json.dumps(idx, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("index.json view_kit badges updated")


if __name__ == "__main__":
    for cid, nf in CASES:
        bake(cid, nf)
    patch_index()
