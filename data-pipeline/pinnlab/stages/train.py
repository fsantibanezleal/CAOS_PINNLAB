"""Stage 3 — train (OFFLINE, heavy SOTA engine): fit the case's PINN with DeepXDE (Adam -> L-BFGS), export the
trained network to ONNX, and verify ONNX-vs-model parity. This is where the deep-research engine is used FOR REAL;
the trained `.onnx` is exactly what the browser (onnxruntime-web) runs in the live lane. The output transform (hard
constraints) is pure tensor ops, so it is captured in the exported graph — the parity check proves it.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..model.analytic import linspace_grid
from ..registry import case_module, get_case


def run(case_id: str, *, seed: int, models_dir: str, sampling: dict | None = None, quick: bool = False) -> dict:
    """Train the case's PINN and export+verify its ONNX. `quick` is the CI smoke path: few iterations, no L-BFGS —
    it exercises the full train -> ONNX -> parity plumbing without waiting for convergence (do NOT bake real
    artifacts with quick=True)."""
    import onnxruntime as ort
    import torch

    import inspect

    case = get_case(case_id)
    mod = case_module(case_id)
    # custom-engine cases (deep ensembles, operator nets) train inside build() and pass quick through; the standard
    # DeepXDE cases keep the plain build(seed) signature.
    built = mod.build(seed, quick=quick) if "quick" in inspect.signature(mod.build).parameters else mod.build(seed)
    model = built["model"]
    d = int(built["input_dim"])
    t = case.train

    if built.get("prebuilt"):
        # the case already trained its own net (e.g. a deep ensemble exported as one [mean,std] graph); the generic
        # Adam->L-BFGS->refine loop does not apply. It must expose `.net` (torch module) + `.predict(X)` for export+parity.
        pass
    else:
        adam_iters = 300 if quick else int(t.get("adam", 12000))
        model.train(iterations=adam_iters, display_every=max(1, adam_iters // 6))
        if (not quick) and t.get("lbfgs", True):
            lw = t.get("loss_weights")
            model.compile("L-BFGS", loss_weights=lw) if lw is not None else model.compile("L-BFGS")
            model.train()

        # optional case-defined refinement (e.g. RAR adaptive sampling for sharp fronts) before export
        if (not quick) and getattr(mod, "refine", None) is not None:
            mod.refine(model, case, seed)

    # export the raw trained net to ONNX (input dim = d coordinates)
    net = model.net
    net.eval()
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    onnx_path = Path(models_dir) / f"{case_id}.onnx"
    dummy = torch.zeros(1, d, dtype=torch.float32)
    # Modern exporter (torch.export/dynamo, requires onnxscript) — the supported path going forward; the TorchScript
    # exporter is deprecated. dynamic batch axis so onnxruntime-web can evaluate any number of query coordinates.
    torch.onnx.export(
        net, (dummy,), str(onnx_path),
        input_names=["coords"], output_names=["u"],
        dynamic_axes={"coords": {0: "n"}, "u": {0: "n"}},
        opset_version=18, dynamo=True, verbose=False,
        external_data=False,  # embed weights -> a single self-contained .onnx for onnxruntime-web
    )

    # parity: ONNX must match model.predict on a random in-domain sample (proves the train->web bridge is faithful)
    rng = np.random.default_rng(seed + 1)
    lo = np.array([case.domain[a][0] for a in case.inputs], dtype=np.float64)
    hi = np.array([case.domain[a][1] for a in case.inputs], dtype=np.float64)
    sample = (lo + (hi - lo) * rng.random((512, d))).astype(np.float64)
    pred_dde = np.asarray(model.predict(sample), dtype=np.float64).reshape(len(sample), -1)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    pred_onnx = np.asarray(
        sess.run(["u"], {"coords": sample.astype(np.float32)})[0], dtype=np.float64
    ).reshape(len(sample), -1)
    parity = float(np.max(np.abs(pred_dde - pred_onnx)))

    # measure the ort-web-proxy inference time on the full eval grid (CPU onnxruntime stands in for ort-web)
    _coords, XY, _shape = linspace_grid(case.domain, case.grid)
    XYf = XY.astype(np.float32)
    sess.run(["u"], {"coords": XYf[:64]})  # warm up
    n_rep = 5
    t0 = time.perf_counter()
    for _ in range(n_rep):
        sess.run(["u"], {"coords": XYf})
    infer_ms = (time.perf_counter() - t0) * 1000.0 / n_rep

    return {
        "model": model,
        "onnx_path": str(onnx_path),
        "onnx_bytes": onnx_path.stat().st_size,
        "parity_max_abs": parity,
        "input_dim": d,
        "infer_ms": infer_ms,
        "opset": 18,
    }
