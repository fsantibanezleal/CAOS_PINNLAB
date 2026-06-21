"""Group A · canonical-benchmark — Darcy-flow OPERATOR learning with a Fourier Neural Operator (FNO).

The one case that does NOT train a single PINN for a single boundary-value problem. It learns the solution OPERATOR
    G: a(x) |-> u(x)   for  -div(a(x) grad u) = 1,  u|_boundary = 0  on (0,1)^2,
over a whole FAMILY of permeability fields a(x) (two-value thresholded Gaussian random fields). One trained FNO maps
any new coefficient field to its pressure field in a single forward pass — no per-instance retraining. Engine: a
compact, self-contained 2D FNO (model/fno.py); data: the Li-et-al. Darcy benchmark generated in-build (datasets/darcy).

Pipeline fit: this is a CUSTOM-ENGINE, FIELD-IO case. It trains + exports its OWN ONNX in build() (field-in: the FNO
maps a coefficient FIELD to a solution FIELD, not coordinates -> a value), so it sets web_drivable=False and ships
lane=PRECOMPUTE — the browser replays a representative baked result (the App output selector shows the input field a,
the FNO prediction u_pred, and the FD reference u_true). The headline metric is the held-out TEST-set relative-L2 (the
real operator-generalization metric); the shipped ONNX is parity-checked against the model. real_or_synthetic =
synthetic (the analytic-coefficient Darcy dataset is the field-standard FNO benchmark; the reference u is a
finite-difference solve, a numerical anchor).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import CaseSpec

N_GRID = 32
WIDTH, MODES, LAYERS = 20, 10, 4
_REPO = Path(__file__).resolve().parents[3]
_STATE: dict[str, float] = {}  # build() stashes test/sample L2 for extra_metrics

CASE = CaseSpec(
    id="bench-darcy-operator",
    category="canonical-benchmark",
    title="Darcy-flow operator learning — Fourier Neural Operator G: a(x) -> u(x)",
    governing_equations=(
        r"-\nabla\!\cdot(a(\mathbf{x})\nabla u)=1,\ u|_{\partial\Omega}=0\ \text{on}\ (0,1)^2;\ "
        r"\text{learn the operator}\ \mathcal{G}_\theta: a\mapsto u\ \text{(FNO)}"
    ),
    method="operator-fno",
    engine="fno-torch",
    real_or_synthetic="synthetic",
    inputs=("x", "y"),
    outputs=("u_pred", "u_true", "a"),  # primary = FNO prediction; + FD reference + the input coefficient field
    domain={"x": (0.0, 1.0), "y": (0.0, 1.0)},
    grid={"x": N_GRID, "y": N_GRID},
    expected_band="one FNO maps any coefficient field a(x) to its pressure field in one pass; held-out test relative-L2 ~8-12%",
    validation_anchor="operator-test-l2",
    train={"lr": 1e-3, "adam": 0},  # bespoke training loop in build(); no DeepXDE Adam/L-BFGS
    notes="Custom-engine FIELD-IO case: trains a real 2D FNO on (a,u) pairs in build(), exports its own field-in ONNX "
          "(parity-checked), web_drivable=False -> lane=precompute. Headline = held-out test relative-L2.",
)


def extra_metrics(sf) -> dict:
    out = {}
    if "test_l2" in _STATE:
        out["l2_relative"] = round(float(_STATE["test_l2"]), 6)        # held-out operator generalization (headline)
        out["sample_l2_relative"] = round(float(_STATE["sample_l2"]), 6)
        out["n_test"] = int(_STATE["n_test"])
        out["ensemble_K"] = 0
    return out


class _Baked:
    """The infer model: returns the 3 baked physical fields (u_pred, u_true, a) at the eval grid (XY ignored — the
    fields are the FNO's representative-sample output, replayed)."""

    def __init__(self, fields):  # fields: np.ndarray [3, H, W] physical units, order (u_pred, u_true, a)
        self._f = np.asarray(fields, dtype=np.float64)

    def predict(self, XY):
        return np.stack([self._f[k].ravel() for k in range(self._f.shape[0])], axis=1)  # [H*W, 3]


def build(seed: int, quick: bool = False) -> dict:
    import time

    import onnxruntime as ort
    import torch

    from ..datasets.darcy import make_dataset
    from ..model.fno import FNO2d

    torch.manual_seed(int(seed))
    n_train, n_test, epochs = (8, 8, 3) if quick else (256, 64, 120)
    Xtr, Ytr, Xte, Yte, stats, (a_raw, u_raw) = make_dataset(seed, N_GRID, n_train, n_test)

    net = FNO2d(in_ch=3, width=WIDTH, modes=MODES, layers=LAYERS)
    opt = torch.optim.Adam(net.parameters(), lr=CASE.train["lr"])

    def rl2(p, y):
        num = torch.norm((p - y).reshape(len(p), -1), dim=1)
        den = torch.norm(y.reshape(len(y), -1), dim=1).clamp_min(1e-9)
        return (num / den).mean()

    bs = 20
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(n_train)
        for b in range(0, n_train, bs):
            idx = perm[b : b + bs]
            opt.zero_grad()
            loss = rl2(net(Xtr[idx]), Ytr[idx])
            loss.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        test_l2 = float(rl2(net(Xte), Yte).item())
        u_pred_n = net(Xte[:1]).numpy()[0, 0]                      # normalized prediction for test sample 0

    # representative sample (test #0) in PHYSICAL units
    u_pred = u_pred_n * stats["u_sd"] + stats["u_mu"]
    a_star = a_raw[n_train, 0]
    u_true = u_raw[n_train, 0]
    sample_l2 = float(np.linalg.norm(u_pred - u_true) / (np.linalg.norm(u_true) + 1e-12))
    _STATE.update({"test_l2": test_l2, "sample_l2": sample_l2, "n_test": n_test})

    # export the FNO's OWN field-in ONNX + parity + a field-forward timing
    onnx_path = _REPO / "models" / f"{CASE.id}.onnx"
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = Xte[:1]
    torch.onnx.export(
        net, (dummy,), str(onnx_path), input_names=["a_grid"], output_names=["u"],
        dynamic_axes={"a_grid": {0: "n"}, "u": {0: "n"}},
        opset_version=18, dynamo=True, verbose=False, external_data=False,
    )
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    k = min(8, n_test)
    with torch.no_grad():
        pt = net(Xte[:k]).numpy()
    ox = np.asarray(sess.run(["u"], {"a_grid": Xte[:k].numpy()})[0])
    parity = float(np.max(np.abs(pt - ox)))
    one = Xte[:1].numpy()
    sess.run(["u"], {"a_grid": one})  # warm up
    t0 = time.perf_counter()
    for _ in range(5):
        sess.run(["u"], {"a_grid": one})
    infer_ms = (time.perf_counter() - t0) * 1000.0 / 5

    fields = np.stack([u_pred, u_true, a_star.astype(np.float64)], axis=0)  # (3, H, W)
    return {
        "model": _Baked(fields),
        "input_dim": 2,
        "prebuilt": True,
        "onnx_path": str(onnx_path),
        "onnx_bytes": onnx_path.stat().st_size,
        "parity_max_abs": parity,
        "infer_ms": infer_ms,
        "opset": 18,
        "web_drivable": False,   # field-IO operator -> not coordinate-drivable in the browser -> lane=precompute
    }
