"""Group A · canonical-benchmark — Darcy-flow OPERATOR learning with a Fourier Neural Operator (FNO).

The one case that does NOT train a single PINN for a single boundary-value problem. It learns the solution OPERATOR
    G: a(x) |-> u(x)   for  -div(a(x) grad u) = 1,  u|_boundary = 0  on (0,1)^2,
over a whole FAMILY of permeability fields a(x) (two-value thresholded Gaussian random fields). One trained FNO maps
any new coefficient field to its pressure field in a single forward pass — no per-instance retraining. Engine: a
compact, self-contained 2D FNO (model/fno.py); data: the Li-et-al. Darcy benchmark generated in-build (datasets/darcy).

Pipeline fit: this is a CUSTOM-ENGINE, FIELD-IO case. It trains + exports its OWN ONNX in build() (field-in: the FNO
maps a coefficient FIELD to a solution FIELD, not coordinates -> a value), so it sets web_drivable=False and ships
lane=PRECOMPUTE — the browser replays each baked result (the App output selector shows the input field a, the FNO
prediction u_pred, and the FD reference u_true). The workbench variants are a DISCRETE family of held-out test samples
the FNO never saw (the point of an operator: generalize to new inputs in one pass). The headline metric is the
held-out TEST-set relative-L2 (the real operator-generalization number); each chip also reports its own sample L2; the
shipped ONNX is parity-checked against the model. real_or_synthetic = synthetic (the analytic-coefficient Darcy
dataset is the field-standard FNO benchmark; the reference u is a finite-difference solve, a numerical anchor).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import CaseSpec, Variant

N_GRID = 32
WIDTH, MODES, LAYERS = 20, 10, 4
N_VIEW = 6                              # held-out samples shown as discrete workbench variants
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
    field_axes=("x", "y"),              # explicit heatmap axes (== inputs; no parameter axis — variants are discrete samples)
    expected_band="one FNO maps any coefficient field a(x) to its pressure field in one pass; held-out test relative-L2 ~5-12%",
    validation_anchor="operator-test-l2",
    train={"lr": 1e-3, "adam": 0},  # bespoke training loop in build(); no DeepXDE Adam/L-BFGS
    notes="Custom-engine FIELD-IO case: trains a real 2D FNO on (a,u) pairs in build(), exports its own field-in ONNX "
          "(parity-checked), web_drivable=False -> lane=precompute. Variants = held-out test samples (discrete). "
          "Headline = held-out test-set relative-L2; each chip also reports its own sample L2.",
)


def variants() -> list[Variant]:
    """The DISCRETE family: six held-out coefficient fields the FNO never saw at training. Each chip shows the SAME
    frozen operator mapping a new a(x) to its pressure field in one forward pass (the point of operator learning)."""
    presets = [
        ("s1", "Held-out a #1", "a fuera de muestra #1",
         "An unseen permeability field — one frozen FNO maps it to its pressure in a single pass.",
         "Un campo de permeabilidad no visto — un FNO congelado lo mapea a su presión en una sola pasada."),
        ("s2", "Held-out a #2", "a fuera de muestra #2",
         "A different channel geometry — same operator, no retraining.",
         "Una geometría de canales distinta — el mismo operador, sin reentrenar."),
        ("s3", "Held-out a #3", "a fuera de muestra #3",
         "More tortuous high-permeability paths — the FNO still recovers the pressure field.",
         "Caminos de alta permeabilidad más tortuosos — el FNO aún recupera el campo de presión."),
        ("s4", "Held-out a #4", "a fuera de muestra #4",
         "A blockier conductivity pattern — tests the operator on coarser interfaces.",
         "Un patrón de conductividad más en bloques — prueba el operador en interfaces más gruesas."),
        ("s5", "Held-out a #5", "a fuera de muestra #5",
         "Thin connected channels — the hardest pressure gradients.",
         "Canales delgados conectados — los gradientes de presión más difíciles."),
        ("s6", "Held-out a #6", "a fuera de muestra #6",
         "Another unseen instance — the operator generalizes across the whole family.",
         "Otra instancia no vista — el operador generaliza sobre toda la familia."),
    ]
    return [Variant(vid, le, ls, {"sample": i}, ne, ns)
            for i, (vid, le, ls, ne, ns) in enumerate(presets)]


def extra_metrics(sf) -> dict:
    out = {}
    if "test_l2" in _STATE:
        out["l2_relative"] = round(float(_STATE["test_l2"]), 6)        # held-out TEST-set mean (headline, same every chip)
        out["sample_l2_relative"] = round(float(_STATE.get("sample_l2", _STATE["test_l2"])), 6)  # this chip's own sample L2
        out["n_test"] = int(_STATE["n_test"])
        out["ensemble_K"] = 0
    return out


class _Baked:
    """The infer model: holds the N_VIEW held-out samples' three baked physical fields each (u_pred, u_true, a) and
    returns them one variant at a time. The pipeline calls predict() EXACTLY ONCE per variant, in variants() order, so
    an advancing index serves sample i on the i-th call (deterministic; the grid carries no variant info)."""

    def __init__(self, fields_per_sample, sample_l2):
        # fields_per_sample: np.ndarray [N_VIEW, 3, H, W] in physical units, order (u_pred, u_true, a)
        self._f = np.asarray(fields_per_sample, dtype=np.float64)
        self._l2 = list(sample_l2)
        self._i = 0

    def predict(self, XY):
        k = min(self._i, self._f.shape[0] - 1)
        _STATE["sample_l2"] = float(self._l2[k])   # expose the CURRENT sample's L2 to extra_metrics
        self._i += 1
        f = self._f[k]
        return np.stack([f[c].ravel() for c in range(f.shape[0])], axis=1)  # [H*W, 3]


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
        test_l2 = float(rl2(net(Xte), Yte).item())                 # held-out test-set mean (headline)
        n_view = min(N_VIEW, n_test)
        u_pred_n = net(Xte[:n_view]).numpy()[:, 0]                  # [n_view, H, W] normalized predictions

    # the first n_view held-out samples in PHYSICAL units (each: u_pred, u_true, a)
    fields_per_sample, sample_l2 = [], []
    for s in range(n_view):
        u_pred = u_pred_n[s] * stats["u_sd"] + stats["u_mu"]
        a_star = a_raw[n_train + s, 0]
        u_true = u_raw[n_train + s, 0]
        sample_l2.append(float(np.linalg.norm(u_pred - u_true) / (np.linalg.norm(u_true) + 1e-12)))
        fields_per_sample.append(np.stack([u_pred, u_true, a_star.astype(np.float64)], axis=0))  # (3,H,W)
    fields_per_sample = np.stack(fields_per_sample, axis=0)         # (n_view, 3, H, W)
    _STATE.update({"test_l2": test_l2, "n_test": n_test})           # sample_l2 set per-call by _Baked

    # export the FNO's OWN field-in ONNX + parity + a field-forward timing
    onnx_path = _REPO / "models" / f"{CASE.id}.onnx"
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = Xte[:1]
    torch.onnx.export(
        net, (dummy,), str(onnx_path), input_names=["a_grid"], output_names=["u"],
        dynamic_axes={"a_grid": {0: "n"}, "u": {0: "n"}},
        opset_version=18, dynamo=True, verbose=False, external_data=False,
    )
    from ..io.formats import strip_onnx_metadata
    strip_onnx_metadata(onnx_path)  # the dynamo exporter embeds the local build path — strip it (clean public artifact)
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

    return {
        "model": _Baked(fields_per_sample, sample_l2),
        "input_dim": 2,
        "prebuilt": True,
        "onnx_path": str(onnx_path),
        "onnx_bytes": onnx_path.stat().st_size,
        "parity_max_abs": parity,
        "infer_ms": infer_ms,
        "opset": 18,
        "web_drivable": False,   # field-IO operator -> not coordinate-drivable in the browser -> lane=precompute
    }
