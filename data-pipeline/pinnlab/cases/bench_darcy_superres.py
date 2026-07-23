"""Group A · canonical-benchmark — ZERO-SHOT SUPER-RESOLUTION: one operator, any grid.

The fourth Darcy operator case. It exercises the FNO's SIGNATURE property, the one that separates a neural
OPERATOR from an ordinary grid-to-grid network (a CNN, a U-Net): DISCRETISATION INVARIANCE. Because every
Fourier layer acts on a truncated set of Fourier MODES, not on pixels, the SAME trained weights apply at ANY
grid resolution. An operator trained only on coarse data can be queried on a finer grid it never saw.

Li, Kovachki, Azizzadenesheli, Liu, Bhattacharya, Stuart, Anandkumar, *Fourier Neural Operator for Parametric
PDEs*, arXiv:2010.08895 (ICLR 2021): "the first ML-based method to successfully model turbulent flows with
zero-shot super-resolution."

WHAT THIS CASE DOES. Trains the FNO ONLY at 32x32, then evaluates it, with NO retraining, at 32x32, 64x64 and
96x96, against a finite-difference reference solved at each resolution. The coefficient random field is
generated with a resolution-consistent correlation length (the Gaussian-filter sigma scales with the grid),
so all three are the SAME physical process sampled more finely, not three different families.

VERIFIED BEFORE BUILDING (spike, 2026-07-15): trained at 32, the held-out error was 0.039 at 32 and 0.070 at
64 (unseen), a 1.8x degradation, which is the property holding. A CNN cannot even be EVALUATED at a different
resolution without changing its architecture; the FNO just runs.

THE VARIANTS are the evaluation resolutions. Each reports the held-out relative-L2 at that resolution, so the
graceful degradation is a number you read, not a claim.

HONEST LIMITS.
- Super-resolution is not free accuracy: error grows with resolution (finer grids resolve features the coarse
  training never taught the operator), and far past the training resolution it degrades. It is "one operator
  serves many grids", not "the operator gets better on finer grids".
- The reference at each resolution is its OWN finite-difference solve, so this measures the operator against
  the discrete truth at that grid, which is the honest comparison.
- The exported ONNX is the 32x32 operator; the higher-resolution evaluations are baked field artifacts (the
  browser replays them; onnxruntime-web would also run the operator at any grid, but the app is a viewer).

Custom FNO engine (like bench-darcy-operator), field-IO, precompute.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import CaseSpec, Variant

BASE = 32                              # the ONLY resolution the operator is trained on
RES = (32, 64, 96)                     # evaluation resolutions (variants)
WIDTH, MODES, LAYERS = 20, 10, 4
N_TRAIN, N_TEST = 256, 24
EPOCHS, BS = 120, 20
SIGMA0 = 3.0                           # coefficient correlation length at BASE, scaled with resolution
U_SCALE = 0.02
_REPO = Path(__file__).resolve().parents[3]
_STATE: dict = {}

CASE = CaseSpec(
    id="bench-darcy-superres",
    system_type="operator-surrogate",
    category="canonical-benchmark",
    title="Zero-shot super-resolution: one operator trained at 32x32, run at any grid",
    governing_equations=(
        r"-\nabla\!\cdot(a(\mathbf{x})\nabla u)=1;\ \ \mathcal{G}_\theta\ \text{entrenado en } 32^2,\ "
        r"\text{evaluado en } 64^2,96^2\ \text{sin reentrenar (invariante a la discretización)}"
    ),
    method="operator-superres",
    engine="fno-torch",
    real_or_synthetic="synthetic",
    inputs=("x", "y"),
    outputs=("u_pred", "u_true", "abs_err"),
    domain={"x": (0.0, 1.0), "y": (0.0, 1.0)},
    grid={"x": BASE, "y": BASE},
    field_axes=("x", "y"),
    expected_band="trained at 32x32, the operator runs at 64x64 and 96x96 it never saw; the error grows "
                  "gracefully (a CNN cannot even be evaluated off its training grid)",
    validation_anchor="operator-test-l2",
    train={"lr": 1e-3, "adam": 0},
    notes="Custom-engine FIELD-IO case: trains the FNO at 32x32, evaluates + bakes at 32/64/96, exports the "
          "32x32 ONNX. Variants are resolutions. web_drivable=False -> lane=precompute.",
)


def variants() -> list[Variant]:
    text = {
        32: ("32x32 (training resolution)", "32x32 (resolución de entrenamiento)",
             "The grid the operator was trained on: the baseline error.",
             "La grilla en que se entrenó el operador: el error de referencia."),
        64: ("64x64 (zero-shot, unseen)", "64x64 (cero-disparo, no vista)",
             "Twice as fine, never seen in training: the same weights just run.",
             "El doble de fina, nunca vista al entrenar: los mismos pesos simplemente corren."),
        96: ("96x96 (zero-shot, unseen)", "96x96 (cero-disparo, no vista)",
             "Three times as fine: the degradation you pay for going far past the training grid.",
             "El triple de fina: la degradación que se paga al alejarse de la grilla de entrenamiento."),
    }
    out = []
    for r in RES:
        le, ls, ne, ns = text[r]
        out.append(Variant(f"r{r}", le, ls, {"resolution": float(r)}, ne, ns))
    return out


def eval_grid():
    """Return the grid for the CURRENT variant, at its native resolution, so each variant renders at 32, 64
    or 96. The pipeline calls this once per variant in variants() order, immediately before predict(); a call
    counter picks the resolution and stashes the index for predict() to serve the matching field."""
    i = int(_STATE.get("eval_call", 0)) % len(RES)
    _STATE["eval_call"] = i + 1
    _STATE["cur"] = i
    r = RES[i] if not _STATE.get("quick") else _STATE["quick_res"][i]
    xs = np.linspace(0.0, 1.0, r)
    xx, yy = np.meshgrid(xs, xs, indexing="ij")
    XY = np.stack([xx.ravel(), yy.ravel()], axis=1)
    return {"x": xs, "y": xs}, XY, (r, r)


def extra_metrics(sf) -> dict:
    i = int(_STATE.get("cur", 0))
    out = {}
    if "l2" in _STATE:
        out["l2_relative"] = round(float(_STATE["l2"][i]), 6)
        out["resolution"] = int(RES[i])
        out["train_resolution"] = int(BASE)
        out["degradation_x"] = round(float(_STATE["l2"][i] / _STATE["l2"][0]), 2)
        out["n_test"] = int(N_TEST)
        out["ensemble_K"] = 0
    return out


class _Baked:
    """Serves the pre-baked fields for the variant selected by eval_grid() (via _STATE['cur']); each variant's
    field is at its OWN resolution, so no reshape mismatch."""

    def __init__(self, fields):
        self._f = [np.asarray(f, dtype=np.float64) for f in fields]     # per-variant [3, res, res] (res varies!)

    def predict(self, XY):
        k = int(_STATE.get("cur", 0)) % len(self._f)
        f = self._f[k]
        return np.stack([f[c].ravel() for c in range(f.shape[0])], axis=1)


def _coeff(rng, n):
    """Resolution-consistent coefficient field: sigma scales with n so the physical correlation length is the
    same at every grid (the SAME process sampled more finely, not a different family)."""
    from scipy.ndimage import gaussian_filter
    g = gaussian_filter(rng.standard_normal((n, n)), sigma=SIGMA0 * n / BASE)
    return np.where(g > 0.0, 12.0, 3.0)


def _solve(a):
    from ..datasets.darcy import _solve_darcy
    return _solve_darcy(a)


def _pack(a, n):
    gx, gy = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n), indexing="ij")
    return np.stack([(a - 7.5) / 4.5, gx, gy])


def build(seed: int, quick: bool = False) -> dict:
    import time

    import onnxruntime as ort
    import torch

    from ..model.fno import FNO2d

    n_train, n_test, epochs = (8, 4, 3) if quick else (N_TRAIN, N_TEST, EPOCHS)
    res = (32, 48, 64) if quick else RES
    _STATE["quick"] = bool(quick)
    _STATE["quick_res"] = res
    _STATE["eval_call"] = 0

    # ---- train ONLY at the base resolution ----
    rng = np.random.default_rng(int(seed))
    atr = [_coeff(rng, BASE) for _ in range(n_train)]
    utr = [_solve(a) for a in atr]
    Xtr = torch.as_tensor(np.stack([_pack(a, BASE) for a in atr]), dtype=torch.float32)
    Utr = torch.as_tensor(np.stack([u[None] for u in utr]), dtype=torch.float32)

    torch.manual_seed(int(seed) + 6)
    net = FNO2d(in_ch=3, width=WIDTH, modes=MODES, layers=LAYERS)
    opt = torch.optim.Adam(net.parameters(), lr=CASE.train["lr"])

    def rl2(p, y):
        return (torch.norm((p - y).reshape(len(p), -1), dim=1)
                / torch.norm(y.reshape(len(y), -1), dim=1).clamp_min(1e-9)).mean()

    for _ in range(epochs):
        perm = torch.randperm(n_train)
        for b in range(0, n_train, BS):
            idx = perm[b:b + BS]
            opt.zero_grad()
            rl2(net(Xtr[idx]) * U_SCALE, Utr[idx]).backward()
            opt.step()
    net.eval()

    # ---- evaluate at each resolution against a finite-difference reference solved at that grid ----
    l2s, fields = [], []
    for r in res:
        g = np.random.default_rng(int(seed) + 100 + r)
        a_te = [_coeff(g, r) for _ in range(n_test)]
        u_te = [_solve(a) for a in a_te]
        Xte = torch.as_tensor(np.stack([_pack(a, r) for a in a_te]), dtype=torch.float32)
        Ute = torch.as_tensor(np.stack([u[None] for u in u_te]), dtype=torch.float32)
        with torch.no_grad():
            pred = (net(Xte) * U_SCALE).numpy()[:, 0]
        true = Ute.numpy()[:, 0]
        num = np.linalg.norm((pred - true).reshape(n_test, -1), axis=1)
        den = np.linalg.norm(true.reshape(n_test, -1), axis=1)
        l2s.append(float((num / (den + 1e-12)).mean()))
        # show sample 0 at this resolution
        fields.append(np.stack([pred[0], true[0], np.abs(pred[0] - true[0])], axis=0))

    _STATE.update({"l2": l2s})

    # ---- export the 32x32 operator's ONNX ----
    onnx_path = _REPO / "models" / f"{CASE.id}.onnx"
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    class _Phys(torch.nn.Module):
        def __init__(self, n):
            super().__init__(); self.n = n

        def forward(self, x):
            return self.n(x) * U_SCALE

    # export at the FIXED 32x32 training grid (batch dynamic only): the exported artifact is the operator at
    # its training resolution; the higher-resolution evaluations are baked FIELD artifacts, not run through
    # this ONNX in the browser. (The FNO's mode-scatter does not export cleanly with dynamic spatial axes;
    # discretisation-invariance is a property of the torch model, demonstrated by the baked 64/96 fields.)
    wrapped = _Phys(net).eval()
    torch.onnx.export(wrapped, (Xtr[:1],), str(onnx_path), input_names=["a_grid"], output_names=["u"],
                      dynamic_axes={"a_grid": {0: "n"}, "u": {0: "n"}},
                      opset_version=18, dynamo=True, verbose=False, external_data=False)
    from ..io.formats import strip_onnx_metadata
    strip_onnx_metadata(onnx_path)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        pt = wrapped(Xtr[:4]).numpy()
    ox = np.asarray(sess.run(["u"], {"a_grid": Xtr[:4].numpy()})[0])
    parity = float(np.max(np.abs(pt - ox)))
    one = Xtr[:1].numpy()
    sess.run(["u"], {"a_grid": one})
    t0 = time.perf_counter()
    for _ in range(5):
        sess.run(["u"], {"a_grid": one})
    infer_ms = (time.perf_counter() - t0) * 1000.0 / 5

    return {
        "model": _Baked(fields),
        "input_dim": 2,
        "prebuilt": True,
        "onnx_path": str(onnx_path),
        "onnx_bytes": onnx_path.stat().st_size,
        "parity_max_abs": parity,
        "infer_ms": infer_ms,
        "opset": 18,
        "web_drivable": False,
    }
