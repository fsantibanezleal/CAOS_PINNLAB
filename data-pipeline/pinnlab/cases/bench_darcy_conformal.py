"""Group A · canonical-benchmark — CONFORMAL PREDICTION: a distribution-free error bar for the operator.

The third Darcy operator case. `bench-darcy-operator` ships a data-driven FNO and `bench-darcy-pino` adds the
equation; both report a single held-out error NUMBER and say nothing about the NEXT instance. This case adds
the guarantee that a surrogate actually needs in practice:

    for a new coefficient field, a band pred(x) +/- q that contains the true field with a stated probability,
    with NO assumption on the error distribution beyond exchangeability.

SPLIT CONFORMAL PREDICTION (Vovk; Lei et al. 2018; and the 2026 neural-operator conformal literature,
arXiv:2606.09923, 2606.17513). The recipe is three lines and needs no retraining:

  1. Hold out a CALIBRATION set the operator never trained on.
  2. Score each calibration instance by its worst pixel error  s_i = max_x |G(a_i)(x) - u_i(x)|.
  3. Take the finite-sample quantile  q = the ceil((n+1)(1-alpha))/n empirical quantile of {s_i}.
     Then for a NEW instance the band  G(a)(x) +/- q  contains the WHOLE true field with probability
     >= 1 - alpha (marginal, over the draw of the calibration + test instances).

VERIFIED BEFORE BUILDING (spike, 2026-07-15). On this exact Darcy FNO the per-instance empirical coverage
landed at 81.5% / 96.5% / 97.5% for targets 80% / 90% / 95%: at or above target every time, which is exactly
what the guarantee requires (it is a lower bound, so over-coverage is allowed and expected from the
whole-field max score).

THE VARIANTS are the target coverage levels. Each shows, on the SAME held-out field, the band for that target
and the empirical coverage achieved over the test set, so the guarantee is something you watch hold rather
than take on faith.

HONESTY (this is the whole point of the case, so it is stated, not buried).
- Coverage is MARGINAL and holds only under EXCHANGEABILITY: the new instance must come from the same
  distribution as the calibration set. A different coefficient family (different sigma, different value pair)
  breaks exchangeability and VOIDS the guarantee. That caveat is precisely what an operator surrogate needs
  attached to it.
- The band is a fixed width q (per-instance whole-field), NOT a per-pixel uncertainty: it is wide enough for
  the hardest pixel, so it over-covers the easy interior. A normalised score would tighten it; kept simple
  here so the guarantee is transparent.
- Conformal quantifies the operator's OWN error against the reference solver; it does not make a wrong
  operator right.

Offline engine only. Custom FNO engine (like bench-darcy-operator), field-IO, precompute.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import CaseSpec, Variant

N_GRID = 32
WIDTH, MODES, LAYERS = 20, 10, 4
N_TRAIN, N_CAL, N_TEST = 128, 128, 200
EPOCHS, BS = 120, 16
TARGETS = (0.80, 0.90, 0.95)         # the coverage levels shown as variants
_REPO = Path(__file__).resolve().parents[3]
_STATE: dict = {}

CASE = CaseSpec(
    id="bench-darcy-conformal",
    system_type="operator-surrogate",
    category="canonical-benchmark",
    title="Conformal prediction: a distribution-free error bar for the operator",
    governing_equations=(
        r"-\nabla\!\cdot(a(\mathbf{x})\nabla u)=1;\ \ q=\text{cuantil}_{1-\alpha}\big(\max_{\mathbf{x}}|\mathcal{G}(a_i)-u_i|\big),\ "
        r"\ \mathbb{P}\big(\max_{\mathbf{x}}|\mathcal{G}(a)-u|\le q\big)\ge 1-\alpha"
    ),
    method="operator-conformal-uq",
    engine="fno-torch",
    real_or_synthetic="synthetic",
    inputs=("x", "y"),
    outputs=("u_pred", "u_true", "abs_err", "in_band"),
    domain={"x": (0.0, 1.0), "y": (0.0, 1.0)},
    grid={"x": N_GRID, "y": N_GRID},
    field_axes=("x", "y"),
    expected_band="the empirical coverage lands at or above each target (80/90/95%), as split conformal guarantees "
                  "under exchangeability",
    validation_anchor="conformal-coverage",
    train={"lr": 1e-3, "adam": 0},
    notes="Custom-engine FIELD-IO case: trains the FNO, calibrates on a held-out split, bakes the coverage band "
          "and the in-band mask. Variants are coverage targets. web_drivable=False -> lane=precompute.",
)


def variants() -> list[Variant]:
    """The variants are the target coverage levels 1 - alpha."""
    text = {
        0.80: ("80% coverage target", "Cobertura objetivo 80%",
               "A tight band: it will miss ~1 field in 5, by design.",
               "Una banda ajustada: fallará ~1 campo de cada 5, por diseño."),
        0.90: ("90% coverage target", "Cobertura objetivo 90%",
               "The common default: the band contains the whole field 9 times in 10.",
               "El valor por defecto habitual: la banda contiene el campo entero 9 de cada 10 veces."),
        0.95: ("95% coverage target", "Cobertura objetivo 95%",
               "A conservative band: wider, misses ~1 field in 20.",
               "Una banda conservadora: más ancha, falla ~1 campo de cada 20."),
    }
    out = []
    for tgt in TARGETS:
        le, ls, ne, ns = text[tgt]
        out.append(Variant(f"c{int(tgt*100)}", le, ls, {"target": tgt}, ne, ns))
    return out


def extra_metrics(sf) -> dict:
    i = int(_STATE.get("cur", 0))
    out = {}
    if "cov" in _STATE:
        out["l2_relative"] = round(float(_STATE["test_l2"]), 6)          # the operator's own error (constant)
        out["target_coverage"] = TARGETS[i]
        out["empirical_coverage"] = round(float(_STATE["cov"][i]), 4)
        out["band_q"] = round(float(_STATE["q"][i]), 5)
        out["n_calibration"] = N_CAL
        out["n_test"] = N_TEST
        out["ensemble_K"] = 0
    return out


class _Baked:
    def __init__(self, fields):
        self._f = np.asarray(fields, dtype=np.float64)   # [n_var, 4, H, W]
        self._i = 0

    def predict(self, XY):
        k = min(self._i, self._f.shape[0] - 1)
        _STATE["cur"] = k
        self._i += 1
        f = self._f[k]
        return np.stack([f[c].ravel() for c in range(f.shape[0])], axis=1)


def build(seed: int, quick: bool = False) -> dict:
    import time

    import onnxruntime as ort
    import torch

    from ..datasets.darcy import _make_coeff, _solve_darcy
    from ..model.fno import FNO2d

    n_train, n_cal, n_test, epochs = (8, 8, 8, 3) if quick else (N_TRAIN, N_CAL, N_TEST, EPOCHS)
    targets = TARGETS
    U_SCALE = 0.02

    rng = np.random.default_rng(int(seed))
    a_all = [_make_coeff(rng, N_GRID) for _ in range(n_train + n_cal + n_test)]
    u_all = [_solve_darcy(a) for a in a_all]
    gx, gy = np.meshgrid(np.linspace(0, 1, N_GRID), np.linspace(0, 1, N_GRID), indexing="ij")

    def pack(a):
        return np.stack([(a - 7.5) / 4.5, gx, gy])

    X = torch.as_tensor(np.stack([pack(a) for a in a_all]), dtype=torch.float32)
    U = torch.as_tensor(np.stack([u[None] for u in u_all]), dtype=torch.float32)
    Xtr, Utr = X[:n_train], U[:n_train]
    Xcal, Ucal = X[n_train:n_train + n_cal], U[n_train:n_train + n_cal]
    Xte, Ute = X[n_train + n_cal:], U[n_train + n_cal:]
    a_raw = np.asarray(a_all)

    torch.manual_seed(int(seed) + 5)
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
    with torch.no_grad():
        test_l2 = float(rl2(net(Xte) * U_SCALE, Ute))
        cal_pred = (net(Xcal) * U_SCALE).numpy()[:, 0]
        te_pred = (net(Xte) * U_SCALE).numpy()[:, 0]

    cal_true = Ucal.numpy()[:, 0]
    te_true = Ute.numpy()[:, 0]
    # per-instance whole-field nonconformity score = worst pixel error
    cal_scores = np.abs(cal_pred - cal_true).reshape(n_cal, -1).max(axis=1)
    te_score = np.abs(te_pred - te_true).reshape(len(te_pred), -1).max(axis=1)

    # the band widths first (they do not depend on the shown field): finite-sample split-conformal quantiles
    qs, covs = [], []
    for tgt in targets:
        level = min(np.ceil((n_cal + 1) * tgt) / n_cal, 1.0)
        q = float(np.quantile(cal_scores, level))
        qs.append(q)
        covs.append(float((te_score <= q).mean()))              # empirical whole-field coverage on the test set

    # Choose ONE held-out field to display, such that the in-band mask is informative: its worst pixel should
    # fall BETWEEN the tightest and widest bands, so at 80% part of the field is OUT (mask shows 0s at the hard
    # pixels) and by 95% it is fully IN. Pick the field whose whole-field score is closest to the midpoint of
    # the smallest and largest q. An easy field (below every q) would be trivially all-1 and illustrate nothing.
    mid = 0.5 * (qs[0] + qs[-1])
    view_idx = int(np.argmin(np.abs(te_score - mid)))
    view_pred = te_pred[view_idx]
    view_true = te_true[view_idx]
    view_err = np.abs(view_pred - view_true)

    fields = []
    for q in qs:
        in_band = (view_err <= q).astype(np.float64)            # 1 where the band contains the truth
        fields.append(np.stack([view_pred, view_true, view_err, in_band], axis=0))

    _STATE.update({"test_l2": test_l2, "q": qs, "cov": covs})

    # export the FNO's field-in ONNX (the operator itself; the conformal step is post-processing)
    onnx_path = _REPO / "models" / f"{CASE.id}.onnx"
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    class _Phys(torch.nn.Module):
        def __init__(self, n):
            super().__init__(); self.n = n

        def forward(self, x):
            return self.n(x) * U_SCALE

    wrapped = _Phys(net).eval()
    torch.onnx.export(wrapped, (Xte[:1],), str(onnx_path), input_names=["a_grid"], output_names=["u"],
                      dynamic_axes={"a_grid": {0: "n"}, "u": {0: "n"}},
                      opset_version=18, dynamo=True, verbose=False, external_data=False)
    from ..io.formats import strip_onnx_metadata
    strip_onnx_metadata(onnx_path)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    k = min(8, n_test)
    with torch.no_grad():
        pt = wrapped(Xte[:k]).numpy()
    ox = np.asarray(sess.run(["u"], {"a_grid": Xte[:k].numpy()})[0])
    parity = float(np.max(np.abs(pt - ox)))
    one = Xte[:1].numpy()
    sess.run(["u"], {"a_grid": one})
    t0 = time.perf_counter()
    for _ in range(5):
        sess.run(["u"], {"a_grid": one})
    infer_ms = (time.perf_counter() - t0) * 1000.0 / 5

    return {
        "model": _Baked(np.stack(fields, axis=0)),
        "input_dim": 2,
        "prebuilt": True,
        "onnx_path": str(onnx_path),
        "onnx_bytes": onnx_path.stat().st_size,
        "parity_max_abs": parity,
        "infer_ms": infer_ms,
        "opset": 18,
        "web_drivable": False,
    }
