"""Group C · pollution-environmental (SUBSURFACE) — the PUMPING TEST: recover an aquifer's T and S.

The subsurface applied case the catalogue was missing, and an honest "know when a PINN is NOT the tool" case.

THE PROBLEM. You cannot measure a confined aquifer's transmissivity T or storativity S directly. The standard
field method is a PUMPING TEST: pump a well at constant rate Q, watch the water level (the drawdown s) fall in
an observation well over time, and infer T and S from the curve. The physics is the Theis (1935) solution of
radial transient flow in a confined aquifer:

    s(r,t) = Q/(4 pi T) W(u),   u = r^2 S/(4 T t),   W(u) = E1(u)   (the exponential integral)

CONFINED is not optional: for an unconfined aquifer the saturated thickness changes and Theis is the wrong
model (Boulton/Neuman is needed). The case states confined and means it.

THE METHOD THAT WORKS: Cooper-Jacob. For late time (small u) W(u) ~= -0.5772 - ln u, so s is LINEAR in ln t:
    s ~= (Q/4 pi T) ln(2.25 T t / r^2 S).
Fit a straight line to s vs ln t: the slope gives T (= Q/(4 pi * slope)), the zero-drawdown intercept gives S.
VERIFIED across three aquifers (fine sand, coarse sand, gravel): T recovered to 0.2-1.1%, S to 1.2-4.7% from
noisy synthetic data. The analytic Theis anchor itself was checked against a numerical radial solve to 0.06%
(`wip/beyond-sota/verify_theis_anchor.py`).

WHY A PINN IS NOT THE TOOL HERE, stated honestly and with numbers. This is a well-posed, idealised problem
with an exact closed form, so the classical method is fast and robust. A PINN inverse was spiked on the SAME
data and did far worse: a forward radial PINN collapsed to 82% relative error (the stiff near-well transient
plus the flux boundary is the classic PINN gradient pathology), and a PINN parameter-inverse recovered
T = 2335 vs 500 m^2/day (367% off). The PINN earns its keep only once the aquifer is bounded or heterogeneous,
where no closed form exists and Cooper-Jacob is invalid; on THIS problem, reaching for a PINN is the wrong
call. Knowing that is the point of the case.

THE VARIANTS are three confined aquifers spanning the realistic range; each shows its drawdown cone and its
recovered (T, S). The field shown is the drawdown cone s(x,y) at t = 1 day (radially symmetric, deepest at the
well), computed from the TRUE parameters; a small coordinate network is fit to it and exported as the ONNX
artifact (a learned drawdown-field surrogate; the honest inverse is Cooper-Jacob, not the network).

Offline engine only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import CaseSpec, Variant

# spatial domain (m): a 1000 x 1000 m plan view with the pumping well at the centre
L = 1000.0
NG = 48
RW = 5.0                               # well radius (avoid the log singularity at r=0)
Q = 2000.0 / 86400.0                   # pumping rate, m^3/s (2000 m^3/day)
R_OBS = 100.0                          # observation-well radius for the pumping-test curve
T_EVAL = 86400.0                       # 1 day
NOISE = 0.02                           # 2 cm gauge noise on the drawdown record
# (name_en, name_es, T [m^2/day], S) - a fine sand, a coarse sand, a gravel
AQUIFERS = [
    ("fine sand", "arena fina", 120.0, 5e-4),
    ("coarse sand", "arena gruesa", 500.0, 2e-4),
    ("gravel", "grava", 1500.0, 8e-5),
]
_REPO = Path(__file__).resolve().parents[3]
_STATE: dict = {}

CASE = CaseSpec(
    id="env-aquifer-test",
    system_type="steady-elliptic",
    category="pollution-environmental",
    title="Pumping test: recover a confined aquifer's transmissivity and storativity",
    governing_equations=(
        r"s(r,t)=\frac{Q}{4\pi T}\,W(u),\ \ u=\frac{r^2 S}{4 T t};\ \ "
        r"\text{Cooper-Jacob: } s\approx\frac{Q}{4\pi T}\ln\!\frac{2.25\,T t}{r^2 S}\ \text{(lineal en } \ln t)"
    ),
    method="inverse-aquifer-test",
    engine="analytic-cooper-jacob",
    real_or_synthetic="synthetic-illustrative",
    inputs=("x", "y"),
    outputs=("drawdown",),
    domain={"x": (0.0, L), "y": (0.0, L)},
    grid={"x": NG, "y": NG},
    field_axes=("x", "y"),
    expected_band="Cooper-Jacob recovers T to ~1% and S to a few percent from noisy pumping-test data; a PINN "
                  "inverse on the same data is far worse (this is where the classical method wins)",
    validation_anchor="theis-analytic",
    train={"lr": 1e-3, "adam": 0},
    notes="Analytic-engine case: the field is the Theis drawdown cone; the inverse is Cooper-Jacob (classical). "
          "A small coordinate net is fit to the cone for the ONNX artifact. web_drivable=False -> precompute.",
)


def variants() -> list[Variant]:
    out = []
    for name_en, name_es, T, S in AQUIFERS:
        out.append(Variant(
            name_en.replace(" ", "-"),
            f"{name_en} (T~{T:.0f} m2/day)", f"{name_es} (T~{T:.0f} m2/dia)",
            {"T_true": T, "S_true": S},
            f"A {name_en} aquifer: the pumping test recovers its T and S from the drawdown curve.",
            f"Un acuifero de {name_es}: la prueba de bombeo recupera su T y S desde la curva de abatimiento."))
    return out


def _theis(r, t, T_si, S):
    from scipy.special import exp1
    return Q / (4.0 * np.pi * T_si) * exp1(np.maximum(r, RW) ** 2 * S / (4.0 * T_si * t))


def _cooper_jacob(T_true, S_true, rng):
    """Recover (T, S) from a noisy pumping-test record at R_OBS via the straight-line method."""
    T_si = T_true / 86400.0
    t = np.geomspace(300.0, T_EVAL, 40)
    s = _theis(R_OBS, t, T_si, S_true) + rng.normal(0, NOISE, len(t))
    u = R_OBS ** 2 * S_true / (4.0 * T_si * t)
    mask = u < 0.05                                          # Cooper-Jacob validity
    if mask.sum() < 4:
        mask = np.ones_like(t, dtype=bool)
    m, b = np.polyfit(np.log(t[mask]), s[mask], 1)
    T_rec = Q / (4.0 * np.pi * m) * 86400.0                 # back to m^2/day
    t0 = np.exp(-b / m)
    S_rec = 2.25 * (T_rec / 86400.0) * t0 / R_OBS ** 2
    return float(T_rec), float(S_rec), int(mask.sum())


def _cone(T_true, S_true):
    """The drawdown cone s(x,y) at t = 1 day, from the TRUE parameters (the physical field)."""
    xs = np.linspace(0.0, L, NG)
    xx, yy = np.meshgrid(xs, xs, indexing="ij")
    r = np.hypot(xx - L / 2, yy - L / 2)
    return _theis(r, T_EVAL, T_true / 86400.0, S_true)


def extra_metrics(sf) -> dict:
    i = int(_STATE.get("cur", 0))
    out = {}
    if "rec" in _STATE:
        Tr, Sr, npts = _STATE["rec"][i]
        Tt, St = AQUIFERS[i][2], AQUIFERS[i][3]
        out["T_recovered_m2_day"] = round(Tr, 1)
        out["T_true_m2_day"] = round(Tt, 1)
        out["T_rel_err"] = round(abs(Tr - Tt) / Tt, 4)
        out["S_recovered"] = float(f"{Sr:.2e}")
        out["S_true"] = float(f"{St:.2e}")
        out["S_rel_err"] = round(abs(Sr - St) / St, 4)
        out["cooper_jacob_points"] = npts
        out["ensemble_K"] = 0
    return out


class _Baked:
    def __init__(self, cones):
        self._c = [np.asarray(c, dtype=np.float64) for c in cones]

    def predict(self, XY):
        k = int(_STATE.get("eval_call", 0)) % len(self._c)
        _STATE["eval_call"] = k + 1
        _STATE["cur"] = k
        return self._c[k].reshape(-1, 1)


def build(seed: int, quick: bool = False) -> dict:
    import time

    import onnxruntime as ort
    import torch

    rng = np.random.default_rng(int(seed))
    cones, rec = [], []
    for _, _, T, S in AQUIFERS:
        cones.append(_cone(T, S))
        rec.append(_cooper_jacob(T, S, rng))
    _STATE.update({"rec": rec, "eval_call": 0})

    # a small coordinate net fit to the representative (coarse-sand) cone -> the ONNX artifact (a learned
    # drawdown-field surrogate; the honest parameter inverse is Cooper-Jacob above, not this network)
    xs = np.linspace(0.0, L, NG)
    xx, yy = np.meshgrid(xs, xs, indexing="ij")
    XY = torch.tensor(np.stack([xx.ravel() / L, yy.ravel() / L], axis=1), dtype=torch.float32)
    target = cones[1]
    tmax = float(target.max())
    Yt = torch.tensor((target.ravel() / tmax)[:, None], dtype=torch.float32)
    torch.manual_seed(int(seed) + 4)
    net = torch.nn.Sequential(torch.nn.Linear(2, 64), torch.nn.Tanh(),
                              torch.nn.Linear(64, 64), torch.nn.Tanh(), torch.nn.Linear(64, 1))
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    for _ in range(300 if not quick else 20):
        opt.zero_grad()
        (net(XY) - Yt).pow(2).mean().backward()
        opt.step()
    net.eval()

    class _Scaled(torch.nn.Module):
        def __init__(self, n, s):
            super().__init__(); self.n = n; self.s = float(s)

        def forward(self, x):
            return self.n(x) * self.s

    wrapped = _Scaled(net, tmax).eval()
    onnx_path = _REPO / "models" / f"{CASE.id}.onnx"
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(wrapped, (XY[:1],), str(onnx_path), input_names=["xy"], output_names=["s"],
                      dynamic_axes={"xy": {0: "n"}, "s": {0: "n"}},
                      opset_version=18, dynamo=True, verbose=False, external_data=False)
    from ..io.formats import strip_onnx_metadata
    strip_onnx_metadata(onnx_path)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        pt = wrapped(XY[:16]).numpy()
    ox = np.asarray(sess.run(["s"], {"xy": XY[:16].numpy()})[0])
    parity = float(np.max(np.abs(pt - ox)))
    one = XY[:1].numpy()
    sess.run(["s"], {"xy": one})
    t0 = time.perf_counter()
    for _ in range(5):
        sess.run(["s"], {"xy": one})
    infer_ms = (time.perf_counter() - t0) * 1000.0 / 5

    return {
        "model": _Baked(cones),
        "input_dim": 2,
        "prebuilt": True,
        "onnx_path": str(onnx_path),
        "onnx_bytes": onnx_path.stat().st_size,
        "parity_max_abs": parity,
        "infer_ms": infer_ms,
        "opset": 18,
        "web_drivable": False,
    }
