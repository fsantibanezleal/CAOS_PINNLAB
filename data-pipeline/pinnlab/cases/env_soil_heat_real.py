"""Group C · pollution-environmental (REAL DATA) — subsurface heat conduction inverse from NOAA USCRN soil temperatures.

The ONLY case trained against a REAL measured dataset. NOAA's U.S. Climate Reference Network reports daily-mean soil
temperature at five depths (5, 10, 20, 50, 100 cm). Subsurface heat conduction is genuinely diffusive:
    T_t = alpha T_zz   on z in [5,100] cm, t over 2019-2021 (station IL_Champaign_9_SW),
the surface signal propagating downward damped + phase-lagged. We pose the INVERSE: take the 5 cm + 100 cm sensors as
real time-varying Dirichlet boundaries, recover the effective thermal diffusivity alpha (a trainable scalar), and
VALIDATE OUT-OF-SAMPLE against the held-out 10/20/50 cm sensors — interior depths never shown to the optimizer.

real_or_synthetic = validated-real: boundaries + held-out anchor are real measurements (no manufactured truth). The
score is the held-out relative-L2 / RMSE in degrees C against the real interior temperatures, plus the recovered alpha
in physical mm^2/s (typical mineral soils ~0.2-0.8 mm^2/s). Data vendored offline by `datasets/uscrn_soil.py`.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .base import CaseSpec, Variant

_DATA_PATH = Path(__file__).resolve().parents[3] / "data" / "reference" / "uscrn" / "soil_temp_il_champaign.json"
_D = json.loads(_DATA_PATH.read_text(encoding="utf-8"))
_MEAN = _D["normalization"]["mean_c"]
_SCALE = _D["normalization"]["scale_c"]
_DEPTHS = _D["depths_cm"]                      # [5,10,20,50,100]
_Z0, _Z1 = float(_DEPTHS[0]), float(_DEPTHS[-1])
_ZETA = {d: (float(d) - _Z0) / (_Z1 - _Z0) for d in _DEPTHS}   # depth(cm) -> zeta in [0,1]
_HOLDOUT = _D["holdout_depths_cm"]             # [10,20,50]
_L = _D["L_meters"]
_SPAN = _D["span_seconds"]
_NT = len(_D["t_grid_days_from_2019_01_01"])

_RECOVERED: dict[str, object] = {}             # build() stashes the trainable log-kappa; extra_metrics reads it post-train


def _norm(d: int) -> np.ndarray:
    return (np.asarray(_D["temp_c"][str(d)], dtype=np.float64) - _MEAN) / _SCALE


CASE = CaseSpec(
    id="env-soil-heat-real",
    category="pollution-environmental",
    title="Subsurface heat conduction — recover soil thermal diffusivity from REAL USCRN temperatures",
    governing_equations=(
        r"T_t=\alpha\,T_{zz},\ z\in[5,100]\,\mathrm{cm};\ \text{5 \& 100 cm = real Dirichlet boundaries},\ "
        r"\text{recover }\alpha,\ \text{validate vs held-out 10/20/50 cm}"
    ),
    method="inverse-parameter-real",
    engine="deepxde",
    real_or_synthetic="validated-real",
    inputs=("z", "t"),
    outputs=("T",),
    domain={"z": (0.0, 1.0), "t": (0.0, 1.0)},
    grid={"z": 49, "t": _NT},
    field_axes=("z", "t"),  # single 2-D heatmap over depth × time; no parametric axis (alpha is recovered, not swept)
    expected_band="damped, phase-lagged seasonal wave with depth; held-out 10/20/50 cm relative-L2 vs REAL temps ~7e-2 (~1 deg C RMSE)",
    validation_anchor="real-data-holdout",
    train={"lr": 1e-3, "adam": 18000, "lbfgs": False, "num_domain": 4000, "num_boundary": 0, "num_initial": 0, "num_test": 2000, "loss_weights": [1, 60, 60, 20]},
    notes="REAL USCRN soil temps; 5/100 cm real Dirichlet boundaries; recovered alpha=kappa*L^2/span (mm^2/s); "
          "held-out 10/20/50 cm out-of-sample validation. Adam-only so the trainable diffusivity is never dropped.",
)


def variants() -> list[Variant]:
    # Single VALIDATED-REAL variant: alpha is recovered (not a network-input knob) and the real surface forcing has no
    # closed form, so there is no honest parametric family — the field is the one measured T(z,t) reconstruction.
    return [Variant(
        "real", "USCRN (real data)", "USCRN (datos reales)", {},
        "Soil-temperature field reconstructed from real 5/100 cm boundaries; validated out-of-sample vs 10/20/50 cm.",
        "Campo de temperatura de suelo reconstruido desde bordes reales 5/100 cm; validado fuera de muestra vs 10/20/50 cm.",
    )]


def extra_metrics(sf) -> dict:
    import math

    from ..model.analytic import l2_relative

    z_axis = np.asarray(sf.axes["z"], dtype=np.float64)
    field = np.asarray(sf.fields["T"], dtype=np.float64)        # (nz, nt) normalized
    preds, reals = [], []
    per_depth = {}
    for d in _HOLDOUT:
        col = np.array([np.interp(_ZETA[d], z_axis, field[:, j]) for j in range(field.shape[1])])
        pred_c = col * _SCALE + _MEAN
        real_c = np.asarray(_D["temp_c"][str(d)], dtype=np.float64)
        preds.append(pred_c)
        reals.append(real_c)
        per_depth[f"holdout_rmse_c_{d}cm"] = round(float(np.sqrt(np.mean((pred_c - real_c) ** 2))), 4)
    pred_all = np.concatenate(preds)
    real_all = np.concatenate(reals)
    rmse = float(np.sqrt(np.mean((pred_all - real_all) ** 2)))
    out = {
        "l2_relative": round(l2_relative(pred_all, real_all), 6),   # held-out relative-L2 vs REAL interior temps
        "holdout_rmse_c": round(rmse, 4),
        "n_holdout_pts": int(pred_all.size),
        **per_depth,
    }
    log_k = _RECOVERED.get("log_kappa")
    if log_k is not None:
        try:
            kappa = math.exp(float(log_k.detach().cpu().numpy()))   # type: ignore[attr-defined]
            alpha = kappa * (_L ** 2) / _SPAN                       # m^2/s
            out["kappa_norm"] = round(kappa, 4)
            out["recovered_alpha_mm2_s"] = round(alpha * 1e6, 5)
        except Exception:
            pass
    return out


def build(seed: int) -> dict:
    import deepxde as dde
    import torch

    dde.config.set_random_seed(int(seed))
    geom = dde.geometry.Interval(0.0, 1.0)
    timedomain = dde.geometry.TimeDomain(0.0, 1.0)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    log_kappa = dde.Variable(3.4)                                  # kappa ~ e^3.4 ~ 30 (normalized diffusivity)
    _RECOVERED["log_kappa"] = log_kappa

    def pde(x, T):
        T_t = dde.grad.jacobian(T, x, i=0, j=1)
        T_zz = dde.grad.hessian(T, x, i=0, j=0)
        return T_t - torch.exp(log_kappa) * T_zz

    t_grid = np.linspace(0.0, 1.0, _NT)
    top_pts = np.column_stack([np.zeros(_NT), t_grid])
    bot_pts = np.column_stack([np.ones(_NT), t_grid])
    bc_top = dde.icbc.PointSetBC(top_pts, _norm(_DEPTHS[0]).reshape(-1, 1), component=0)
    bc_bot = dde.icbc.PointSetBC(bot_pts, _norm(_DEPTHS[-1]).reshape(-1, 1), component=0)

    # initial profile at t=0 from the REAL 5-sensor depth profile (interpolated across zeta)
    z_ic = np.linspace(0.0, 1.0, 41)
    zeta_s = np.array([_ZETA[d] for d in _DEPTHS])
    t0_s = np.array([_norm(d)[0] for d in _DEPTHS])
    ic_val = np.interp(z_ic, zeta_s, t0_s).reshape(-1, 1)
    ic_pts = np.column_stack([z_ic, np.zeros(41)])
    bc_ic = dde.icbc.PointSetBC(ic_pts, ic_val, component=0)

    anchors = np.vstack([top_pts, bot_pts, ic_pts])
    t = CASE.train
    data = dde.data.TimePDE(
        geomtime, pde, [bc_top, bc_bot, bc_ic],
        num_domain=t["num_domain"], num_boundary=t["num_boundary"], num_initial=t["num_initial"],
        anchors=anchors, num_test=t["num_test"],
    )
    net = dde.nn.FNN([2] + [48] * 4 + [1], "tanh", "Glorot normal")
    model = dde.Model(data, net)
    model.compile("adam", lr=t["lr"], loss_weights=t["loss_weights"], external_trainable_variables=[log_kappa])
    return {"model": model, "input_dim": 2}
