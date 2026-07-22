"""Group A · canonical-benchmark — PINO: the PHYSICS-INFORMED neural operator on the Darcy family.

The companion to `bench-darcy-operator`. That case learns the Darcy solution operator from SOLVED PAIRS only
(data-driven FNO). This one adds the governing equation to the operator's own training loss, which is the
Physics-Informed Neural Operator of Li et al. (arXiv:2111.03794, ACM/IMS J. Data Science 1(3) 2024,
doi:10.1145/3648506), and asks the question that actually matters in practice:

    how many SOLVED instances do you need, if the operator also knows the equation?

Everything is held identical between the two lanes -- same FNO architecture (width 20, 10 modes, 4 layers),
same seed, same epochs, same held-out test set -- so the ONLY difference is the loss:

    data-only FNO :  L_data
    PINO          :  L_data  +  lambda * L_pde        (+ L_pde on VIRTUAL instances, Algorithm 1)

The virtual instances are the point: a fresh coefficient field a' ~ mu costs a Gaussian filter and a
threshold, with NO solve, so the equation supplies an unlimited training signal. In the paper's words, "we
have access to the unlimited dataset by sampling new a_j in each iteration".

THE RESIDUAL ON A GRID (paper Section 3.3). An operator emits u on a grid, so there is no autograd in x. We
use the finite-volume divergence form with harmonic-mean face conductivities -- the correct face average for
a discontinuous coefficient -- implemented and VERIFIED in `model/pino.py` (fed the reference solution it
returns rms 3.6e-14; fed u = 0 it returns exactly f = 1). The FFT route is NOT the default here: this problem
is non-periodic (u = 0 on the boundary) with a piecewise-constant coefficient, and measured on a reference
field the spectral Laplacian route misses by ~6x the source term, exactly the failure the paper warns about
("spectral methods require smoothness and uniform grids").

WHAT THIS CASE MAY AND MAY NOT CLAIM. The residual uses the same stencil as the reference solver, so driving
it to zero IS solving the reference's discrete system: "PINO matches the reference" is therefore NOT a
discretisation-accuracy result. What the physics term buys, and all we claim, is DATA EFFICIENCY, measured on
our own grid. The paper's headline numbers (20x error and 25x speedup vs PINN on Kolmogorov flow; 400x vs a
GPU pseudo-spectral solver) are the paper's, on the paper's problems, and are cited as such -- never as ours.
And for ONE well-posed forward instance a classical sparse direct solve still wins; the operator's argument is
many-query amortisation.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import CaseSpec, Variant

N_GRID = 32
WIDTH, MODES, LAYERS = 20, 10, 4
POOL, N_TEST = 128, 32               # labelled pool available, and the held-out test set
BUDGETS = (0, 8, 32, 128)            # the label budgets shown as variants (0 = no labels at all)
EPOCHS, BS = 80, 8
LAMBDA_PDE = 1.0                     # fallback weight on the PDE loss (paper's lambda)
BALANCE_EVERY = 25                   # steps between gradient-norm re-balances of that weight
_REPO = Path(__file__).resolve().parents[3]
_STATE: dict = {}

CASE = CaseSpec(
    id="bench-darcy-pino",
    category="canonical-benchmark",
    title="PINO: the operator that also knows the equation (Darcy, data-efficiency)",
    governing_equations=(
        r"-\nabla\!\cdot(a(\mathbf{x})\nabla u)=1,\ u|_{\partial\Omega}=0\ \text{en}\ (0,1)^2;\ "
        r"\min_\theta\ \mathcal{L}_{\text{data}} + \lambda\,\mathbb{E}_{a'\sim\mu}\|\mathcal{R}(a',\mathcal{G}_\theta(a'))\|^2"
    ),
    method="operator-pino",
    engine="pino-torch",
    real_or_synthetic="synthetic",
    inputs=("x", "y"),
    outputs=("u_pino", "u_true", "u_fno", "a"),
    domain={"x": (0.0, 1.0), "y": (0.0, 1.0)},
    grid={"x": N_GRID, "y": N_GRID},
    field_axes=("x", "y"),
    expected_band="with the equation in the loss the operator needs far fewer solved instances; at ZERO labels a "
                  "data-only operator is untrained while PINO still reaches a usable field",
    validation_anchor="operator-test-l2",
    train={"lr": 1e-3, "adam": 0},
    notes="Custom-engine FIELD-IO case (like bench-darcy-operator): trains BOTH lanes in build(), exports the "
          "PINO operator's own field-in ONNX (physical units), web_drivable=False -> lane=precompute. Variants "
          "are LABEL BUDGETS, so the workbench shows the data-efficiency curve directly.",
)


def variants() -> list[Variant]:
    """The variants are LABEL BUDGETS: how many solved instances each lane was allowed to see."""
    text = {
        0: ("No solved instances at all", "Sin instancias resueltas",
            "Zero labels: the data-only operator has no signal whatsoever, while PINO still has the equation.",
            "Cero etiquetas: el operador solo-datos no tiene senal alguna, mientras PINO aun tiene la ecuacion."),
        8: ("8 solved instances", "8 instancias resueltas",
            "A budget you could actually afford when each solve is expensive.",
            "Un presupuesto que si podrias pagar cuando cada resolucion es cara."),
        32: ("32 solved instances", "32 instancias resueltas",
             "A moderate budget: both lanes work, the physics term still helps.",
             "Un presupuesto moderado: ambas vias funcionan, el termino fisico aun ayuda."),
        128: ("128 solved instances", "128 instancias resueltas",
              "Data-rich: the regime where a data-only operator is at its strongest.",
              "Rico en datos: el regimen donde el operador solo-datos esta en su mejor momento."),
    }
    out = []
    for n in BUDGETS:
        le, ls, ne, ns = text[n]
        out.append(Variant(f"n{n}", le, ls, {"n_labels": float(n)}, ne, ns))
    return out


def extra_metrics(sf) -> dict:
    """Per-variant metrics: this budget's held-out test L2 for BOTH lanes, so the contrast is a number."""
    i = int(_STATE.get("cur", 0))
    out = {}
    if "pino_l2" in _STATE:
        out["l2_relative"] = round(float(_STATE["pino_l2"][i]), 6)       # headline = the PINO lane
        out["fno_data_only_l2"] = round(float(_STATE["fno_l2"][i]), 6)
        out["n_labels"] = int(BUDGETS[i])
        out["n_test"] = int(N_TEST)
        out["train_s_pino"] = round(float(_STATE["t_pino"][i]), 1)
        out["train_s_fno"] = round(float(_STATE["t_fno"][i]), 1)
        out["lambda_balanced"] = round(float(_STATE["lam"][i]), 2)
        out["ensemble_K"] = 0
    return out


class _Baked:
    """Serves the baked fields one variant at a time, in `variants()` order (the pipeline calls predict()
    exactly once per variant)."""

    def __init__(self, fields):
        self._f = np.asarray(fields, dtype=np.float64)   # [n_var, 4, H, W]
        self._i = 0

    def predict(self, XY):
        k = min(self._i, self._f.shape[0] - 1)
        _STATE["cur"] = k
        self._i += 1
        f = self._f[k]
        return np.stack([f[c].ravel() for c in range(f.shape[0])], axis=1)


class _Physical(__import__("torch").nn.Module):
    """Export wrapper: the ONNX emits u in PHYSICAL units (the net's output times the fixed scale)."""

    def __init__(self, net, scale: float):
        super().__init__()
        self.net = net
        self.scale = float(scale)

    def forward(self, x):
        return self.net(x) * self.scale


def build(seed: int, quick: bool = False) -> dict:
    import time

    import onnxruntime as ort
    import torch

    from ..datasets.darcy import _make_coeff, _solve_darcy
    from ..model.fno import FNO2d
    from ..model.pino import (U_SCALE, balance_lambda, pde_loss, relative_l2, test_time_optimize,
                              to_physical)

    pool, n_test, epochs = (8, 4, 2) if quick else (POOL, N_TEST, EPOCHS)
    # quick mode must keep the SAME NUMBER of budgets as variants() advertises, or the workbench shows
    # repeated fields; only the sizes shrink (and they must stay within the tiny pool).
    budgets = (0, 2, 4, 8) if quick else BUDGETS
    h = 1.0 / (N_GRID - 1)

    # ---- the family: labelled pool + held-out test set (the ONLY solves we pay for) ----
    rng = np.random.default_rng(int(seed))
    a_all = [_make_coeff(rng, N_GRID) for _ in range(pool + n_test)]
    u_all = [_solve_darcy(a) for a in a_all]

    gx, gy = np.meshgrid(np.linspace(0, 1, N_GRID), np.linspace(0, 1, N_GRID), indexing="ij")

    def pack(a):
        # fixed normalisation constants (the coefficient takes the two known values 3 and 12), NOT statistics
        # fitted to solved instances -- so a zero-label run really is zero-label
        return np.stack([(a - 7.5) / 4.5, gx, gy])

    X = torch.as_tensor(np.stack([pack(a) for a in a_all]), dtype=torch.float32)
    A = torch.as_tensor(np.stack([a[None] for a in a_all]), dtype=torch.float32)
    U = torch.as_tensor(np.stack([u[None] for u in u_all]), dtype=torch.float32)
    Xte, Ate, Ute = X[pool:], A[pool:], U[pool:]

    def virtual(g, k):
        """Fresh coefficient fields, NO solve: the unlimited PDE-loss dataset (paper Algorithm 1)."""
        av = np.stack([_make_coeff(g, N_GRID) for _ in range(k)])
        return (torch.as_tensor(np.stack([pack(x) for x in av]), dtype=torch.float32),
                torch.as_tensor(av[:, None], dtype=torch.float32))

    def train_lane(n_labels: int, use_physics: bool):
        torch.manual_seed(int(seed) + 7)
        g = np.random.default_rng(int(seed) + 1000)
        net = FNO2d(in_ch=3, width=WIDTH, modes=MODES, layers=LAYERS)
        opt = torch.optim.Adam(net.parameters(), lr=CASE.train["lr"])
        lam = None                               # gradient-balanced weight on the physics term
        step = 0
        t0 = time.perf_counter()
        for _ in range(epochs):
            if n_labels > 0:
                perm = torch.randperm(n_labels)
                for b in range(0, n_labels, BS):
                    idx = perm[b:b + BS]
                    opt.zero_grad()
                    ld = relative_l2(to_physical(net(X[idx])), U[idx])
                    if use_physics:
                        lp, _, _ = pde_loss(net(X[idx]), A[idx], h)
                        # re-balance occasionally: two extra backward passes, so not every step
                        if step % BALANCE_EVERY == 0:
                            lam = balance_lambda(net, ld, lp, lam)
                        loss = ld + (lam if lam is not None else LAMBDA_PDE) * lp
                    else:
                        loss = ld
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                    opt.step()
                    step += 1
            if use_physics:                      # virtual instances: equation only, no labels
                xv, av = virtual(g, BS)
                opt.zero_grad()
                lp, _, _ = pde_loss(net(xv), av, h)
                # no data term here, so the balanced weight does not apply: the equation IS the objective
                lp.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
        net.eval()
        with torch.no_grad():
            l2 = float(relative_l2(to_physical(net(Xte)), Ute))
        return net, l2, time.perf_counter() - t0, (float(lam) if lam is not None else LAMBDA_PDE)

    pino_l2, fno_l2, t_pino, t_fno, fields, lam_used = [], [], [], [], [], []
    best_net = None
    for n in budgets:
        net_f, e_f, tf, _ = train_lane(n, False)
        net_p, e_p, tp, lam_p = train_lane(n, True)
        lam_used.append(lam_p)
        fno_l2.append(e_f); pino_l2.append(e_p); t_fno.append(tf); t_pino.append(tp)
        best_net = net_p
        with torch.no_grad():                    # the SAME held-out instance for every budget
            up = to_physical(net_p(Xte[:1])).numpy()[0, 0]
            uf = to_physical(net_f(Xte[:1])).numpy()[0, 0]
        fields.append(np.stack([up, Ute[0, 0].numpy(), uf, Ate[0, 0].numpy()], axis=0))

    # ---- phase 2: test-time optimization on the held-out instance, with the anchor loss ----
    u_tto, hist = test_time_optimize(best_net, Xte[:1], Ate[:1], h,
                                     steps=(5 if quick else 120), lr=1e-4, anchor_weight=1.0)
    tto_l2 = float(np.linalg.norm(u_tto.numpy()[0, 0] - Ute[0, 0].numpy())
                   / (np.linalg.norm(Ute[0, 0].numpy()) + 1e-12))

    _STATE.update({"pino_l2": pino_l2, "fno_l2": fno_l2, "t_pino": t_pino, "t_fno": t_fno,
                   "budgets": list(budgets), "tto_l2": tto_l2, "tto_hist": hist,
                   "lam": lam_used})

    # ---- export the PINO operator's own ONNX, in physical units ----
    onnx_path = _REPO / "models" / f"{CASE.id}.onnx"
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    wrapped = _Physical(best_net, U_SCALE).eval()
    torch.onnx.export(
        wrapped, (Xte[:1],), str(onnx_path), input_names=["a_grid"], output_names=["u"],
        dynamic_axes={"a_grid": {0: "n"}, "u": {0: "n"}},
        opset_version=18, dynamo=True, verbose=False, external_data=False,
    )
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
