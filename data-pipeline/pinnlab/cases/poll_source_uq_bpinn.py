"""Group C · pollution-environmental — Bayesian PINN (deep ensemble) for pollutant diffusion with UNCERTAINTY.

A dissolved pollutant diffuses in 1D, c_t = D c_xx on x in [0,1], t in [0,1], c=0 at the walls, true field
    c*(x,t) = e^{-D pi^2 t} sin(pi x)   (the fundamental diffusion mode, D=0.1).
We are given only a HANDFUL of sparse, noisy sensor readings (not the full initial condition). A single PINN would
report one answer with no error bars; instead we train a DEEP ENSEMBLE of K independently-initialized PINNs — the
recognized cheap approximation to Bayesian inference (Lakshminarayanan 2017). The predictive MEAN tracks c*; the
ensemble STD is the epistemic uncertainty — it stays small near sensors and the c=0 walls, and GROWS where data is
sparse. The whole ensemble is exported as ONE ONNX graph emitting [mean, std], so the live lane ships a single file.

real_or_synthetic = synthetic-illustrative: a UQ demonstrator on a manufactured field (analytic c*), not a measured
dataset. Primary score = relative-L2 of the mean vs c*; UQ quality = 2-sigma calibration (fraction of the grid where
|mean - c*| <= 2 std) reported in extra_metrics.
"""
from __future__ import annotations

import numpy as np

from .base import CaseSpec

D = 0.1
N_OBS = 24
NOISE = 0.02
K_MEMBERS = 5


def analytic(xt: np.ndarray) -> np.ndarray:
    xt = np.asarray(xt, dtype=np.float64)
    x = xt[:, 0:1]
    t = xt[:, 1:2]
    return np.exp(-D * np.pi ** 2 * t) * np.sin(np.pi * x)


CASE = CaseSpec(
    id="poll-source-uq-bpinn",
    system_type="time-evol-1d",
    category="pollution-environmental",
    title="Bayesian PINN (deep ensemble) — pollutant diffusion with epistemic uncertainty",
    governing_equations=(
        r"c_t = D\,c_{xx},\ c|_{x=0,1}=0,\ c^*=e^{-D\pi^2 t}\sin(\pi x);\ "
        r"\text{deep ensemble} \to [\text{mean},\ \text{std}]\ \text{(epistemic UQ from sparse noisy sensors)}"
    ),
    method="bayesian-ensemble-uq",
    engine="deepxde",
    real_or_synthetic="synthetic-illustrative",
    inputs=("x", "t"),
    outputs=("c", "c_std"),  # c = predictive mean (scored vs c*); c_std = epistemic uncertainty
    domain={"x": (0.0, 1.0), "t": (0.0, 1.0)},
    grid={"x": 61, "t": 61},
    expected_band="mean tracks c* (relative-L2 < 5e-2); std small near sensors/walls, larger in data-sparse regions; ~95% 2-sigma calibration",
    validation_anchor="analytic",
    train={"lr": 1e-3, "adam": 6000, "lbfgs": False, "num_domain": 1500, "num_boundary": 0, "num_test": 2000},
    notes="Deep ensemble of K=5 independently-initialized PINNs (approx. Bayesian); hard c=0 walls; N=24 sparse noisy "
          "sensors; exported as a single ONNX emitting [mean, std]. Custom-engine (prebuilt) case. "
          "Single honest benchmark variant: the UQ knob (sensor noise/sparsity) is a TRAINING-DATA knob, not a network "
          "input, so a discrete family would need K separate ONNX exports the share-one-ONNX pipeline does not support; "
          "and a parameter sweep would erase the sparse-sensor UQ story (ADR-0016 §9.A). Not parametric.",
)


def extra_metrics(sf) -> dict:
    from ..model.analytic import linspace_grid

    _coords, XT, _shape = linspace_grid(CASE.domain, CASE.grid)
    truth = analytic(XT).reshape(-1)
    mean = np.asarray(sf.fields["c"], dtype=np.float64).reshape(-1)
    std = np.asarray(sf.fields["c_std"], dtype=np.float64).reshape(-1)
    within2 = float(np.mean(np.abs(mean - truth) <= 2.0 * std + 1e-9))
    return {
        "uq_calibration_2sigma": round(within2, 4),
        "uq_mean_std": round(float(std.mean()), 6),
        "uq_max_std": round(float(std.max()), 6),
        "ensemble_K": K_MEMBERS,
    }


class _Ensemble:
    """Torch module: stack K trained sub-nets, emit [mean, std] per query point (the exported live graph)."""

    def __init__(self, nets):
        import torch

        class _Mod(torch.nn.Module):
            def __init__(self, members):
                super().__init__()
                self.members = torch.nn.ModuleList(members)

            def forward(self, x):
                preds = torch.stack([m(x) for m in self.members], dim=0)  # (K, N, 1)
                mean = preds.mean(dim=0)
                std = preds.std(dim=0, unbiased=False)
                return torch.cat([mean, std], dim=1)  # (N, 2)

        self.net = _Mod(nets).eval()

    def predict(self, X):
        import torch

        with torch.no_grad():
            return self.net(torch.as_tensor(np.asarray(X), dtype=torch.float32)).cpu().numpy()


def build(seed: int, quick: bool = False) -> dict:
    import deepxde as dde

    geom = dde.geometry.Interval(0.0, 1.0)
    timedomain = dde.geometry.TimeDomain(0.0, 1.0)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    def pde(x, c):
        c_t = dde.grad.jacobian(c, x, i=0, j=1)
        c_xx = dde.grad.hessian(c, x, i=0, j=0)
        return c_t - D * c_xx

    # a pool of candidate sensor sites; each member bootstraps (bags) a subset + its OWN noise realization, so members
    # disagree most where data is sparse — that disagreement IS the epistemic uncertainty (bagging + random init).
    pool_rng = np.random.default_rng(seed)
    pool_xt = pool_rng.uniform([0.05, 0.0], [0.95, 1.0], size=(2 * N_OBS, 2))

    k = 2 if quick else K_MEMBERS
    iters = 200 if quick else int(CASE.train["adam"])
    nets = []
    for m in range(k):
        dde.config.set_random_seed(int(seed) + 1000 * (m + 1))  # distinct init per member -> epistemic spread
        mrng = np.random.default_rng(seed + 7 * (m + 1))
        pick = mrng.integers(0, len(pool_xt), size=N_OBS)        # bootstrap sample of sensors (with replacement)
        ob_xt = pool_xt[pick]
        ob_c = analytic(ob_xt) + NOISE * mrng.standard_normal((N_OBS, 1))
        observe = dde.icbc.PointSetBC(ob_xt, ob_c, component=0)
        data = dde.data.TimePDE(
            geomtime, pde, [observe],
            num_domain=CASE.train["num_domain"], num_boundary=0, anchors=ob_xt, num_test=CASE.train["num_test"],
        )
        net = dde.nn.FNN([2] + [24] * 3 + [1], "tanh", "Glorot normal")
        net.apply_output_transform(lambda x, y: x[:, 0:1] * (1.0 - x[:, 0:1]) * y)  # hard c=0 at x=0,1
        model = dde.Model(data, net)
        model.compile("adam", lr=CASE.train["lr"], loss_weights=[1, 30])
        model.train(iterations=iters, display_every=max(1, iters // 3))
        nets.append(net.eval())

    return {"model": _Ensemble(nets), "input_dim": 2, "prebuilt": True}
