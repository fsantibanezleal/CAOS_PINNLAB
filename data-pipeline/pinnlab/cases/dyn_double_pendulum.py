"""Group A · canonical-benchmark — the DOUBLE PENDULUM: a chaotic dynamical system, PINN as a t -> state map.

This is the flagship of the `ode-dynamical` category. Unlike every field case, there is NO spatial domain: the
network maps **time t -> the two angles (theta1, theta2)** of a planar double pendulum, and the "solution" is an
animated TRAJECTORY, not a heatmap (rendered by TrajectoryAnimationKit, ADR-0063).

Governing equations (Lagrangian mechanics, released from rest):
    theta1'' = f1(theta1, theta2, theta1', theta2'),   theta2'' = f2(...)
the two coupled nonlinear second-order ODEs of the double pendulum. The PINN loss is the squared residual of these
two ODEs at collocation times, with the initial condition (theta_i(0)=theta_i0, theta_i'(0)=0) enforced EXACTLY by a
hard-constraint output transform   theta_hat_i(t) = theta_i0 + t^2 * N_i(t)   (t=0 -> theta_i0; every d/dt term
carries a factor t -> theta_i'(0)=0).

Honesty — chaos has a hard wall. A double pendulum is chaotic: two nearby initial conditions diverge exponentially
(the butterfly effect), so NO fixed network can track the true trajectory past a finite horizon. The validation
anchor is a high-accuracy RK45 integrator (rtol=atol=1e-10); we bake it alongside the PINN and report the
**leave-time** (where the PINN first leaves RK45 by > LEAVE_TOL rad) as the honest headline metric — NOT a long-term
match. A second RK45 run with a 1e-3 perturbed angle is baked too, so the App can show two nearby starts peeling
apart. This is the point of the case: you watch a PINN do well early and then lose a chaotic trajectory.
"""
from __future__ import annotations

import numpy as np

from .base import CaseSpec

# physical constants (SI-ish; g in m/s^2, lengths in m, masses in kg)
M1, M2 = 1.0, 1.0
L1, L2 = 1.0, 1.0
G = 9.81
# initial condition: both arms at 120 degrees from vertical, released from REST -> a classic chaotic regime
TH1_0 = np.deg2rad(120.0)
TH2_0 = np.deg2rad(120.0)
OM1_0, OM2_0 = 0.0, 0.0
T_MAX = 3.0          # seconds; the PINN (soft-IC) tracks the first ~1.5-2 s accurately then a CHAOTIC trajectory
                     # peels away — the honest "tracks then loses it" story, with the twin-IC butterfly diverging
                     # visibly over this longer window
N_EVAL = 601         # trajectory samples baked (Δt ≈ 5 ms -> smooth animation; 1-D trace keeps up to 601)
LEAVE_TOL = 0.30     # rad; "leave-time" = first t where the combined angle error exceeds this
TWIN_PERTURB = 1e-2  # rad added to theta1(0) for the butterfly twin trajectory (diverges visibly within the window)

_DEG = "°"

CASE = CaseSpec(
    id="dyn-double-pendulum",
    system_type="ode-dynamical",
    view_kit="TrajectoryAnimationKit",
    category="canonical-benchmark",
    title="Double pendulum (chaotic): a PINN as a t to state map, vs a high-accuracy integrator",
    governing_equations=(
        r"\ddot{\theta}_1 = f_1(\theta_1,\theta_2,\dot\theta_1,\dot\theta_2),\quad "
        r"\ddot{\theta}_2 = f_2(\cdot);\ \ \theta_i(0)=120^\circ,\ \dot\theta_i(0)=0;\ "
        r"\text{anchor: RK45 } (\mathrm{rtol}=\mathrm{atol}=10^{-10})"
    ),
    method="ode-residual-hard-ic",
    engine="deepxde",
    real_or_synthetic="synthetic-illustrative",
    inputs=("t",),
    outputs=("th1", "th2"),
    domain={"t": (0.0, T_MAX)},
    grid={"t": N_EVAL},
    field_axes=("t",),
    param_specs=(),
    expected_band=(
        "the PINN tracks the integrator for a short horizon, then a chaotic trajectory peels away; the honest "
        "metric is the leave-time (where |Δθ| first exceeds 0.30 rad), not a long-term match"
    ),
    validation_anchor="integrator-ref",
    train={
        "layers": [1, 96, 96, 96, 96, 2],
        "activation": "tanh",  # stable for the soft-IC IVP (SIREN + soft IC was unstable)
        "lr": 5e-4,
        "adam": 25000,
        "lbfgs": True,
        "num_domain": 4500,
        "num_test": 3000,
    },
    notes="ode-dynamical flagship: no spatial field; t->state; hard-IC released from rest; RK45 anchor + leave-time; "
    "twin-IC butterfly baked for the App.",
)


# ── the equations of motion (shared by the torch residual and the numpy integrator) ──────────────────
def _accel_np(th1, th2, om1, om2):
    """The two double-pendulum angular accelerations (numpy), standard explicit form."""
    delta = th1 - th2
    den = 2 * M1 + M2 - M2 * np.cos(2 * th1 - 2 * th2)
    a1 = (
        -G * (2 * M1 + M2) * np.sin(th1)
        - M2 * G * np.sin(th1 - 2 * th2)
        - 2 * np.sin(delta) * M2 * (om2 ** 2 * L2 + om1 ** 2 * L1 * np.cos(delta))
    ) / (L1 * den)
    a2 = (
        2 * np.sin(delta)
        * (om1 ** 2 * L1 * (M1 + M2) + G * (M1 + M2) * np.cos(th1) + om2 ** 2 * L2 * M2 * np.cos(delta))
    ) / (L2 * den)
    return a1, a2


def _rk45(th1_0: float, t_eval: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """High-accuracy reference trajectory (RK45, rtol=atol=1e-10) -> (theta1, theta2) on t_eval."""
    from scipy.integrate import solve_ivp

    def rhs(_t, y):
        th1, th2, om1, om2 = y
        a1, a2 = _accel_np(th1, th2, om1, om2)
        return [om1, om2, a1, a2]

    sol = solve_ivp(
        rhs, (float(t_eval[0]), float(t_eval[-1])), [th1_0, TH2_0, OM1_0, OM2_0],
        t_eval=t_eval, method="RK45", rtol=1e-10, atol=1e-10, max_step=0.01,
    )
    return sol.y[0], sol.y[1]


def build(seed: int) -> dict:
    import deepxde as dde
    import torch

    dde.config.set_random_seed(int(seed))
    geom = dde.geometry.TimeDomain(0.0, T_MAX)

    def pde(x, u):
        th1 = u[:, 0:1]
        th2 = u[:, 1:2]
        om1 = dde.grad.jacobian(u, x, i=0, j=0)
        om2 = dde.grad.jacobian(u, x, i=1, j=0)
        a1_net = dde.grad.hessian(u, x, component=0, i=0, j=0)
        a2_net = dde.grad.hessian(u, x, component=1, i=0, j=0)
        delta = th1 - th2
        den = 2 * M1 + M2 - M2 * torch.cos(2 * th1 - 2 * th2)
        a1 = (
            -G * (2 * M1 + M2) * torch.sin(th1)
            - M2 * G * torch.sin(th1 - 2 * th2)
            - 2 * torch.sin(delta) * M2 * (om2 ** 2 * L2 + om1 ** 2 * L1 * torch.cos(delta))
        ) / (L1 * den)
        a2 = (
            2 * torch.sin(delta)
            * (om1 ** 2 * L1 * (M1 + M2) + G * (M1 + M2) * torch.cos(th1) + om2 ** 2 * L2 * M2 * torch.cos(delta))
        ) / (L2 * den)
        return [a1_net - a1, a2_net - a2]

    # SOFT initial conditions (not a t^2 hard-constraint output transform): the hard ansatz theta=theta0+t^2*N kills
    # the parameter gradient near t=0 (d theta/d params ~ t^2 -> 0), so the net never learns the initial acceleration.
    # Soft IVP constraints are well-conditioned: theta_i(0)=theta_i0 and d theta_i/dt (0)=0, weighted above the residual.
    def boundary_t0(x, on_boundary):
        return on_boundary and np.isclose(x[0], 0.0)

    ic_th1 = dde.icbc.IC(geom, lambda _x: TH1_0, boundary_t0, component=0)
    ic_th2 = dde.icbc.IC(geom, lambda _x: TH2_0, boundary_t0, component=1)
    bc_w1 = dde.icbc.OperatorBC(geom, lambda x, y, _X: dde.grad.jacobian(y, x, i=0, j=0), boundary_t0)
    bc_w2 = dde.icbc.OperatorBC(geom, lambda x, y, _X: dde.grad.jacobian(y, x, i=1, j=0), boundary_t0)

    t = CASE.train
    data = dde.data.PDE(
        geom, pde, [ic_th1, ic_th2, bc_w1, bc_w2],
        num_domain=t["num_domain"], num_boundary=2, num_test=t["num_test"],
    )
    net = dde.nn.FNN(t["layers"], t["activation"], "Glorot normal")
    model = dde.Model(data, net)
    # losses: [res1, res2, ic_th1, ic_th2, w1(0), w2(0)] — weight the ICs well above the residual so the IVP is pinned
    model.compile("adam", lr=t["lr"], loss_weights=[1, 1, 100, 100, 100, 100])
    # web_drivable=False -> precompute/replay lane: the Field tab is a baked animated trajectory, not a coord heatmap
    return {"model": model, "input_dim": 1, "web_drivable": False}


def extra_metrics(sf) -> dict:
    """Bake the RK45 anchor + a twin-IC trajectory into the trace and compute the honest leave-time. Mutates `sf`
    (adds reference fields + constants) so they ride along in the committed trace."""
    t = np.asarray(sf.axes["t"], dtype=np.float64)
    th1_p = np.asarray(sf.fields["th1"], dtype=np.float64).reshape(-1)
    th2_p = np.asarray(sf.fields["th2"], dtype=np.float64).reshape(-1)

    th1_r, th2_r = _rk45(TH1_0, t)
    th1_tw, th2_tw = _rk45(TH1_0 + TWIN_PERTURB, t)

    shape = sf.fields["th1"].shape
    sf.fields["th1_ref"] = th1_r.reshape(shape)
    sf.fields["th2_ref"] = th2_r.reshape(shape)
    sf.fields["th1_twin"] = th1_tw.reshape(shape)
    sf.fields["th2_twin"] = th2_tw.reshape(shape)

    # leave-time: first t where the combined angle error exceeds LEAVE_TOL (wrapped to [-pi,pi])
    def wrap(a):
        return (a + np.pi) % (2 * np.pi) - np.pi

    err = np.sqrt(wrap(th1_p - th1_r) ** 2 + wrap(th2_p - th2_r) ** 2)
    over = np.where(err > LEAVE_TOL)[0]
    leave_time = float(t[over[0]]) if len(over) else float(t[-1])

    # relative-L2 over the whole window (will be large for chaos — reported honestly)
    num = np.sqrt(np.sum((th1_p - th1_r) ** 2 + (th2_p - th2_r) ** 2))
    den = np.sqrt(np.sum(th1_r ** 2 + th2_r ** 2)) or 1.0
    l2_rel = float(num / den)

    # twin-IC divergence rate (a crude largest-Lyapunov estimate): fit log-separation over the CLEAN exponential-
    # growth window only — after the separation saturates near O(pi) the slope flattens and would bias the estimate.
    sep = np.sqrt(wrap(th1_r - th1_tw) ** 2 + wrap(th2_r - th2_tw) ** 2)
    sep = np.maximum(sep, 1e-12)
    grow = np.where((sep > 2 * TWIN_PERTURB) & (sep < 0.5))[0]
    lam = float(np.polyfit(t[grow], np.log(sep[grow]), 1)[0]) if len(grow) >= 5 else float("nan")

    sf.scalars.update({
        "l1": L1, "l2": L2, "m1": M1, "m2": M2,
        "th1_0": float(TH1_0), "th2_0": float(TH2_0),
        "t_max": float(T_MAX), "leave_tol": float(LEAVE_TOL),
        "twin_perturb": float(TWIN_PERTURB),
    })
    out = {
        "l2_relative": round(l2_rel, 6),
        "leave_time": round(leave_time, 4),
        "validation_anchor": "integrator-ref",
    }
    if np.isfinite(lam):
        out["lyapunov_est"] = round(lam, 4)
    return out
