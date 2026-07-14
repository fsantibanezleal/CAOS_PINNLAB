"""Group D · industrial-fluids-heat — HIDDEN VELOCITY FROM DYE: the Hidden-Fluid-Mechanics mechanism (Raissi,
Yazdani & Karniadakis, Science 2020, DOI 10.1126/science.aaw4741) at CPU-lane scale. The flagship of the
estimation reframe (issue #48): you can only SEE the dye; the PINN estimates the CURRENT underneath.

Truth: a steady incompressible cellular flow on (0,1)^2,
    u* = A sin(pi x) cos(pi y),   v* = -A cos(pi x) sin(pi y),   A = 1.5   (div-free; walls are streamlines),
advecting-diffusing a Gaussian dye blob:  c_t + u* c_x + v* c_y = D lap(c),  D = 0.02, no-flux walls.
The dye field c(x,y,t) is produced by a seeded explicit FD solve (central differences are legitimate here:
cell Peclet = A dx / D ~ 0.59 < 2; upwind would inject numerical diffusion comparable to D and corrupt the
truth), CFL-safe dt, stability-ASSERTED (boundedness + mass conservation), never trusted blindly.

The PINN sees ONLY ~800 sparse noisy space-time dye samples + the physics (transport residual with KNOWN D +
soft incompressibility). NO velocity data, NO IC/BC on c: that absence is the point (HFM is "agnostic to the
geometry or the initial and boundary conditions"). Net (x,y,t) -> (u, v, c); primary score = recovered u vs u*.

HONESTY: velocity is identifiable only where the dye has gradients. extra_metrics bakes the dye-swept mask and
reports the u error inside the swept region vs the dead zones separately; the full-grid number is published
as-is even though the never-dyed zones inflate it.
"""
from __future__ import annotations

import numpy as np

from .base import CaseSpec, ParamSpec, Variant

A_FLOW = 1.5
D_COEF = 0.02
X0, Y0 = 0.5, 0.2
S0_SQ = 0.006
N_OBS = 800
NOISE_FRAC = 0.005  # of max |c|
HOLDOUT_FRAC = 0.2  # never trained on; scored post-bake by build_hidden_velocity_validation.py

_NFD = 129  # FD grid per axis
_DT = 4.0e-4
_NT_SAVE = 51  # saved time slices (t = 0 .. 1)


def u_true(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    return A_FLOW * np.sin(np.pi * X[:, 0:1]) * np.cos(np.pi * X[:, 1:2])


def v_true(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    return -A_FLOW * np.cos(np.pi * X[:, 0:1]) * np.sin(np.pi * X[:, 1:2])


CASE = CaseSpec(
    id="ind-hidden-velocity",
    system_type="inverse-assim",
    view_kit="HiddenFlowKit",
    category="industrial-fluids-heat",
    title="Hidden velocity from dye: estimate the current from passive-scalar observations (HFM mechanism)",
    governing_equations=(
        r"c_t + \mathbf{u}\cdot\nabla c = D\nabla^2 c,\ \ \nabla\cdot\mathbf{u}=0;\ \ "
        r"\text{recover }\mathbf{u}(x,y)\text{ from sparse noisy }c\text{ samples};\ "
        r"\mathbf{u}^*=A(\sin\pi x\cos\pi y,\,-\cos\pi x\sin\pi y)"
    ),
    method="inverse-hidden-state",
    engine="deepxde",
    real_or_synthetic="synthetic",
    inputs=("x", "y", "t"),
    outputs=("u", "v", "c"),  # u is PRIMARY -> l2_relative scores the recovered current vs u*
    domain={"x": (0.0, 1.0), "y": (0.0, 1.0), "t": (0.0, 1.0)},
    grid={"x": 81, "y": 81},
    field_axes=("x", "y"),
    param_specs=(ParamSpec("t", "Time t", "Tiempo t", 0.5, 0.0, 1.0, 0.02),),
    expected_band=(
        "recovered current accurate INSIDE the dye-swept region (u rel-L2 there well below the full-grid number); "
        "unidentifiable in never-dyed dead zones, and the full-grid L2 honestly includes them"
    ),
    validation_anchor="analytic",
    train={
        "lr": 1e-3,
        "adam": 15000,
        "lbfgs": True,
        "num_domain": 8000,
        "num_test": 4000,
        "loss_weights": [1, 1, 1, 1, 60],  # [transport, incompressibility, u_t=0, v_t=0 (steady), dye data]
    },
    notes=(
        "HFM mechanism (Science 2020) at CPU scale: dye observations + transport physics ONLY (no velocity data, "
        "no IC/BC on c); D known; soft incompressibility; seeded FD dye truth (central, Pe_cell 0.59, CFL-safe, "
        "stability-asserted); swept-vs-dead-zone honesty split."
    ),
)


def analytic(xyt: np.ndarray) -> np.ndarray:
    """Primary output is u -> score the recovered current against the closed-form u* (t-independent)."""
    return u_true(xyt)


def variants() -> list[Variant]:
    presets = [
        ("t02", 0.2, "t=0.2 — the blob barely moved; the current is only starting to reveal itself.", "t=0.2 — el parche apenas se movió; la corriente apenas empieza a revelarse."),
        ("t04", 0.4, "t=0.4 — the dye arcs with the vortex; the swept region grows.", "t=0.4 — el tinte se arquea con el vórtice; la región barrida crece."),
        ("t05", 0.5, "t=0.5 — mid-window; the reference regime.", "t=0.5 — mitad de la ventana; el régimen de referencia."),
        ("t06", 0.6, "t=0.6 — over half the ring swept; the recovery is at its best there.", "t=0.6 — más de media vuelta barrida; la recuperación es mejor allí."),
        ("t08", 0.8, "t=0.8 — broad and fading dye; late-time information is weaker.", "t=0.8 — tinte ancho y tenue; la información tardía es más débil."),
        ("t10", 1.0, "t=1.0 — the window ends; dead zones never saw dye and stay unidentifiable.", "t=1.0 — termina la ventana; las zonas muertas nunca vieron tinte y siguen sin identificarse."),
    ]
    return [Variant(vid, f"t={tv:g}", f"t={tv:g}", {"t": tv}, en, es) for vid, tv, en, es in presets]


# ---------------------------------------------------------------------------
# The seeded FD dye truth (module-cached: build(), extra_metrics() and the validation tool share one solve).
# ---------------------------------------------------------------------------
_FD: dict | None = None


def fd_dye_truth() -> dict:
    """Explicit FD solve of the forward transport on the closed-form flow. Returns
    { 'x','y' (nodes), 't_save', 'c' [nt_save, nx, ny] } with stability ASSERTED, cached per process."""
    global _FD
    if _FD is not None:
        return _FD
    n = _NFD
    xs = np.linspace(0.0, 1.0, n)
    ys = np.linspace(0.0, 1.0, n)
    dx = xs[1] - xs[0]
    XX, YY = np.meshgrid(xs, ys, indexing="ij")
    U = A_FLOW * np.sin(np.pi * XX) * np.cos(np.pi * YY)
    V = -A_FLOW * np.cos(np.pi * XX) * np.sin(np.pi * YY)
    # stability preconditions (fail loudly, never bake garbage): cell Peclet for central + CFL for dt
    pe_cell = float(np.abs([U, V]).max()) * dx / D_COEF
    assert pe_cell < 2.0, f"cell Peclet {pe_cell:.2f} >= 2: central differencing invalid"
    assert _DT < dx / np.abs([U, V]).max() and _DT < dx * dx / (4.0 * D_COEF), "dt violates CFL"

    c = np.exp(-((XX - X0) ** 2 + (YY - Y0) ** 2) / (2.0 * S0_SQ))
    mass0 = float(c.sum())
    n_steps = int(round(1.0 / _DT))
    save_every = n_steps // (_NT_SAVE - 1)
    saved = [c.copy()]
    for step in range(1, n_steps + 1):
        # ghost cells by reflection = no-flux walls (advective normal flux already vanishes: U=0 at x-walls, V=0 at y-walls)
        cp = np.pad(c, 1, mode="edge")
        c_x = (cp[2:, 1:-1] - cp[:-2, 1:-1]) / (2 * dx)
        c_y = (cp[1:-1, 2:] - cp[1:-1, :-2]) / (2 * dx)
        lap = (cp[2:, 1:-1] + cp[:-2, 1:-1] + cp[1:-1, 2:] + cp[1:-1, :-2] - 4 * c) / (dx * dx)
        c = c + _DT * (-U * c_x - V * c_y + D_COEF * lap)
        if step % save_every == 0 and len(saved) < _NT_SAVE:
            saved.append(c.copy())
    while len(saved) < _NT_SAVE:
        saved.append(c.copy())
    C = np.stack(saved)  # [nt, nx, ny]
    # stability postconditions: bounded (no blow-up/negative overshoot beyond tolerance) + mass conserved (no-flux)
    assert np.isfinite(C).all(), "FD dye solve produced non-finite values"
    assert float(C.max()) <= 1.05 and float(C.min()) >= -0.02, f"FD dye out of physical bounds [{C.min():.3f},{C.max():.3f}]"
    mass_drift = abs(float(C[-1].sum()) - mass0) / mass0
    assert mass_drift < 0.01, f"FD mass drift {mass_drift:.4f} > 1%: no-flux walls broken"
    _FD = {"x": xs, "y": ys, "t_save": np.linspace(0.0, 1.0, _NT_SAVE), "c": C}
    return _FD


def _sample_fd(pts: np.ndarray) -> np.ndarray:
    """Trilinear sample of the FD dye field at [N,3] (x,y,t) points -> [N,1]."""
    fd = fd_dye_truth()
    xs, ys, ts, C = fd["x"], fd["y"], fd["t_save"], fd["c"]

    def frac_idx(v, nodes):
        f = (np.clip(v, nodes[0], nodes[-1]) - nodes[0]) / (nodes[-1] - nodes[0]) * (len(nodes) - 1)
        i0 = np.clip(np.floor(f).astype(int), 0, len(nodes) - 2)
        return i0, f - i0

    ix, fx = frac_idx(pts[:, 0], xs)
    iy, fy = frac_idx(pts[:, 1], ys)
    it, ft = frac_idx(pts[:, 2], ts)
    out = np.zeros(len(pts))
    for dt_ in (0, 1):
        wt = np.where(dt_ == 0, 1 - ft, ft)
        for dx_ in (0, 1):
            wx = np.where(dx_ == 0, 1 - fx, fx)
            for dy_ in (0, 1):
                wy = np.where(dy_ == 0, 1 - fy, fy)
                out += wt * wx * wy * C[it + dt_, ix + dx_, iy + dy_]
    return out[:, None]


# the seeded dye samples, shared by build() (training) / extra_metrics (overlay) / the validation tool (holdout)
_OBS: dict | None = None


def dye_observations(seed: int) -> dict:
    """Seeded space-time dye samples, split train/holdout. Cached per process for one seed."""
    global _OBS
    if _OBS is not None and _OBS["seed"] == seed:
        return _OBS
    rng = np.random.default_rng(seed)
    pts = np.column_stack([
        rng.uniform(0.02, 0.98, N_OBS),
        rng.uniform(0.02, 0.98, N_OBS),
        rng.uniform(0.0, 1.0, N_OBS),
    ])
    c_clean = _sample_fd(pts)
    noise = NOISE_FRAC * float(np.abs(c_clean).max())
    c_obs = c_clean + noise * rng.standard_normal(c_clean.shape)
    n_hold = int(round(HOLDOUT_FRAC * N_OBS))
    idx = rng.permutation(N_OBS)
    hold, tr = idx[:n_hold], idx[n_hold:]
    _OBS = {"seed": seed, "train_xyz": pts[tr], "train_c": c_obs[tr], "hold_xyz": pts[hold], "hold_c": c_obs[hold]}
    return _OBS


def swept_mask_on_grid() -> np.ndarray:
    """The dye-swept identifiability mask on the case grid: max over t of |grad c_FD| above a threshold.
    Velocity is informed by the transport equation ONLY where the dye has gradients."""
    from ..model.analytic import linspace_grid

    fd = fd_dye_truth()
    C = fd["c"]
    dx = fd["x"][1] - fd["x"][0]
    gx = np.gradient(C, dx, axis=1)
    gy = np.gradient(C, dx, axis=2)
    gmax = np.sqrt(gx ** 2 + gy ** 2).max(axis=0)  # [nx, ny] on the FD grid
    thr = 0.05 * float(gmax.max())
    field_dom = {a: CASE.domain[a] for a in CASE.axes}  # (x,y) only: t is a parameter axis
    _coords, XY, shape = linspace_grid(field_dom, CASE.grid)
    # nearest-node lookup FD -> case grid (both uniform on [0,1])
    ix = np.clip(np.round(XY[:, 0] * (_NFD - 1)).astype(int), 0, _NFD - 1)
    iy = np.clip(np.round(XY[:, 1] * (_NFD - 1)).astype(int), 0, _NFD - 1)
    return (gmax[ix, iy] > thr).reshape(shape)


def extra_metrics(sf) -> dict:
    """Bake the truth fields + the swept mask + the observation overlay, and report the HONEST error split:
    the recovered current inside the dye-swept region vs the never-dyed dead zones."""
    from ..model.analytic import linspace_grid

    field_dom = {a: CASE.domain[a] for a in CASE.axes}
    _coords, XY, shape = linspace_grid(field_dom, CASE.grid)
    ut = u_true(XY).reshape(shape)
    vt = v_true(XY).reshape(shape)
    sf.fields["u_true"] = ut
    sf.fields["v_true"] = vt
    mask = swept_mask_on_grid()
    sf.fields["swept"] = mask.astype(float)

    du = np.hypot(sf.fields["u"] - ut, sf.fields["v"] - vt)
    st = np.hypot(ut, vt)
    denom = float(np.sqrt(np.mean(st ** 2)))
    err_swept = float(np.sqrt(np.mean(du[mask] ** 2))) / denom
    err_dead = float(np.sqrt(np.mean(du[~mask] ** 2))) / denom if (~mask).any() else 0.0

    obs = _OBS
    if obs is not None:
        sf.inverse = {
            "param": "u",
            "observe_output": "c",
            "observations": [
                [round(float(x), 4), round(float(y), 4), round(float(t), 4), round(float(cv), 4)]
                for (x, y, t), cv in zip(obs["train_xyz"], obs["train_c"].reshape(-1))
            ][:250],  # a readable overlay subset; the full set is in the training record
        }
    return {
        "speed_rel_rmse_swept": round(err_swept, 6),
        "speed_rel_rmse_dead": round(err_dead, 6),
        "swept_area_frac": round(float(mask.mean()), 4),
    }


def build(seed: int) -> dict:
    import deepxde as dde

    dde.config.set_random_seed(int(seed))
    geom = dde.geometry.Rectangle([0.0, 0.0], [1.0, 1.0])
    timedomain = dde.geometry.TimeDomain(0.0, 1.0)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    def pde(x, y):
        u = y[:, 0:1]
        v = y[:, 1:2]
        c_t = dde.grad.jacobian(y, x, i=2, j=2)
        c_x = dde.grad.jacobian(y, x, i=2, j=0)
        c_y = dde.grad.jacobian(y, x, i=2, j=1)
        c_xx = dde.grad.hessian(y, x, component=2, i=0, j=0)
        c_yy = dde.grad.hessian(y, x, component=2, i=1, j=1)
        u_x = dde.grad.jacobian(y, x, i=0, j=0)
        v_y = dde.grad.jacobian(y, x, i=1, j=1)
        # STEADY-FLOW assumption (stated in the case doc): the current does not depend on t. Without this the
        # net wastes its freedom on a time-varying u that sparse dye data cannot pin (measured: 38-60% swept
        # error); enforcing u_t = v_t = 0 aggregates the dye information from ALL times into ONE steady field.
        u_t = dde.grad.jacobian(y, x, i=0, j=2)
        v_t = dde.grad.jacobian(y, x, i=1, j=2)
        transport = c_t + u * c_x + v * c_y - D_COEF * (c_xx + c_yy)
        incompress = u_x + v_y
        return [transport, incompress, u_t, v_t]

    obs = dye_observations(seed)
    observe_c = dde.icbc.PointSetBC(obs["train_xyz"], obs["train_c"], component=2)

    t = CASE.train
    data = dde.data.TimePDE(
        geomtime, pde, [observe_c],
        num_domain=t["num_domain"], anchors=obs["train_xyz"], num_test=t["num_test"],
    )
    net = dde.nn.FNN([3] + [64] * 4 + [3], "tanh", "Glorot normal")
    model = dde.Model(data, net)
    model.compile("adam", lr=t["lr"], loss_weights=t["loss_weights"])
    return {"model": model, "input_dim": 3}
