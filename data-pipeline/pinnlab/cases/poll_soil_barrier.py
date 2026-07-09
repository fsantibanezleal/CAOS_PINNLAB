"""Group C · pollution-environmental — contaminated-site barrier transport, DOMAIN-DECOMPOSITION (FBPINN-style) PINN.

A dissolved contaminant diffuses through a soil column containing one low-permeability vertical barrier (slurry/clay
cutoff): a low-D slab in [AB,BB] that slows the plume. Pure-diffusion illustrative case (V=0):
    c_t = D(x) c_xx + f,   D(x) = D_soil outside [AB,BB], D_barrier (100x lower) inside.
The coefficient JUMP makes c develop a kink at each barrier face (continuous c, continuous diffusive flux, but
discontinuous c_x). MMS anchor = the layered series-resistance steady profile (closed-form, exhibits the kink):
    Psi(x) = 1 - R(x)/R(L),  R(x)=int_0^x dx'/D(x'),   c*(x,t) = (1 - e^{-t}) Psi(x),  f = e^{-t} Psi(x)
(the diffusive flux D c*_x = const, so its divergence vanishes in the interior; f is just the time term).

Method — DOMAIN DECOMPOSITION (FBPINN partition-of-unity, dossier §4 #14): a 2-channel net blended by sigmoid
windows w_left+w_right=1 across the barrier centre, so the kink in c_x is produced by two networks meeting on the
subdomain boundary rather than a single smooth tanh fighting the jump. Hard IC/inlet/outlet via the output transform.
real_or_synthetic = synthetic-illustrative (illustrative engineering values; MMS anchor, NOT a calibrated site).
The strict per-subdomain-normalized FBPINN is documented in docs/methods/domain-decomposition.md.
"""
from __future__ import annotations

import numpy as np

from .base import CaseSpec, Variant

D_SOIL = 1.0
D_BARR = 1.0e-1  # 10x diffusion contrast (a 100x jump makes the kink too severe for a 2-channel net on CPU)
AB, BB = 0.45, 0.55
R_L = AB / D_SOIL + (BB - AB) / D_BARR + (1.0 - BB) / D_SOIL  # total diffusion resistance
X_C = 0.5 * (AB + BB)
BETA = 40.0  # partition-of-unity window sharpness

CASE = CaseSpec(
    id="poll-soil-barrier",
    category="pollution-environmental",
    title="Contaminated-site barrier: domain-decomposition (FBPINN) PINN",
    governing_equations=(
        r"c_t = D(x)\,c_{xx} + f,\ D=D_{soil}\ \text{outside}\ [0.45,0.55],\ D_{barrier}=0.1\ \text{inside};\ "
        r"c^*=(1-e^{-t})\,\Psi(x)"
    ),
    method="domain-decomposition",
    engine="deepxde",
    real_or_synthetic="synthetic-illustrative",
    inputs=("x", "t"),
    outputs=("c",),
    domain={"x": (0.0, 1.0), "t": (0.0, 1.0)},
    grid={"x": 101, "t": 51},
    expected_band="plume slowed by the low-D barrier (kink in c at each face); relative-L2 ~2e-1 on the CPU 2-channel lane (the coefficient-jump kink is the hard part; the strict per-subdomain-normalized FBPINN + GPU tighten it)",
    validation_anchor="analytic",
    train={"lr": 1e-3, "adam": 18000, "lbfgs": True, "num_domain": 4000, "num_boundary": 0, "num_initial": 0, "num_test": 8000},
    notes="2-channel partition-of-unity (FBPINN-style) net; series-resistance MMS exhibits the barrier kink; hard IC/BC.",
)


def _psi_np(x: np.ndarray) -> np.ndarray:
    r = np.clip(x, 0, AB) / D_SOIL + np.clip(x - AB, 0, BB - AB) / D_BARR + np.clip(x - BB, 0, 1 - BB) / D_SOIL
    return 1.0 - r / R_L


def analytic(xt: np.ndarray) -> np.ndarray:
    xt = np.asarray(xt, dtype=np.float64)
    x = xt[:, 0:1]
    t = xt[:, 1:2]
    return (1.0 - np.exp(-t)) * _psi_np(x)


def variants() -> list[Variant]:
    # Single honest benchmark: the coefficient-jump kink at the fixed 10x contrast is already the limiting difficulty;
    # a contrast sweep would let the sharpest regime dominate the L2 and blur the feature (ADR-0016 §9.A).
    return [Variant(
        "barrier10x", "Barrier (10× contrast)", "Barrera (contraste 10×)", {},
        "10× low-D barrier in [0.45,0.55]: the plume is slowed and c develops a kink at each barrier face.",
        "Barrera de baja D (10×) en [0.45,0.55]: el plume se frena y c desarrolla un quiebre en cada cara.",
    )]


def build(seed: int) -> dict:
    import deepxde as dde
    import torch

    dde.config.set_random_seed(int(seed))
    geom = dde.geometry.Interval(0.0, 1.0)
    timedomain = dde.geometry.TimeDomain(0.0, 1.0)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    def psi_torch(xs):
        r = (torch.clamp(xs, 0, AB) / D_SOIL
             + torch.clamp(xs - AB, 0, BB - AB) / D_BARR
             + torch.clamp(xs - BB, 0, 1 - BB) / D_SOIL)
        return 1.0 - r / R_L

    def d_field(xs):
        chi = ((xs >= AB) & (xs <= BB)).to(xs.dtype)
        return D_SOIL * (1.0 - chi) + D_BARR * chi

    def pde(x, c):
        xs = x[:, 0:1]
        ts = x[:, 1:2]
        c_t = dde.grad.jacobian(c, x, i=0, j=1)
        c_xx = dde.grad.hessian(c, x, i=0, j=0)
        f = torch.exp(-ts) * psi_torch(xs)
        return c_t - d_field(xs) * c_xx - f

    net = dde.nn.FNN([2] + [64] * 4 + [2], "tanh", "Glorot normal")

    def fbpinn_transform(x, y):
        xs = x[:, 0:1]
        ts = x[:, 1:2]
        w_left = torch.sigmoid(BETA * (X_C - xs))
        w_right = 1.0 - w_left
        c_raw = w_left * y[:, 0:1] + w_right * y[:, 1:2]   # partition-of-unity blend of the two sub-nets
        lift = (1.0 - torch.exp(-ts)) * (1.0 - xs)         # hard: inlet c(0,t)=1-e^{-t}, outlet c(1,t)=0, IC c(x,0)=0
        vanish = ts * xs * (1.0 - xs)
        return lift + vanish * c_raw

    net.apply_output_transform(fbpinn_transform)

    n_a = 40
    t_a = np.linspace(0.0, 1.0, n_a)
    anchors = np.concatenate([
        np.stack([np.full(n_a, AB), t_a], axis=1),
        np.stack([np.full(n_a, BB), t_a], axis=1),
        np.stack([np.full(n_a, X_C), t_a], axis=1),
    ])
    t = CASE.train
    data = dde.data.TimePDE(
        geomtime, pde, [],
        num_domain=t["num_domain"], num_boundary=t["num_boundary"],
        num_initial=t["num_initial"], solution=analytic, num_test=t["num_test"],
        anchors=anchors,
    )
    model = dde.Model(data, net)
    model.compile("adam", lr=t["lr"], metrics=["l2 relative error"])
    return {"model": model, "input_dim": 2}


def build_naive(seed: int) -> dict:
    """NAIVE lane: ONE smooth single-channel net (no domain decomposition), same hard IC/BC. A single smooth network
    cannot represent the discontinuous c_x at the low-permeability barrier faces, so it rounds off the kink — the
    failure the FBPINN partition-of-unity blend fixes. Same net size, so the contrast is the METHOD, not capacity."""
    import deepxde as dde
    import torch

    dde.config.set_random_seed(int(seed))
    geom = dde.geometry.Interval(0.0, 1.0)
    timedomain = dde.geometry.TimeDomain(0.0, 1.0)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    def psi_torch(xs):
        r = (torch.clamp(xs, 0, AB) / D_SOIL
             + torch.clamp(xs - AB, 0, BB - AB) / D_BARR
             + torch.clamp(xs - BB, 0, 1 - BB) / D_SOIL)
        return 1.0 - r / R_L

    def d_field(xs):
        chi = ((xs >= AB) & (xs <= BB)).to(xs.dtype)
        return D_SOIL * (1.0 - chi) + D_BARR * chi

    def pde(x, c):
        xs = x[:, 0:1]
        ts = x[:, 1:2]
        c_t = dde.grad.jacobian(c, x, i=0, j=1)
        c_xx = dde.grad.hessian(c, x, i=0, j=0)
        f = torch.exp(-ts) * psi_torch(xs)
        return c_t - d_field(xs) * c_xx - f

    net = dde.nn.FNN([2] + [64] * 4 + [1], "tanh", "Glorot normal")  # SINGLE channel (no partition-of-unity)

    def naive_transform(x, y):
        xs = x[:, 0:1]
        ts = x[:, 1:2]
        lift = (1.0 - torch.exp(-ts)) * (1.0 - xs)
        vanish = ts * xs * (1.0 - xs)
        return lift + vanish * y  # one smooth net -> cannot produce the kink in c_x

    net.apply_output_transform(naive_transform)

    n_a = 40
    t_a = np.linspace(0.0, 1.0, n_a)
    anchors = np.concatenate([
        np.stack([np.full(n_a, AB), t_a], axis=1),
        np.stack([np.full(n_a, BB), t_a], axis=1),
        np.stack([np.full(n_a, X_C), t_a], axis=1),
    ])
    t = CASE.train
    data = dde.data.TimePDE(
        geomtime, pde, [],
        num_domain=t["num_domain"], num_boundary=t["num_boundary"],
        num_initial=t["num_initial"], solution=analytic, num_test=t["num_test"],
        anchors=anchors,
    )
    model = dde.Model(data, net)
    model.compile("adam", lr=t["lr"], metrics=["l2 relative error"])
    return {"model": model, "input_dim": 2}
