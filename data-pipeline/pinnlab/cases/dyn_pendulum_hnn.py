"""Group A · canonical-benchmark — STRUCTURE-PRESERVING learning: does the model respect the energy?

The companion to `dyn-double-pendulum`. That case asks how long a PINN can track a chaotic trajectory. This
one asks a different and sharper question:

    two models that fit the dynamics EQUALLY WELL: does either of them respect the conserved energy?

This is the family the catalogue was missing entirely. Before this case, the only mechanical system in
PINN-Lab was a plain residual PINN, so nothing in it conserved energy by construction, and none of the
geometric / structure-preserving literature (Hamiltonian and Lagrangian networks, symplectic integrators, Lie
group and equivariant models) was exercised anywhere.

THE TWO LANES. Identical width, identical training data, identical seed, identical number of steps, and the
SAME RK4 integrator for the rollout. Only the OUTPUT STRUCTURE differs:

    mlp  (unstructured)  state -> d(state)/dt        free to output ANY vector field
    hnn  (structured)    state -> scalar H;  dz/dt = J grad H     symplectic by construction

Any field of the form J grad H conserves H along its own flow, because J is antisymmetric. So for the HNN,
energy conservation is not something the optimiser has to get right: it is a property of the parameterisation.

MEASURED (see `wip/beyond-sota/plan-2026-07-15.md`), in a LOW-ENERGY bounded regime (released from rest at
40 deg, quasi-periodic, not chaotic): both lanes reach the same order of derivative fit (1.0e-4 vs 1.2e-5) and
the structured lane holds energy ~100x tighter over 8 s (7.44% vs 0.07% drift). That gap is not accuracy, it
is structure. The high-energy CHAOTIC regime is `dyn-double-pendulum`'s job; conservation is demonstrated here
where it can be shown honestly.

CANONICAL COORDINATES ARE NOT OPTIONAL. An HNN is Hamiltonian only in (q, p). The momenta are NOT the angular
velocities: p1 = 2w1 + w2 cos(d), p2 = w2 + w1 cos(d) for unit masses and lengths. Feeding an HNN
(theta, omega) and calling it Hamiltonian is a common and silent error; the conversion lives in `model/hnn.py`.

HONEST LIMITS. This is the LOW-ENERGY bounded regime; at high energy the double pendulum is chaotic and the
trajectory leaves the training distribution within a second, where even a structured model's learned H is
unconstrained and drifts (the HNN still beats the baseline there by ~27x, but both are bad in absolute
terms). And 0.07% is small but not zero, because the HNN conserves its OWN learned H (not the true H) and is
rolled out with RK4, which is not itself a symplectic integrator.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import CaseSpec, Variant
from .dyn_double_pendulum import G, _accel_np

TH1_0 = np.deg2rad(40.0)   # shown IC: released from rest at 40 deg -> bounded, quasi-periodic (NOT chaotic)
TH2_0 = np.deg2rad(40.0)
TRAIN_ANGLE_DEG = (15.0, 55.0)   # training trajectories drawn from the same low-energy regime

T_MAX = 8.0   # long horizon: at low energy the motion is periodic, so conservation shows over many swings
N_EVAL = 601
WIDTH = 200
STEPS = 8000
BATCH = 512
N_TRAIN_TRAJ = 12
LEAVE_TOL = 0.30
_REPO = Path(__file__).resolve().parents[3]
_STATE: dict = {}

CASE = CaseSpec(
    id="dyn-pendulum-hnn",
    system_type="ode-dynamical",
    category="canonical-benchmark",
    title="Structure-preserving learning: a Hamiltonian network vs an unstructured one",
    governing_equations=(
        r"\dot{q}=\partial H/\partial p,\ \ \dot{p}=-\partial H/\partial q\ \ (\dot{z}=J\,\nabla H);\ "
        r"H(q,p)\ \text{conservada exactamente porque } J \text{ es antisimétrica}"
    ),
    method="hamiltonian-symplectic",
    engine="hnn-torch",
    real_or_synthetic="synthetic-illustrative",
    inputs=("t",),
    outputs=("th1", "th2"),
    domain={"t": (0.0, T_MAX)},
    grid={"t": N_EVAL},
    field_axes=("t",),
    param_specs=(),
    expected_band=(
        "both lanes fit the derivative equally well; the structured lane drifts far less in energy, while BOTH "
        "eventually lose the chaotic trajectory"
    ),
    validation_anchor="integrator-ref",
    train={"lr": 1e-3, "adam": 0},
    notes="Custom-engine ode-dynamical case: trains BOTH lanes in build() and rolls each out with the SAME RK4, "
          "so the integrator can never explain the difference. Variants ARE the two models.",
)


def variants() -> list[Variant]:
    """The variants are the two MODELS, so the structure question is the case's own axis."""
    return [
        Variant("hnn", "Hamiltonian network (structured)", "Red hamiltoniana (estructurada)",
                {"structured": 1.0},
                "Outputs a single scalar H and takes the symplectic gradient, so the energy is conserved by construction.",
                "Entrega un único escalar H y toma el gradiente simpléctico, así la energía se conserva por construcción."),
        Variant("mlp", "Unstructured network (baseline)", "Red sin estructura (referencia)",
                {"structured": 0.0},
                "Outputs the four time-derivatives directly, free to produce any vector field at all.",
                "Entrega directamente las cuatro derivadas temporales, libre de producir cualquier campo vectorial."),
    ]


def _p_from_omega_np(th1, th2, om1, om2):
    c = np.cos(th1 - th2)
    return 2.0 * om1 + om2 * c, om2 + om1 * c


def _energy_np(th1, th2, p1, p2):
    d = th1 - th2
    s, c = np.sin(d), np.cos(d)
    return (p1 ** 2 + 2.0 * p2 ** 2 - 2.0 * p1 * p2 * c) / (2.0 * (1.0 + s ** 2)) \
        - 2.0 * G * np.cos(th1) - G * np.cos(th2)


def _rk45_ref(th1_0, th2_0, t_eval):
    from scipy.integrate import solve_ivp
    sol = solve_ivp(lambda _t, y: [y[2], y[3], *_accel_np(y[0], y[1], y[2], y[3])],
                    (float(t_eval[0]), float(t_eval[-1])), [th1_0, th2_0, 0.0, 0.0],
                    t_eval=t_eval, method="RK45", rtol=1e-11, atol=1e-11, max_step=0.01)
    return sol.y


def extra_metrics(sf) -> dict:
    """Inject the RK45 anchor into the trace and report THIS lane's energy drift and leave-time."""
    t = np.asarray(sf.axes["t"], dtype=np.float64)
    shape = sf.fields["th1"].shape
    ref = _STATE["ref"]
    sf.fields["th1_ref"] = ref[0].reshape(shape)
    sf.fields["th2_ref"] = ref[1].reshape(shape)

    i = int(_STATE.get("cur", 0))
    sf.fields["energy"] = _STATE["energy"][i].reshape(shape)           # this model's total energy over time
    sf.fields["energy_ref"] = np.full(shape, _STATE["e0"])             # the constant true energy
    th1_p = np.asarray(sf.fields["th1"], dtype=np.float64).reshape(-1)
    th2_p = np.asarray(sf.fields["th2"], dtype=np.float64).reshape(-1)

    def wrap(a):
        return (a + np.pi) % (2 * np.pi) - np.pi

    err = np.sqrt(wrap(th1_p - ref[0]) ** 2 + wrap(th2_p - ref[1]) ** 2)
    over = np.where(err > LEAVE_TOL)[0]
    # trajectory relative-L2 over the window; large for BOTH lanes because the system is chaotic, so it is
    # reported honestly rather than used as the headline (the headline is the energy drift)
    num = np.sqrt(np.sum(wrap(th1_p - ref[0]) ** 2 + wrap(th2_p - ref[1]) ** 2))
    den = np.sqrt(np.sum(ref[0] ** 2 + ref[1] ** 2)) or 1.0
    return {
        "l2_relative": round(float(num / den), 6),
        "energy_drift_rel": round(float(_STATE["drift"][i]), 6),
        "fit_loss": float(f"{_STATE['loss'][i]:.3e}"),
        "leave_time_s": round(float(t[over[0]]) if len(over) else float(t[-1]), 3),
        "theta_err_1s": round(float(abs(wrap(th1_p[100] - ref[0][100]))), 4),
        "train_s": round(float(_STATE["train_s"][i]), 1),
        "ensemble_K": 0,
    }


class _Baked:
    def __init__(self, traj):
        self._t = np.asarray(traj, dtype=np.float64)   # [n_var, N, 2]
        self._i = 0

    def predict(self, T):
        k = min(self._i, self._t.shape[0] - 1)
        _STATE["cur"] = k
        self._i += 1
        return self._t[k]


def build(seed: int, quick: bool = False) -> dict:
    import time

    import torch

    from ..model.hnn import (HNN, StateMLP, hamiltonian_analytic, p_from_omega, relative_energy_drift,
                             rk4_rollout, symplectic_grad)

    steps = 60 if quick else STEPS
    n_traj = 3 if quick else N_TRAIN_TRAJ
    t_eval = np.linspace(0.0, T_MAX, N_EVAL)
    dt = float(t_eval[1] - t_eval[0])
    rng = np.random.default_rng(int(seed))

    # ---- training states: several RK45 trajectories, derivative from Hamilton's equations ----
    states = []
    for _ in range(n_traj):
        y = _rk45_ref(np.deg2rad(rng.uniform(*TRAIN_ANGLE_DEG)), np.deg2rad(rng.uniform(*TRAIN_ANGLE_DEG)), t_eval)
        p1, p2 = _p_from_omega_np(y[0], y[1], y[2], y[3])
        states.append(np.stack([y[0], y[1], p1, p2], axis=1))
    Z = torch.as_tensor(np.concatenate(states), dtype=torch.float64)
    dZ = symplectic_grad(lambda x: hamiltonian_analytic(x, G), Z.clone()).detach()
    mu, sd = Z.mean(0, keepdim=True), Z.std(0, keepdim=True)
    # SCALAR output scale: a per-component scale would break the coupling of J grad H
    dscale = float(dZ.abs().mean())
    dZn = dZ / dscale

    def train(structured: bool):
        torch.manual_seed(int(seed) + 3)
        if structured:
            net = HNN(mu, sd, dscale, width=WIDTH)
            f = net
        else:  # noqa: RET505
            body = StateMLP(width=WIDTH, out=4)
            net = body
            f = lambda z: body((z - mu) / sd) * dscale   # noqa: E731  (same normalisation as the HNN)
        opt = torch.optim.Adam(net.parameters(), lr=CASE.train["lr"])
        t0 = time.perf_counter()
        loss = torch.tensor(0.0)
        for _ in range(steps):
            idx = torch.randperm(len(Z))[:BATCH]
            opt.zero_grad()
            loss = ((f(Z[idx].clone()) / dscale - dZn[idx]) ** 2).mean()
            loss.backward()
            opt.step()
        return f, net, float(loss), time.perf_counter() - t0

    # ---- the held-out initial condition, and the reference ----
    ref = _rk45_ref(TH1_0, TH2_0, t_eval)
    p10, p20 = _p_from_omega_np(TH1_0, TH2_0, 0.0, 0.0)
    z0 = torch.tensor([[TH1_0, TH2_0, p10, p20]], dtype=torch.float64)

    traj, drift, losses, times, energy = [], [], [], [], []
    hnn_net = None
    for structured in (True, False):                    # order MUST match variants(): hnn, then mlp
        f, net, loss, ts = train(structured)
        if structured:
            hnn_net = net
        roll = rk4_rollout(f, z0.clone(), dt, N_EVAL - 1)
        traj.append(roll[:, :2].detach().numpy())
        energy.append(hamiltonian_analytic(roll, G).detach().numpy())   # E(t) along THIS model's rollout
        drift.append(relative_energy_drift(roll, G))
        losses.append(loss)
        times.append(ts)
    e0 = float(hamiltonian_analytic(z0.clone(), G).item())              # the constant true energy

    _STATE.update({"ref": ref, "drift": drift, "loss": losses, "train_s": times,
                   "t": t_eval, "traj": [x.copy() for x in traj], "energy": energy, "e0": e0})

    # ---- export the LEARNED ENERGY SURFACE, state (4) -> H, the HNN's core object ----
    # HNN.forward takes a symplectic gradient via autograd.grad, which ONNX cannot represent; the scalar
    # energy network can be exported, and it is the physically meaningful artifact: the Hamiltonian the model
    # actually learned, which is what makes its conservation property true.
    import onnxruntime as ort

    class _EnergyNet(torch.nn.Module):
        def __init__(self, hnn):
            super().__init__()
            self.hnn = hnn

        def forward(self, z):
            return self.hnn.energy(z).unsqueeze(-1)

    onnx_path = _REPO / "models" / f"{CASE.id}.onnx"
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    wrapped = _EnergyNet(hnn_net).eval().to(torch.float32)
    probe = Z[:8].to(torch.float32)
    torch.onnx.export(
        wrapped, (probe,), str(onnx_path), input_names=["state"], output_names=["H"],
        dynamic_axes={"state": {0: "n"}, "H": {0: "n"}},
        opset_version=18, dynamo=True, verbose=False, external_data=False,
    )
    from ..io.formats import strip_onnx_metadata
    strip_onnx_metadata(onnx_path)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        pt = wrapped(probe).numpy()
    ox = np.asarray(sess.run(["H"], {"state": probe.numpy()})[0])
    parity = float(np.max(np.abs(pt - ox)))
    t0 = time.perf_counter()
    for _ in range(5):
        sess.run(["H"], {"state": probe[:1].numpy()})
    infer_ms = (time.perf_counter() - t0) * 1000.0 / 5

    return {
        "model": _Baked(np.stack(traj, axis=0)),
        "input_dim": 1,
        "prebuilt": True,
        "onnx_path": str(onnx_path),
        "onnx_bytes": onnx_path.stat().st_size,
        "parity_max_abs": parity,
        "infer_ms": infer_ms,
        "opset": 18,
        "web_drivable": False,
    }
