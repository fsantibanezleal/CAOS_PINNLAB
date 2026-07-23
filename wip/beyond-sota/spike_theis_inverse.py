"""Spike: recover T and S from a pumping test (the real groundwater inverse), PINN-style.

The forward Theis PINN collapsed (82% error): the steep near-well transient plus the flux BC is a stiff PINN
problem, exactly the failure mode the critique names. The INVERSE is better conditioned: OBSERVED drawdown at
a few wells pins the field, so the network cannot collapse, and the PDE residual with unknown (T, S) recovers
the parameters. This is also the real task, since T and S cannot be measured directly.

Setup: synthetic drawdown at 3 observation radii over time from the exact Theis solution + a few cm of noise,
with KNOWN true (T, S). The PINN maps (ln r, ln t) -> s, fits the observations, and satisfies the radial PDE
with T, S as trainable log-variables. Grade: recovered (T, S) vs truth.
"""
import time

import numpy as np
import torch
from scipy.special import exp1

T_true = 500.0 / 86400.0
S_true = 2e-4
Q = 2000.0 / 86400.0
t0, t1 = 60.0, 86400.0
OBS_R = [30.0, 100.0, 300.0]      # observation wells
NOISE = 0.02                       # 2 cm gauge noise


def theis(r, t):
    return Q / (4.0 * np.pi * T_true) * exp1(r ** 2 * S_true / (4.0 * T_true * t))


rng = np.random.default_rng(0)
# observations: dense in time at each well
t_obs = np.geomspace(t0, t1, 40)
obs = []
for r in OBS_R:
    s = theis(r, t_obs) + rng.normal(0, NOISE, len(t_obs))
    for tt, ss in zip(t_obs, s):
        obs.append((np.log(r), np.log(tt), ss))
obs = np.array(obs)
SC = float(theis(min(OBS_R), t1))

xo = torch.as_tensor(obs[:, 0]); tauo = torch.as_tensor(obs[:, 1]); so = torch.as_tensor(obs[:, 2])
xw, xR = np.log(10.0), np.log(1000.0)
tauA, tauB = np.log(t0), np.log(t1)

torch.manual_seed(0)
net = torch.nn.Sequential(
    torch.nn.Linear(2, 64), torch.nn.Tanh(), torch.nn.Linear(64, 64), torch.nn.Tanh(),
    torch.nn.Linear(64, 64), torch.nn.Tanh(), torch.nn.Linear(64, 1)).double()
# unknowns as log-variables, initialised an order of magnitude off
logT = torch.tensor(np.log(T_true * 5.0), dtype=torch.float64, requires_grad=True)
logS = torch.tensor(np.log(S_true * 0.2), dtype=torch.float64, requires_grad=True)
opt = torch.optim.Adam(list(net.parameters()) + [logT, logS], lr=3e-3)


def feat(x, tau):
    return torch.stack([(x - xw) / (xR - xw) * 2 - 1, (tau - tauA) / (tauB - tauA) * 2 - 1], dim=1)


def s_of(x, tau):
    return SC * net(feat(x, tau)).squeeze(-1)


t0f = time.perf_counter()
hist = []
for it in range(8000):
    opt.zero_grad()
    Ldata = ((s_of(xo, tauo) - so) ** 2).mean() / SC ** 2
    x = torch.rand(1500, dtype=torch.float64) * (xR - xw) + xw
    tau = torch.rand(1500, dtype=torch.float64) * (tauB - tauA) + tauA
    x = x.requires_grad_(True); tau = tau.requires_grad_(True)
    s = s_of(x, tau)
    s_x = torch.autograd.grad(s, x, torch.ones_like(s), create_graph=True)[0]
    s_xx = torch.autograd.grad(s_x, x, torch.ones_like(s_x), create_graph=True)[0]
    s_tau = torch.autograd.grad(s, tau, torch.ones_like(s), create_graph=True)[0]
    Tt, Ss = torch.exp(logT), torch.exp(logS)
    res = Ss * s_tau / torch.exp(tau) - Tt * torch.exp(-2 * x) * s_xx
    Lpde = (res ** 2).mean() / (Q / (4 * np.pi * T_true) * S_true / t1) ** 2
    loss = Ldata + 0.1 * Lpde
    loss.backward(); opt.step()
    if it % 1000 == 0:
        hist.append((it, float(torch.exp(logT) * 86400), float(torch.exp(logS))))

Trec = float(torch.exp(logT).item() * 86400)
Srec = float(torch.exp(logS).item())
print(f"trained in {time.perf_counter()-t0f:.0f}s\n")
print(f"{'':>12} {'true':>12} {'recovered':>12} {'rel err':>9}")
print(f"{'T (m2/day)':>12} {T_true*86400:>12.1f} {Trec:>12.1f} {abs(Trec-T_true*86400)/(T_true*86400):>8.1%}")
print(f"{'S':>12} {S_true:>12.2e} {Srec:>12.2e} {abs(Srec-S_true)/S_true:>8.1%}")
ok = abs(Trec-T_true*86400)/(T_true*86400) < 0.15 and abs(Srec-S_true)/S_true < 0.30
print("\nINVERSE", "OK (buildable)" if ok else "NEEDS WORK")
