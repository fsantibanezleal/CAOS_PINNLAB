"""Spike: can a plain PyTorch PINN solve the radial Theis drawdown, and how close does it get?

Confined aquifer, radial, S ds/dt = T (s_rr + s_r/r), well flux at r=rw, far-field s->0, s(r,0)=0.
Theis is the exact solution, so this is a FORWARD PINN we can grade to the closed form. Key moves that make
it tractable:
  - log-radius input x = ln(r): Theis drawdown is ~linear in ln(r) (Cooper-Jacob), so the network sees an
    easy coordinate; the Laplacian transforms as (s_rr + s_r/r) = e^{-2x} s_xx.
  - log-time input for the same reason; the cone grows over decades of time.
  - a hard IC/-far-field-friendly output scaling.
Time-boxed: if this converges to a few percent of Theis, the case is buildable.
"""
import sys
import time

import numpy as np
import torch
from scipy.special import exp1

# confined-aquifer parameters (SI), same as the verified anchor
T = 500.0 / 86400.0
S = 2e-4
Q = 2000.0 / 86400.0
rw, R = 0.3, 3000.0
t0, t1 = 60.0, 86400.0     # 1 min .. 1 day


def theis(r, t):
    return Q / (4.0 * np.pi * T) * exp1(r ** 2 * S / (4.0 * T * t))


# work in x = ln r, tau = ln t. drawdown scale from Theis at the well end
SC = float(theis(rw, t1))
xw, xR = np.log(rw), np.log(R)
tauA, tauB = np.log(t0), np.log(t1)


def net_make():
    return torch.nn.Sequential(
        torch.nn.Linear(2, 64), torch.nn.Tanh(),
        torch.nn.Linear(64, 64), torch.nn.Tanh(),
        torch.nn.Linear(64, 64), torch.nn.Tanh(),
        torch.nn.Linear(64, 1)).double()


def features(x, tau):
    return torch.stack([(x - xw) / (xR - xw) * 2 - 1, (tau - tauA) / (tauB - tauA) * 2 - 1], dim=1)


torch.manual_seed(0)
net = net_make()
opt = torch.optim.Adam(net.parameters(), lr=2e-3)


def s_of(x, tau):
    return SC * net(features(x, tau)).squeeze(-1)


def pde_res(x, tau):
    x = x.requires_grad_(True); tau = tau.requires_grad_(True)
    s = s_of(x, tau)
    s_x = torch.autograd.grad(s, x, torch.ones_like(s), create_graph=True)[0]
    s_xx = torch.autograd.grad(s_x, x, torch.ones_like(s_x), create_graph=True)[0]
    s_tau = torch.autograd.grad(s, tau, torch.ones_like(s), create_graph=True)[0]
    r2 = torch.exp(2 * x); t = torch.exp(tau)
    # S ds/dt = T (s_rr + s_r/r);  with x=ln r: s_r=e^{-x}s_x, s_rr+s_r/r = e^{-2x} s_xx ; ds/dt = s_tau/t
    return S * s_tau / t - T * torch.exp(-2 * x) * s_xx


t0f = time.perf_counter()
for it in range(6000):
    opt.zero_grad()
    x = torch.rand(2000, dtype=torch.float64) * (xR - xw) + xw
    tau = torch.rand(2000, dtype=torch.float64) * (tauB - tauA) + tauA
    Lpde = (pde_res(x, tau) ** 2).mean() / (Q / (4 * np.pi * T) * S / t1) ** 2   # normalise
    # well flux: 2 pi rw T ds/dr = Q  -> ds/dr|_rw = Q/(2 pi rw T); ds/dr = e^{-x} s_x
    xwb = torch.full((256,), xw, dtype=torch.float64)
    taub = torch.rand(256, dtype=torch.float64) * (tauB - tauA) + tauA
    xwb = xwb.requires_grad_(True)
    swb = s_of(xwb, taub)
    s_xw = torch.autograd.grad(swb, xwb, torch.ones_like(swb), create_graph=True)[0]
    dsdr = torch.exp(-xwb) * s_xw
    Lwell = ((dsdr + Q / (2 * np.pi * rw * T)) ** 2).mean() * (rw ** 2)    # target: T*2pi*rw*dsdr=-Q (inflow)
    # far field s(R)=0 and IC s(.,t0)~Theis(t0) as a mild anchor (early time cone is tiny)
    xf = torch.rand(256, dtype=torch.float64) * (xR - xw) + xw
    Lfar = (s_of(torch.full((256,), xR, dtype=torch.float64), torch.rand(256, dtype=torch.float64)*(tauB-tauA)+tauA) ** 2).mean() / SC ** 2
    xi = torch.rand(400, dtype=torch.float64) * (xR - xw) + xw
    s_ic = s_of(xi, torch.full((400,), tauA, dtype=torch.float64))
    s_ic_true = torch.as_tensor(theis(np.exp(xi.numpy()), t0), dtype=torch.float64)
    Lic = ((s_ic - s_ic_true) ** 2).mean() / SC ** 2
    loss = Lpde + Lwell + Lfar + Lic
    loss.backward(); opt.step()

# grade against Theis on a grid away from the well
net.eval()
with torch.no_grad():
    rr = np.geomspace(rw * 3, R * 0.5, 40)
    tt = np.geomspace(t0 * 2, t1, 40)
    RR, TT = np.meshgrid(rr, tt)
    xg = torch.as_tensor(np.log(RR).ravel()); tg = torch.as_tensor(np.log(TT).ravel())
    sp = s_of(xg, tg).numpy().reshape(RR.shape)
    sa = theis(RR, TT)
    rel = np.linalg.norm(sp - sa) / np.linalg.norm(sa)
print(f"trained in {time.perf_counter()-t0f:.0f}s; PINN vs Theis relative-L2 = {rel:.3%}")
for robs in (30.0, 100.0, 300.0):
    with torch.no_grad():
        sp1 = float(s_of(torch.tensor([np.log(robs)]), torch.tensor([np.log(t1)])).item())
    print(f"  r={robs:>4.0f} m, t=1d: PINN {sp1:.3f} m vs Theis {theis(robs, t1):.3f} m")
print("THEIS PINN", "OK (buildable)" if rel < 0.05 else "NEEDS WORK")
