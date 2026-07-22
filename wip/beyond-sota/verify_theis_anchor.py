"""Verify the Theis analytic solution against a numerical radial finite-difference solve.

Theis (1935), confined aquifer, fully-penetrating well of constant discharge Q into an infinite homogeneous
aquifer:
    s(r,t) = Q/(4 pi T) * W(u),   u = r^2 S / (4 T t),   W(u) = E1(u)  (exponential integral)

If the analytic solution is right and the well BC is handled correctly, a numerical radial solve of
    S ds/dt = T (1/r) d/dr( r ds/dr )   with a constant-flux well at the inner radius
must agree with it away from the singular origin. This checks BOTH the analytic form and the BC before any
PINN case is built on it.
"""
import numpy as np
from scipy.special import exp1  # E1, the exponential integral = Theis well function W(u)

# realistic confined-aquifer parameters (SI): a moderately transmissive sand aquifer
T = 500.0 / 86400.0   # transmissivity 500 m^2/day -> m^2/s
S = 2e-4              # storativity (confined: small, 1e-5..1e-3)
Q = 2000.0 / 86400.0  # pumping 2000 m^3/day -> m^3/s


def theis(r, t):
    u = r ** 2 * S / (4.0 * T * t)
    return Q / (4.0 * np.pi * T) * exp1(u)


# ---- numerical radial solve on a log-spaced grid, implicit Euler ----
rw, rmax = 0.1, 5000.0
nr = 400
r = np.geomspace(rw, rmax, nr)
t_end = 1.0 * 86400.0     # 1 day
nt = 400
dt = t_end / nt

s = np.zeros(nr)
# control-volume radial diffusion with a constant-flux inner BC: Q into the well = 2 pi rw T ds/dr|_rw
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# face radii and volumes for a radial control-volume scheme
rf = np.sqrt(r[:-1] * r[1:])                       # geometric-mean face radii
vol = np.zeros(nr)
vol[1:-1] = 0.5 * (rf[1:] ** 2 - rf[:-1] ** 2)     # ~ integral r dr over the cell (times 2pi dropped consistently)
vol[0] = 0.5 * (rf[0] ** 2 - rw ** 2)
vol[-1] = 0.5 * (rmax ** 2 - rf[-1] ** 2)

for _ in range(nt):
    A = sp.lil_matrix((nr, nr))
    b = S * vol * s / dt
    for i in range(nr):
        A[i, i] += S * vol[i] / dt
        if i > 0:
            w = T * rf[i - 1] / (r[i] - r[i - 1])
            A[i, i] += w; A[i, i - 1] -= w
        if i < nr - 1:
            e = T * rf[i] / (r[i + 1] - r[i])
            A[i, i] += e; A[i, i + 1] -= e
    b[0] += Q / (2.0 * np.pi)          # constant-flux well (the 2 pi is the angular measure)
    A[-1, :] = 0; A[-1, -1] = 1.0; b[-1] = 0.0     # far-field s = 0
    s = spla.spsolve(A.tocsr(), b)

# ---- compare at observation radii away from the singular origin ----
print(f"T={T*86400:.0f} m2/day  S={S:.0e}  Q={Q*86400:.0f} m3/day  t=1 day\n")
print(f"{'r (m)':>7} {'u':>10} {'Theis s (m)':>12} {'numeric s (m)':>14} {'rel err':>9}")
maxerr = 0.0
for robs in (10.0, 30.0, 100.0, 300.0, 1000.0):
    sa = theis(robs, t_end)
    sn = float(np.interp(robs, r, s))
    e = abs(sa - sn) / max(abs(sa), 1e-12)
    maxerr = max(maxerr, e)
    u = robs ** 2 * S / (4 * T * t_end)
    print(f"{robs:>7.0f} {u:>10.3e} {sa:>12.4f} {sn:>14.4f} {e:>8.2%}")
print(f"\nmax relative error away from the well: {maxerr:.2%}")
print("THEIS ANCHOR", "OK" if maxerr < 0.05 else "CHECK (grid/BC)")
