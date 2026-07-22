# dyn-pendulum-hnn — Structure-preserving learning: does the model respect the energy?

**Method:** `hamiltonian-symplectic` · **Engine:** `hnn-torch` · **Category:** canonical-benchmark ·
**Label:** synthetic-illustrative · **Lane:** precompute

The companion to [`dyn-double-pendulum`](dyn-double-pendulum.md). That case asks how long a network can track a
chaotic trajectory. This one asks a sharper question, and it is the whole thesis of geometric,
structure-preserving learning (the subject of the RSS 2026 workshop "The Geometry of Motion"):

> **Two models that fit the dynamics equally well: does either respect the energy the system conserves?**

This is the family the catalogue was missing entirely. Before this case, the only mechanical system in
PINN-Lab was a plain residual PINN, so nothing conserved energy by construction, and no Hamiltonian /
Lagrangian / symplectic / Lie-group method was exercised anywhere.

## The two lanes

Identical width, training data, seed, number of steps, and the **same RK4 integrator** for the rollout. Only
the **output structure** differs:

| lane | maps | free to output |
|---|---|---|
| unstructured MLP | state -> d(state)/dt | any vector field at all |
| Hamiltonian network | state -> scalar H, then dz/dt = J grad H | only fields that conserve H |

Any field of the form dz/dt = J grad H conserves H exactly along its own flow, because J is antisymmetric
(dH/dt = grad H . J grad H = 0). So for the HNN, energy conservation is not something the optimiser has to get
right: it is a property of the parameterisation.

## The regime, and why it is the low-energy one

The pendulum is released from rest at **40 degrees** (not the 120 degrees of the chaotic sibling case), so the
motion is **bounded and quasi-periodic**. This is deliberate. At high energy the double pendulum is chaotic,
the trajectory leaves the training distribution within a second, and off that distribution even a structured
model's learned H is unconstrained and drifts. Conservation can only be demonstrated honestly where the model
is actually accurate, and the chaotic "tracks then loses it" story is already `dyn-double-pendulum`'s job.

## Results (measured, 8 s horizon)

| lane | derivative fit loss | energy drift over 8 s | trajectory L2 | theta err @1s |
|---|---|---|---|---|
| **Hamiltonian network** | 1.24e-05 | **0.07%** | 0.013 | 0.001 rad |
| unstructured MLP | 1.02e-04 | **7.44%** | 0.084 | 0.011 rad |

Both fit the derivative to the same order; the structured lane holds energy about **100x** tighter. The
difference is the invariant, not the fit. Both track the trajectory for the whole window (the motion is not
chaotic here), so this is a clean like-for-like comparison.

## Canonical coordinates are not optional

An HNN is Hamiltonian only in canonical coordinates (q, p). The momenta are **not** the angular velocities:
for unit masses and lengths, p1 = 2 w1 + w2 cos(d), p2 = w2 + w1 cos(d). Feeding an HNN (theta, omega) and
calling it Hamiltonian is a common, silent error. The conversion lives in `model/hnn.py`, and the analytic
Hamiltonian used to generate the training signal was verified before use: it matches E = T + V to
**1.07e-14** over 2000 random states, and RK45 conserves it to **8.9e-10** over 10 s.

## What this case does NOT claim

- **Not long-horizon prediction.** Conserving energy keeps the solution on the right energy surface; it does
  not defeat chaos. At the high-energy chaotic initial condition both lanes lose the trajectory within a
  second (the HNN still wins by ~27x, but both are bad in absolute terms).
- **Not exact conservation.** 0.07% is small but not zero: the HNN conserves its OWN learned H, not the true
  H, and it is rolled out with RK4, which is not itself symplectic. A leapfrog integrator would tighten it.
- **The exported ONNX is the learned energy surface** (state -> H), not the dynamics: the HNN's forward pass
  takes a symplectic gradient via autograd, which ONNX cannot represent, but the scalar H it differentiates
  is the physically meaningful object and exports cleanly (parity < 1e-5).

## References

- Greydanus, Dzamba, Yosinski. *Hamiltonian Neural Networks.* NeurIPS 2019, arXiv:1906.01563.
- Cranmer et al. *Lagrangian Neural Networks.* arXiv:2003.04630 (the coordinate-free cousin, not implemented here).
- RSS 2026 workshop, "The Geometry of Motion: Physics-Informed Structures for Learning and Control."

Working notes, including the two fixes tried (a uniform box that made it worse, then the low-energy regime
that fixed it) and the chaotic-regime numbers: `wip/beyond-sota/plan-2026-07-15.md` §C.
