# Case design — `ctrl-zero-source` (control)

## Decision: **PARAMETRIC** (6 variants), amplitude knob `a ∈ [0, 1]` as a network input.

The case is the archetype's mandatory negative control: `-∇²u = 0` on `(0,1)²`, `u|∂Ω = 0 ⇒ u ≡ 0`. As shipped it is
visually dead (a flat-zero heatmap), which is exactly the "dead viz" failure the playbook warns about. The honest,
rigorous upgrade for a **control** case is a **manufactured-solution (MMS) verification sweep**: pick an exact `u*`,
derive the source `f = -∇²u*`, impose it, and the relative-L2 vs `u*` is the *true* solver error. We make the
manufactured **amplitude** `a` a network input so one trained net + one ONNX sweeps the whole family in Live.

Crucially, the family is built to **contain the original degenerate control** as its `a=0` member (`f≡0 ⇒ u≡0`), so
the case keeps its identity and purpose (the engine recovers a known answer, including the exact zero) and gains a
genuine workbench. This is the conservative choice that satisfies the parametric rule honestly: the knob is a real
structural amplitude, surfaced in `inputs` + `param_specs` and NOT in `field_axes`, and the anchor is provably exact
for every `a` by substitution.

### Why not the alternatives
- **Single benchmark:** rejected — a flat-zero field is a dead workbench, and a rock-solid closed-form family *does*
  exist here (MMS). Going single would understate what the control can honestly demonstrate.
- **Time-scrubber:** N/A — the case is elliptic `(x,y)`, no time axis.
- **Discrete:** the amplitude is genuinely continuous (a slider), so a continuous parametric input is the natural fit;
  the 6 variants are presets on that continuum (including the exact-zero endpoint).
- **Shape-varying knob (mode-mixing via `a` and `a²`):** considered and rejected — a single scalar driving both
  amplitude and shape is pedagogically muddy for a *control*. A clean amplitude knob over a fixed, non-trivial
  two-mode shape is the textbook MMS and keeps the `a=0` degenerate endpoint exact.

## Analytic anchor (closed form) + substitution proof

**Manufactured solution** (fixed two-mode shape `g`, scaled by amplitude `a`):

    u*(x,y;a) = a · g(x,y),   g(x,y) = sin(πx)·sin(πy) + ½·sin(2πx)·sin(2πy)

**Zero Dirichlet BC for all `a`:** every mode `sin(kπx)sin(mπy)` (k,m ∈ {1,2}) vanishes at x∈{0,1} and y∈{0,1}, so
`u*|∂Ω = 0` for every `a`. ✓ (preserves the control's hard-zero boundary).

**Source by substitution.** Using `∇²[sin(kπx)sin(mπy)] = -(k²+m²)π²·sin(kπx)sin(mπy)`:

    ∇²g = -(1²+1²)π²·sin(πx)sin(πy) − ½·(2²+2²)π²·sin(2πx)sin(2πy)
        = -2π²·sin(πx)sin(πy) − 4π²·sin(2πx)sin(2πy)

so the source that makes `u*` the exact solution of `-∇²u = f` is

    f(x,y;a) = -∇²u* = a·( 2π²·sin(πx)sin(πy) + 4π²·sin(2πx)sin(2πy) ).

**Proof it solves the PDE for ALL a:** by construction `-∇²(a·g) = a·(-∇²g) = f(·;a)` identically, and `u*=a·g=0` on
∂Ω. Hence `u*` is the unique solution of `-∇²u=f(·;a)`, `u|∂Ω=0`, for every `a`. At `a=0`: `f≡0`, `u*≡0` — the exact
original degenerate control, recovered as the family's endpoint. (Verified symbolically with sympy: `expand(-∇²u*)`
matches `f` exactly, and `u*|∂Ω=0` at sampled boundary points.) The relative-L2 vs `u*` is therefore the true error;
at `a=0` `l2_relative` returns `||pred||` (per `model/analytic.py`), exactly the original control's metric.

## Exact case `.py` edits (orchestrator applies)

Preserve the existing hard-constraint method/engine (DeepXDE FNN + hard-zero output transform). Replace the metadata,
add `analytic`/`variants`, and inject the parametric source `f` into the PDE. Full target file:

```python
"""Control · manufactured-solution (MMS) verification of the Poisson operator, PARAMETRIC in the source amplitude a,
HARD-CONSTRAINT PINN (zero Dirichlet boundary baked exactly).

Governing equation:
    -nabla^2 u = f(x,y;a)  on (0,1)^2,   u|_{boundary} = 0.
Manufactured exact solution (validation anchor, closed form for ANY a) — the method of manufactured solutions:
    u*(x,y;a) = a * g(x,y),   g(x,y) = sin(pi x) sin(pi y) + 1/2 sin(2 pi x) sin(2 pi y),
which vanishes on the whole boundary for every a. Substituting into -nabla^2 gives the imposed source
    f(x,y;a) = a * ( 2 pi^2 sin(pi x) sin(pi y) + 4 pi^2 sin(2 pi x) sin(2 pi y) ).
At a=0 this is the archetype's mandatory degenerate negative control (f == 0 => u == 0): the engine must run without
crashing and return a flat-zero field, with relative-L2 = ||pred|| ~ 0. The amplitude a is a NETWORK INPUT, so ONE
trained net covers the whole family and the web `Live` tab sweeps a continuously via the shared ONNX — watch the field
fade to flat zero as a -> 0 (the degenerate control as the limit of the family).

Method — HARD CONSTRAINTS: the zero Dirichlet boundary is satisfied exactly by the output transform
x(1-x)y(1-y) * N, so there is no boundary loss term; the relative-L2 vs the manufactured u* is the true error.
"""
from __future__ import annotations

import numpy as np

from .base import CaseSpec, ParamSpec, Variant

A_MIN, A_MAX = 0.0, 1.0
PI = np.pi


def _g(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fixed two-mode manufactured shape (vanishes on the boundary)."""
    return np.sin(PI * x) * np.sin(PI * y) + 0.5 * np.sin(2.0 * PI * x) * np.sin(2.0 * PI * y)


CASE = CaseSpec(
    id="ctrl-zero-source",
    category="control",
    title="Manufactured-solution control — parametric source amplitude (Poisson, hard-zero boundary)",
    governing_equations=(
        r"-\nabla^2 u = f(x,y;a)\ \text{on}\ (0,1)^2,\ u|_{\partial\Omega}=0,\ "
        r"u^*=a\big(\sin\pi x\sin\pi y+\tfrac12\sin 2\pi x\sin 2\pi y\big)"
    ),
    method="hard-constraints",
    engine="deepxde",
    real_or_synthetic="synthetic",
    inputs=("x", "y", "a"),
    outputs=("u",),
    domain={"x": (0.0, 1.0), "y": (0.0, 1.0), "a": (A_MIN, A_MAX)},
    grid={"x": 41, "y": 41},
    field_axes=("x", "y"),
    param_specs=(ParamSpec("a", "Source amplitude a", "Amplitud de fuente a", 1.0, A_MIN, A_MAX, 0.05),),
    expected_band="a=0 recovers the degenerate control (flat zero); a=1 a two-mode field; relative-L2 vs manufactured u* < 1e-2",
    validation_anchor="analytic",
    train={
        "layers": [3, 48, 48, 48, 48, 1],
        "activation": "tanh",
        "lr": 1e-3,
        "adam": 12000,
        "lbfgs": True,
        "num_domain": 4000,
        "num_test": 4000,
    },
    notes="MMS verification family containing the archetype's degenerate zero control at a=0; amplitude a is a network input; hard-zero boundary via output transform; relative-L2 vs manufactured u* is the true error.",
)


def analytic(xya: np.ndarray) -> np.ndarray:
    """u*(x,y;a) = a * g(x,y), on [N,3] (x,y,a) -> [N,1]."""
    xya = np.asarray(xya, dtype=np.float64)
    x, y, a = xya[:, 0:1], xya[:, 1:2], xya[:, 2:3]
    return a * _g(x, y)


def variants() -> list[Variant]:
    presets = [
        ("a00", 0.0, "Degenerate control (a=0) — zero source, identically-zero field.", "Control degenerado (a=0) — fuente cero, campo idénticamente cero."),
        ("a02", 0.2, "a=0.2 — the manufactured field switching on, faint.", "a=0.2 — el campo manufacturado encendiéndose, tenue."),
        ("a04", 0.4, "a=0.4 — two-mode structure clearly visible.", "a=0.4 — estructura de dos modos claramente visible."),
        ("a06", 0.6, "a=0.6 — stronger lobes.", "a=0.6 — lóbulos más marcados."),
        ("a08", 0.8, "a=0.8 — near full amplitude.", "a=0.8 — casi amplitud plena."),
        ("a10", 1.0, "Full amplitude (a=1) — dominant fundamental lobe with a finer second-mode ripple.", "Amplitud plena (a=1) — lóbulo fundamental dominante con ondulación más fina del segundo modo."),
    ]
    return [Variant(vid, f"a={av:g}", f"a={av:g}", {"a": av}, en, es) for vid, av, en, es in presets]


def build(seed: int) -> dict:
    import deepxde as dde
    import torch

    dde.config.set_random_seed(int(seed))
    geom = dde.geometry.Hypercube([0.0, 0.0, A_MIN], [1.0, 1.0, A_MAX])

    def pde(x, u):
        u_xx = dde.grad.hessian(u, x, i=0, j=0)
        u_yy = dde.grad.hessian(u, x, i=1, j=1)
        xs, ys, a = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        f = a * (
            2.0 * np.pi ** 2 * torch.sin(np.pi * xs) * torch.sin(np.pi * ys)
            + 4.0 * np.pi ** 2 * torch.sin(2.0 * np.pi * xs) * torch.sin(2.0 * np.pi * ys)
        )
        return -u_xx - u_yy - f  # -lap u = f

    t = CASE.train
    net = dde.nn.FNN(t["layers"], t["activation"], "Glorot uniform")
    net.apply_output_transform(
        lambda x, u: x[:, 0:1] * (1.0 - x[:, 0:1]) * x[:, 1:2] * (1.0 - x[:, 1:2]) * u
    )
    data = dde.data.PDE(
        geom, pde, [],
        num_domain=t["num_domain"], num_boundary=0, solution=analytic, num_test=t["num_test"],
    )
    model = dde.Model(data, net)
    model.compile("adam", lr=t["lr"], metrics=["l2 relative error"])
    return {"model": model, "input_dim": 3}
```

Notes for the orchestrator:
- `field_axes=("x","y")`, `a` is the only parameter axis → `param_specs` has exactly `a` (matches `base.__post_init__`
  invariant: `set(inputs)-set(axes) == {a} == spec_keys`). `grid` is over field axes only. ✓
- The hard-zero output transform is **unchanged** from the original case — the method is preserved; only the PDE source
  and the input dimension (2→3, adding `a`) change.
- `solution=analytic` enables DeepXDE's `l2 relative error` metric per variant; no L-BFGS-from-RAR loop needed (smooth
  problem). One final L-BFGS polish if `lbfgs=True` in the standard pipeline; otherwise Adam alone already suffices.

## Bake recipe

- **Net:** `[3, 48, 48, 48, 48, 1]`, `tanh`, Glorot uniform (small — this is a smooth, low-frequency problem; no need
  for the big burgers/wave nets).
- **Sampling:** `num_domain=4000`, `num_boundary=0` (BC is hard), `num_test=4000`, over the Hypercube `[0,0,0]→[1,1,1]`.
- **Optimizer:** Adam `lr=1e-3` × **12k** iters, then **one** L-BFGS polish. **No RAR** (no sharp fronts; smooth sines).
- **Smoke test first:** `--quick` (~30 s) to catch a structural bug before the real bake. Expected real bake: fast
  (~few min, small net + no RAR + no L-BFGS long tail to worry about — cap L-BFGS maxiter if it idles).
- **Expected relative-L2 band:** `< 1e-2` across all `a>0` variants (likely `~1e-3`, comparable to bench-poisson2d);
  `a=0` reports `||pred|| ~ 1e-4` (the degenerate control's metric). ONNX parity `~1e-7` (pure tensor transform).

## Registry line (orchestrator adds to `frontend/src/content/cases/registry.tsx`)

```ts
import { ZeroSourceContext } from "./ZeroSourceContext";
// ... in CASE_CONTEXT:
"ctrl-zero-source": ZeroSourceContext,
```

## Risks / fallback

- **Spectral bias on the 2nd mode:** the `sin(2πx)sin(2πy)` term is higher-frequency; if the net underfits it at full
  amplitude, the L2 could rise. Mitigation: the modes are still low (k,m ≤ 2) and the field is smooth — the 48×4 net
  with 12k Adam should be ample (cf. bench-poisson2d <0.2% with a comparable setup). Fallback if it underfits: bump to
  `[3,64,64,64,64,1]` and/or drop the 2nd mode to a pure single-mode `g=sin(πx)sin(πy)` (still a valid MMS family
  containing the degenerate control at a=0, just a simpler heatmap).
- **`a=0` trivial member:** the net must output exactly the residual norm there; the hard-zero transform already forces
  `u=0` on ∂Ω, and with `f=0` at `a=0` the trivial `u≡0` is the global minimum — converges immediately. ✓
- **Honesty:** labeled `synthetic` (a control represents no physical data); the Context states plainly its job is to
  measure pipeline correctness against an exact truth, never dressed up as physics.
```
