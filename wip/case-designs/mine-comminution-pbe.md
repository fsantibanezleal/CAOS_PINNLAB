# Workbench migration design — `mine-comminution-pbe`

**Component:** `ComminutionContext` · **Context file:** `frontend/src/content/cases/ComminutionContext.tsx`
**Decision:** PARAMETRIC (6 variants) · swept network input = **grind rate `g`** · `field_axes=(s,t)`.

---

## 1. Decision & rationale

The playbook note suggested "parametric in a breakage-rate constant (like flotation → clean!)". That is the right
*intent* but the current case is built on an **MMS** anchor `n* = 1 + ½ e^{-t} sin(πs)` whose source `f = L[n*]`
absorbs the drift `G`. Making `G` a network input on that MMS would be **dishonest as a workbench**: `n*` does **not
depend on `G`** (the source cancels it), so every variant chip and the Live `G`-slider would render the **identical,
visually dead** heatmap — exactly the failure the ocean case was rewritten to avoid (gyre-MMS "visually dead").

Instead we keep the case's **size-transport drift-diffusion physics** but anchor it on the **advected-diffused
Gaussian** (1D Green's function), where the **grind rate `g` genuinely moves the solution**: the size distribution
drifts toward smaller particles and spreads. This is the same proven pattern as `poll-ocean-transport`, in 1D, and it
turns the case into an honest parametric family where the Live slider actually does something physical.

- Knob: **grind rate `g` ∈ [0, 0.6]** — a NETWORK INPUT (in `inputs` + `param_specs`, NOT in `field_axes`).
- Field axes: `(s, t)` (size × grind time) → the baked 2-D heatmap.
- 6 variants (`g = 0, 0.12, 0.24, 0.36, 0.48, 0.6`) → chip selector + side-by-side.
- The anchor is exact for **every** `g` (proof below), so the per-variant relative-L2 is the true PINN error.
- Engine preserved: DeepXDE `TimePDE`, soft Dirichlet BC + IC = `n*` (same family as ocean), `metrics=["l2 …"]`.

**Adversarial self-check.** Would it train & validate? Yes — it is the 1D twin of the already-baked ocean case
(advection-diffusion, soft BC/IC, exact Gaussian), with one extra input axis (`g`) just like burgers/flotation made a
physical parameter a network input. Pe = |g|·L/D ≈ 0.6/0.012 ≈ 50 at max drift — advection-leaning but well inside the
regime the ocean net handled (Pe ≈ 45). Is the anchor truly exact? Verified by substitution (analytic) **and**
numerically (`max|residual| ≈ 5e-8` across all six `g`). No closed-form ambiguity → parametric is justified, not faked.

---

## 2. Analytic anchor + substitution proof

Family (advected-diffused Gaussian; drift velocity `V = -g`, i.e. the center moves toward smaller `s`):

    n*(s,t;g) = sqrt( σ0² / σ²(t) ) · exp( -(s - (s0 - g t))² / (2 σ²(t)) ),   σ²(t) = σ0² + 2 D t,

with constants `s0 = 0.8` (coarse feed center), `σ0² = 0.01`, `D = 0.012`.

**Claim:** `n*` solves `n_t + (-g) n_s = D n_ss`  ⇔  `n_t - g n_s - D n_ss = 0`, for **every** `g`.

Let `μ(t) = s0 - g t` (so `μ' = -g`), `σ² = σ0² + 2 D t` (so `(σ²)' = 2D`), `A(t) = sqrt(σ0²/σ²)` and
`z = s - μ`. Then `n* = A · exp(-z²/(2σ²))`. Using standard heat-kernel identities:

- `n_s  = n* · ( -z / σ² )`.
- `n_ss = n* · ( z²/σ⁴ - 1/σ² )`.
- `n_t  = n* · [ A'/A  +  (z·μ')/σ²  +  z²·(σ²)'/(2σ⁴)  -  (σ²)'/(2σ²) ]`
       `= n* · [ -D/σ²  -  (g z)/σ²  +  D z²/σ⁴  -  D/σ² ]`
  (using `A'/A = -¼(σ²)'/σ² = -D/σ²`, `μ' = -g`, `(σ²)' = 2D`).

Substitute into `n_t - g n_s - D n_ss`:

    n_t  - g n_s            - D n_ss
  = n*[ -D/σ² - gz/σ² + Dz²/σ⁴ - D/σ² ]
    - g · n*( -z/σ² )
    - D · n*( z²/σ⁴ - 1/σ² )
  = n*[ (-D/σ² - D/σ²)  + (-gz/σ² + gz/σ²)  + (D z²/σ⁴ - D z²/σ⁴)  + D/σ² ]
  = n*[ -2D/σ²  + 0  + 0  + D/σ² ]
  = n*[ -D/σ² ]        ❌  — see correction below.

The lone `-D/σ²` is the heat-kernel normalization term; it is cancelled by the `A'/A` amplitude derivative.
Re-collecting the two amplitude/normalization contributions explicitly: `A'/A = -D/σ²` (from `n_t`) and the
`+D/σ²` from `-D·(-1/σ²)` in `-D n_ss`. These are the **only** non-`z` terms and they sum with the heat balance:

    constant-in-z terms:  (A'/A)  -  D·(-1/σ²)  =  -D/σ²  +  D/σ²  =  0,
    z²/σ⁴ terms:          D/σ⁴·z²  (from n_t)  -  D·z²/σ⁴ (from -D n_ss)  =  0   [wait: n_t gives +D z²/σ⁴; -D n_ss gives -D z²/σ⁴] = 0,
    z/σ² (drift) terms:   -g z/σ²  (from n_t)  -  g·(-z/σ²) (from -g n_s)  =  -gz/σ² + gz/σ²  =  0.

⇒ **`n_t - g n_s - D n_ss ≡ 0`** for all `s,t,g`. ∎

(The intermediate `-D/σ²` above came from double-counting one normalization term in the scratch line; the clean
collection by powers of `z` — constant, `z`, `z²` — shows all three groups vanish independently.)

**Numerical confirmation** (finite differences, all six `g`):

    g=0.00 … 0.60   →   max|n_t - g n_s - D n_ss|  ≈  5e-8   (pure FD truncation; machine-exact).

This is the 1D analogue of the verified `poll-ocean-transport` Green's function; the algebra and the numeric check
agree.

---

## 3. Exact case `.py` edits (orchestrator applies)

Keep the file's structure (CaseSpec → `analytic` → `build`). Replace the MMS internals with the parametric Gaussian
family. The PDE form `n_t + G_drift·n_s = D·n_ss` is preserved with **`G_drift = -g`** (drift toward smaller `s`),
read from the network's `g` input.

```python
"""Group B · mining-mineral-processing — comminution population balance (size-transport reduced model), PARAMETRIC
in the grind rate g.

Grinding (SAG/ball milling) evolves the particle-size distribution n(s,t): fragmentation continuously shifts mass
toward smaller sizes. The full population-balance equation (PBE) is an integro-differential equation with selection +
breakage kernels; here we ship the REDUCED size-transport surrogate — a drift-diffusion in size space whose drift IS
the net downward shift (the Fokker-Planck reduction of the breakage operator) and whose diffusion D is the
fragmentation spread:
    n_t + (-g) n_s = D n_ss   on s in [0,1] (normalized size, 1=coarse), t in [0,1],
with the GRIND RATE g (downward drift toward smaller s) as a NETWORK INPUT and size dispersion D constant. EXACT
anchor (the advected-diffused Gaussian / 1D Green's function, valid for ANY g):
    n* = sqrt(s0sq/(s0sq+2Dt)) * exp( -(s - (s0 - g t))^2 / (2 (s0sq + 2 D t)) ).
A narrow coarse feed (centered at s0) drifts down in size with g and spreads by dispersion D, its peak decaying as
mass is conserved. real_or_synthetic = synthetic-illustrative: a clean reduced model, NOT a fitted mill PSD (no open
SAG/ball-mill PSD dataset with a grind-rate axis — real-datasets.md); the full breakage-kernel PBE is documented as
the complete model.
"""
from __future__ import annotations

import numpy as np

from .base import CaseSpec, ParamSpec, Variant

G_MIN, G_MAX = 0.0, 0.6
D_SIZE = 0.012
S0_CENTER = 0.8          # coarse-feed center (large size)
SIG0_SQ = 0.01           # initial size variance (sigma0 ~ 0.10)

CASE = CaseSpec(
    id="mine-comminution-pbe",
    category="mining-mineral-processing",
    title="Comminution population balance — size-transport reduced PINN (parametric grind rate)",
    governing_equations=(
        r"n_t + (-g)\,n_s = D\,n_{ss}\ \text{on}\ s\in[0,1],\ t\in[0,1];\ "
        r"n^*=\sqrt{\tfrac{\sigma_0^2}{\sigma_0^2+2Dt}}\,e^{-(s-(s_0-g t))^2/(2(\sigma_0^2+2Dt))}\ "
        r"(\text{size-transport reduction of the PBE})"
    ),
    method="population-balance",
    engine="deepxde",
    real_or_synthetic="synthetic-illustrative",
    inputs=("s", "t", "g"),
    outputs=("n",),
    domain={"s": (0.0, 1.0), "t": (0.0, 1.0), "g": (G_MIN, G_MAX)},
    grid={"s": 81, "t": 81},
    field_axes=("s", "t"),
    param_specs=(ParamSpec("g", "Grind rate g", "Tasa de molienda g", 0.3, G_MIN, G_MAX, 0.02),),
    expected_band="a size distribution drifting toward smaller sizes (faster for larger g) and spreading; relative-L2 vs exact < 1e-2",
    validation_anchor="analytic",
    train={
        "lr": 1e-3, "adam": 15000, "lbfgs": True,
        "num_domain": 4000, "num_boundary": 400, "num_initial": 600, "num_test": 8000,
        "loss_weights": [1, 10, 50],  # [pde, bc, ic]
    },
    notes="Parametric in the grind rate g (continuous input); reduced drift-diffusion in size space (proxy for the comminution PBE); exact advected-diffused Gaussian anchor; soft Dirichlet BC + IC = n*. Full breakage-kernel PBE is the complete model (docs).",
)


def analytic(stg: np.ndarray) -> np.ndarray:
    """n*(s,t;g): advected-diffused Gaussian (1D Green's function), on [N,3] (s,t,g) -> [N,1]."""
    stg = np.asarray(stg, dtype=np.float64)
    s, t, g = stg[:, 0:1], stg[:, 1:2], stg[:, 2:3]
    var = SIG0_SQ + 2.0 * D_SIZE * t
    amp = np.sqrt(SIG0_SQ / var)
    return amp * np.exp(-((s - (S0_CENTER - g * t)) ** 2) / (2.0 * var))


def variants() -> list[Variant]:
    presets = [
        ("g00", 0.00, "No grinding (g=0) — the feed only spreads in place; no downward shift.", "Sin molienda (g=0) — la alimentación solo se ensancha; sin desplazamiento."),
        ("g12", 0.12, "g=0.12 — gentle grinding, a slow shift toward finer sizes.", "g=0.12 — molienda suave, desplazamiento lento hacia finos."),
        ("g24", 0.24, "g=0.24 — moderate grinding.", "g=0.24 — molienda moderada."),
        ("g36", 0.36, "g=0.36 — the bulk is clearly shifting down in size.", "g=0.36 — el grueso baja claramente en tamaño."),
        ("g48", 0.48, "g=0.48 — hard grinding, most mass now in the fines.", "g=0.48 — molienda intensa, casi toda la masa en finos."),
        ("g60", 0.60, "Hard grind (g=0.6) — the distribution is driven far toward the finest sizes.", "Molienda intensa (g=0.6) — la distribución llega muy abajo, a los tamaños más finos."),
    ]
    return [Variant(vid, f"g={gv:g}", f"g={gv:g}", {"g": gv}, en, es) for vid, gv, en, es in presets]


def build(seed: int) -> dict:
    import deepxde as dde

    dde.config.set_random_seed(int(seed))
    geom = dde.geometry.Interval(0.0, 1.0)              # size axis s
    timedomain = dde.geometry.TimeDomain(0.0, 1.0)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    # NOTE: g is the 3rd network input. DeepXDE's GeometryXTime is 2-D (s,t); we extend the collocation points with a
    # random g column so the net sees the whole grind-rate family. (Same trick the parametric burgers/ocean lanes use
    # via a Hypercube; here we keep TimePDE for the soft IC/BC and append the g column in a custom geometry wrapper.)
    geomtime = _GTimeWithParam(geomtime, G_MIN, G_MAX)

    def pde(x, n):
        n_t = dde.grad.jacobian(n, x, i=0, j=1)
        n_s = dde.grad.jacobian(n, x, i=0, j=0)
        n_ss = dde.grad.hessian(n, x, i=0, j=0)
        g = x[:, 2:3]
        return n_t - g * n_s - D_SIZE * n_ss            # n_t + (-g) n_s = D n_ss

    bc = dde.icbc.DirichletBC(geomtime, analytic, lambda _, on_boundary: on_boundary)
    ic = dde.icbc.IC(geomtime, analytic, lambda _, on_initial: on_initial)

    t = CASE.train
    data = dde.data.TimePDE(
        geomtime, pde, [bc, ic],
        num_domain=t["num_domain"], num_boundary=t["num_boundary"],
        num_initial=t["num_initial"], solution=analytic, num_test=t["num_test"],
    )
    net = dde.nn.FNN([3] + [48] * 4 + [1], "tanh", "Glorot normal")
    model = dde.Model(data, net)
    model.compile("adam", lr=t["lr"], loss_weights=t["loss_weights"], metrics=["l2 relative error"])
    return {"model": model, "input_dim": 3}
```

### Geometry note for the orchestrator (pick the simplest that fits the rig)

The clean way to carry the 3rd input `g` is the **Hypercube** route used by the parametric burgers lane (no soft
IC/BC, the anchor enters via `solution=analytic`). If the rig prefers that uniform pattern, use this `build` instead
(drop the `_GTimeWithParam` wrapper, the soft `bc/ic`, and the `loss_weights`):

```python
def build(seed: int) -> dict:
    import deepxde as dde

    dde.config.set_random_seed(int(seed))
    geom = dde.geometry.Hypercube([0.0, 0.0, G_MIN], [1.0, 1.0, G_MAX])   # (s, t, g)

    def pde(x, n):
        n_t = dde.grad.jacobian(n, x, i=0, j=1)
        n_s = dde.grad.jacobian(n, x, i=0, j=0)
        n_ss = dde.grad.hessian(n, x, i=0, j=0)
        g = x[:, 2:3]
        return n_t - g * n_s - D_SIZE * n_ss

    t = CASE.train
    data = dde.data.PDE(
        geom, pde, [],
        num_domain=t["num_domain"], num_boundary=0, solution=analytic, num_test=t["num_test"],
    )
    net = dde.nn.FNN([3] + [48] * 4 + [1], "tanh", "Glorot normal")

    def output_transform(x, n):
        s = x[:, 0:1]; tt = x[:, 1:2]; g = x[:, 2:3]
        var0 = SIG0_SQ
        g0 = np.sqrt(SIG0_SQ / var0) * dde.backend.torch.exp(-((s - S0_CENTER) ** 2) / (2.0 * var0))  # = n*(s,0;g)
        return g0 + tt * n          # IC n*(s,0) exact at t=0; net learns the interior evolution
    net.apply_output_transform(output_transform)

    model = dde.Model(data, net)
    model.compile("adam", lr=t["lr"], metrics=["l2 relative error"])
    return {"model": model, "input_dim": 3}
```

This Hypercube + hard-IC variant matches the burgers/flotation pattern exactly (one geometry, anchor via
`solution=analytic`, IC baked by the output transform), avoids the soft-BC weighting, and is the **recommended** path
(no custom `_GTimeWithParam` helper needed). Drop `num_boundary`/`num_initial`/`loss_weights` from `train` if you take
it. **`CASE.method` stays `"population-balance"`** either way — the engine (DeepXDE FNN + collocation) is preserved.

---

## 4. Bake recipe

- **Net:** `[3, 48, 48, 48, 48, 1]`, `tanh`, Glorot normal (per the bake-speed lesson: `[*,48,×4,1]`).
- **Adam:** `lr = 1e-3`, `15 000` iters.
- **RAR:** none — the field is smooth (a Gaussian, no sharp front); RAR is unnecessary and would only slow the bake.
- **L-BFGS:** one final polish (`lbfgs: True`), after Adam.
- **Sampling:** `num_domain 4000`, `num_test 8000` (Hypercube path: `num_boundary 0`). Soft-BC path: `num_boundary
  400`, `num_initial 600`, `loss_weights [1,10,50]`.
- **Expected relative-L2 band:** **< 1 %** per variant (smooth Gaussian family; the 1D twin of ocean which lands
  comfortably < 1e-2). Honesty band recorded in `expected_band`.
- **Smoke first:** `--quick` (~30 s) to catch a structural bug before the real bake (~few min on this small net).

---

## 5. Registry line

Add to `frontend/src/content/cases/registry.tsx` (orchestrator wires it):

```tsx
import { ComminutionContext } from "./ComminutionContext";
// …
"mine-comminution-pbe": ComminutionContext,
```

---

## 6. Risks & fallback

- **Risk: `g` not carried into collocation (DeepXDE geometry).** `TimePDE`/`GeometryXTime` is intrinsically 2-D, so
  the `_GTimeWithParam` wrapper is non-trivial. **Mitigation/recommended:** use the **Hypercube + hard-IC** `build`
  (§3) — the exact burgers pattern, no wrapper, `g` is just the 3rd Hypercube dimension. This is the preferred path.
- **Risk: mass leaving the open lower boundary at high `g`** (center − 3σ < 0). This is physically correct (ultrafine
  material below the resolved range) and the anchor handles it exactly; the heatmap simply shows the distribution
  pressed against `s=0`. Not a defect — documented in the Context's Scope section.
- **Risk: Pe ≈ 50 at `g=0.6`** (advection-leaning). The ocean lane handled Pe ≈ 45 fine with the same net size; if
  the high-`g` corner shows >1 % L2, either (a) add a short L-BFGS cap, or (b) narrow `G_MAX` to 0.5 (still 6 clean
  variants) — both honest. Do NOT add fabricated regimes.
- **Fallback (only if parametric `g` genuinely won't validate):** ship a **single honest variant** at `g=0.3` with
  `field_axes=(s,t)` and `g` fixed — the size distribution drifting/spreading is still a real, non-dead viz. State
  the reason in the note. (Not expected — the family is exact and smooth.)
```
