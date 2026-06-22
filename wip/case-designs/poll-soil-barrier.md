# Workbench design — `poll-soil-barrier`

Contaminated-site barrier transport: pure diffusion through a low-permeability vertical barrier (a clay/slurry cutoff),
solved with a domain-decomposition (FBPINN-style) two-channel PINN. The coefficient jump puts a **kink** in the
concentration derivative at each barrier face — the distinctive feature this case showcases.

## Decision: **SINGLE honest benchmark variant**

- **`decision = single`**, `n_variants = 1`, `param_key = ""`.
- `field_axes` = the inputs `(x, t)` (default; no swept network-input knob, no `param_specs`).
- The case already has the correct single-variant **shape** (no `field_axes`/`param_specs` override). The only change
  is to add an explicit one-variant `variants()` so the workbench chip reads as a named benchmark rather than the bare
  "default". The existing FBPINN method, MMS anchor, geometry and bake plan are **unchanged**.

### Why not parametric (the adversarial check)

The natural physical knob is the **diffusion contrast** `ρ = D_barrier / D_soil`. The MMS anchor is provably exact for
*any* ρ (proof below), so a closed-form parametric family *exists*. But it would **not train or validate well**, and it
would weaken the case:

1. **The kink is the whole point and it is already hard.** At the fixed 10× contrast the FBPINN two-channel lane on CPU
   already sits at ~`2e-1` relative L2 — the coefficient-jump kink is the limiting difficulty. The kink sharpness scales
   as `1/ρ` (inside the barrier `c_x = -1/(D·R_L)`), so a contrast *sweep* forces **one** network to span from mild to
   very severe kinks. The sharpest (smallest-ρ) regime dominates the L2; the parametric band would be **worse** than the
   fixed case, not better — this is the spectral-bias-across-severities failure the playbook already flags for parametric
   Helmholtz.
2. **The hard-constraint baseline becomes ρ-dependent.** `Ψ(x;ρ)` and the source `f(x,t;ρ)` both depend on ρ through
   `R_L(ρ) = a/D_soil + (b−a)/(D_soil·ρ) + (1−b)/D_soil`. Making ρ a network input means re-deriving `Ψ` from the ρ
   column inside the output transform and the PDE residual — doable, but it adds risk to the one thing that currently
   works (the exact lift/vanish baking) for no accuracy gain.
3. **Honesty (ADR-0016 §9.A).** Fabricating regimes to hit a chip count is the "toy" failure. A contrast sweep here
   would *blur* the very feature the case exists to demonstrate. The conservative, honest choice is a fixed-parameter
   benchmark — exactly the Allen-Cahn precedent (a sharp, single-feature stiff case → single variant).

**Fallback if a sweep is ever wanted:** keep ρ in a *narrow* band `ρ∈[0.1,0.5]` (mild kinks only), expect ~`1e-1`,
honestly labeled — but the single benchmark is the recommended ship.

## Analytic anchor (series-resistance MMS) + substitution proof

PDE (pure diffusion, piecewise-constant coefficient, source `f`):

```
c_t = D(x) c_xx + f,   D(x) = D_soil for x∉[a,b],  D_barrier for x∈[a,b],   a=0.45, b=0.55
```

Define the accumulated diffusive resistance and the stationary series-resistance profile:

```
R(x) = ∫_0^x dx'/D(x'),    Ψ(x) = 1 − R(x)/R(L),    R(L) = a/D_soil + (b−a)/D_barrier + (1−b)/D_soil
c*(x,t) = (1 − e^{-t}) Ψ(x),    f(x,t) = e^{-t} Ψ(x)
```

**Proof it solves the PDE for ALL D-values (hence any contrast ρ):**

- `R'(x) = 1/D(x)`  ⟹  `Ψ'(x) = −R'(x)/R(L) = −1/(D(x) R(L))`.
- Therefore the diffusive flux is **constant**: `D(x) Ψ'(x) = −1/R(L) = const` everywhere.
- Hence `d/dx[ D(x) Ψ'(x) ] = 0` in the interior of each layer, i.e. `D(x) Ψ''(x) = 0` (away from the faces).
- Time derivative: `c*_t = e^{-t} Ψ(x)`.
- Diffusion term: `D(x) c*_xx = (1 − e^{-t}) · D(x) Ψ''(x) = 0` in each interior.
- Residual: `c*_t − D(x) c*_xx − f = e^{-t}Ψ − 0 − e^{-t}Ψ = 0.`  ∎

`c*` is continuous and `D c*_x = −(1−e^{-t})/R(L)` is continuous (the physical flux-matching condition), while `c*_x`
itself jumps at each face — the **kink**. At `t=0`, `c*≡0` (clean start); inlet `c*(0,t)=(1−e^{-t})·1=1−e^{-t}`, outlet
`c*(1,t)=(1−e^{-t})·0=0`. These match the hard-constraint output transform exactly.

## Exact case `.py` edits (orchestrator applies)

The current file is already correct for a single variant: `inputs=("x","t")`, no `field_axes`, no `param_specs`, so
`CASE.axes == ("x","t")` and `param_specs` is empty — `__post_init__` passes. **Keep `CASE`, `analytic`, `build` exactly
as they are** (the FBPINN two-channel net, the `fbpinn_transform`, the barrier-face anchors, the `TimePDE` with
`solution=analytic`). The only addition is an explicit single-variant `variants()`.

Add the import of `Variant` and the `variants()` function:

```python
# change the existing import line:
from .base import CaseSpec, Variant


def variants() -> list[Variant]:
    # Single honest benchmark: the diffusion contrast is a physical knob with a closed-form anchor for any value,
    # but the coefficient-jump kink is already the limiting difficulty at the fixed 10x contrast (a sweep would let
    # the sharpest regime dominate the L2 and blur the feature). One fixed-parameter benchmark variant (ADR-0016 §9.A).
    return [
        Variant(
            "barrier10x",
            "Barrier (10× contrast)",
            "Barrera (contraste 10×)",
            {},  # no swept parameter; the field is c(x,t) on the (x,t) grid
            "10× low-D barrier in [0.45,0.55]: the plume is slowed and c develops a kink at each barrier face.",
            "Barrera de baja D (10×) en [0.45,0.55]: el plume se frena y c desarrolla un quiebre en cada cara.",
        ),
    ]
```

No other edits. (If the orchestrator's pipeline prefers omitting `variants()` entirely for single-variant cases — the
Allen-Cahn convention — that is also acceptable; this explicit one-variant form just gives the workbench chip a
meaningful name. Do **not** add `field_axes` or `param_specs`.)

## Bake recipe

Keep the case's existing plan (already in `CASE.train`); it is the validated CPU FBPINN lane:

- **Net:** `FNN([2] + [64]*4 + [2], "tanh", "Glorot normal")` — 2-channel (two output heads blended by the
  partition-of-unity windows). Keep as-is.
- **Output transform:** the existing `fbpinn_transform` (sigmoid partition-of-unity blend + hard lift/vanish baking
  inlet/outlet/IC). Keep as-is.
- **Optimizer:** Adam, `lr=1e-3`, **18 000** iterations, then **one** final **L-BFGS** polish.
- **RAR:** none — instead, the existing **barrier-face anchors** (40 points each on `x=a`, `x=b`, `x=x_c`) concentrate
  collocation on the kink. (Optional future: a few RAR rounds targeting the residual at the faces; not required to ship.)
- **Sampling:** `num_domain=4000`, `num_boundary=0`, `num_initial=0`, `num_test=8000`, `solution=analytic`
  (so the reported metric is the true relative-L2 vs the MMS).
- **`--quick` smoke first** (~30 s) to catch a structural bug before the ~real bake.
- **Expected band:** relative-L2 **~`2e-1`** on the CPU two-channel lane (the coefficient-jump kink is the hard part).
  This is the honest band — the strict per-subdomain-normalized FBPINN + GPU would tighten it; do **not** quote a
  smaller number.

## Registry line

Add to `frontend/src/content/cases/registry.tsx` (orchestrator, not this agent):

```tsx
import { SoilBarrierContext } from "./SoilBarrierContext";
// ...
"poll-soil-barrier": SoilBarrierContext,
```

## Risks / fallback

- **Risk:** the ~`2e-1` band looks high next to the `<1%` analytic cases. **Mitigation:** the Context and
  `expected_band` state plainly *why* (coefficient-jump kink, CPU two-channel lane) and that GPU + strict FBPINN tighten
  it — honesty over a flattering number. Do not silently shrink the band.
- **Risk:** the single chip looks thin vs parametric cases. **Mitigation:** this is the correct, honest call (Allen-Cahn
  precedent); the Context's "Why a single variant" paragraph justifies it explicitly so it does not read as laziness.
- **Risk (if a sweep is later attempted):** ρ-dependent `R_L` must thread through `psi_torch`, the source `f`, and the
  `lift` baseline; the sharpest regime dominates L2. Stay in `ρ∈[0.1,0.5]` and accept ~`1e-1` if pursued. Not
  recommended.
