# ind-helmholtz — workbench migration design

**Case:** `ind-helmholtz` — 2D Helmholtz (high-wavenumber), Fourier-feature PINN (spectral-bias showcase).
**Context component:** `HelmholtzContext` (written: `frontend/src/content/cases/HelmholtzContext.tsx`).

## Decision: SINGLE honest benchmark variant (fixed wavenumber n = 3, k0 = 6π)

Keep the case as a **single fixed-wavenumber benchmark**, not a parametric sweep. The distinctive method
(random Fourier-feature embedding defeating spectral bias) is the star of this case and is best demonstrated at one
genuinely-hard frequency. This mirrors the `bench-allencahn` precedent (single honest benchmark where the method, not
a parameter family, is the point).

### Why not parametric in the wavenumber n

The MMS family `u* = sin(k0 x) sin(k0 y)` with `k0 = 2π n` *is* closed-form exact for any n (proof below), so an
analytic anchor for a parametric sweep is available in principle. We reject parametric anyway, conservatively:

1. **Spectral bias across a frequency BAND for one net is hard.** The frozen Gaussian Fourier matrix `B` is tuned for
   a single target frequency via `SIGMAS = (1, n)`. A continuous-n net must span the whole band with one fixed `B`;
   accuracy degrades toward the sharp end (the bias re-emerges higher in the band). The playbook flags this explicitly:
   *"parametric wavenumber n is HARD … either keep narrow n∈[2,4] (accept ~1e-1, honestly labeled) or ship the fixed-n
   Fourier-feature showcase single-variant."*
2. **Uneven per-variant error undermines the showcase.** A sweep whose worst chip is ~1e-1 while its best is ~1e-2
   teaches the spectral-bias lesson *worse* than one fixed hard frequency where the Fourier map is unambiguously the
   hero. Honesty over chip count (ADR-0016 §9.A).
3. **No other honest discrete/time axis exists.** Helmholtz is steady-state (frequency domain) — there is no time to
   scrub, and no second physical knob with a clean closed form worth a discrete set.

Adversarial self-check: would parametric train+validate? Yes, the anchor is exact, so it *would* validate — but with
marginal, uneven accuracy that weakens the pedagogical point. The conservative choice is single. The parametric path is
retained below as an explicit fallback (e.g. for a future GPU + frequency-annealing lane).

## Analytic anchor + substitution proof

Anchor (MMS, exact — DeepXDE sign convention `-u_xx - u_yy - k0^2 u - f = 0`):

```
u*(x,y) = sin(k0 x) sin(k0 y),   k0 = 2π n,   n = 3,   f = k0^2 sin(k0 x) sin(k0 y)
```

Proof it solves the PDE for any n:
- u*_xx = -k0^2 sin(k0 x) sin(k0 y) = -k0^2 u*
- u*_yy = -k0^2 sin(k0 x) sin(k0 y) = -k0^2 u*
- ∇²u* = u*_xx + u*_yy = -2 k0^2 u*
- Residual: -u*_xx - u*_yy - k0^2 u* - f = 2 k0^2 u* - k0^2 u* - f = k0^2 u* - f.
  With f = k0^2 u*, residual ≡ 0. ✔ (holds for ANY n, hence the parametric fallback is sound.)

Boundary: for k0 = 2π n with integer n, sin(k0·0) = sin(0) = 0 and sin(k0·1) = sin(2π n) = 0, so u* = 0 on all four
edges → satisfies the homogeneous Dirichlet BC exactly. ✔

This is the case's **existing** anchor — the migration keeps it. The validation anchor stays `analytic`, so the
per-variant relative-L2 is the true PINN error.

## Exact case `.py` edits (orchestrator applies)

The current `ind_helmholtz.py` is **already structurally a single-variant workbench case**: `inputs = ("x","y")`,
no `param_specs`, no `field_axes` (so `axes` defaults to `inputs` = the heatmap), `validation_anchor = "analytic"`,
and the Fourier-feature method + soft Dirichlet BC are intact. No `variants()` function → base ships the single
"default" variant, which is exactly what we want.

**Required edits: none to the structure or method.** Only an optional cosmetic clarification (does not change physics,
bake, or ONNX):

```python
# OPTIONAL — clarify in the title that this is a fixed-wavenumber benchmark (purely a label change):
title="2D Helmholtz (fixed high-wavenumber n=3) — Fourier-feature PINN",

# Everything else stays exactly as-is:
#   inputs=("x","y"); outputs=("u",); domain={"x":(0,1),"y":(0,1)}; grid={"x":121,"y":121}
#   NO field_axes (axes defaults to inputs)  ·  NO param_specs  ·  NO variants()
#   validation_anchor="analytic"
#   method="fourier-features"; SIGMAS=(1.0, float(N_WAVES)); N_FEATURES=64; soft Dirichlet, loss_weights=[1,100]
#   analytic(xy) and build(seed) UNCHANGED.
```

If the orchestrator prefers an explicit single `variants()` for uniformity with other migrated cases, add (optional,
equivalent to the base default):

```python
from .base import CaseSpec  # (already imported); add Variant if using the explicit form:
# from .base import CaseSpec, Variant
def variants():
    return [Variant(
        "n3", "n=3 (k0=6π)", "n=3 (k0=6π)", {},
        "Fixed high-wavenumber standing wave; Fourier features defeat the spectral-bias plateau.",
        "Onda estacionaria de número de onda alto fijo; las características de Fourier vencen la meseta de sesgo espectral.",
    )]
```

This `variants()` is purely presentational (empty `params`, single entry). Prefer omitting it (base default) unless the
shell needs an explicit chip label.

## Bake recipe (single bake; orchestrator runs)

The case's `train` dict already encodes the recipe — keep it; it is well-matched to the Fourier-feature lane:

- **Net:** `dde.nn.FNN([in_dim] + [128]*4 + [1], "tanh", "Glorot uniform")` with `in_dim = 2*N_FEATURES*len(SIGMAS) =
  2*64*2 = 256` after the frozen Fourier feature transform. Keep `apply_feature_transform(fourier_features)`.
- **Fourier map:** frozen Gaussian `B`, seeded under `torch.manual_seed(0)`, scales `SIGMAS=(1, 3)`, `N_FEATURES=64`.
  Identical `B` for train / parity / ONNX export (do NOT reseed between stages).
- **BC:** soft Dirichlet (`u=0`), `loss_weights=[1, 100]` (PDE, BC). No hard constraint (it would fight the oscillation
  near the boundary).
- **Adam:** `lr=1e-3`, `25000` iters.
- **RAR:** none (the solution is globally oscillatory, not a localised front — uniform collocation `~12 pts/wavelength`,
  `nx = 12*N_WAVES = 36`, `num_domain = 36*36 = 1296`, `num_boundary = 4*36 = 144` — is appropriate; RAR adds nothing).
- **L-BFGS:** one final polish (`lbfgs=True`).
- **num_test:** 4000 (analytic).
- **Expected relative-L2 band:** ~1e-1 on the CPU lane (Fourier features lift the spectral-bias plateau; GPU + frequency
  annealing would tighten further). Honestly labeled in `expected_band` — do NOT claim sub-1e-2; that would be
  dishonest for this CPU high-wavenumber lane.

`--quick` smoke first (~30 s) to catch a structural bug before the real bake. Do not pair with another heavy bake (this
one is a 256-wide-input MLP, moderately heavy).

## Registry line (orchestrator adds to `registry.tsx`)

```ts
import { HelmholtzContext } from "./HelmholtzContext";
// ...
"ind-helmholtz": HelmholtzContext,
```

(plus the corresponding `index.json` regeneration — out of this subagent's scope.)

## Risks / fallback

- **Risk: the ~1e-1 band reads as "bad" next to the <1% benchmarks.** Mitigation: the Context and `expected_band`
  state explicitly that high-wavenumber-on-CPU is intrinsically harder and that ~1e-1 is the honest, expected band for
  this lane; the *visual* (a crisp 3×3 lobe grid vs. a blurred low-frequency collapse) is the success criterion, not a
  sub-1% number. This is honesty, not weakness.
- **Risk: ONNX parity drift if `B` is reseeded.** Mitigation: `B` is built under `torch.manual_seed(0)` independent of
  the case seed, so train / parity / export share the identical frozen matrix. Keep that ordering.
- **Fallback (only if a future GPU/annealing lane wants a sweep):** go parametric in `n` — the substitution proof above
  holds for any n, so the anchor stays exact. Edits would be: `inputs=("x","y","n")`,
  `domain` add `"n":(2.0,4.0)`, `field_axes=("x","y")`,
  `param_specs=(ParamSpec("n","Wavenumber n","Número de onda n",3.0,2.0,4.0,0.25),)`,
  `analytic(xyn)` reads `k0 = 2π·n` per row, the `pde` reads `k0 = 2π·x[:,2:3]` and forms `f` per row, the Fourier
  `SIGMAS` widen to cover the band (e.g. `(1, 2, 4)`), and `variants()` lists `n ∈ {2.0, 2.5, 3.0, 3.5, 4.0, ...}`
  (≥6). Expect ~1e-1 with uneven per-n error (worst at n=4). NOT recommended for the current CPU lane.
