# Case design — `bench-navier-cavity` (2D lid-driven cavity, steady incompressible Navier-Stokes)

## Decision: SINGLE honest benchmark variant

Migrate the lid-driven cavity to the workbench as a **single-variant benchmark** at fixed `Re = 100`, validated
against the Ghia, Ghia & Shin (1982) centerlines. **Not parametric, not a time-scrubber, not a discrete sweep.**

### Rationale (adversarial check applied)

- **Parametric-in-Reynolds is rejected.** The lid-driven cavity has **no closed-form solution for any Re** — that is
  precisely why Ghia 1982 exists as a digitized numerical reference. There is no analytic anchor that can be proven by
  substitution for a continuous Reynolds family, so the parametric bar (a network-input knob with an exact anchor for
  *all* values) cannot be met honestly. Manufacturing a Reynolds sweep would fabricate regimes with no verifiable
  anchor — the "toy" failure (ADR-0016 §9.A). The migration-state note already flagged this: *"parametric Reynolds has
  no closed form → single benchmark."*
- **Discrete-Re set is rejected too.** Ghia's table is defined at discrete Re (100, 400, 1000, …), but (a) each Re
  needs its **own** trained net — it is not one shared net + one ONNX sweeping an input — and (b) higher Re on a soft-BC
  primitive-variable DeepXDE PINN on CPU is markedly harder/less stable. Only Re=100 is reliably reproducible in this
  lane, so a multi-Re chip bar would be padding with weak/uncertain variants. Honesty over chip count.
- **Time-scrubber is N/A.** The problem is *steady* (time-independent). No `t` axis exists.
- **Single benchmark is correct.** Coupled (u,v,p), three residuals, pressure gauge, real digitized validation anchor —
  this is exactly the kind of case the playbook earmarks for one honest variant (cf. `bench-allencahn`).

### Field / param structure

The field is the 2-D map over `(x, y)` with three outputs `(u, v, p)`. Since `field_axes` defaults to `inputs` and
`inputs == ("x","y")`, **no `field_axes`, no `param_specs` are needed** — the case is already structurally a
single-variant non-parametric case. The only change is to add an explicit single-element `variants()` so the chip bar
shows a clean, named "Re=100 benchmark" label instead of the generic `default` fallback (mirrors how the single
Allen-Cahn case reads, and gives the App a meaningful chip + bilingual note). Everything else in the existing case —
the SOTA method (multi-output loss-weighting + L-BFGS), the regularized lid, the pressure gauge, the Ghia
`extra_metrics` — is preserved untouched.

## Analytic anchor

**There is no closed-form / MMS solution.** The validation anchor is the **Ghia, Ghia & Shin (1982)** Re=100
benchmark (digitized centerline velocities), already embedded in the case as `GHIA_Y/GHIA_U` (u along the vertical
centerline x=0.5) and `GHIA_X/GHIA_V` (v along the horizontal centerline y=0.5). This is a *numerical/FEM-class
reference*, not an analytic family.

- **Substitution-proof status:** N/A by construction — no exact solution exists to substitute. The reported metric is
  not a closed-form relative-L2; it is the relative-L2 of the PINN velocity vs the Ghia digitized centerlines, computed
  in `extra_metrics`:
  `l2_relative = 0.5 * (||u_PINN(0.5,·) − GHIA_U|| / ||GHIA_U||  +  ||v_PINN(·,0.5) − GHIA_V|| / ||GHIA_V||)`.
- **Honesty label:** `real_or_synthetic = "synthetic-illustrative"` (reduced-fidelity CPU numerical solution, not
  experimental data, not spectral-fidelity). Keep `validation_anchor = "benchmark-ghia"`.

Reference (DOI-backed): U. Ghia, K. N. Ghia, C. T. Shin, *High-Re Solutions for Incompressible Flow Using the
Navier–Stokes Equations and a Multigrid Method*, J. Comput. Phys. 48 (1982) 387–411.
DOI: `10.1016/0021-9991(82)90058-4`.

## Exact case `.py` edits (orchestrator applies)

The case is already correct apart from one additive change. **Do NOT change `inputs`, `outputs`, `domain`, `grid`,
`build()`, `pde`, the BCs, the pressure gauge, `train`, or `extra_metrics`.** The method/engine stay exactly as is.

1. **Import `Variant`** (the `ParamSpec` is not needed — single, non-parametric):

```python
from .base import CaseSpec, Variant
```

2. **Add a single explicit `variants()`** (place it after `CASE` / alongside `build`), so the chip bar shows a named
   benchmark instead of the generic `default`:

```python
def variants() -> list[Variant]:
    """Single honest benchmark: the cavity has no closed-form solution for any Re, and only Re=100 is reliably
    reproducible in the CPU primitive-variable lane (validated against the Ghia 1982 centerlines). A Reynolds
    sweep would fabricate regimes with no verifiable anchor, so this case ships one rigorous variant."""
    return [
        Variant(
            "re100",
            "Re = 100 (Ghia benchmark)",
            "Re = 100 (benchmark de Ghia)",
            {},  # non-parametric: no network-input knob; the field is the (u,v,p) map over (x,y)
            "Primary vortex + corner eddies; u,v on the centerlines validated vs Ghia 1982.",
            "Vórtice primario + remolinos de esquina; u,v en las líneas centrales validados vs Ghia 1982.",
        ),
    ]
```

That is the **only** edit. `field_axes` stays unset (defaults to `("x","y")`), `param_specs` stays empty, and
`base.__post_init__` passes because `param_axes = inputs − axes = {} == spec_keys = {}`.

## Bake recipe

Keep the existing SOTA recipe in the case (already encoded in `CASE.train`); no parametric domain, so it is cheaper
than the parametric cases.

- **Net:** FNN `[2, 64, 64, 64, 64, 64, 3]`, activation `tanh`, Glorot uniform (as in `build()`).
- **Optimizer:** Adam, `lr = 1e-3`, `adam = 20000` iters, then **one** final **L-BFGS** polish (no RAR — the steady
  cavity has no moving front to chase; RAR buys little here and adds cost).
- **Sampling:** `num_domain = 2601`, `num_boundary = 400`, `num_test = 10000`.
- **Loss weights:** `[1, 1, 1, 10, 10, 10, 10, 10]` — the three PDE residuals at weight 1, the five BC/gauge terms
  up-weighted x10 (lid-u, lid-v, wall-u, wall-v, pressure-gauge).
- **ONNX:** single shared net `(x,y) -> (u,v,p)`, `input_dim = 2`; parity max-abs expected ~1e-6..1e-7 (pure tanh FNN,
  no exotic transform).
- **Smoke first:** `--quick` (~30 s) to catch a structural bug, then the real bake.
- **Expected band (honest):** primary vortex + faint corner eddies; relative-L2 vs the Ghia u/v centerlines **a few
  percent** (roughly 3–8%) on the CPU lane — competent and qualitatively correct, *not* spectral. Label
  synthetic-illustrative. A GPU / stream-function-vorticity lane would tighten it; cite, do not claim.

Bake-speed note: single non-parametric (x,y) domain with a `[2,64×5,3]` net + one L-BFGS is one of the lighter bakes;
safe to pair with a tiny case, not with another heavy parametric bake (torch core oversubscription).

## Registry line (frontend `registry.tsx`)

```
"bench-navier-cavity": NavierCavityContext,
```

(and the matching `import { NavierCavityContext } from "./NavierCavityContext";`). The orchestrator owns this edit.

## Risks / fallback

- **Convergence at the lid corners.** The regularized lid `16 x^2 (1-x)^2` already tames the corner singularity; if
  loss stalls, raise BC/gauge weights or `adam` iters before touching the net — do **not** add RAR (no front to chase).
- **Pressure gauge drift.** Pressure is only defined up to a constant; the single-point pin `p(0,0)=0` plus the x10
  gauge weight handles this. If `p` looks globally offset, confirm the gauge term is in the loss weights (it is, the
  8th entry). The Ghia metric is velocity-only, so a pressure constant offset does not corrupt the reported L2.
- **Accuracy honesty.** If the CPU bake lands above ~8% rel-L2, keep the honest label and note it in the band — do not
  silently re-anchor or relabel. Fallback: if even Re=100 is shaky, the case still ships as the single benchmark with
  the achieved (honestly reported) error; never widen to a Re sweep to "look richer".
- **No fabricated variants.** Explicitly do not add Re=400/1000 chips unless a separate, individually-validated GPU
  bake is produced for each — out of scope for this CPU migration.
```
