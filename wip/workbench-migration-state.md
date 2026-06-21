# Workbench migration — state & playbook

Migrating every case to the SimLab-style **workbench** (ADR-0016 §9): a variant bar + four sub-tabs
(**Field / Live / Charts / Context**) + a deep bilingual Context, driven by a `manifest/v2` (per-variant field
traces + one shared ONNX). The web shell, components and pipeline are done; this tracks the per-case migration.

## The proven recipe (use this)

- **Parametric where a closed form exists.** Make the physical knob a *network input* (in `inputs` + `param_specs`,
  NOT in `field_axes`). One trained net + one ONNX → the Live tab sweeps the knob continuously. Bake ≥6 variants.
- **Hard-constraint output transform** that bakes IC/BC exactly (pure tensor ops → survives ONNX; parity ~1e-7).
  When the transform already encodes the solution's structure at `t=0` (a baseline `g`), the net only learns the
  *interior correction* → converges fast and is accurate even at sharp extremes (burgers ν=0.02 → L2 0.08%).
- **Analytic / MMS anchor** in `solution=analytic` → the per-variant relative-L2 is the true error.
- **Geometry:** `dde.geometry.Hypercube([...lo],[...hi])` over (field axes + param), `dde.data.PDE(..., solution=analytic)`.
  For soft-BC time cases use `GeometryXTime` + `dde.icbc` (ocean).
- **Time-scrubber pattern** for `(x,y,t)` cases: `field_axes=(x,y)`, the swept param is **t**, variants = time
  snapshots, Live = a time scrubber (ocean-transport).
- **Single honest variant** where no meaningful parametric family exists (stiff dataset-anchored; the symmetric
  Allen-Cahn front is *stationary*) — never faked into regimes (ADR-0016 §9.A).

### Bake-speed lessons (important)
- **Net size dominates.** `[*,96,96,96,96,96,1]` + RAR + **two** L-BFGS calls on a parametric domain is ~50 min.
  Prefer `[*,48–64,×4,1]`; keep RAR to ≤4 rounds; **one** final L-BFGS (move it out of the RAR loop).
- **L-BFGS runs to its maxiter** (~15k) even once converged (loss ~1e-6) — the long tail. Consider capping
  `model.train(..., maxiter=...)` for the final polish on big nets.
- **Do not run two heavy bakes in parallel** — torch oversubscribes the cores and *both* crawl. One at a time, or
  pair one heavy + one tiny (flotation).
- Always `--quick` smoke-test first (~30 s) to catch a structural bug before the real bake.

## Status

**DONE + committed + screenshot-verified (0 console errors, KaTeX renders, heatmaps correct):**

| Case | Param | Variants | L2 (max) | Notes |
|------|-------|----------|----------|-------|
| bench-poisson2d | source mode `k` | 6 | <0.2% | MMS, hard BC |
| bench-heat1d | diffusivity `α` | 6 | <0.15% | hard IC+BC |
| bench-wave1d | speed `c`∈[0.5,2] | 6 | <0.32% | SIREN, hard IC×2+BC; needed the bigger net |
| bench-burgers1d | viscosity `ν`∈[0.02,0.08] | 6 | 1.2% | traveling-shock tanh family, hard-constraint + RAR |
| bench-allencahn | — (single) | 1 | 0.41% | stiff, stationary front → single honest benchmark; RAR (4 rounds) |
| mine-flotation-kinetics | `k` is a field axis | 1 | 0.08% | full-family map C(k,t); hard IC |

**In flight:** `poll-ocean-transport` — rewritten as the advected-diffused Gaussian (time-scrubber, 6 t-snapshots),
baking. The old gyre-MMS (decaying sin) was visually dead; replaced with a genuinely moving/spreading exact solution.

## Remaining (12) — parametrization notes

- **ind-helmholtz** — parametric wavenumber `n` is HARD (spectral bias across a frequency range for one net). Either
  keep narrow `n∈[2,4]` (accept ~1e-1, honestly labeled) or ship the fixed-`n` Fourier-feature showcase single-variant.
- **bench-navier-cavity** — lid-driven cavity; parametric Reynolds has no closed form → single benchmark, or a
  numerically-referenced family. Coupled u-v-p; heavy.
- **bench-darcy-operator** — FNO operator: variants = held-out test samples (already a different lane).
- **ind-heat2d-inverse** — inverse: variants could be noise levels or true-parameter values.
- **env-soil-heat-real** — REAL data: single validated-real variant.
- **poll-source-uq-bpinn** — UQ: variants = the [mean, std] views or noise levels.
- **poll-soil-barrier** — FBPINN kink; parametric in the diffusion contrast is possible but the kink is already
  ~2e-1 at fixed contrast → likely single-variant.
- **poll-tailings-seepage** — Richards/Gardner seepage; check for a parametric closed form (suction parameter).
- **mine-heap-leach-rt** — reactive transport; parametric Péclet/Damköhler if an MMS exists, else single.
- **mine-thickener-settling** — Bürger–Concha; parametric in initial concentration or settling parameter.
- **mine-comminution-pbe** — population balance; parametric in a breakage-rate constant (like flotation → clean!).
- **ctrl-zero-source** — validation/control case; likely simple + fast, parametric if it has a knob.

Each remaining case = `--quick` smoke → real bake → deep bilingual Context → register in `registry.tsx` → add to
`index.json` → `npm run build` → commit. Index is regenerated from the migrated list (see the snippet used in commits).

## Verification harness
`CAOS_MANAGE/tools/visual-verify/pinnlab-check.mjs` — start `npm run preview` (serves dist on :4173), run the script;
it walks the App workbench + sub-tabs + all content pages and reports console/page errors + screenshots. 0 errors = green.

## Do NOT
- Merge develop→main until a critical mass of cases is migrated (don't regress the live site from N thin cases to a
  handful of deep ones without intent). main is the live `pinnlab.fasl-work.com`.
- Run `Set-Content -Encoding utf8` on repo files (BOM). Use the Write/Edit tools.
