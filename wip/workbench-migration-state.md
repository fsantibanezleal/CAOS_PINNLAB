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
- **RAR can DE-stabilise stiff operators.** On `mine-thickener-settling` (degenerate Bürger-Concha diffusion) RAR
  chased pathological residual spikes and the L2 *climbed* (0.11 → 0.32). Fix: drop RAR, widen the sharp front
  (`W` 0.06 → 0.10) and lean on Adam(24k) → L-BFGS → L2 0.4%. RAR is for clean sharp fronts (burgers), not stiff
  degenerate diffusion.
- **`dde.icbc.IC` needs a `GeometryXTime`.** On a parametric `Hypercube([z,t,param])` it throws
  (`'Hypercube' has no attribute 'on_initial'`). Use a single `DirichletBC(geom, u*, on_boundary)` over the whole
  cube boundary — the `t=0` face is a boundary face, so it enforces the IC too (u* is exact there).

## Status — 10 committed + verified, 1 baking, 8 designed

**DONE + committed + screenshot-verified (0 console errors, KaTeX renders, heatmaps correct):**

| Case | Kind | Variants | L2 (max) | Notes |
|------|------|----------|----------|-------|
| bench-poisson2d | parametric `k` | 6 | <0.2% | MMS, hard BC |
| bench-heat1d | parametric `α` | 6 | <0.15% | hard IC+BC |
| bench-wave1d | parametric `c`∈[0.5,2] | 6 | <0.32% | SIREN, hard IC×2+BC; needed the bigger net |
| bench-burgers1d | parametric `ν`∈[0.02,0.08] | 6 | 1.2% | traveling-shock tanh family, hard-constraint + RAR |
| bench-allencahn | single | 1 | 0.41% | stiff, stationary front → single honest benchmark; RAR (4 rounds) |
| mine-flotation-kinetics | single (k is field axis) | 1 | 0.08% | full-family map C(k,t); hard IC |
| poll-ocean-transport | time-scrubber `t` | 6 | 0.19% | advected-diffused Gaussian (exact, moves+spreads); Live = time scrubber |
| ctrl-zero-source | parametric `a`∈[0,1] | 6 | a≥0.2:<0.15% | MMS two-mode family containing the degenerate a=0 control |
| poll-tailings-seepage | parametric `α`∈[1,2.5] | 6 | 0.26% | Kirchhoff exact family (sympy-verified, ψ<0 strict); Richards/Gardner |
| poll-soil-barrier | single | 1 | 0.19% | FBPINN kink — honestly high (~2e-1), CPU 2-channel lane, labeled |
| ind-heat2d-inverse | single (inverse) | 1 | k:4.0% | recover k(x,y) from 100 sparse noisy sensors; multi-output PFNN |
| mine-thickener-settling | parametric `R`∈[0.3,0.9] | 6 | 0.41% | Bürger-Concha settling; exact descending tanh; no RAR, W widened to 0.10 |
| mine-heap-leach-rt | time-scrubber `t` | 6 | cA 1e-4 / cB 2e-4 | 2-species advection-diffusion-reaction MMS; multi-output; ocean pattern, baked clean |

**13 committed.** Remaining to bake (6): mine-comminution-pbe (parametric g), bench-darcy-operator (discrete FNO
samples — neuraloperator lane), ind-helmholtz (single), bench-navier-cavity (single), env-soil-heat-real (single REAL),
poll-source-uq-bpinn (single UQ). All have committed Contexts + design notes in wip/case-designs/.

**Designed (Contexts committed + design notes in wip/case-designs/, ready to bake):** the workflow
`wip/migrate-remaining-cases.workflow.mjs` (12 parallel agents) produced honest, adversarially-anchor-checked designs.
Remaining to bake (8): mine-thickener-settling (parametric R), mine-comminution-pbe (parametric g),
mine-heap-leach-rt (time-scrubber t), bench-darcy-operator (discrete FNO samples), ind-helmholtz (single),
bench-navier-cavity (single), env-soil-heat-real (single REAL), poll-source-uq-bpinn (single UQ). Each design note
has the exact .py edits + the analytic anchor + the bake recipe — baking is mechanical (apply → --quick → bake →
register in registry.tsx + index → build → screenshot-verify → commit).

### Bake-order tip for continuation
Single-variant cases are the fastest to APPLY (just add `variants()` + `field_axes`); their bake cost = their existing
recipe. The parametric ones (thickener, comminution, heap-leach) need the .py rewrites in their design notes. Bake one
at a time (CPU contention) — pair a heavy one with nothing, or smoke-test (`--quick`) others meanwhile.

## Remaining design notes — parametrization summary (superseded by wip/case-designs/)

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
