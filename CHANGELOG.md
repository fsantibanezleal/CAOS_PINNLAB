# Changelog

All notable changes to **PINN-Lab**. Format: `X.XX.XXX` (display), see `pinnlab.__version__`. Keep `0.x` while on
synthetic/benchmark data. Tag every release.

## [0.20.006] (2026-07-09) UI: Highlights + Domain as compact dropdowns (#24)

The Highlights and Domain selectors in the left rail are now compact dropdowns instead of expanded chip/button lists,
freeing vertical space (the Case list + Regime + View switch all sit higher). Full re-validation via the new Domain
dropdown: all 20 cases, light + dark, 0 console errors.

## [0.20.005] (2026-07-09) UI polish (#24): equation top-strip (never cut) + both-mode field probe

Addressing the flagged UI issues:
- **Right column -> top context strip**: the governing equation now lives in a full-width row above the stage, so
  it is NEVER cut (the whole Navier-Stokes equation displays); method / engine / L2 / ONNX + the expected band sit
  beside it. The visualization stage is wider (no 320px right rail).
- **Both-mode field probe**: the two side graphs FOLLOW the cursor by default; a single click PINS them at a spot;
  a double click RELEASES back to follow. Verified: follows-on-hover, pins-on-click, releases-on-double-click.
- Full re-validation: all 20 cases, light + dark, 0 console errors.

## [0.20.004] (2026-07-09) darcy FNO-vs-FD comparison + full validation; honest call on wave1d (issue #25)

- **darcy-operator**: a real comparison with ZERO training - the Fourier Neural Operator's one-pass prediction vs the
  classical finite-difference reference (both already baked), relative-L2 = 2.5% on a held-out permeability field.
- **Honest call**: a fair wave1d test (BOTH SIREN and tanh trained fresh at the hard regime c=2) lands them both at
  ~11% - SIREN's spectral-bias edge is not visible at a feasible CPU budget. Rather than ship a fabricated contrast,
  wave1d keeps its honest standard-vs-PINN comparison. The genuinely dramatic naive-fails (Helmholtz, allencahn,
  heat2d-inverse) are fundamental and regime-independent; those are the real ones.
- **Full validation**: all 20 cases render (17 with a Compare view, all with a Field view), light + dark, 0 console errors.

## [0.20.003] (2026-07-09) the DATA-DRIVEN contrast: heat2d-inverse pure-physics vs physics+data (issue #25)

Answering "is it data-driven?" with real computed content. For the 2D inverse conductivity problem:
- **pure physics (NO sensor data)** leaves k(x,y) underdetermined -> the recovered field is nowhere near k*
  (L2 = 356% vs the analytic standard);
- **physics + ~100 sparse T sensors** recovers it (L2 = 4.0%).
Shown as standard | pure-physics | physics+data + error maps. The DATA is what makes the inverse solvable - the
concrete demonstration of the hybrid data+physics PINN, computed, not asserted.

## [0.20.002] (2026-07-09) two more REAL naive-vs-fix ladders: allencahn (collapse) + soil-barrier (issue #25)

Fast training path (~1-3 min/case, no L-BFGS) so the naive contrast is real without the hour-long runs.
- **allencahn**: the textbook PINN failure, computed and shown. The naive SOFT PINN collapses to a metastable state
  and smears the sharp +/-1 transition layers (L2 = 95% vs the spectral-reference standard); the hard-constraint +
  RAR fix tracks them (L2 = 0.4%). standard | naive | adapted + error maps.
- **soil-barrier**: the single-domain naive vs the FBPINN domain-decomposition, both against the MMS standard. Honest
  outcome shown as-is: on the CPU lane both land near 19% (the FBPINN edge here is subtle, at the kink) - not dressed up.
- Generic `build_naive_lane.py` adds a naive lane to any case with `build_naive()` + an existing comparison.

## [0.20.001] (2026-07-09) Helmholtz flagship: the FULL real ladder (naive vs Fourier) + diagnostics (issue #25)

The reference case computed end-to-end, all REAL, nothing staged:
- **Compare view**: standard (classical FDM solve) | **naive plain-tanh PINN (120.8% error, visibly broken by
  spectral bias)** | **Fourier-feature PINN (9.3%, matches)** | analytic (MMS), plus the two error maps. You can SEE
  the naive lane fail to resolve the high-wavenumber pattern while the Fourier lane nails it.
- **Diagnostics view**: the wavenumber sweep (naive: 3% at n=1 -> ~100% at n>=2; Fourier stays low) and the radial
  spectral energy - the WHY behind the failure. Real numbers from real training runs.
- Naive ONNX exported for a future Live on/off toggle.

## [0.20.000] (2026-07-09) COMPARE view: the standard PDE solution vs the PINN, on real baked data (issue #25)

The offline pipeline now COMPUTES the comparison the docs describe (it did not before: each case baked one thin PINN
field, so there was nothing rich to show). First deliverable: every case with a closed-form reference gets a real
"standard vs PINN" comparison, baked with NO retraining (the PINN field is already baked; the standard is analytic).

- **Compare view (default)** per case: the STANDARD solution vs the adapted PINN vs their pointwise error map, with
  a shared hover probe that reads every lane at a point, and the real baked relative-L2 headline. 14 cases so far.
- Honest, not staged: e.g. the domain-decomposition (FBPINN) barrier case shows a real 19.1% error concentrated at
  the barrier faces; the easy forward cases show <1% (the PINN genuinely matches the standard there).
- `CompareKit` + `DiagnosticsKit` are manifest-driven, so they light up for every case as its pipeline computes a
  comparison. Manifest gains `comparison` + `diagnostics` blocks.
- Next: the Helmholtz flagship full ladder (naive tanh vs Fourier + wavenumber sweep + spectrum) is computing now.

## [0.19.000] (2026-07-09) full-width App shell (mirrors Lidar3D): left rail + stage + right stats

The App content was boxed into a narrow centered column while the header and footer used the full width. The App is
now a FULL-WIDTH workbench mirroring CAOS_RES_Lidar3D's `.work` shell, so the content finally uses the whole viewport.

- **`main.content` is full-width**; the App route (`.content-app`) fills the viewport, the doc routes (`.content-doc`)
  keep a centered, readable 1280px column. No more double `max-width` cap on the App (was 1280 then 1180).
- **Three-column App (`.pl-work` grid: 300px / 1fr / 320px, full viewport height):**
  - LEFT rail: the case selection (Highlights with the REAL case starred, Domain chips, a scrollable Case list),
    then the Regime chips + the note, then the View switch (Field / Live / Charts / Context).
  - CENTER stage: the active view, full width. The field heatmap, streamlines and the inverse panels get much more room.
  - RIGHT rail: the governing equation + the method / engine / L2 / ONNX-parity metrics + the expected band.
  - Every control from the old stacked layout is preserved, just re-slotted into the three columns; nothing removed.
- **Fixed a latent crash** exposed by the new layout: switching to a VectorField case (navier-cavity) could pair the
  new manifest with the PREVIOUS case's trace for one render and crash the whole App. The trace is now reset the
  moment the case changes, plus a defensive guard in VectorFieldKit.
- Verified: all 20 cases in the new shell, light + dark, every view, 0 console errors; doc pages still centered.

## [0.18.003] (2026-07-09) both modes together on every evolution case: PLAY + probe, and a content sweep

Correction of 0.18.001/0.18.002, which had DELETED the play button when adding the probe. Both now coexist:

- **Unified field view (`FieldView`):** every scalar-field case now has, at once,
  - the **play button** (animate the time evolution forward, paused by default, one pass then stop, pauses on a
    hidden tab), for the 10 cases with a real time axis (allencahn, burgers1d, heat1d, wave1d, soil-heat-real,
    comminution, flotation, thickener, soil-barrier, tailings);
  - the **click-to-pin probe**: click the map to lock a location; the crosshair, value read-out and the two
    line-cut graphs follow the pin. Play and the pin share ONE time-cursor (play advances it, click/scrub sets it).
  - **real axis labels** on the map and on both graphs (a time axis is tagged "(time)"), and the spatial graph
    overlays **dashed = initial state (t=0)** against **solid = selected time**, with a legend.
  - Steady cases (darcy, poisson, helmholtz, zero-source, ocean, heap-leach) correctly show no play button (no time
    dimension), just the probe + two labelled cross-sections. Consistent behavior across all cases.
- **Inverse case (`heat2d-inverse`) redesigned:** recovered k (with the ~100 sparse sensor dots), true k* on the same
  color scale, the error |k-k*|, and measured T are shown SIDE BY SIDE, so it reads as an assimilation result, not a
  bare heatmap.
- **Content standards sweep (ADR-0067):** removed every em-dash / en-dash from the 20 case titles, the variant
  notes, the captions and hints, and the deep Context / Methodology / Implementation / Experiments prose (661
  occurrences across source + manifests). No em-dashes remain in the app content.
- Verified with a per-case live audit (all 20 cases, light + dark): play button present on the time cases, probe +
  two labelled graphs + legend on every field case, 4 panels + 100 sensor dots on the inverse, 0 console errors.

## [0.18.002] (2026-07-09) initial-vs-selected overlay on the field probe + cache-busting

- **Initial-vs-selected overlay:** in the space cross-section panel, the profile at the selected time (solid) is now
  overlaid with the **initial state t=0 (dashed)** — the initial-vs-selected comparison — combined with the value
  read-out + the two panels + the dimension labels, on every field case (`FieldView`).
- **Cache-busting:** every artifact fetch (index / manifest / trace / onnx) is stamped `?v=APP_VERSION`, so a new
  deploy never serves stale cached data (the data files are not content-hashed). Hard-refresh once to pick up the
  new JS shell; after that, data stays in sync with the build.

## [0.18.001] — 2026-07-09 — consistent probe + labelled dimensions (the space-time heatmap is the EVOLUTION)

Owner feedback: the play-button (animated) field cases had lost the value-at-cursor + the two cross-section graphs
that the plain heatmap cases (darcy, poisson) keep, and it was unclear whether a heatmap showed the evolution or a
static final state. Fixed by REPLACING the animation with the heatmap+probe on those cases and labelling dimensions:

- **Reverted the 4 animated field cases** (allencahn, burgers1d, wave1d, ocean-transport) to HeatmapKit — so every
  field case again shows the crosshair + value-at-cursor read-out + the two line-cut profiles (u along each axis).
  No play button; consistent with darcy/poisson. (The double pendulum stays a trajectory — it has no field.)
- **Axis-dimension labels on the heatmap** (FieldView): the horizontal + vertical axis names are drawn, and a time
  axis is marked "t — time →".
- **A dimension caption** states exactly what you are seeing: for a time-dependent PDE, "SPACE–TIME field u(x,t):
  the whole evolution in one image … reading up the time axis shows how it evolves. It is NOT a static final
  state."; for a steady field, "no time dimension"; for a param-time case, "spatial snapshot at t=… (use the regime
  chips to step through time)."

## [0.18.000] — 2026-07-09 — the HYBRID data+physics rung (the practical winner), cited + honestly scoped

Answering "why aren't we featuring the data-driven PINN?" — from a verified deep-research pass on hybrid CFD PINNs
(`wip/web-review/hybrid-pinn-research-2026-07-09.md`). The Methodology ladder now foregrounds the data-hybrid rung:

- **Honest-scope panel** updated with the verified facts: the current generation of PINNs has NOT beaten the finite
  element method [Grossmann 2024]; pure-physics PINNs hit ~100% error as the regime hardens [Krishnapriyan 2021] and
  stall on complex CFD geometry — and the practical fix is the HYBRID data+physics PINN. Cites Grossmann 2024,
  Krishnapriyan 2021, Hidden Fluid Mechanics (Raissi 2020).
- **"Inverse & UQ" reframed to "Hybrid data + physics … (where PINNs win)"** with the SCAFFOLD mechanism (the
  data term fixes an optimization pathology — PDE-residual gradients dominate, NTK K_rr ≫ K_uu — anchoring the net to
  the realizable solution among the infinitely many satisfying the PDE), the seminal HFM example, and the honest
  scope (data ASSIMILATION / reconstruction, not prediction from nothing; the "beats classical CFD" claim is NOT
  supported). heat2d-inverse / soil-heat-real / uq-bpinn named as exactly this rung.
- **New operator rung** — physics-informed DeepONet [Wang 2021], with the OOD-distribution-bound limit. 4 new refs.

Next: a dedicated hybrid-CFD flow-reconstruction case (sparse sensors + NS residual) with a pure-physics baseline
that visibly stalls (the S-bend contrast), ONNX-exported.

## [0.17.000] — 2026-07-09 — Highlights strip: the REAL-data + data-hybrid cases stop being lost

The one case trained on REAL measured data (`env-soil-heat-real`) was buried as one card among 20. The App now
opens with a **Highlights strip** above the domain groups, surfacing the standouts that pure-forward-physics cases
drown out — **REAL data first, starred + green** (`soil-heat-real`, NOAA soil temperatures), then the hybrid
data+physics (`heat2d-inverse`), data+uncertainty (`source-uq-bpinn`), and chaotic-dynamics (`double-pendulum`)
cases. Clicking a highlight jumps to it in its domain. This elevates exactly where PINNs genuinely win
(data-hybrid / real-data), not just the synthetic forward benchmarks.

## [0.16.000] — 2026-07-09 — Methodology as a classical → SOTA → beyond-SOTA ladder (cited)

Driven by a verified deep-research pass (117 agents, 21 adversarially-verified claims;
`wip/web-review/sota-research-2026-06-26.md`). The Methodology page becomes an explicit ladder, honestly framed:

- **Honest-scope panel** — for a single well-posed FORWARD solve a tuned classical FEM/FVM/spectral solver is
  usually faster/more accurate (community consensus, NOT overclaimed — the exact "PINNs win at X" framing was
  refuted against its cited source, so it is stated as the community view). The two VERIFIED hard limits: standard
  PINNs fail on chaotic/turbulent regimes (only causal training cracked Lorenz/KS/NS, over a finite pre-Lyapunov
  window) and struggle on large/multi-scale domains. Cites Krishnapriyan 2021 + Wang 2024.
- **Per-family "→ SOTA frontier + candidate-novel proposal + honest limit"** notes with DOIs: RAD/RAR-D, causal
  weighting (+ PINN-Lab's Lyapunov-horizon leave-time as the candidate discipline), Residual-Based Attention (RBA),
  PirateNets + PIKANs, XPINN/FBPINN, SPINN.
- **New method entry: Dynamical systems & chaos (the Lyapunov horizon)** — the double pendulum's ode-residual +
  soft-IC + RK45 anchor + leave-time. 15 references, all with DOIs. Persisted research reports committed.

## [0.15.000] — 2026-06-26 — App reorganized: domain groups + functionality cards (not 20 flat tabs)

The App section is restructured from a single flat 20-tab strip into a legible three-level navigation:

- **Level 1 — scenario domains.** A group nav by physics `category` (Canonical benchmarks 8 · Mining 4 · Pollution 5
  · Industrial 2 · Control 1), each with a count + a one-line "what this domain is" intro.
- **Level 2 — functionality cards.** Within a domain, a grid of case cards; each shows the case + its
  **functionality badges**: the view-kit (Time evolution / Trajectory / Vector flow / Inverse overlay / …), the
  data-honesty label (synthetic / synthetic-illustrative / **REAL data**), and the SOTA method. The groupings of
  functionalities are now legible at a glance.
- **Level 3 — the workbench** (Field / Live / Charts / Context) — unchanged; lands on the first case (ADR-0016).

`index.json` is enriched with per-case `system_type` / `view_kit` / `method` / `real_or_synthetic`
(`build_index` + `contract.ts`) so the cards render badges without loading every manifest.

## [0.14.000] — 2026-06-26 — InverseOverlayKit: the recovered field + its sparse evidence

- **InverseOverlayKit** (`ind-heat2d-inverse`, inverse-assim): the recovered conductivity field `k(x,y)` with the
  **100 sparse noisy T sensors overlaid as dots** — the evidence the PDE prior interpolates between. Toggle recovered
  `k` / true `k*` / `|k−k*|` / `T`; hover read-out; k-L2 = 4.0% (worst near the boundary where `|∇T|` is small).
- The case now bakes the sensor positions + measured T into `trace.inverse` and the true `k*` as a field (the case
  already trained fine — a low-risk re-bake, not new physics). Static (no animation).

## [0.13.000] — 2026-06-26 — vector-flow + UQ kits (no retraining; static, no bomb)

Two more `system_type → view_kit` mappings, both from already-baked artifacts (no retraining), both static (no
animation, no autoplay):

- **VectorFieldKit** (`bench-navier-cavity`, vector-flow): RK4 **streamlines** + optional quiver over a scalar
  background (speed / pressure / vorticity) + hover read-out of (u, v, |U|, p). Reveals the recirculating lid-driven
  cavity vortex that a per-scalar heatmap hid.
- **UQBandKit** (`poll-source-uq-bpinn`, uq-bayesian): mean `c(x)` with a filled **±1σ / ±2σ band** at a paused time
  slider + the σ(x) magnitude curve + the calibration coverage (@2σ = 99.97%, K=5 ensemble). Band at true scale —
  honest about a well-calibrated (thin) uncertainty; the σ curve shows where it concentrates.

## [0.12.000] — 2026-06-26 — the dynamical-systems category: an animated chaotic double pendulum + leaner field UX

The 20th case is the **double pendulum** — the `ode-dynamical` flagship and the first case with no spatial field:
the PINN maps `t → (θ₁, θ₂)` and the App animates the swinging linkage. Plus a UX cut: animation is kept only where
the motion teaches something.

### Added — `dyn-double-pendulum` + `TrajectoryAnimationKit`
- New ODE/trajectory path in the pipeline (a trajectory is a 1-D-in-`t` multi-output trace; the 1-D trace keeps up
  to 601 samples; `web_drivable=False` → precompute lane). Soft-IC IVP (a `t²` hard-IC was rejected: it kills the
  gradient near `t=0`). Engine: DeepXDE, tanh, Adam→L-BFGS; ONNX parity ~1.6e-6.
- **RK45 anchor** (`rtol=atol=1e-10`) + a twin-IC trajectory baked alongside the PINN. Honest headline:
  **leave-time = 1.99 s** (the PINN tracks to ~0.02 rad then diverges; L2 = 9.3% over 3 s) — chaos past the Lyapunov
  horizon, shown not hidden (the PINN arm turns red after the leave-time; the butterfly panel diverges).
- `TrajectoryAnimationKit`: animated linkage (PINN ghost over RK45) + θ₁–θ₂ phase portrait + butterfly |Δθ| (semilog)
  + angles-vs-`t` with the leave-time marked. Deep bilingual Context + per-case doc.

### Changed — animation only where it aports
- Per the owner's call: **4 cases stay animated** (allencahn front, burgers1d shock, wave1d oscillation,
  ocean-transport plume); the other **9 revert to the static heatmap** (the x–t carpet already shows their
  evolution — the scrub added little). Kept kits lead with a clear "Press ▶ to…" caption.

## [0.11.001] — 2026-06-26 — animation is PAUSED by default (no autoplay; kill the compute bomb)

Hotfix. The animated kits autoplayed on load with an infinite loop — a `requestAnimationFrame` replay that pinned
a CPU core continuously and never stopped, even unattended. Now:

- **No autoplay.** Every animated kit (TimeEvolution / SpatioTemporal / Trajectory) starts **paused** on the first
  (meaningful) frame; nothing computes until the user presses Play.
- **No infinite loop.** A Play runs ONCE through the timeline and stops at the end (looping is opt-in per case);
  pressing Play at the end restarts from the start.
- **Hidden-tab safety.** Animation stops the instant the browser tab is hidden (`visibilitychange` → pause), so
  there is no background CPU when the page is not in view.

## [0.11.000] — 2026-06-26 — per-category view kits: the catalogue stops looking identical (animation)

Cases no longer all render as one static heatmap. ADR-0063 introduces an orthogonal **`system_type` →
`view_kit`** axis: the Field tab picks a render kit by *kind of system*, and the time-evolving cases now
**animate**. Driven by a 5-agent deep-research pass (`wip/app-redesign/`). Phase 0 + Phase 1 of a 5-phase plan.

### Added — the view-kit architecture (ADR-0063)
- **`system_type` + `view_kit`** on `CaseSpec` / the manifest / `contract.ts`; `FieldView` refactored into
  **`HeatmapKit`** (the default, pixel-identical), one of several kits selected via `kits/registry.ts`. Unknown/
  absent `view_kit` falls back to `HeatmapKit` — zero regression for un-migrated cases.
- **`TimeEvolutionKit`** (11 cases): plays `u(space)` forward over the baked time axis with a dashed initial-frame
  ghost and a y-scale locked across the run; the `[space,t]` carpet is a click-to-seek bar. **No retraining.**
- **`SpatioTemporalKit`** (2 cases: ocean-transport, heap-leach): animates the 2-D field over its time-snapshot
  variants with a colorbar fixed across frames + hover read-out. **No retraining.**
- Shared primitives: `useAnimator` (rAF, wall-clock speed, honest stop-at-end), `Transport` (play/scrub/speed/
  loop), `HeatCanvas`, `LineProfile`; `viridis` extracted to `lib/colormap.ts`.
- Verified with Playwright (preview): all 13 autoplay + pause works + **0 console/page errors**; the 6 steady/
  vector/inverse cases unchanged (their kits arrive in Phases 2–4: dynamical-systems/double-pendulum, vector-flow,
  UQ, mode-shape, inverse).

### Changed — honesty remediation (audited against the engine)
- Anchors described as **analytic/numerical**, never "FEM" — no case uses a finite-element anchor (kept the honest
  "not a replacement for FEM/FVM"). **neuraloperator** is the documented *reference*, not a dependency (the Darcy
  case ships a self-contained FNO). Case **count = 19** everywhere (was ~20 / 18).

## [0.10.000] — 2026-06-22 — the SimLab-style per-case workbench, all 19 cases, deployed

The defining release: every case is now a **workbench**, the whole catalogue is migrated, and the app is live at
**https://pinnlab.fasl-work.com**.

### Added — per-case workbench (manifest/v2) across all 19 cases
- **A workbench per case** mirroring CAOS_SIMLAB (ADR-0016 §9): a **variant bar** (parameter regimes as chips + lane
  badge + bilingual note) + **four sub-tabs — Field / Live / Charts / Context**. The **Field** view is an interactive
  viridis heatmap with a value read-out at the cursor + colorbar + two line-cut profiles; **Live** re-evaluates the
  exported ONNX in-browser (onnxruntime-web) as you move the parameter slider; **Charts** is a clickable per-variant
  relative-L2 comparison; **Context** is a deep bilingual (EN/ES) write-up (the problem → components → formalization in
  KaTeX → scope → what each variant shows → how to read & use the viz).
- **manifest/v2** (`pinnlab.manifest/v2`): each case bakes one compact field trace **per variant**, with `param_specs`
  + `field_axes`; a **parametric** case makes its physical knob a *network input* so ONE trained net + ONE ONNX drives
  the whole family and the Live tab sweeps it continuously (stronger than SimLab's pre-simulated regimes).
- **Parametric families (6 variants, Live sweep):** poisson (source mode `k`), heat1d (diffusivity `α`), wave1d
  (speed `c`, SIREN), burgers1d (viscosity `ν`, traveling-shock + RAR), ctrl-zero-source (amplitude `a`, contains the
  degenerate `a=0` control), tailings-seepage (Gardner sorptive `α`, exact **Kirchhoff** family), thickener-settling
  (descent rate `R`, Bürger-Concha), comminution-pbe (grind rate `g`). **Time-scrubbers** (param = `t`):
  ocean-transport (advected-diffused Gaussian), heap-leach-rt (2-species reactive transport).
- **Single honest benchmarks** (no fabricated regimes — ADR-0016 §9.A): allencahn (stiff, stationary front),
  flotation-kinetics (full-family `C(k,t)` map), soil-barrier (FBPINN kink), helmholtz (high-wavenumber Fourier
  features), navier-cavity (Ghia Re=100), heat2d-inverse (recover `k(x,y)` from sparse sensors), source-uq-bpinn
  (deep-ensemble UQ), **env-soil-heat-real (REAL NOAA USCRN data** — recovered `α=0.30 mm²/s`, held-out RMSE 1.03 °C).
  darcy-operator ships a **discrete** family of 6 held-out FNO test samples (operator generalization 5.5 %).
- **Honest measured bands** per case (relative-L2 ≤0.4 % for the clean analytic families; ~10 % for the spectral-bias
  Helmholtz / ~17 % for the CPU Navier-cavity — labeled honestly, GPU lanes noted). ONNX parity `< 1e-4` everywhere.

### Added — deep content + docs
- Content pages rewritten to depth: Introduction (PINN loss in KaTeX), Implementation (5 sub-tabs of the build/web/
  design flows), Methodology (9 SOTA method families + DOIs), Experiments (per-category narrative). **Zero internal
  repo paths in the rendered UI.**
- `docs/` entry point (`docs/README.md`) + a real `guides/` set (getting-started, pipeline, web-app, add-a-case,
  interpreting-results) + `frameworks/` & `methods/` landings + a runnable `docs/frameworks/deepxde/example.py`.
- Root `README.md` rewritten for PINN-Lab (was the generic template boilerplate).

### Fixed — green CI + clean public artifacts
- Updated `tests/*` + `scripts/check_artifacts.py` to the **manifest/v2** schema (per-variant `trace`/`metrics`;
  parametric poisson `input_dim=3`).
- `strip_onnx_metadata()` (io/formats) wired into the ONNX export — the dynamo exporter embedded the local build path
  in graph metadata; stripped from all 19 `.onnx` (inference/parity unaffected) so no local-machine path ships.
- ruff clean (E702/F401/E731). CI (guards + test) is green.

## [0.07.000] — 2026-06-21

### Added — `bench-darcy-operator` (case #19, the operator-learning method family)
- A compact, self-contained **2D Fourier Neural Operator** (`model/fno.py`) learns the Darcy solution operator
  $\mathcal{G}: a(x)\mapsto u(x)$ for $-\nabla\!\cdot(a\nabla u)=1$, $u|_{\partial\Omega}=0$, over a family of two-value
  thresholded-GRF coefficient fields (the Li-et-al. benchmark). Dataset generated in-build via a seeded SciPy
  finite-difference solver (`datasets/darcy.py`, 256 train / 64 test). Spectral convolution in **real arithmetic** so
  the operator exports cleanly to ONNX (parity 1.7e-6).
- Measured (seed 42): **held-out test relative-L2 = 5.6 %** (genuine operator generalization over 64 unseen $a$),
  sample relative-L2 2.4 %. **lane = precompute** (a field-IO operator is not browser-coordinate-drivable; the App
  replays a representative baked result — 3 switchable fields: input $a$, FNO $u_{\text{pred}}$, FD $u_{\text{true}}$).
  `synthetic` (analytic-coefficient benchmark, FD numerical anchor). This closes the last unexercised SOTA family.
- **Engine: generic FIELD-IO hooks** (ADR-0057-compatible, like the prebuilt path) — a custom-engine case may train +
  export its OWN field-shaped ONNX in `build()` (train.py passes it through), and the lane gate gains a `web_drivable`
  term so a field-IO operator is honestly classified precompute regardless of ONNX size/speed.
- `scipy` pinned in `data-pipeline/requirements.txt`; `docs/cases/bench-darcy-operator.md`.

### Fixed (0.06.001) — web shell now complies with ADR-0016 + ADR-0011
- **English is the default language** for the app (ADR-0011) — removed the navigator-language auto-detect.
- Header carries the mandatory external icon-links (ADR-0016): **GitHub + personal site + portfolio**
  (`lib/links.ts`, lucide-react icons, separator before the Info/language/theme actions); added the **footer**
  (attribution + a CAOS research investigation + the 3 links + license + version). Mirrors the CAOS_SEISMIC shell.

## [0.06.000] — 2026-06-21

### Added — ⓘ "How it was built" architecture panel (implements ADR-0058)
- A header **ⓘ button** opens a tabbed modal that proves the app is real, at COMPLETE depth — **6 hand-authored
  theme-aware SVG diagrams** (every colour a CSS variable; fetched + inlined so they repaint with light/dark) paired
  with **bilingual EN/ES** explanations:
  - **Overview** — system map + the design/build lifecycle (what runs offline / precompute / web).
  - **Web app** — the static SPA + the App-page live-inference flow.
  - **Offline pipeline** — the 6 deterministic stages + the 2 data contracts.
  - **Train → ONNX → web** — the bridge + the parity guarantee.
  - **Live vs precompute** — the measured lane gate.
  - **Methods & honesty** — the 18 cases × SOTA methods + the honesty taxonomy.
- `ArchitectureModal.tsx` + `architecture-tabs.ts` + `arch.*` i18n + `public/svg/tech/*.svg`. Verified dark+light with
  `tools/visual-verify/render-svg.mjs` (per-SVG) and the in-app modal harness (all 6 tabs, both themes, zero JS errors).
- This is the binding **ADR-0058** standard (Veta/Circuita pattern) applied to this non-shell app — a product is not
  "done" without it.

## [0.05.000] — 2026-06-21

Uncertainty quantification — the last unexercised SOTA method family — plus the generic engine path that enables it.

### Added — `poll-source-uq-bpinn` (case #18, Bayesian PINN / deep ensemble)
- Trains **K=5 independently-initialized PINNs with per-member bagging** (deep ensembles ≈ approximate Bayesian
  inference) for 1D pollutant diffusion from **24 sparse noisy sensors**. The predictive **mean** tracks the analytic
  field (relative-L2 **1.2 %**) and the ensemble **std** is the epistemic uncertainty — small near sensors/walls,
  ~2.7× larger in data-sparse regions. **2σ calibration = 100 %** (well-calibrated, not overconfident).
- Exported as **one self-contained ONNX emitting `[mean, std]`** → lane **live** (101 KB, 3.8 ms, parity 2.4e-7).
- Honesty `synthetic-illustrative` (UQ demonstrator on a manufactured field).

### Added — engine: prebuilt-engine path
- `train.py` now supports **custom-engine cases** (`build()` returns `{"model", "input_dim", "prebuilt": True}`): the
  case trains its own net (a deep ensemble here; FNO operator nets later), so the generic Adam→L-BFGS→refine loop is
  skipped while ONNX export + parity + the lane gate still apply. `build(seed)` may optionally accept `quick=` (passed
  only to builds that declare it), so custom-engine cases get a cheap CI path.
- `docs/cases/poll-source-uq-bpinn.md`.

## [0.04.000] — 2026-06-21

The first REAL-data case. Everything prior validates against closed-form / reduced-model truth; this milestone adds a
case trained on, and validated against, real measured observations — plus the engine plumbing for it.

### Added — `env-soil-heat-real` (case #17, the flagship real-data inverse)
- Recovers **soil thermal diffusivity** from **NOAA USCRN** daily soil temperatures (station IL_Champaign_9_SW,
  2019–2021, depths 5/10/20/50/100 cm). The 1D heat equation $T_t=\alpha T_{zz}$ with the **5 cm + 100 cm sensors as
  real time-varying Dirichlet boundaries** and the diffusivity a **trainable scalar** (`dde.Variable`,
  `external_trainable_variables`; Adam-only so the generic L-BFGS recompile never drops it).
- **Out-of-sample validation**: the **10/20/50 cm sensors are held out** and scored in `evaluate` against the real
  interior temperatures. Measured (seed 42): recovered **α = 0.30 mm²/s** (textbook moist-mineral-soil range),
  held-out **RMSE = 1.05 °C** (10 cm 1.26 · 20 cm 1.06 · 50 cm 0.75), **relative-L2 = 6.9 %**, lane **live** (40 KB,
  0.7 ms, parity 1.4e-6). Honesty flag **`validated-real`** (new green "real data" tag, EN/ES).

### Added — engine + ingestion
- `datasets/uscrn_soil.py` — documented fetcher that vendors the real USCRN data offline
  (`data/reference/uscrn/soil_temp_il_champaign.json`, schema `pinnlab.dataset.uscrn/v1`), so training is reproducible
  and CI needs no network.
- **Data-fit validation mode** in `evaluate`: a case with `validation_anchor="real-data-holdout"` and no analytic
  anchor scores via `extra_metrics` (held-out interpolation of the baked field vs real observations) and reports the
  recovered physical parameter.
- `docs/cases/` (landing + `env-soil-heat-real.md`); management dossier `real-datasets.md` updated to record the
  shipped real-data case (and why USGS nested-piezometer heads were rejected as non-diffusive, USCRN soil heat chosen).

### Notes
- The USGS groundwater path was explored and dropped: the nested-piezometer head profile is decoupled by aquifer
  layering (not a homogeneous-diffusion fit). OpenAQ air-source inversion remains a documented future real-data case.

## [0.03.000] — 2026-06-21

The full case catalogue (16 cases), the interactive web app, the architecture wiki, and the live GitHub Pages
deployment. Every metric below is the committed manifest's measured value (relative-L2 vs the validation anchor);
lanes are derived from measurements (ONNX size · ort-web inference time · trace bytes), never hand-set.

### Added — engine generalizations (what made the catalogue possible)
- **Validation anchors**: analytic, vendored numerical reference (`reference_on_grid`), and benchmark (Ghia
  centerline) — all leakage-safe in `evaluate`. Custom `eval_grid()`, `extra_metrics()`, and multi-output reshape
  hooks so vector fields (u,v,p) and non-default grids flow through unchanged.
- **SOTA training surface**: hard constraints (output transforms), Fourier-feature input transforms, SIREN,
  RAR residual-adaptive refinement hook (`refine`), per-term `loss_weights`, **MMS** (method of manufactured
  solutions) sources for cases with no closed-form solution, **parametric PINNs** (a parameter as a network input),
  **FBPINN** 2-channel domain decomposition, and **inverse** cases (a 2nd network output for the unknown field +
  `PointSetBC` observations through the ingestion contract).

### Added — cases (8 canonical + 8 mining/pollution/industrial/control differentiators)
- **Canonical benchmark (6)**: `bench-poisson2d` (hard-constraint, **L2 5e-6**), `bench-heat1d` (**9.1e-5**),
  `bench-wave1d` (**4.9e-5**), `bench-burgers1d` (RAR + Burgers.npz, **9.7e-3**), `bench-allencahn`
  (RAR + Allen_Cahn.npz, **1.2e-3**), `bench-navier-cavity` (3-output, Ghia anchor, **L2 1.7e-1**, honest
  CPU-limited).
- **Industrial fluids/heat (2)**: `ind-helmholtz` (Fourier features, **1.0e-1**, CPU-limited), `ind-heat2d-inverse`
  (PFNN inverse k+T from observations, **4.0e-2**).
- **Mining / mineral-processing (4)**: `mine-heap-leach-rt` (2-species reactive transport, MMS, **7.9e-5**),
  `mine-thickener-settling` (Bürger–Concha degenerate flux, **5.1e-3**), `mine-flotation-kinetics` (parametric
  first-order kinetics C(k,t), **7.6e-4**), `mine-comminution-pbe` (size-transport reduced population balance,
  **1.8e-4**).
- **Pollution / environmental (3)**: `poll-ocean-transport` (advection–diffusion, **3.1e-4**), `poll-soil-barrier`
  (FBPINN 2-channel, D-contrast, **1.9e-1**, honest CPU-limited), `poll-tailings-seepage` (Richards/Gardner
  unsaturated seepage, **6.7e-4**).
- **Control (1)**: `ctrl-zero-source` (degenerate/zero-source sanity anchor, **2.1e-4**).
- All 16 cases classify **lane=live** (ONNX 20–40 KB, ort-web infer < 1 ms, parity ~1e-7); honesty flags
  (`synthetic` vs `synthetic-illustrative`) set per case; CPU-limited cases labeled, not hidden.

### Added — web app (Vite + React 19 + TS)
- Six pages (App · Introduction · Methodology · Implementation · Experiments · Benchmark), HashRouter, zustand,
  i18next (EN/ES), KaTeX equations, canvas viridis heatmap, **onnxruntime-web live inference** with baked-trace
  replay. `lib/contract.ts` mirrors the artifact schema so contract drift fails the build.

### Added — docs wiki + deploy
- `docs/architecture/` (overview · the-gate · data-contracts · staged-pipeline · train→ONNX→web · deploy ·
  determinism), `docs/frameworks/` (DeepXDE, PhysicsNeMo, neuraloperator, jax-pi, NeuralPDE.jl, PINA),
  `docs/methods/` (adaptive sampling · causal-curriculum · loss-weighting · architectures · domain-decomposition ·
  variational-scalable · optimization · operator-learning · inverse-UQ).
- **Live on GitHub Pages** (Actions): https://fsantibanezleal.github.io/CAOS_PINNLAB/ — `base: "./"`, `copy-data`
  prebuild, `github-pages` environment branch-policy cleared. CI smoke (`--quick`) + lane re-derivation guard.

## [0.02.000] — 2026-06-20

### Changed
- **Specialized the engine to Physics-Informed Neural Networks**, replacing the template's example SIR body. Primary
  engine: **DeepXDE 1.15.0** (PyTorch backend); trained PINNs export to **ONNX (opset 18)** for onnxruntime-web.
- The two data contracts re-cast for PINNs: ingestion = observation tables (inverse cases), artifact = solution
  fields. The lane gate now decides live-vs-precompute from **ONNX size + ort-web inference time + artifact bytes**.

### Added
- `cases/bench-poisson2d` — 2D Poisson (Dirichlet) via a **hard-constraint PINN**, analytic validation anchor.
  Verified end-to-end: **lane=live, relative-L2 = 5e-6 vs analytic, ONNX parity = 2.4e-7, onnx = 48 KB self-contained.**
- Core: `io/schema` (ObservationRow, SolutionField), `io/contract` (observation ingestion + outlier policy),
  `core/trace` (`pinnlab.field/v1`), `core/manifest` (`pinnlab.manifest/v1`), `core/gate` (ONNX/ort-web lane),
  `model/analytic` (numpy reference helpers — Pyodide-safe).
- Stages: preprocess / feature_extraction / **train (DeepXDE → ONNX + parity check)** / infer / evaluate
  (relative-L2 vs analytic) / export. CLI `python -m pinnlab.pipeline`.
- Precompute-lane requirements pinned + mapped to docs (numpy, deepxde, torch, onnx, onnxruntime, onnxscript);
  documented GPU lane (`requirements-gpu.txt`).
- Tests rewritten to PINN (contract, smoke, manifest, gate) — **10 passing**.

### Removed
- Stray `models/surrogate.json` (an SIR-example artifact the template copy carried in).

## [0.01.000] — 2026-06-20

### Added
- Initial instantiation from the CAOS product-repo template (ADR-0057): the frozen base — `data-pipeline` package,
  the two data contracts, the named staged pipeline, the seeded RNG, the manifest/trace, the measured lane gate, the
  cases-by-category registry, tests, and CI — before the PINN engine replaced the example body in 0.02.000.
