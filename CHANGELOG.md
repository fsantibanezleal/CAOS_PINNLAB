# Changelog

All notable changes to **PINN-Lab**. Format: `X.XX.XXX` (display), see `pinnlab.__version__`. Keep `0.x` while on
synthetic/benchmark data. Tag every release.

## [0.28.000] (2026-07-15) Structure-preserving learning: a Hamiltonian neural network

The catalogue had no geometric / structure-preserving method, and its one mechanical system
(`dyn-double-pendulum`) was a plain residual PINN that conserves nothing by construction. This adds the family
that the RSS 2026 "Geometry of Motion" workshop is about.

**New case `dyn-pendulum-hnn`** (`hamiltonian-symplectic`): two models trained on identical data, size, seed,
steps and the same RK4 rollout, differing ONLY in output structure. An unstructured MLP emits the vector
field directly; a Hamiltonian network emits a scalar H and takes the symplectic gradient dz/dt = J grad H,
which conserves H exactly because J is antisymmetric.

- **`model/hnn.py`**: the symplectic gradient, the canonical momenta (NOT the angular velocities), the RK4
  rollout and the energy-drift metric. The analytic Hamiltonian was verified before use: it matches E = T + V
  to 1.07e-14 over 2000 states and RK45 conserves it to 8.9e-10 over 10 s.
- **Measured, low-energy bounded regime**: both lanes fit the derivative to the same order (1.2e-5 vs 1.0e-4)
  and the structured lane holds energy 100x tighter over 8 s (0.07% vs 7.44%). The gap is structure, not
  accuracy. Two earlier attempts are recorded honestly in the plan: a uniform-box coverage that made it worse,
  and the chaotic high-energy regime where both lanes' absolute drift blows up (HNN still wins ~27x).
- **The exported ONNX is the learned energy surface** (state -> H); the symplectic gradient in the forward
  pass is not ONNX-representable, but the scalar H it differentiates is the physically meaningful object.

**FIX (pre-existing): the trajectory kit's side panels collapsed to 2px.** `.traj-kit svg { height: 100% }`
resolved against a collapsing flex column, so the phase-portrait and butterfly panels were invisible in the
Results tab for BOTH trajectory cases (the shipped double pendulum included). Now 218px. The kit is also
manifest-aware: with no twin trajectory it drops the chaos/leave-time framing, relabels "PINN" as "model",
and shows an energy-vs-time panel in place of the butterfly.

The case landed COMPLETE: pipeline + estimate + bilingual context + verdict + docs; index rebuilt to 23
cases; fit gate passes 540 checks.

## [0.27.000] (2026-07-15) PINO: a physics-informed neural operator, and the overclaim it exposed

The catalogue's operator lane was data-driven only. A domain expert asked publicly whether we had looked at
PINO as an FNO variant; auditing that question found the docs already claimed `bench-darcy-operator`
"exercises all three methods on the same family: DeepONet, FNO, and PINO", while `PINO` appeared **zero
times** in `data-pipeline/` and `frontend/src/`. Only the case doc was honest about it.

**New case `bench-darcy-pino`** (`operator-pino`), the matched companion to the data-driven FNO case: same
family, architecture, seed, epochs and test set, so only the loss differs.

- **`model/pino.py`**: the paper's two phases (arXiv:2111.03794). Pre-training with the data loss plus the PDE
  residual on VIRTUAL instances that cost no solve (Algorithm 1), then test-time optimization on a single
  instance with the anchor loss. Residual by finite volume with harmonic-mean faces, **verified before use**:
  rms 3.6e-14 on the reference solution, exactly f = 1 for u = 0.
- **The FFT route is documented as unusable here, with a number**: this problem is non-periodic with a
  discontinuous coefficient, and the spectral Laplacian misses by ~6x the source term.
- **Gradient-norm balanced lambda**, after measuring a real pathology: the two loss terms are balanced in
  VALUE (1.07 vs 1.00) but their gradient norms differ by **38.6x**, so a fixed lambda = 1 gave the equation
  ~2.6% of the update.
- **Hard boundary constraint**, which was the decisive fix: the residual is interior-only and Darcy is unique
  only with its BC, so a soft penalty let the physics converge around wrong boundary values. Enforcing
  u = 0 exactly moved 32 labels from -0.5% to **+45.3%**. The exported ONNX applies the same constraint.

Measured (held-out relative-L2 over 32 unseen fields): **+34.1%** at 0 labels, **-177.6%** at 8,
**+45.3%** at 32, **+48.1%** at 128. The 8-label regression is shipped and explained rather than hidden: it
is the standing "physics-informed training is slow" objection, reproduced inside our own operator.

Also in this release:

- **`rebuild_index.py`**: regenerate `manifests/index.json` from the registry without retraining the whole
  catalogue (previously only `run_all()` wrote it, so adding a case meant a full rebuild). It refuses to list
  a case that has no manifest.
- **FIX: dead evidence link.** The Results panel listed "Live" for every case, but the Live tab is gated on
  `lane === "live"`, so both field-IO operator cases pointed at a tab that does not exist.
- Docs corrected: the DeepONet/PINO "exercises all three" claim is replaced by a status table (DeepONet is
  marked NOT implemented), and the literature's "far less data hunger" line now carries what we actually
  measured.
- Research persisted under `wip/beyond-sota/`: the primary-source transcription, the claims-vs-engine audit,
  the plan with all three configurations tried, and the full survey.

## [0.26.009] (2026-07-15) FIX: the equations rendered English inside the Spanish app

Each case ships one `governing_equations` LaTeX string. The math is language-neutral, but the prose inside
`\text{...}` was English only, so the Spanish app rendered "rate constant", "recover k from sparse T obs",
"validate vs held-out 10/20/50 cm", "degenerate above", "deep ensemble". 18 of the 21 cases carried such
annotations (24 distinct phrases).

- **New step** `data-pipeline/localize_equations.py`: rewrites ONLY the `\text{...}` payloads through an
  explicit phrase map and bakes `governing_equations_es` beside the untouched original. Training-free and
  deterministic. It FAILS LOUDLY on an unknown annotation, so a new case cannot silently ship English.
- **Schema/app**: `governing_equations_es?` added to the contract; the equation row prefers it in Spanish and
  falls back for the 3 pure-math cases that need no twin.
- Verified in-browser across 10 cases: no English left in any Spanish equation; the math is byte-identical.
  The fit gate still passes its 508 tab-checks with the longer Spanish strings.

## [0.26.008] (2026-07-15) FIX: English leaked into the Spanish answers

The estimate schema localized `question`, `why` and `label` but carried a single `value`, so every value that
embeds prose rendered in English inside the Spanish app: "39.4%, R=90% not reached in window", "not in
window", "356%: unrecoverable", "0.96 to 1.17", "0.077 at t = 1.00", "t = 0.70 (exact 0.702)", "n/a",
"within the window: no", "σ = ... at (x=...)".

- **Schema**: `EstimateItem` gains optional `value_es` / `values_es`; the app prefers them in Spanish and
  falls back to `value` / `values`, which is what every pure-number value keeps using (no ES twin needed).
- **Pipeline**: `build_estimates.py` `item()` takes `value_es` / `values_es`, and the 10 prose-carrying sites
  now bake both readings. Re-baked training-free from the committed artifacts: the diff adds `value_es` /
  `values_es` only, and **no metric, L2 or ONNX-parity number changed**.
- Verified in the browser across both languages: no English in any Spanish value; the fit gate still passes
  its 508 tab-checks with the longer Spanish strings.

## [0.26.007] (2026-07-15) FIX: the field map was clipped on its left edge (and its colorbar scale hidden)

The `fv2` field kit pinned `.fv2-mapcol` to `fit.w`, the MAP's width. The row inside it is the map plus a
6px gap plus the colorbar (measured: 416 + 6 + 46 = 468 against a 468px row), so the row overflowed the
column by ~52px and, being centered, spilled 26px past each edge. The left spill fell outside
`.pl-res-viz` (`overflow:hidden`) and cut the map's left edge; the colorbar's scale labels were lost with
it, so the map had no readable value scale. Reproduced live at every device pixel ratio (1, 2, 3), with and
without the equation row, so it was hitting real users, on the default landing case (flotation).

Fix: size the map column to its content instead of to the map width alone. Map size is unchanged (canvas
still 414px); nothing clips (measured left overflow 26px -> 0 in Results, -15 in Field). The standing fit
gate still passes its 508 tab-checks.

## [0.26.006] (2026-07-15) The app lands on a mining case by default

- **Default landing = mining.** A fresh visit (no `?case` deep link) now opens on `mine-flotation-kinetics` in
  the Mining & mineral processing domain, instead of the first canonical-benchmark case. Flotation is the
  emblematic mineral-processing operation and its recovery-vs-time curve is the most legible first view; deep
  links still win over the default. Verified with a headless landing check (domain = mining, case =
  flotation-kinetics, hash `#/`).
- **README:** dropped the "not a demo" negation from the opening line (state what it is); the honest-scope
  bullet (not a FEM/FVM replacement, not a digital twin) is unchanged.

## [0.26.005] (2026-07-15) Content corrections: every case audited vs its docs

Each case was audited against its `docs/cases/*.md` and the baked traces; contradictions and overclaims were
corrected so the in-app answer/verdict matches the documented physics and the numbers the pipeline actually
produced.

## [0.26.004] (2026-07-15) Presentation overhaul of the text surfaces

Bigger type on the reading surfaces, a two-column Context panel that uses the full width, and removal of the
empty voids that made the text surfaces feel hollow.

## [0.26.003] (2026-07-15) Stage-fill layout crisis fixed

The visualization kits now FILL the stage: no empty space, no overflow, no clipped captions. Closes the map-not-
filling / tiny-floater regressions surfaced in the live review.

## [0.26.002] (2026-07-15) Visual-identity pass + citation infrastructure

A luminous answer hero, added depth, and refined regime chips; the Context panel goes full-width and the caption
is unclamped. New citation infrastructure (`citations.ts`, `Cite` / `Refs`).

## [0.26.001] (2026-07-15) FIX: the heatmap no longer collapses to a few pixels (issue #57)

The field map appeared at a good size then shrank to a tiny few-pixel version when the profile plots were
present (Results, Field, Live). Cause: the contain-fit hook measured the map column, whose width was set from
the fit result: a ResizeObserver feedback loop that converged toward zero while the profile column took the
width. Fixed: the fit now measures the STABLE body row and reserves ~47% of the width for the profiles, so the
map is sized from the real available space and claims its own width. Verified stable over time (canvas
~300px, no shrink) and the fit gate still passes 508 checks.

## [0.26.000] (2026-07-15) THE REVIEW PLAN COMPLETE: fit gate + content arc closed (issue #49)

Closes the real-review remediation plan (wip/web-review/real-review-and-plan-2026-07-15.md), E1-E6.

- **E5 the standing fit gate**: tools/visual-verify/pl-fit-audit.mjs is committed and PASSES 508 tab-checks
  (21 cases x every tab x 1366x768 AND 1920x1080 x dark+light): every tab fits with no scroll, Results-first
  order, rail non-overflow, and every visualization tab renders a viz. The last fit residuals were fixed:
  the HeatmapKit output-chip row now shares the fitted column (darcy), the field hint lines were dropped,
  and the replay-only cases (darcy operator, chaotic pendulum) no longer show an empty Live tab (Live is
  absent for the precompute lane, honestly).
- **E4 content arc closed**: Experiments gains a "what was actually run" section narrating the real runs
  (the fair-budget protocol and the wave1d inversion it caught, the identifiability sweep 356%->12.6%, the
  soil real-data out-of-sample holdout, the flagship dye holdout 0.78% with the swept/dead honesty split, the
  captured training dynamics, and the published negative results). With Introduction (destaled), Methodology
  (the win/lose scope + estimators-in-the-wild), Implementation (the two worlds + Engineering lessons) and
  Benchmark (how-to-read-these-numbers), the five pages are now one honest reading arc with hand-offs.
- 51 contract tests green.
## [0.25.003] (2026-07-15) Header/footer font aligned to the shell ADR

The header and footer declared "Inter"/"JetBrains Mono" (never loaded), so on viewers with those fonts
installed the shell drifted from the other CAOS apps (which use the system stack per ADR-0016 + the
CAOS_SEISMIC/LDA-HSI design system). Aligned the app-wide font to the exemplar system stack (ui-sans-serif,
system-ui, ...; ui-monospace, ...) with antialiasing, so the header/footer render consistently with the rest
of the portfolio. Header verified single-line and ADR-0016-shaped (brand, route-nav, actions).

## [0.25.002] (2026-07-15) RESULTS made rich again: the full interactive viz + an enriched answer (issue #49)

The interim v0.25.001 Results showed ONE static key image (a regression). Restored: the Results tab now shows
the case's FULL interactive visualization (the real fitted kit: hover/pin/zoom fields, animated spatiotemporal
spills, the multi-panel hidden-flow / inverse overlays, etc.) beside a substantially richer answer column: THE
QUESTION, THE ANSWER, the computed values as prominent stat tiles, THE VERDICT (from the baked numbers), and
the evidence links. Both fit the stage at 1366x768 and 1920x1080 (no scroll). The space is now used: columns
rebalanced, the visualization gets the width. (Context already carries the Summary cards + the deep sectioned
theory; verified rich for all 21 cases.)

## [0.25.001] (2026-07-15) CONTENT ARC + Results key-graph fix (issue #49)

- **The five doc pages made faithful to v0.25 and deepened**: Introduction drops the removed "story selector"
  reference and states the real tab structure; the win/lose material stays as honest documentation pointing at
  the cases + Methodology. Implementation's Web-flow updated to the real workbench (Results-first tabs, the
  contain-fit engine, no scroll) and gains a new **Engineering lessons** sub-tab telling the real, documented
  failures that left rules in the pipeline (the diverged FDM nearly baked, the Allen-Cahn metastable trap, the
  hidden-velocity steady-flow fix that a first run failed, the compute-bomb fix, the test that clobbered
  canonical artifacts). Benchmark gains a **"how to read these numbers"** section (anchor classes, ONNX parity
  is not a physics error, the honest high-error cases, the data-honesty label, the honest dashes) and the
  reading-arc hand-off across the five pages.
- **Results key graph fixed**: a leftover `align-items: start` collapsed the grid cell so the key graph
  rendered at 85px; now it fills the Results stage (360px), markers and all, at every case.
- 51 contract tests green; all tabs still fit at 1366x768 and 1920x1080.

## [0.25.000] (2026-07-15) THE LAYOUT OVERHAUL: an instrument that fits the screen (issue #49, real review)

After a full measured review (wip/web-review/real-review-and-plan-2026-07-15.md: the no-scroll principle was
violated in every tab of every case, canvases hard-capped, the layout unstable between cases), the App was
rebuilt to a single coherent contract, honoring the owner's decisions:

- **A stable shell for every case**: the story stepper is GONE; the left rail has one purpose set (search,
  Domain dropdown, CASE dropdown, the selected case's card, the constraint chips, the regime control, once);
  the governing EQUATION is a full-width strip above the tabs, always visible (the case's identity); the
  header is single-line.
- **Tabs in the owner-fixed order (Results, Context, then the evidence)**, sitting ON the stage.
- **Every tab fits the stage at full view: no page scroll** (validated at 1366x768 and 1920x1080). A new
  contain-fit sizing engine (useFitBox) sizes every map/chart to the measured stage box, so nothing is
  hard-capped and overlays (crosshair, markers, dots) stay truthful. Where a view has more than fits it
  becomes a sub-view: Context is a section switcher (Summary cards + the deep prose split by its own
  headings, full-width columns); Compare is a fitted single row of LARGE panels with Fields/Errors/Evolution
  sub-views (never two rows of thumbnails); Diagnostics is a chart pager; Training, Live, Field, the vector /
  spatiotemporal / inverse / hidden-flow / trajectory kits all fit.
- **Results = the result**: the answer sentence + values + verdict + evidence links on the left, ONE fitted
  key graph on the right (the full interactive kit lives in Field); no scroll.
- The Charts tab is now "Regimes"; the redundant duplicated regime control and the duplicated equation were
  removed.

Content depth of the five doc pages (Implementation/Experiments rebuild, the reading arc) is the next slice,
tracked in issue #49.

## [0.24.001] (2026-07-14) INTERACTIVE VIZ: chart solo + drag-zoom, field wheel-zoom, frame markers, responsive (issue #49 S4/S5)

- **Diagnostic charts are interactive**: click a legend entry to SOLO a series (click again for all); drag
  horizontally to zoom an x-range (double-click resets, a reset chip appears); the crosshair hover values stay.
  Applies to every line comparison (Ghia centerlines, sensor sweeps, held-out profiles, the flagship's
  recovered-current profiles) and the wavenumber/spectrum charts.
- **The field maps wheel-zoom**: scroll to zoom around the cursor (the zoom is a data-window slice, so the
  crosshair, read-out, markers and axis mapping stay truthful; a badge shows the factor; double-click resets
  and releases the pin). Click-to-pin / follow-cursor behavior unchanged.
- **Markers on the animated frame kits**: the ocean checkpoint now draws on the drifting-spill frames
  (SpatioTemporalKit).
- **Responsive**: below 900px the rail stacks above the stage and the stepper wraps.
- Verified: interactions exercised end-to-end (solo dims, zoom + reset, badge, frame markers) with 0 console
  errors; full structural validator 21 cases x 2 themes green.

## [0.24.000] (2026-07-14) THE INVESTIGATION WORKBENCH + THE HFM FLAGSHIP (issues #48, #49 S1-S3)

The owner's full review (persisted in `wip/web-review/unified-remediation-plan-2026-07-14.md`) found the app a
solver gallery, not a product: this release restructures it into an investigation instrument and ships the 21st
case.

- **RESULTS-FIRST WORKBENCH (owner-fixed tab order: Results, Context, then the evidence)**: every case now
  opens on a designed Results screen: THE SITUATION (a concrete scenario with stakes) -> what the model
  calculates (what the axes and colors physically mean) -> what we know vs what we seek -> how the PINN helps
  under DECLARED assumptions -> THE ANSWER as a plain-language sentence with its values -> THE VERDICT derived
  from the baked numbers (usable for X, not for Y) -> the evidence index (what each remaining tab proves).
  New content layers for all 21 cases, EN/ES, zero unexplained jargon (`scenarios.ts`, `results.ts`).
- **THE LAYOUT IS AN INSTRUMENT**: the story is a visible 9-chapter stepper (entry lands on chapter 1, never
  an arbitrary case); tabs sit ON the stage they control; regime chips sit directly above the stage; the left
  rail is selection only (search across names + questions; only the case list scrolls); one-line meta footer.
  Measured against the previous release: rail overflow 220px -> 0; the stage starts at 247px instead of 404px
  and gets 616px instead of 466px; the App page no longer scrolls at 1500x950.
- **THE ANSWER DRAWN ON THE FIELD**: `build_estimates.py` now bakes structured marker coordinates (34 markers
  across 12 cases) and the field kits draw them: heat1d's half-cooling instant, burgers' arrival, allencahn's
  final walls, the navier vortex core, the helmholtz receiver, the flagship's recovered-vs-true circulation
  centers, the barrier's half-rise, poisson's per-mode peaks, the thickener's mid-height crossings. The number
  and the picture finally reference each other.
- **Charts tab fixed**: single-regime cases no longer show a meaningless one-bar chart; they show every
  measured number of the bake, each explained in plain language.
- **CASE 21, THE FLAGSHIP: `ind-hidden-velocity` (issue #48)**: the Hidden Fluid Mechanics mechanism (Science
  2020) at CPU scale: the whole hidden current estimated from ~640 sparse noisy dye samples + transport
  physics ONLY (no velocity data, no IC/BC). Seeded FD dye truth with asserted stability (central scheme legal
  at cell Peclet 0.59; CFL-checked; mass conserved). The first training run was REJECTED (38-60% error even in
  the dye-swept region) and diagnosed: a time-varying velocity that sparse dye cannot pin; the declared
  steady-flow assumption (u_t = v_t = 0 residuals) fixed it and is documented as teaching content. Shipped
  honest numbers: 16.4% current error inside the swept region vs 37.5% in the never-dyed dead zones (the
  unidentifiability split is part of the answer), recovered circulation center (0.47, 0.50) vs true
  (0.50, 0.50), held-out dye RMSE 0.78% on 160 never-trained samples. New HiddenFlowKit view; 9th story
  chapter; docs page; Methodology/constraints/registry wired; catalogue counts swept 20 -> 21.
- Validation: 51 artifact contract tests green; the structural validator now ASSERTS Results-first order,
  Results content on every case, rail non-overflow and page fit (21 cases x 2 themes, 0 console errors).
- Remaining from the unified plan (tracked open in issue #49): chart zoom/solo interactivity, the shared
  probe/time cursor across views, marker wiring on the frame-animation kits, the jargon sweep of Context
  prose.

## [0.23.000] (2026-07-13) THE ESTIMATION REFRAME: every case asks a question and answers it with a computed estimate (issue #46)

The catalogue's framing moves from solver-centric to estimation-centric: the reason to use a PINN is to ESTIMATE
something you cannot measure directly, and every case now leads with that. Plan + adversarially verified research
record in `wip/web-review/estimation-reframe-plan-2026-07-10.md`.

- **THE ANSWERS, BAKED (new pipeline tool `build_estimates.py`)**: a training-free baker computes each case's
  quantity of interest from the already-validated artifacts (front tracking, threshold crossings, integration,
  argmax, stream-function integration, exceedance from mean/sigma) and writes a manifest `estimate` block
  (question EN/ES, honest why-a-PINN line, answer items; per-variant where the family is the point) + patches
  `index.json` with per-case questions. Cross-validated where closed forms exist: heat1d half-cooling matches
  ln2/(alpha pi^2) on all six alphas (0.70 vs 0.702 ... 0.07 vs 0.070); wave periods exact (T = 2/c); burgers
  front arrival 0.80-0.82 vs exact 0.80; flotation recovery matches 1-exp(-k) and t90 = ln10/k within 2%;
  darcy FNO peak heads within 4% of finite differences. Physical sanity: thickener settling monotone in R,
  comminution passing fraction monotone in grind rate, navier vortex core at (0.61, 0.75) via the stream
  function (a plain speed minimum finds a corner eddy instead), pendulum twin-divergence 2.11 s agrees with
  the PINN leave-time 1.99 s, zero-source RMS 5.2e-4. The engine's `lyapunov_est` summary is NOT surfaced
  (it fails a back-of-envelope consistency check; the twin leave-time replaces it).
- **ANSWER CARD (web)**: every case opens with THE QUESTION then THE ESTIMATE (computed offline), plus the
  why-a-PINN line; per-variant answers react to the regime chips and clicking a value chip switches the regime.
- **Case list + story**: each case in the Domain list shows its engineering question under the name; story
  chapters lead with the case's question, then the computed evidence line.
- **Introduction**: opens with "Every case answers an engineering question" (the PDE as measuring instrument),
  with three computed examples (soil diffusivity 0.304 mm2/s at ~1 degC holdout; defect map 4% vs 356%;
  spill arrival t = 0.44).
- **Methodology**: new "PINNs as estimators in the wild" section transcribed from the verified research record
  (Raissi 2019 data-driven discovery; Hidden Fluid Mechanics, Science 2020; Nature Reviews Physics 2021 Key
  Point + Tomo-BOS espresso; subsurface conductivity from sparse heads incl. the no-direct-measurements
  unsaturated law, WRR 2020; seismic full-waveform inversion with honest limits, JGR 2022; two-stage Bayesian
  source inversion, ASCE), each mapped to the catalogue case that teaches the same mechanism; 5 new references.
  Fixed two off-by-one BEYOND citations (inverse-uq linked the fluids review instead of HFM; operator-learning
  indexed past the reference list and rendered "doi:undefined").
- **Benchmark**: new "what it estimates" column (the QoI + value per case, question on hover).
- **Contract tests**: 22 new tests (estimate coverage on all 20 cases, question/why/item shape, value XOR
  per-variant values, variant-id keys, index questions).
- Phase D of the plan (an HFM-style velocimetry-from-dye flagship case) is deferred to a dedicated compute
  session, recorded in the plan and issue #46: real training runs are not rushed into a release.

## [0.22.000] (2026-07-10) FUNDAMENTALS: the identifiability sweep, why-PDEs, the information budget, constraint anatomy (issue #44)

Answering the owner's fundamental questions with computed content and taught concepts (review persisted in
wip/web-review/fundamentals-review-2026-07-10.md):

- **THE IDENTIFIABILITY SWEEP (new bake)**: heat2d-inverse trained at n = 0/10/25/50/100 sensors at ONE fair fast
  budget: recovered-k error = 356% / 17.3% / 16.3% / 13.6% / 12.6%. The computed answer to "how much information
  does a PINN need": the cliff is between 0 and 10 sensors (ANY anchoring data restores identifiability of the
  field unknown), then returns diminish; the fully-trained n=100 run reaches 4.0% (Compare). Shown as a log-y curve
  in the Diagnostics view (`build_identifiability_sweep.py` + `build_n_sensors()`; XYChart gained yLog).
- **CONSTRAINT ANATOMY**: every case now shows "What pins the solution" chips under its equation: BC / IC / PARAM /
  UNKNOWN / DATA / ANCHOR, color-coded, bilingual, sourced from each case definition. A PDE alone has infinitely
  many solutions; the chips teach what makes each problem well-posed.
- **Introduction RESORTED with the fundamentals**: Why PDEs matter (local laws, the three predictions) -> the loss
  -> What a PINN attempts to reach (zero-loss = the solution for well-posed problems; the solution-as-object target)
  -> The information budget (four regimes + the chaos ceiling, all with the catalogue's computed numbers) -> the
  story -> is/isn't.
- **Docs audit (repo-wide)**: README gains the method-ladder & dynamics section + the tests-never-write-canonical
  rule; docs index cross-links the ladder; pipeline.py docstring points to the ladder tools + re-apply order; the
  architecture-modal Overview text (EN/ES) mentions the ladder bakes; all 20 governing-equation strings audited
  (correct); heat2d case doc + the ladder architecture doc record the identifiability result.
- Full validation: 20 cases x 2 themes 0 console errors; 26 artifact contract tests pass.

## [0.21.003] (2026-07-10) polish round: Compare colorbar, pendulum/UQ axes, story labels, snapshot consistency (issue #42)

- **Compare view: a shared labeled colorbar** for the value lanes - the viridis scale finally has a numeric anchor
  (max / mid / min), and it uses the previously-empty right space.
- **Pendulum (TracePlot) + UQ band charts** brought to the plot standard: nice-tick gridded axes with smart number
  formats on the phase portrait, butterfly, angles and the sigma band (they still had bare min/max exponentials).
- **Story labels shortened** so the 8 chapters no longer truncate in the rail select; the chapter blurb carries the detail.
- **Snapshot-to-PNG consistency**: the animated hero, the animated ladder and the Training L2 chart now have the
  same export button as the panels.
- **Side profiles**: numeric x end-ticks (y ticks landed in #40).
- **aria-labels** on the chart SVGs; the hero tightened to 720px (more fits above the fold).
- Full validation: 20 cases x 2 themes, 0 console errors.

## [0.21.002] (2026-07-10) visual-quality pass: real axes everywhere, the equation truly never cut (issue #40)

Owner: "why are the graphs so ugly now?" - confirmed by live screenshots and fixed per the dataviz procedure
(recessive grids, nice ticks, smart number formats, render-and-look):

- **Shared plot helpers** (lib/plot.ts): nice 1/2/5 tick generation + smart formatting (plain decimals in the human
  range, exponentials only outside it, k-suffixed iteration counts).
- **The animated hero** now has real axes: 4-6 ticks per axis with plain-decimal labels, a recessive grid, bounded
  width (780px), a calm title with a proper current-vs-initial legend, and no mono debug text. The carpet + temporal
  graph fit with it.
- **The equation strip** finally cannot clip: the equation owns a FULL-WIDTH row (verified: the whole heat1d
  u* = e^(-alpha pi^2 t) sin(pi x) renders); method/engine/L2/parity + the expected band moved to the second row.
- **Training L2 chart**: index-spaced checkpoints (0/250/.../12k evenly, no more 6-points-in-a-corner), percent
  y-ticks on the log scale, integer iteration read-out, panels enlarged to ~400px.
- **Compare**: uniform fixed 240px panels across the value AND error rows (no more mismatched sizes), captions
  baseline-aligned; the animated-ladder chart got the same real axes.
- **Side profiles**: numeric y-ticks + grid (they had none).
- **Colorbar/read-outs**: smart formatting ("1" instead of "1.0e+0").
- **Bug fix**: the App now reacts to URL param changes AFTER mount (external hash navigation / back-forward sync).
- Full validation: 20 cases x 2 themes, 0 console errors; before/after screenshots in the issue.

## [0.21.001] (2026-07-10) improvement pass: shareable deep links, story-to-view, Benchmark ladder, ladder diagram, artifact tests, sandboxed test suite (issue #38)

- **Shareable deep links**: #/?case=<id>&view=<view> - open a link and land exactly there; every navigation keeps
  the URL in sync. One mechanism also powers the story.
- **Story chapters land on the DEMONSTRATING view** (helmholtz -> Training, allencahn -> Training, soil-heat-real ->
  Diagnostics, ...) with a one-line chapter blurb under the selector.
- **Benchmark page: the ladder columns** - naive-vs-standard and fix-vs-standard L2 per case (from the baked
  comparisons; manifests now carry the summary so no heavy trace fetches), each case linking to its Compare view.
  A dash means the lane does not exist (a contrast is never fabricated).
- **Architecture modal: "Method ladder & dynamics" tab** with a new hand-authored theme-aware SVG (pipeline lanes ->
  baked artifacts -> web views + the real results strip), EN/ES.
- **allencahn Training extended to 12k checkpoints**: the naive soft lane oscillates at 77-114% and never converges;
  the hard-constraint lane descends monotonically 63% -> 34% (short-budget; the fully-trained 0.4% is in Compare).
- **Artifact contract tests** (25 tests): comparison/training/frames/diagnostics files exist, keys + shapes + L2
  lengths consistent, manifest summaries match traces, coverage floors enforced.
- **CRITICAL test-suite fix**: the pipeline smoke tests wrote the CANONICAL artifacts (a quick run clobbered
  poisson2d/heat1d bakes and silently stripped comparison blocks from manifests - caught by the standing rule,
  restored from git). The whole suite is now SANDBOXED via an autouse conftest fixture redirecting
  DERIVED/MANIFESTS/MODELS to tmp dirs. 34 tests pass with zero canonical writes.
- Full validation: 20 cases x 2 themes, 0 console errors.

## [0.21.000] (2026-07-10) DYNAMICS: the app now SHOWS motion + the when-PINNs-win/lose story (issue #36)

Owner review: "why are no dynamics observed?" Root cause: we computed the right content but presented its dynamics
as stills (the space-time carpet encodes motion instead of showing it). Grounded in a deep-research pass
(wip/web-review/dynamics-research-2026-07-10.md, 13 adversarially-verified findings): animated field evolution +
quantitative L2 is the verified state of practice; use-case-first + failure-mode-and-remedy is the right primary
axis for an educational catalogue.

- **Animated evolution hero** (Field + Live): for every time case the dominant element is now the MOVING profile
  u(space) at time t (locked y-scale, ghost = initial state), driven by the transport bar; the space-time map is
  demoted to the seek/probe carpet. Both-mode probe retained. 10 cases.
- **CompareEvolution**: PLAY the method ladder - standard (grey) vs naive (red) vs adapted (green) profiles evolving
  TOGETHER on one locked scale; you watch the naive lane peel away from the standard. Pure replay of the baked
  comparisons; lights up on 10 (space,t) comparisons.
- **2-D evolution frames**: offline ONNX bakes (ocean 24 frames; heap-leach 16 x 2 species) -> the Field view plays
  the 2-D field evolving (SpatioTemporalKit, colour scale fixed across frames).
- **TrainingKit - "watch it learn"** (BEYOND the surveyed state of practice - the catalogue's novel presentation
  element): REAL checkpoint fields, naive vs adapted side by side + live L2 + the L2-vs-iteration curve.
  helmholtz: the naive tanh lane never leaves ~100% at ANY checkpoint (at iter 500 it is literally a flat blank at
  100.3% while the Fourier lane already shows the pattern at 25.9% -> 9.0% at 12k) - spectral bias made visible as
  a training pathology (Krishnapriyan 2021). allencahn: the naive soft lane plateaus ~95-114% while the
  hard-constraint lane descends 63% -> 45% at the short 6k budget (the fully-trained 0.4% lives in Compare).
- **THE STORY**: an 8-chapter "when PINNs win / lose" selector in the App rail (each chapter lands on the case whose
  computed Compare/Training/Diagnostics view DEMONSTRATES it) + the storyline written into the Introduction.
  Chapter 1 says it honestly: on the easy forward problem the classical solver already wins.
- All animations: paused by default, play-once, loop opt-in, halt on hidden tab.
- Persisted: research dossier + evaluation (wip/web-review/), the dynamics layer in docs/architecture/
  method-ladder-comparison.md, Training sections in the helmholtz/allencahn case docs.
- Full validation: 20 cases x 2 themes - 10 animated ladders, 10 evolution heroes, 2 frame animations, 2 Training
  views, all empirically verified to MOVE (polyline/canvas-hash change under play), 0 console errors.

## [0.20.009] (2026-07-10) persist the capability in docs + Experiments page + a click-to-enlarge detail popup

Persisting all the deep-pass content and one more graph enrichment:
- **docs/architecture/method-ladder-comparison.md** (new): the capability end to end - the lanes (standard / naive /
  adapted / data-driven), the pipeline ladder tools, the CompareKit/DiagnosticsKit, the manifest schema, the fairness
  rule, and the real per-case results. Linked from the architecture README.
- **Per-case docs**: a Comparison/validation section added to the 7 ladder cases (helmholtz, allencahn, heat2d-inverse,
  soil-barrier, navier, soil-heat-real, darcy) with the real numbers.
- **Experiments page**: a new "method ladder" panel surfacing the standard-vs-naive-vs-fix results (it had none) +
  the honesty note.
- **Detail popup**: click any Compare panel to enlarge it in a modal (with its own snapshot button).
- Full re-validation: all 20 cases, light + dark, 0 console errors.

## [0.20.008] (2026-07-10) snapshot-to-PNG on every comparison panel + diagnostics chart (issue #25)

Interactive-graph enrichment: a ⤓ button on each Compare panel (the heatmaps) and each Diagnostics chart exports it
to a PNG (2x, on a solid background) for slides / papers. Self-contained (canvas.toDataURL / SVG serialization, no
dependency). Verified: the button triggers a real download; full re-validation 0 errors, light + dark.

This completes the deep pass: the offline pipeline computes real comparisons/validations, and ALL 20 cases show them
(17 Compare views incl. 4 naive-vs-fix ladders + darcy FNO-vs-FD; navier Ghia-centerline + soil-heat-real held-out
real-sensor Diagnostics; the double pendulum's PINN-vs-RK45 trajectory), with the full-width top-strip layout, the
both-mode probe, dropdown nav, and per-panel snapshot. Nothing invented; honest numbers throughout.

## [0.20.007] (2026-07-10) navier Ghia validation + soil-heat-real held-out real-sensor validation (issue #25)

The last two cases get real comparison/validation views (done PROPERLY, not the rushed FDM that diverged):
- **navier-cavity**: a Diagnostics view with the canonical Ghia-Ghia-Shin (1982) Re=100 CENTERLINE validation - the
  PINN velocity along both cavity centerlines vs Ghia's tabulated benchmark points (u RMSE 0.053, v RMSE 0.029).
  Robust (no finicky field solver); the streamlines stay as the Field view.
- **soil-heat-real**: a Diagnostics view validating the reconstruction against the REAL measured USCRN temperatures
  at the HELD-OUT depths (10/20/50 cm, never shown to the optimizer) over 2019-2021 - measured points vs the PINN
  out-of-sample curve, RMSE 1.24 / 1.05 / 0.75 degC (matching the case's own metrics).
- A generic XY chart (benchmark points vs model curve) added to the DiagnosticsKit.

**Every one of the 20 cases now has a real comparison/validation**: 17 Compare views (4 naive-vs-fix ladders),
navier + soil-heat-real Diagnostics, and the double pendulum's PINN-vs-RK45 trajectory. Full validation: 0 errors, light + dark.

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
