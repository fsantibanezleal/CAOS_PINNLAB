# THE ESTIMATION REFRAME: why use a PINN at all (deep analysis + plan, 2026-07-10)

Owner critique (verbatim intent): the content is poor because the app shows PDEs and PINN errors, but never answers
WHY use a PINN, what is the point. PDEs in practice are used to ESTIMATE parameters and metrics; the PINN can be a
PROXY for estimation. The web must motivate the interest for PINNs and make each case's USE understandable: the kind
of problem that can be evaluated.

## 1. The diagnosis (why the content is poor despite being correct)

The app is SOLVER-CENTRIC: every case presents "the PDE, the PINN's solution, the error vs a standard". That framing
answers "how well does a PINN solve equations", a question only method researchers ask. The audience question is
ESTIMATION-CENTRIC: "I cannot measure X directly; can physics + the measurements I do have give me X?" In practice a
PDE is an INSTRUMENT: you observe what is cheap (a few temperatures, a dye video, pressures at wells, a settling
interface) and the PDE transports that information to what you actually need (a diffusivity, a permeability map, a
breakthrough time, a recovery, an exceedance probability). The PINN is one machine for doing that transport: it
fuses the residual, the conditions, and the data into a single differentiable estimator. Our catalogue already
CONTAINS this story (soil-heat-real literally estimates alpha = 0.305 mm2/s from real sensors; heat2d estimates a
conductivity MAP; the identifiability sweep quantifies how much data the estimate needs) but the presentation leads
with the equation, not the question. That is the poverty: no case says what QUESTION it answers.

## 2. Why use a PINN: the honest answer (to be told on the web)

- For a single well-posed FORWARD solve: you usually should NOT: a classical solver is exact and faster (our own
  chapter 1). Honesty stays.
- A PINN earns its place when the problem is an ESTIMATION problem:
  1. INVERSE parameter/field estimation is NATIVE: the unknown (a scalar alpha, a field k(x,y)) is just another
     output/trainable, and sparse noisy data enters the same loss: no adjoint derivation, no mesh, no iterated
     forward solves (classical inversion wraps an optimizer around MANY forward solves).
  2. HIDDEN-STATE estimation: infer what you cannot see from what you can (the Science 2020 Hidden Fluid Mechanics
     pattern: velocity + pressure from dye images): the PDE is the bridge between observable and unobservable.
  3. AMORTIZED estimation: one parametric net (or operator) answers the question for a WHOLE family instantly
     (our Live sliders, the darcy FNO): screening designs/scenarios where classical would re-solve each time.
  4. DIFFERENTIABLE estimation: gradients of the estimate w.r.t. anything (design variables, sensor positions) come
     free: the object plugs into optimization and sensor-placement studies.
  5. HONEST UNCERTAINTY: ensembles give the estimate WITH error bars where data is thin (source-uq).

## 3. Per-case reframe: THE QUESTION -> THE ESTIMATE (the quantity of interest)

Legend: computable = the QoI can be computed NOW from the baked artifacts (no new training).

| case | THE QUESTION (who asks it) | THE ESTIMATE (QoI) | computable from bakes | why a PINN (honest) |
|---|---|---|---|---|
| env-soil-heat-real | What is this soil's thermal diffusivity? (geothermal design, agronomy: you cannot measure alpha in situ) | alpha = 0.305 mm2/s, validated out-of-sample ~1 degC | DONE (baked) | the flagship PROXY: two boundary temperature series + physics -> the property |
| ind-heat2d-inverse | Where is this plate insulating/conducting? (non-destructive characterization, defect detection) | the conductivity MAP k(x,y) + where it is trustworthy | DONE + identifiability curve | field unknown = native PINN inverse; classical needs adjoint + mesh |
| poll-source-uq-bpinn | Is concentration at point P above the regulatory limit, with what confidence? (compliance) | exceedance probability from mean +- sigma at P | YES (mean/std baked) | estimate WITH honest uncertainty from sparse sensors |
| dyn-double-pendulum | How far ahead can ANY surrogate predict this machine? (predictability budget) | leave-time = 1.99 s vs the Lyapunov horizon | DONE (baked) | the honest ceiling of estimation |
| mine-thickener-settling | How long to settle below level z*, what final bed height? (thickener SIZING: the Kynch method is literally this) | mudline trajectory, settling time, bed height per R | YES (track the front in phi(z,t)) | one parametric net = the whole flux-family design chart |
| mine-flotation-kinetics | What recovery in t minutes; what residence time for target recovery? (circuit design) | R(t) = 1 - C; t*(R_target, k) | YES (C(k,t) baked) | the k-family in one net = the design curve |
| mine-heap-leach-rt | When does the reagent break through at the base; how much leaves? (irrigation planning) | breakthrough time at z=1; outlet flux | YES (frames baked) | amortized: t is an input; scenario screening |
| mine-comminution-pbe | What grind rate reaches X% passing size s* at time t? (mill sizing, P80-style) | passing fraction below s* over time per g | YES (integrate n(s,t)) | parametric family = the operating chart |
| poll-ocean-transport | When does the spill reach the coast point, at what peak concentration? (emergency response) | arrival time + peak c at (x*,y*) | YES (frames baked) | fast what-if per release scenario |
| poll-soil-barrier | How much does the cutoff wall delay the plume? (remediation design) | breakthrough delay at the outlet | YES (c(x,t) baked) | the D-jump kink needs the FBPINN mechanics |
| poll-tailings-seepage | What seepage flux through the dam for material alpha? (dam safety) | flux K(psi)(psi_z+1) at the base per alpha | YES (finite-diff on psi) | parametric alpha = material screening |
| bench-heat1d | How long until the centerline cools below half? (quench/food/electronics cooling) | t_half(alpha); analytic check ln 2/(alpha pi^2) | YES + closed-form check | one net answers for EVERY alpha |
| bench-burgers1d | When does the front arrive at x*? (flood/pressure-front arrival) | arrival time t(x*; nu) | YES (track u=0.5 crossing) | parametric nu family |
| bench-allencahn | Where are the domain walls at t=1; how fast do they move? (microstructure coarsening) | interface positions/velocity | YES (zero crossings) | the stiff-dynamics lesson feeds the estimate |
| bench-navier-cavity | Where is the vortex core and how strong? (mixing quality) | vortex-core location (+ Ghia's own core benchmark) | YES (u=v=0 interior point) | the coupled-field lesson |
| bench-poisson2d | What peak value and where, per source mode? (hot-spot/stress screening) | u_max + argmax per k | YES | honest: classical wins the single solve; the family is the value |
| bench-wave1d | What oscillation frequency does the string have at speed c? | frequency c/2 from the baked period | YES | parametric family |
| ind-helmholtz | What field amplitude at a receiver at frequency k0? (acoustics/EM at a target) | u at a probe point | YES | the spectral-bias lesson: naive estimators fail at high k |
| bench-darcy-operator | Screen 1000 geology maps: peak pressure per map, instantly | peak head + location per held-out a(x) | YES | amortized estimation: 1 pass vs 1 solve per map |
| ctrl-zero-source | Does the machinery return zero when the answer IS zero? | the null estimate (u = 0) | DONE | the estimator's negative control |

Every case HAS a real question; three already estimate; the rest need their QoI computed from existing bakes (a
training-free baker) and the question TOLD.

## 4. The plan (phased; each phase a validated, screenshot-QA'd release)

### Phase A: compute THE ANSWERS (training-free baker: build_estimates.py)
For each case x variant, compute the QoI above from the baked traces/frames (front tracking, integration, finite
differences, argmax, threshold crossings, exceedance from mean/sigma). Write into the manifest as
`estimate: { question_en/es, qoi_label_en/es, value, unit, detail_en/es, per_variant? }`. All REAL numbers derived
from the already-validated fields: no assertion, no training. Contract-test the block.

### Phase B: tell THE QUESTION on the web
- **Answer card** (new, top of the workbench stage, every view): "THE QUESTION" (one sentence, who asks it) ->
  "THE ESTIMATE": the QoI value(s) with unit, reacting to the regime chips where per-variant. The FIRST thing a
  visitor reads on a case is the question it answers, not the operator.
- **Why-a-PINN line** per case (honest: forward cases say the classical solver also does this; the value is the
  family/portability; inverse cases say this is the native tool).
- **Case list**: show the question fragment (not just the method) under each case name.
- **Introduction**: add the estimation framing UP TOP: "A PDE is an instrument: it transports cheap measurements to
  the quantity you need. The catalogue's real content is the QUESTIONS each case answers." Rework the story
  selector labels to lead with questions where natural.
- **Methodology**: a new "PINNs as estimators in the wild" section from the deep-research findings (HFM velocity
  from dye, elastography moduli, permeability from heads, battery parameters, seismic velocity models), each mapped
  to the catalogue case that teaches the same mechanism. Citations with DOI/arXiv.

### Phase C: estimation-first navigation polish
- Benchmark table gains the QoI column (the estimate per case).
- The story chapters cross-link question -> case (labels already short; blurbs updated to the question).

### Phase D (research-dependent, only if the findings justify): one NEW flagship estimation case
Candidate: a Hidden-Fluid-Mechanics-style "estimate the velocity field from passive scalar frames" mini-case (the
Science 2020 pattern), IF feasible on the CPU lane. Decide AFTER reading the research dossier; never fabricate.

## 5. Research grounding
Deep-research workflow running (PINN-as-estimator applications with QoI/measurement/why-PINN per class + honest
limitations). Findings + citations will be appended here and drive the Methodology section (Phase B) and the Phase D
decision.

## 6. Relation to previous plans
- Supersedes the presentation emphasis of dynamics-evaluation-2026-07-10.md (motion) and
  fundamentals-review-2026-07-10.md (constraints/budget): both remain valid layers; this reframe adds the MISSING
  first layer: the question. Order of a case page becomes: QUESTION -> ANSWER (estimate) -> how (the ladder,
  the dynamics, the constraints) -> how much information (identifiability).
