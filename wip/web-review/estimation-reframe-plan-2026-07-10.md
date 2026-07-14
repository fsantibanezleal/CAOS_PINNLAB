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



## 5b. RESEARCH FINDINGS (deep-research workflow wf_54a23afd-263: 20 adversarially-verified claims, 3-0 votes; synthesis step rate-limited so claims listed directly)

Answering the owner's question 'did you search for cases where PDEs/PINNs estimate a parameter or metric?': yes, and the verified record follows. Every claim carries its primary source + verbatim quote.

**1.** The foundational 2019 PINN paper explicitly defines two problem classes, and the second, 'data-driven discovery of partial differential equations', is exactly the estimation-instrument use: inferring unknown PDE parameters/coefficients from measurement data. This is the primary-source basis for research-question item 8 (system identification) and the template all later application classes (thermal, permeability, elastography, seismic) instantiate.
   - Source: https://www.sciencedirect.com/science/article/abs/pii/S0021999118307125
   - Quote: "In this work, we present our developments in the context of solving two main classes of problems: data-driven solution and data-driven discovery of partial differential equations."

**2.** The core mechanism that makes a PINN usable as a proxy for unmeasurable quantities is stated in the opening sentence: the network is trained on supervised data while being constrained to satisfy governing nonlinear PDEs, so physics acts as regularization that lets indirect/partial observations determine hidden fields and parameters, without meshing or a discretized forward solver in an outer optimization loop as in classical FEM-plus-adjoint inversion.
   - Source: https://www.sciencedirect.com/science/article/abs/pii/S0021999118307125
   - Quote: "We introduce physics-informed neural networks – neural networks that are trained to solve supervised learning tasks while respecting any given laws of physics described by general nonlinear partial differential equations."

**3.** Raissi, Yazdani and Karniadakis (Science 367(6481):1026-1030, 2020, DOI 10.1126/science.aaw4741) introduce Hidden Fluid Mechanics (HFM), a physics-informed deep-learning framework that estimates velocity and pressure fields from flow-visualization images (passive scalar/concentration transport data) by encoding the Navier-Stokes equations into the neural network, because extracting these fields directly from images is otherwise challenging.
   - Source: https://pubmed.ncbi.nlm.nih.gov/32001523/
   - Quote: "Although such flow patterns can be, in principle, described by the Navier-Stokes equations, extracting the velocity and pressure fields directly from the images is challenging. We addressed this problem by developing hidden fluid mechanics (HFM), a physics-informed deep-learning framework capable of encoding the Navier-Stokes equations into the neural networks"

**4.** The stated advantage of HFM over classical CFD-based inversion is that it does not require specification of the flow geometry or the initial and boundary conditions, which classical forward-solver-plus-optimization approaches typically need.
   - Source: https://pubmed.ncbi.nlm.nih.gov/32001523/
   - Quote: "a physics-informed deep-learning framework capable of encoding the Navier-Stokes equations into the neural networks while being agnostic to the geometry or the initial and boundary conditions."

**5.** The authors explicitly position HFM as an estimation instrument (a proxy) for quantities that cannot be measured directly, demonstrating it on several physical and biomedical flow problems rather than on a deployed industrial system, i.e. an academic demonstration.
   - Source: https://pubmed.ncbi.nlm.nih.gov/32001523/
   - Quote: "We demonstrate HFM for several physical and biomedical problems by extracting quantitative information for which direct measurements may not be possible."

**6.** The field's authoritative 2021 review (by the PINN originators) states as a Key Point that PINNs are effective and efficient specifically for ill-posed and inverse problems, i.e., the estimation-instrument setting where parameters, fields, or boundary conditions are unknown, and that scalability to large problems requires combining PINNs with domain decomposition.
   - Source: https://www.nature.com/articles/s42254-021-00314-5
   - Quote: "Physics-informed neural networks are effective and efficient for ill-posed and inverse problems, and combined with domain decomposition are scalable to large problems."

**7.** The review's stated motivation for preferring physics-informed learning over classical inversion pipelines (FEM plus optimization loops, adjoint methods) is cost and implementation burden: it asserts that solving inverse problems with hidden physics by conventional numerical means is often prohibitively expensive and demands special formulations and elaborate codes.
   - Source: https://www.nature.com/articles/s42254-021-00314-5
   - Quote: "Moreover, solving inverse problems with hidden physics is often prohibitively expensive and requires different formulations and elaborate computer codes."

**8.** As a flagship hidden-fluid-mechanics estimation example, the review's Figure 2 shows a PINN inferring the 3D flow field over an espresso cup from Tomo-BOS (tomographic background-oriented schlieren) imaging data, i.e., velocity estimated from an indirectly measured scalar field on real experimental data; this is a laboratory/academic demonstration rather than an industrial deployment, in the same class as Raissi et al.'s Hidden Fluid Mechanics (velocity and pressure from flow visualizations), which the review cites.
   - Source: https://www.nature.com/articles/s42254-021-00314-5
   - Quote: "Inferring the 3D flow over an espresso cup based using the Tomo-BOS imaging system and physics-informed neural networks (PINNs)."

**9.** The survey characterizes PINNs (as introduced by Raissi et al. 2019) as a single framework that handles both forward solution of governing models and inverse problems in which the model parameters themselves are learned from observable data, i.e., the PINN functions as a parameter-estimation instrument, which is the core framing of the research question.
   - Source: https://dl.acm.org/doi/10.1007/s10915-022-01939-z
   - Quote: "They created physics-informed neural networks (PINNs) which can handle both forward problems of estimating the solutions of governing mathematical models and inverse problems, where the model parameters are learnt from observable data."

**10.** The survey's stated reason to prefer PINNs over classical inversion pipelines is code/workflow unification: the same code that solves the forward problem solves the inverse problem with minimal modification (no separate adjoint or FEM+optimization loop machinery), and it names characterizing fluid flows from sensor data (the Hidden Fluid Mechanics use case) as the canonical inverse example.
   - Source: https://dl.acm.org/doi/10.1007/s10915-022-01939-z
   - Quote: "In addition to solving differential equations (the forward problem), PINNs may be used to solve inverse problems such as characterizing fluid flows from sensor data. In fact, the same code used to solve forward problems can be used to solve inverse problems with minimal modification."

**11.** SINDy estimates the governing equations of a dynamical system themselves (which nonlinear terms are active and their coefficients) directly from measurement data, using sparse regression over a library of candidate functions rather than assuming a prior model structure. This defines the estimation task (quantity of interest = ODE terms/coefficients; data = state time series) that later PINN-based system-identification work targets.
   - Source: https://www.researchgate.net/publication/301627530_Sparse_Identification_of_Nonlinear_Dynamics_SINDy
   - Quote: "In particular, we use sparse regression to determine the fewest terms in the dynamic governing equations required to accurately represent the data."

**12.** The classical SINDy pipeline requires time derivatives of the measured state; when only the state is measured, derivatives must be obtained by numerical differentiation of noisy data, which the paper mitigates with total-variation regularized differentiation. This derivative-from-noisy-data bottleneck is the documented weakness that motivates PINN reformulations (differentiating a smooth network surrogate via autodiff instead of differentiating the raw data), i.e. it answers the 'why a PINN rather than classical identification' axis.
   - Source: https://www.researchgate.net/publication/301627530_Sparse_Identification_of_Nonlinear_Dynamics_SINDy
   - Quote: "we use the total variation regularized derivative to de-noise the derivative"

**13.** For saturated subsurface flow, the paper estimates the spatially distributed hydraulic conductivity field by approximating both conductivity and hydraulic head with two separate deep neural networks, trained jointly on sparse measurements of conductivity and head plus the Darcy's-law PDE constraint (residual minimized at selected points in the domain).
   - Source: https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2019WR026731
   - Quote: "For saturated flow, we approximate hydraulic conductivity and head with two DNNs and use Darcy's law in addition to measurements of hydraulic conductivity and head to train these DNNs."

**14.** For unsaturated flow governed by the Richards equation, the PINN estimates the unsaturated conductivity function (a constitutive relationship) using capillary pressure measurements ONLY, explicitly assuming zero direct measurements of unsaturated conductivity because that quantity is difficult to measure in the field, i.e. the PINN acts as a proxy instrument for an unmeasurable quantity.
   - Source: https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2019WR026731
   - Quote: "For unsaturated flow, we approximate unsaturated conductivity function and capillary pressure with DNNs and train these DNNs using measurements of capillary pressure and the Richards equation. Because it is difficult to measure unsaturated conductivity in the field, we assume that no measurements of unsaturated conductivity are available."

**15.** The stated rationale for the PINN over purely data-driven regression is that embedding the PDE constraint (Darcy or Richards residual at collocation points) increases accuracy when the target function is only sparsely observed and makes estimation possible at all when the function of interest has no direct measurements.
   - Source: https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2019WR026731
   - Quote: "We demonstrate that physics constraints increase the accuracy of DNN approximations of sparsely observed functions and allow for training DNNs when no direct measurements of the functions of interest are available."

**16.** On the saturated hydraulic conductivity estimation problem, the physics-informed DNN is claimed to be more accurate than the state-of-the-art classical inverse method used as baseline, maximum a posteriori (MAP) probability estimation, which is the paper's direct 'why PINN rather than classical inversion' evidence.
   - Source: https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2019WR026731
   - Quote: "For the saturated conductivity estimation problem, we show that the physics-informed DNN method is more accurate than the state-of-the-art maximum a posteriori probability method."

**17.** The paper presents what its authors state is the first full waveform inversion (FWI) for seismological applications using PINNs, and all validation is on synthetic case studies (2D acoustic wave equation; homogeneous, ellipsoidal-anomaly crosswell, teleseismic plane-wave, and nine-source checkerboard models), i.e. an academic demonstration rather than a deployed field system.
   - Source: https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2021JB023120
   - Quote: "Nevertheless, to our knowledge, we present the first FWI for seismological applications using PINNs. In this study, we focus on the development of acoustic FWI with PINNs and demonstrate its practical application to various synthetic case studies."

**18.** Why a PINN instead of classical inversion: the method is meshless and the inversions recovered anomaly location, dimensions, and magnitude even from deliberately poor, uneducated initial guesses, relieving the starting-model dependence of classical gradient-based FWI; a priori structural knowledge (e.g. purely depth-dependent velocity) can be encoded directly in the network parameterization.
   - Source: https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2021JB023120
   - Quote: "PINNs are a meshless method. They offer great flexibility in terms of implementation and if a priori knowledge is available for inverse problems. Moreover, our results show that PINNs perform well even without a priori knowledge or in the absence of an educated starting model."

**19.** Honest limitations are stated: for the pure forward problem, spectral element or finite difference solvers remain more efficient and accurate than PINNs; loss-term weighting and network sizing are heuristic (wrong choices can converge to a wrong solution); large computational domains incur large GPU memory costs; and inverted models recover smoothed rather than sharp discontinuities (e.g. the step-function ellipsoidal anomaly is retrieved with smeared boundaries).
   - Source: https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2021JB023120
   - Quote: "We find that the current state-of-the-art PINNs provide good results for the forward model, even though spectral element or finite difference methods are more efficient and accurate."

**20.** The paper uses a two-stage Bayesian PINN (TSBPINN) as an estimation instrument to invert groundwater contaminant SOURCE PARAMETERS (position and intensity) from noisy concentration observations, with a Gaussian likelihood modeling observational noise and posterior distributions providing uncertainty quantification, i.e., the PINN estimates a hidden quantity (the buried source) that cannot be measured directly.
   - Source: https://ascelibrary.org/doi/10.1061/JWRMD5.WRENG-7084
   - Quote: "A Bayesian probabilistic architecture models observational noise as Gaussian likelihood functions, enabling concurrent inversion of source parameters (position/intensity) and uncertainty propagation analysis through posterior distributions."

(11 verification agents on the last two claims hit a session rate limit; those claims are EXCLUDED (only 3-0-verified claims listed).)

## 6. Relation to previous plans
- Supersedes the presentation emphasis of dynamics-evaluation-2026-07-10.md (motion) and
  fundamentals-review-2026-07-10.md (constraints/budget): both remain valid layers; this reframe adds the MISSING
  first layer: the question. Order of a case page becomes: QUESTION -> ANSWER (estimate) -> how (the ladder,
  the dynamics, the constraints) -> how much information (identifiability).
