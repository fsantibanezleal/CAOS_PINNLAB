# Deep review against the FUNDAMENTALS (2026-07-10)

Owner questions: why are PDEs relevant; what is the target of a PINN; what is it attempting to reach; what side
conditions constrain the result; how much constraint/extra information does a PINN need to give a significant
result. This review answers each with the app's own computed numbers, then audits whether the app TEACHES them.

## 1. Why PDEs are relevant

A PDE is the local form of a physical law: it states that the rate of change at a point is determined by what
happens in its immediate neighbourhood (fluxes, gradients, curvatures). Conservation of mass, momentum and energy,
Fourier's law, Fick's law, Darcy's law, Newton's second law on continua: all become PDEs. They are relevant because
solving them buys the three predictions engineering runs on:
- the FUTURE of a system (time evolution: heat1d, allencahn, ocean-transport, thickener),
- the UNSEEN INTERIOR between measurements (soil-heat-real reconstructs 10/20/50 cm from 5/100 cm sensors),
- the RESPONSE to design/parameter changes (the Live sliders: viscosity, diffusivity, grind rate; darcy's operator).
Closed forms exist only for idealized cases; real geometry/coefficients force numerical solution. That is why
solvers matter at all, and why a new solver class (PINNs) is worth a catalogue.

## 2. What a PINN targets / attempts to reach

Formally: find network weights theta minimizing
L(theta) = lambda_r sum |N[u_theta](x_i)|^2 (the PDE residual at collocation points)
         + lambda_b sum |B[u_theta](x_j)|^2 (boundary/initial conditions)
         + lambda_d sum |u_theta(x_k) - u_k|^2 (observed data, when any).
The point it is attempting to REACH is the function where every term vanishes simultaneously. For a well-posed
forward problem that zero-loss function IS the classical solution; the attempt happens through non-convex
optimization, which is exactly where the pathologies live (Krishnapriyan 2021: failure is a training problem, not
an expressivity problem: our Training view shows the naive Helmholtz lane never leaving ~100% while information-
identical Fourier features converge to 9%).

Practically, the target is NOT to out-solve FEM on a single forward problem (story chapter 1: the classical solver
already wins there, poisson adapted_vs_std = 0.04%, and FDM is exact and faster). The real target is the solution
AS AN OBJECT: mesh-free, differentiable anywhere, parametric (one net = a whole family: our Live tab), fusable with
sparse data (inverse/assimilation), invertible for hidden coefficients, exportable (our ONNX in the browser).

## 3. What constrains the result (the anatomy of a well-posed problem)

A PDE alone has infinitely many solutions. What pins the solution down:
- the OPERATOR + its coefficients (alpha, nu, k0, a(x), k(x,y)): known in forward problems, the unknown in inverse,
- the DOMAIN geometry,
- BOUNDARY conditions (Dirichlet u=0 walls in poisson/helmholtz; no-slip + moving lid in navier; REAL measured
  temperatures as Dirichlet data in soil-heat-real),
- INITIAL conditions for evolution problems (allencahn x^2 cos(pi x); pendulum theta(0)=120 deg, theta'(0)=0),
- FORCING/source terms,
- OBSERVED DATA, which substitutes for missing information in inverse problems,
plus the solver-side constraints that decide whether the optimizer can actually reach the constrained solution:
- HOW conditions are imposed (soft penalties vs hard output-transforms: the allencahn naive-vs-fix IS this ablation),
- the architecture's spectral bias (helmholtz tanh vs Fourier),
- collocation sampling (RAR chasing allencahn's moving layers),
- loss weighting (navier's 10x BC/gauge weights).

## 4. How much information does a PINN need for a SIGNIFICANT result (the information budget)

The catalogue's computed answers, by regime:
- FORWARD WELL-POSED: physics + BC + IC are mathematically COMPLETE: zero observations needed. Evidence: poisson
  0.04%, heat1d 0.15%, burgers 0.08% vs standard with no data at all.
- BUT information-complete is not trainable: with the SAME complete information the naive Helmholtz PINN delivers
  121% (spectral bias) and the naive soft allencahn 95% (metastable collapse). In hard forward problems the missing
  ingredient is not more information but the right inductive bias / constraint mechanism / sampling: the fix ladder.
- INVERSE, SCALAR unknown: small data suffices. soil-heat-real recovers ONE scalar diffusivity from two real
  boundary time-series and predicts three held-out interior depths at ~1 degC RMSE.
- INVERSE, FIELD unknown: interior data is ESSENTIAL and scales with the unknown's dimensionality. heat2d-inverse
  computed: 0 sensors = 356% (k underdetermined, the pure-physics lane is meaningless); ~100 sparse noisy sensors
  = 4%. Identifiability is also LOCAL: k is well-determined where |grad T| is large and poorly determined where the
  temperature gradient vanishes (the case's own expected_band records this).
- DEGENERATE CONTROL: zero-source (a=0 -> u=0) is the sanity floor: the machinery must return the trivial solution.
- THE CEILING: no amount of information buys long-horizon prediction of chaos: the pendulum's leave-time ~2 s is a
  Lyapunov limit, not a data or training deficit. Past it, only re-anchoring on new observations helps.
- HONESTY UNDER SCARCITY: with sparse data the significant result INCLUDES the uncertainty: the ensemble's sigma
  grows exactly where sensors are absent (source-uq).

## 5. Audit: does the app TEACH this? (content, flow, cases, viz, interactivity)

What works (verified): the story axis (win/lose), the computed contrasts (Compare/Training/Diagnostics), motion
everywhere the content is motion, both-mode probes, deep links, honest labels, per-case equations WITH their BC/IC
rendered, the inverse case showing its sensors as dots.

GAPS found by this lens (audited in code, counts are real):
- A. WHY PDEs: zero mentions in the Introduction (grep: 0 hits for conservation/law/predict). The app starts at
  "PINNs embed the governing equation" and assumes the reader already cares about PDEs.
- B. THE TARGET stated crisply: the loss is shown, but nowhere states "the point reached at zero loss IS the
  solution for a well-posed problem; the practical target is the solution as a differentiable, data-fusable object".
- C. CONSTRAINT ANATOMY: BCs/ICs appear inside each equation string but are never dissected (operator vs BC vs IC
  vs data vs solver-side constraints). Only the inverse case VISUALIZES where its information lives (sensor dots);
  forward cases never mark that the domain edges carry the BCs or that the t=0 row is the IC.
- D. THE INFORMATION BUDGET: the evidence exists in fragments (0 vs 100 sensors; complete-info naive failures) but
  the direct quantitative answer, an IDENTIFIABILITY CURVE (recovery error vs number of sensors, N = 0..100), was
  designed in the ladder document and never computed. This is the single most valuable missing computation.
- E. FLOW: no 60-second primer connecting PDE -> conditions -> PINN -> result for a first-time visitor; the story
  selector assumes the concepts.

## 6. Proposed implementation (pending owner go)

1. COMPUTE the heat2d-inverse sensor-count sweep (N = 0, 10, 25, 50, 100; short trainings) -> a Diagnostics
   identifiability curve "recovered-k error vs number of sensors": the computed answer to question 5.
2. Introduction: a new opening section "Why PDEs, what a PINN reaches, and the information budget" (EN/ES), using
   the catalogue's own numbers; states the four regimes (forward-complete, hard-forward, inverse-scalar,
   inverse-field) + the chaos ceiling.
3. Constraint-anatomy strip: per case, compact chips naming operator / BC / IC / parameter / data constraints (a
   short structured `constraints` field per case; the strip renders them color-coded).
4. Mark the information sources on the fields: edge highlight = BC, a t=0 tick = IC (the inverse dots already exist).
