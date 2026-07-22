{
  "summary": "Deep research: neural operators (PINO/FNO family), geometric & structure-preserving physics-informed ML, PINN failure-mode fixes, and applied industrial/subsurface problems",
  "agentCount": 13,
  "logs": [
    "[stall] agent \"deepen:pinnfixes\" stalled (no progress) after 400s — retrying (1/5)",
    "[stall] agent \"survey:uq-inverse\" stalled (no progress) after 3107s — retrying (1/5)",
    "collected 101 methods, 256 sources",
    "[stall] agent \"synthesize:plan\" stalled (no progress) after 1390s — retrying (1/5)"
  ],
  "result": {
    "plan": "I verified the repo state directly; the brief's premise is out of date in one decisive way and the shipped artifact contradicts the plan it was built from. Both facts drive this plan.\n\n---\n\n# PINN-Lab: implementation plan, operator & structure-preserving expansion\n\n**Verified against the working tree** (`D:\\_Repos\\Research_Caos\\CAOS_PINNLAB`, branch `develop`, HEAD `a0ac5d5`) on 2026-07-22. Every repo number below is read from a committed manifest or measured in-session, not recalled.\n\n## 0. Two corrections to the brief before anything is planned\n\n**0.1 PINO is not absent. It is built, shipped as an artifact, and half-landed.** Commit `7e713bc` added `data-pipeline/pinnlab/model/pino.py` (241 lines: FD divergence residual with harmonic-mean faces, spectral Laplacian cross-check, hard-BC mask, gradient-norm λ balancing, test-time optimization with the anchor loss) and `data-pipeline/pinnlab/cases/bench_darcy_pino.py` (284 lines). `models/bench-darcy-pino.onnx` exists (2,625,415 B, parity 1e-8, opset 18, dynamo). Four variant artifacts exist in `data/derived/bench-darcy-pino/`. The catalogue is 22 pipeline cases, not 21.\n\nWhat is missing is the landing: **there is no `docs/cases/bench-darcy-pino.md`** and **no `frontend/src/content/cases/DarcyPinoContext.tsx`** (21 context files, no PINO one). Under the repo's own rule that a unit lands complete in one commit, this unit is incomplete.\n\n**0.2 The shipped PINO result does not support the claim the repo makes about it.** `data/derived/manifests/bench-darcy-pino.json`, held-out relative L2 over 32 test instances:\n\n| n_labels | PINO | data-only FNO | verdict | λ chosen by the balancer | train s (PINO / FNO) |\n|---|---|---|---|---|---|\n| 0 | 0.590 | 1.074 | PINO \"wins\" against an untrained net | 1.0 | 26.7 / 0.3 |\n| 8 | **0.344** | **0.152** | **FNO wins by 2.3x** | 29.3 | 84.7 / 28.5 |\n| 32 | **0.177** | **0.105** | **FNO wins by 1.7x** | 13.3 | 229.9 / 89.0 |\n| 128 | 0.057 | 0.071 | PINO wins by 1.24x | 3.33 | 791.4 / 396.9 |\n\n`wip/beyond-sota/plan-2026-07-15.md` §B.4 records a spike table where PINO wins at all six budgets, and commit `b95fab7` says \"PINO beats data-only FNO at every label budget\". **The built case shows the opposite in the middle of the range.** `docs/methods/operator-learning.md:208` marks PINO \"shipped\" and line 163 claims \"far less data hunger\". That claim is currently false on our own grid.\n\nThe mechanism is legible from the table: `balance_lambda` equalises gradient norms, so it drove λ to 29.3 at n=8, leaving the data term ~3% of the update. The heuristic that was added to fix the zero-label regime destroyed the mid-label regime, and no validation split ever checked it. This is the exact \"a claim consistent everywhere can still be false\" failure. **Section 4 is therefore a repair plan first and an extension plan second, and it blocks every other tier.**\n\n**0.3 A third fact that changes the cost-benefit of half the research findings.** Measured in-session on this machine: the classical reference solve `_solve_darcy` at 32x32 takes **42.4 ms**; the exported FNO ONNX inference takes **35.05 ms** (manifest). At the catalogue's grid size the neural operator has **no speed advantage at all** over the sparse direct solve, and our reference assembly is a Python double loop over `lil_matrix` (`datasets/darcy.py:31-49`), so 42 ms *overstates* the classical cost. At 64x64 the solve is 163 ms. Any speed claim in this product has to be earned at a grid size we do not currently run.\n\n---\n\n## 1. Gap table: what the catalogue claims vs the real 2026 state of the art\n\n| Family | Catalogue today (verified) | Real 2026 state of the art | Blunt assessment |\n|---|---|---|---|\n| **Operator learning** | 2 cases. `bench-darcy-operator` (data-only FNO, test L2 0.055, 64 test instances). `bench-darcy-pino` (built, undocumented, unwired, and losing at 2 of 4 budgets). | FNO is a 2021 baseline. The live axes are basis choice (HNO, arXiv:2606.24851), mode mixing (HO-FNO, arXiv:2606.28122), alias-free construction (CNO, arXiv:2302.01178), local+global kernels (arXiv:2402.16845), factorization (TFNO), and training schedules (iFNO, arXiv:2211.15188). | **Thin, and one method deep.** One backbone, one dataset family, one 32x32 grid. `model/fno.py` is 70 lines and hard-codes width/modes/layers per case. There is no operator *comparison* anywhere. |\n| **Physics loss on an operator** | Exists (`model/pino.py`), verified to machine precision (residual 3.6e-14 on the reference solution), but the case built on it underperforms the data-only baseline in the regime that matters. | PINO (arXiv:2111.03794) is settled; the 2026 work is on *which backbone* takes the physics loss (arXiv:2606.06164 finds CViT the most stable across DeepONet/FNO/CViT and five PDE systems) and on non-periodic residuals (FC-PINO, arXiv:2211.15960; `FCLegendre`/`FourierDiff` now ship in neuraloperator). | **The engine is right, the experiment is wrong.** Fixable in one unit of work. See §4. |\n| **UQ on operators** | **Zero.** `poll-source-uq-bpinn` is a Bayesian ensemble on a *coordinate* PINN; `ensemble_K: 0` appears in every operator manifest. | Split conformal prediction on neural operators is a dense 2026 literature: arXiv:2606.09923, 2606.08654, 2606.17513, 2606.29440, 2607.17297, 2602.08215. | **The single worst gap by value-per-hour.** We ship a surrogate with a headline error number and no bound on the next instance. |\n| **Geometric / structure-preserving** | **Zero.** `dyn-double-pendulum` is a plain residual PINN; energy is not conserved by construction. | HNN/LNN/DeLaN/SympNets are mature; the operator-side version is invariant projection (EP-FNO, arXiv:2606.14913) and hard divergence-free constraints at industrial scale (AB-UPT, arXiv:2502.09692). | **Absent, and the one mechanical case is the wrong model for its own system.** |\n| **PINN failure modes as a documented axis** | Remedies are shipped (Fourier features, hard constraints, RAR, loss weighting, causal curriculum docs) but never framed as failure modes. Ironically, §0.2 is an *unlabelled, unmeasured instance* of the gradient pathology firing in our own repo. | arXiv:2606.06164 establishes that gradient conflict and causality violation carry over from PINNs into PINOs, and that PINN-era mitigations still work there. | **Absent as an axis, and we just walked into it.** Our own λ=29.3 datum is the best teaching material in the repo. |\n| **Irregular geometry** | **Zero.** Every case is a rectangle or an interval. | Geo-FNO (arXiv:2207.05209), GINO (arXiv:2309.00583), OTNO (arXiv:2507.20065), Transolver (arXiv:2402.02366). | Absent. Also *correctly* deferred: none of our physics questions currently need it. Catalogue it, do not build it. |\n| **Subsurface groundwater** | **Zero.** `poll-tailings-seepage` is Richards (unsaturated); no well, no transient pumping, no analytic anchor. | Theis (1935) is the canonical anchor; U-FNO / Nested FNO (arXiv:2109.03697, 2210.17051) are the closest published product analogue. | Absent, and it is the one applied gap with a *closed-form* validation reference. Highest applied priority. |\n| **Foundation models / transfer** | Zero, and correctly so. | Poseidon, DPOT, MPP, PDE-Transformer, CoDA-NO. | Out of reach for pretraining; finetuning is realistic; **browser export is not** (0.5B params). Catalogue only. |\n| **Generative / sparse-observation solvers** | Zero. | DiffusionPDE (arXiv:2406.17763); arXiv:2602.12274 reports 7.7% vs 86.9% relative error at 25% observation for CO2 flow. | Absent. Genuinely the right shape for sparse boreholes, genuinely too slow for the browser. Stretch. |\n| **Browser deployment of operators** | `core/gate.py:25-29` forces **every** field-IO operator to `precompute` regardless of size or speed, via `web_drivable=False`. | — | **The WebGPU op-coverage analysis in the research findings buys this product nothing today.** No operator can be live no matter how export-friendly its ops are. That is a product decision to revisit (§2, T2.4), not a method problem. |\n\n---\n\n## 2. The new method ladder\n\nOrdered by (audience value x feasibility). Tier 0 blocks everything. Time estimates: measured where marked **measured**, otherwise extrapolated from the measured CPU baseline (`bench-darcy-pino` full build = 1,899,713 ms ≈ 31.7 min CPU for 8 trainings at 80 epochs on 32x32) and marked *(est.)*.\n\n### Tier 0 — Repair the PINO unit. Blocks all other tiers.\n\nNot a new method; the honesty precondition for adding any. Full detail in §4.\n\n---\n\n### Tier 1.1 — Split conformal prediction for neural operators\n\n**Highest value per hour in the entire plan.**\n\n**Formulation.** Given trained $\\mathcal{G}_\\theta$ and a calibration set $\\{(a_i,u_i)\\}_{i=1}^n$ disjoint from training, score $s_i(x) = |\\mathcal{G}_\\theta(a_i)(x) - u_i(x)|/\\hat\\sigma(x)$, then\n\n$$\\hat q = \\mathrm{Quantile}_{\\lceil (n+1)(1-\\alpha)\\rceil/n}\\big(\\{s_i\\}\\big), \\qquad C(a)(x) = \\big[\\mathcal{G}_\\theta(a)(x) \\pm \\hat q\\,\\hat\\sigma(x)\\big]$$\n\nwith $\\mathbb{P}(u(x)\\in C(a)(x)) \\ge 1-\\alpha$ under exchangeability. Whole-field coverage uses the max-score $s_i = \\sup_x |\\cdot|/\\hat\\sigma(x)$.\n\n**What it buys, with a number.** arXiv:2606.09923 reports 89.1% empirical coverage at $\\alpha=0.1$ on an FNO, with a 68%/32% epistemic/aleatoric split. arXiv:2607.17297 reports point-adaptive normalized calibration cutting mean interval width by 22.68% on pressure fields, and an out-of-fold protocol cutting coverage standard deviation from 10.41 to 3.10 percentage points. These are abstract-level figures per the findings; **our own coverage number is what we publish.**\n\n**Citation.** arXiv:2606.09923; arXiv:2607.17297; arXiv:2606.29440; arXiv:2602.08215 (Sobolev-space guarantees).\n\n**Recipe.** No training. Split the existing `bench-darcy-operator` pool: it already solves 64 test instances. Add 64 calibration instances (64 x 42 ms = **2.7 s of solver time**). Calibration is a `numpy.quantile` over held-out residuals: **< 1 s**. New module `model/conformal.py` (~80 lines) + `calibration.json` per operator case.\n\n**ONNX export path.** Unchanged. Ship $\\hat q$ and $\\hat\\sigma$ as constants in the manifest and apply the band client-side; no graph change, no new ops.\n\n**Validation anchor.** `coverage-at-alpha`: empirical coverage on a *third* held-out split vs nominal $1-\\alpha$, at $\\alpha \\in \\{0.2, 0.1, 0.05\\}$, plus mean interval width. Report the marginal-coverage caveat and a deliberate distribution-shift arm (calibrate on $\\sigma=3$ GRF, test on $\\sigma=1.5$) to *show* the guarantee degrading.\n\n---\n\n### Tier 1.2 — Localized integral and differential kernels on the FNO\n\n**Cheapest accuracy upgrade available, and it is pure `Conv`.**\n\n**Formulation.** Add a local branch to each Fourier layer: $v_{t+1} = \\sigma\\big(Wv_t + \\mathcal{K}_{\\text{spectral}}v_t + \\mathcal{K}_{\\text{local}}v_t\\big)$. The differential kernel is a CNN stencil $\\sum_{|i|\\le r} w_i\\,v(x+ih)$ with weights scaled $w_i/h^n$ **and constrained $\\sum_i w_i = 0$**, so it converges to an $n$-th order differential operator under refinement. The local integral kernel is $(\\mathcal{K}v)(x)=\\int_{B_r(x)}\\kappa_\\phi(x,y)v(y)dy$ with $\\kappa_\\phi(x,y)=\\sum_b\\phi_b\\psi_b(x-y)$ in a fixed continuous basis.\n\n**What it buys.** \"Adding our layers to FNOs significantly improves their performance, reducing the relative L2-error by **34-72%**\" on turbulent 2D Navier-Stokes and spherical shallow water (arXiv:2402.16845, abstract). Range is over the paper's experiments.\n\n**Citation.** arXiv:2402.16845, Liu-Schiaffini, Berner, Bonev, Kurth, Azizzadenesheli, Anandkumar, ICML 2024.\n\n**Recipe.** Extend `model/fno.py` with a `LocalKernel2d` module (~40 lines) and a `local=True` flag on `FNO2d`. Retrain `bench-darcy-operator` and the repaired `bench-darcy-pino` with and without it, same seed, same 80 epochs, same 32x32. Cost is roughly the measured baseline plus a Conv per layer: **~4 min CPU for the operator case** *(est., baseline 3.4 min measured)*.\n\n**ONNX export path.** `Conv` only. Verified WebGPU-present per the findings; irrelevant to our gate today (see T2.4) but relevant to artifact size.\n\n**Validation anchor.** Held-out relative L2 on the *same* 64 test instances, paired with the existing 0.055 baseline. **Honest caveat to write into the doc: Darcy on a smooth-ish 32x32 grid is not the multi-scale regime where the paper's 34-72% was measured. If we get 5%, we report 5%.**\n\n---\n\n### Tier 1.3 — Structure-preserving dynamics: Hamiltonian model + invariant projection\n\nCloses the largest *conceptual* gap (zero geometric methods) with a metric that cannot be fudged: energy drift.\n\n**Formulation (coordinate lane, the double pendulum).** Learn $H_\\theta(q,p)$ and integrate $\\dot q = \\partial H_\\theta/\\partial p$, $\\dot p = -\\partial H_\\theta/\\partial q$, so the flow is symplectic by construction and $H_\\theta$ is conserved along it exactly, versus the current plain residual PINN in which nothing is conserved.\n\n**Formulation (operator lane, invariant projection).** After each autoregressive step, project onto the invariant level set:\n\n$$u_{n+1} = \\Pi_{\\mathcal{I}}\\big(u_n + \\mathcal{G}_\\theta(u_n)\\big), \\qquad \\Pi(v) = v - \\frac{\\mathcal{I}(v)-\\mathcal{I}(u_0)}{\\|\\nabla\\mathcal{I}(v)\\|^2}\\,\\nabla\\mathcal{I}(v)$$\n\none Newton step, a handful of reductions and elementwise ops.\n\n**What it buys.** arXiv:2606.14913 reports improved long-time stability and more accurate propagation of soliton and coherent wave structures versus a standard FNO, specifically preventing \"drift in conserved quantities, phase error, and loss of qualitative accuracy\", validated on Zakharov-Kuznetsov, Kadomtsev-Petviashvili and sine-Gordon. **The abstract gives no quantitative bound** — do not invent one; our energy-drift number is ours. Industrial precedent for hard constraints at scale: AB-UPT enforces divergence-free \"without degradation in performance\" (arXiv:2502.09692).\n\n**Citation.** arXiv:2606.14913 (projection); arXiv:2502.09692 (hard-constraint precedent). HNN/LNN/SympNet selection per `wip/beyond-sota/plan-2026-07-15.md` §C.\n\n**Recipe.** New `model/hamiltonian.py` (~120 lines). Ladder on the *existing* `dyn-double-pendulum` (naive = shipped residual PINN, adapted = Hamiltonian) plus one new non-chaotic conservative case (§3.4) because chaos limits what a trajectory metric can say. Coordinate-lane training is MLP-scale: **2-5 min CPU per arm** *(est.)*.\n\n**ONNX export path.** The Hamiltonian net is an MLP; the symplectic integrator step is `Mul/Add/Sub` and the gradient of $H_\\theta$ must be **written analytically into the exported graph or the rollout done client-side** — `torch.autograd` is not available in ONNX Runtime. This is the one real export task in Tier 1; budget a day. The projection is `ReduceSum/Mul/Sub/Div`.\n\n**Validation anchor.** $|E(t)-E(0)|/E(0)$ over the window, plus the existing leave-time against RK45. For the projected operator lane: invariant drift over 500 rollout steps, naive vs projected.\n\n---\n\n### Tier 2.1 — Fourier continuation (`FCLegendre` / `FourierDiff`)\n\nDirectly upgrades the PINO unit's headline honesty caveat, and it is **already written** in neuraloperator.\n\n**Formulation.** Extend $f$ on $[0,1]$ to a smooth periodic $f^{\\text{ext}}$ on $[0,1+\\delta]$ by fitting Legendre/Gram basis polynomials over $n_{\\text{additional pts}}$ extension points; then $\\widehat{\\partial_x^n f}(k) = (i2\\pi k/L)^n\\hat f(k)$ on the extended field, restricted back to $[0,1]$.\n\n**What it buys for *us* specifically.** Our own measurement (`plan-2026-07-15.md` §B.3): on a reference Darcy field the naive spectral Laplacian gives rms residual **6.0**, i.e. six times the source term $f=1$, exactly the periodicity failure the PINO paper warns about. FC is the documented fix. **Crucially, `FCLegendre`/`FourierDiff` live in the loss, never in the exported graph** — so their FFTs are irrelevant to any deployment constraint. The findings note the neuraloperator docs show qualitative agreement plots, not error tables, so **we cannot cite an accuracy figure for the library implementation**; we measure our own rms residual against the FD reference.\n\n**Citation.** neuraloperator 2.0.0 example gallery, Layers section; FC-PINO, Maust et al. arXiv:2211.15960.\n\n**Recipe.** Add `spectral_laplacian_fc` beside the existing `spectral_laplacian` in `model/pino.py`. Sweep `fc_degree` and `n_additional_pts`; the pass/fail is rms residual on the *known* reference solution, target within 1-2 orders of the FD route's measured **3.6e-14**. **Zero training** for the diagnostic: **minutes**.\n\n**ONNX export path.** None required.\n\n**Validation anchor.** `residual-rms-on-reference`: feed the exact reference $u$, the residual must be ~0; feed $1.1u$, it must be ~10%; feed $u=0$, it must be exactly $f$. This is the three-way check already used for the FD residual and it is the reason that residual is trustworthy.\n\n---\n\n### Tier 2.2 — Convolutional Neural Operator (CNO)\n\nThe correct answer to aliasing, and the most export-friendly operator on the list.\n\n**Formulation.** Operate on band-limited $\\mathcal{B}_w$ with alias-free activations $\\Sigma_w(f) = \\mathcal{P}_w\\big(\\sigma(\\mathcal{U}_{w\\to 2w}f)\\big)$: upsample to $2w$, apply the pointwise nonlinearity, low-pass back to $w$. Rationale: $\\sigma(x)=x^2$ takes $K$ modes to $2K$, so resolve at $2w$ before projecting. U-Net-shaped stack around it. Universality proved.\n\n**What it buys.** The paper reports CNOs \"significantly outperform baselines\" on the RepresentativePDEBenchmarks suite with emphasis on robustness and the low-data regime. **The abstract carries no headline percentage — the tables are in the body. Do not quote a number without opening the PDF.** The concrete buy for us: alias-free *by construction* is a stronger answer than \"keep n_modes below N/2\", and it costs nothing at export.\n\n**Citation.** arXiv:2302.01178, Raonic et al., NeurIPS 2023. Subsurface application: arXiv:2509.20238 (velocity -> RTM image, gradient-based inversion without adjoint-state solves).\n\n**Recipe.** New `model/cno.py` (~150 lines). Train on the same Darcy family, 32x32 and 64x64, 80 epochs. CNN-scale: **~5-8 min CPU** *(est.)*, faster than the FNO lane since there is no FFT.\n\n**ONNX export path.** `Conv` + `Resize` + activation, no FFT anywhere. Works on the legacy TorchScript exporter as well as dynamo, so it sidesteps the complex-parameter question entirely. **One caveat to test:** the sinc/Fourier-zero-pad upsampler must be expressed as `Resize` (linear/cubic) or a fixed-weight transposed convolution; an FFT-based upsampler must not be exported.\n\n**Validation anchor.** Same 64 held-out instances, paired against the FNO's 0.055. Plus the super-resolution arm: train at 32x32, evaluate at 64x64, report both. This is where alias-free construction should show and where FNO should degrade.\n\n---\n\n### Tier 2.3 — Hartley Neural Operator (HNO) as a basis-selection *control*\n\nThe value here is the **selection rule**, not a win.\n\n**Formulation.** $\\mathrm{cas}(\\theta)=\\cos\\theta+\\sin\\theta$; $\\mathcal{H}[v](k)=\\sum_n v(n)\\mathrm{cas}(2\\pi kn/N)$, real-to-real, $\\mathcal{H}^{-1}=\\tfrac1N\\mathcal{H}$. Layer $(\\mathcal{K}v)=\\mathcal{H}^{-1}(R\\odot\\mathcal{H}v)$ with $R\\in\\mathbb{R}^{K\\times d_v\\times d_v}$ **real**, versus FNO's complex $R$. Iso-parametric at equal width; $2\\times$ the retained frequency corners since the real Hartley spectrum has no conjugate-symmetry halving.\n\n**Honest caveat, which is the paper's own point.** A real diagonal multiplier in Hartley space is a true convolution only for **even** kernels, because $\\mathcal{H}[f*g](k)=\\tfrac12[H_fH_g - H_f(-k)H_g(-k) + H_f(k)H_g(-k) + H_f(-k)H_g(k)]$. Hence the rule: **HNO for elliptic problems (symmetric Green's function), FNO for time-dependent ones.**\n\n**What it buys.** The paper reports **no explicit speed or accuracy improvement numbers**; its contribution is the predictive selection rule plus an iso-parametric comparison. **Never write \"HNO beats FNO\".** It explicitly expects FNO to stay preferable for time-dependent operators. Very recent (June 2026), no independent reproduction.\n\n**Citation.** arXiv:2606.24851, Sulskis & Ravi.\n\n**Recipe.** `model/hno.py` (~70 lines), simpler than our existing `SpectralConv2d` because there is no real/imag split to carry. At truncated modes ($k_{\\max}\\sim 12$-$16$) the `cas` matrix is tiny: one small real matmul per axis. Same cost order as the equal-width FNO: **~3-4 min CPU** *(est.)*.\n\n**ONNX export path.** `MatMul`/`Einsum`/`Mul`/`Add`/`Slice`/`Concat` only. No `torch.cfloat` parameter, no `view_as_real` bridge, no DFT.\n\n**Validation anchor.** A **two-arm** study, which is the whole point: HNO vs FNO on (a) Darcy/Poisson (elliptic, symmetric Green's function) and (b) `bench-heat1d`/`bench-wave1d` (time-dependent). The publishable result is *the rule reproducing or not reproducing on our problems* — a null result is a valid outcome and must be shipped as one.\n\n---\n\n### Tier 2.4 — Product decision: make field-IO operators live\n\n**Not a method. The precondition for any of the export-friendliness in Tiers 2.2-2.3 to matter.** `core/gate.py:25-29` sends every operator to `precompute` on `web_drivable=False`, so a CNO with perfect WebGPU op coverage still ships as a baked PNG-equivalent. The unlock is a browser affordance where the user *paints or perturbs a coefficient field* and the ONNX runs on it live. Our FNO artifact is 2.62 MB against a 4 MiB gate and 35 ms against a 120 ms gate — **it already passes both numeric gates.** Decide this before investing in export-friendly architectures.\n\n---\n\n### Tier 2.5 — iFNO schedule + TFNO factorization\n\n**Formulation.** iFNO: grow $k_{\\max}$ when $\\sum_{k\\le k_{\\max}}\\|R(k)\\|_F^2 / \\sum_k\\|R(k)\\|_F^2$ crosses a threshold, and refine training resolution on a schedule. TFNO: store $R$ in Tucker form at rank 0.1.\n\n**What it buys.** iFNO reports **10% reduction in testing error, 20% fewer frequency modes, 30% faster training** (arXiv:2211.15188). TFNO gives parameter reduction with the FNO interface unchanged.\n\n**Citation.** arXiv:2211.15188, George, Zhao, Kossaifi, Li, Anandkumar; TFNO via `factorization='Tucker', rank=0.1`.\n\n**Recipe.** A training callback; the exported model is a plain FNO. Effectively free.\n\n**Limit to state.** The mode-growth criterion will happily grow $k_{\\max}$ toward Nyquist at the *training* resolution, which is exactly what breaks super-resolution. **Cap it below $N/2$ explicitly** and say so.\n\n---\n\n### Tier 3.1 — U-FNO + FiLM conditioning for the two-phase subsurface case\n\n**Formulation.** $v_{t+1}=\\sigma\\big(Wv_t + \\mathcal{K}v_t + \\mathcal{U}(v_t)\\big)$ with $\\mathcal{U}$ a small U-Net supplying the sharp front features a mode-truncated spectral layer smears. FiLM for scalar controls: $v\\mapsto\\gamma(s)\\odot v + \\beta(s)$ with $s$ = injection rate, viscosity ratio. **This is how a UI slider is wired into an operator correctly**, instead of retraining per setting.\n\n**What it buys.** U-FNO is superior to both FNO and CNN benchmarks on gas saturation and pressure buildup, \"requiring only a third of the training data to achieve the equivalent accuracy as CNN\" (arXiv:2109.03697). FiLM conditioning on scalar inputs reduces gas-saturation error **21%** versus plain U-FNO (arXiv:2511.20543). Benchmark arXiv:2606.07215 finds U-FNO best for transient saturation, plain FNO best for pressure.\n\n**Citation.** arXiv:2109.03697 (Wen, Li, Azizzadenesheli, Anandkumar, Benson, *Adv. Water Resour.* 2022); arXiv:2511.20543; arXiv:2606.07215.\n\n**Recipe.** `model/ufno.py` (~110 lines) = existing `FNO2d` + a 3-level U-Net branch. Data is the real cost: a 1D/radial two-phase solver in `datasets/twophase.py`, 256 training instances. **~15-25 min CPU** *(est.)*, less on GPU.\n\n**ONNX export path.** FNO story + `Conv`/`Resize`. FiLM is `Mul`/`Add`.\n\n**Validation anchor.** Buckley-Leverett analytic front (§3.5) — a genuine closed form, not a solver-vs-solver comparison.\n\n---\n\n### Tier 3.2 — Derivative-informed training (DINO / DIFNO)\n\nThe largest *capability* addition: turns a forward-replay surrogate into something usable for inversion.\n\n**Formulation.**\n$$\\mathcal{L}_{H^1}=\\mathbb{E}_{m\\sim\\mu}\\Big[\\|\\mathcal{G}_\\theta(m)-\\mathcal{G}(m)\\|^2_{\\mathcal{U}} + \\gamma\\|D\\mathcal{G}_\\theta(m)-D\\mathcal{G}(m)\\|^2_{\\mathrm{HS}}\\Big]$$\nFull Jacobians are never formed: restrict the derivative term to a derivative-informed subspace from the dominant eigenpairs of $\\mathbb{E}_m[D\\mathcal{G}(m)^*D\\mathcal{G}(m)]$, making cost independent of discretization dimension. **At inference the model and the graph are unchanged.**\n\n**What it buys.** arXiv:2305.20053 reports $10^3$-$10^7\\times$ execution-time reductions for PDE-constrained optimization under high-dimensional uncertainty; arXiv:2312.14810 reports >$1000\\times$ for Bayesian optimal experimental design including offline construction; arXiv:2403.08220 reports effective posterior samples 3-9$\\times$ faster than geometric MCMC with break-even after 10-25 effective samples. **Per the findings these are abstract-level figures from the arXiv listing, not tables read in full text — label them as such in any doc.**\n\n**Citation.** arXiv:2206.10745 (O'Leary-Roseberry, Chen, Villa, Ghattas); arXiv:2512.14086 (DIFNO); arXiv:2509.13620 (DeFINO, subsurface); arXiv:2504.08730 (error analysis).\n\n**Recipe.** Feasible **only because we generate our own data**: our `_solve_darcy` is differentiable through `torch.func.jvp` if reimplemented in torch, giving the tangent action for free. That reimplementation (a torch sparse solve or a fixed-point/CG inner loop) is the real work — budget 2 days. Then the reduced-basis eigenproblem is computed once offline.\n\n**ONNX export path.** Identical to the plain model. Derivative supervision lives only in the loss.\n\n**Validation anchor.** Not forward L2 — **inversion quality**: recover $a(x)$ from sparse head observations using the surrogate's Jacobian, versus using the data-only surrogate's Jacobian, versus the true adjoint. The honest metric is inversion error at fixed compute budget.\n\n---\n\n### Tier 3.3 — HO-FNO (explicit mode mixer)\n\n**Formulation.** FNO's spectral layer is diagonal in the mode index, but a quadratic nonlinearity $uu_x$ is a *convolution* in mode space: $\\widehat{uu_x}(k)=\\sum_{k_1+k_2=k}\\hat u(k_1)(ik_2)\\hat u(k_2)$. HO-FNO parameterizes that directly:\n$$\\hat w_l(k)=\\sum_j[R^{(1)}(k)]_{lj}\\hat v_j(k) + \\sum_{k_1+k_2=k}\\sum_{j_1,j_2}[T(k_1,k_2)]_{lj_1j_2}\\hat v_{j_1}(k_1)\\hat v_{j_2}(k_2)+\\cdots$$\n\n**What it buys.** The abstract reports consistent improvement over other spectral neural operators and the headline \"a single HO-FNO layer outperforms FNO models with up to **16 layers**\" on Poisson with polynomial forcing. Fewer layers is directly a smaller artifact. **The headline is on a specific polynomial-forcing Poisson setup, not a general claim; June 2026, no independent reproduction.**\n\n**Citation.** arXiv:2606.28122, Colagrande, Caillon, Feillet, Allauzen.\n\n**Recipe.** Batched real einsum over the truncated index set. **Quadratic in retained modes per axis** — viable only because we truncate hard at $K\\sim10$. Natural target: `bench-burgers1d` and `bench-allencahn`, which have exactly the quadratic/cubic nonlinearities the mixer is designed for.\n\n**ONNX export path.** `Einsum`/`MatMul`/`Mul`/`Add`.\n\n**Validation anchor.** Held-out L2 at *matched parameter count*, and a layer-count sweep to test the single-layer claim on our problems.\n\n---\n\n### Tier 4 — Catalogue only, do not build\n\nDocumented in `docs/methods/` with citations and an explicit \"not implemented\" label, per the audit rule that broke the DeepONet/PINO claim in the first place:\n\n- **CViT as a PINO backbone** (arXiv:2606.06164) — the correct citation for \"PINO does not mean FNO\", and the source of the failure-mode axis. Cite it in §4's doc; do not build.\n- **Geo-FNO / GINO / OTNO** (arXiv:2207.05209, 2309.00583, 2507.20065) — no case needs irregular geometry. Note the export reality: GINO's radius query is data-dependent and does **not** export as a static graph.\n- **PDE foundation models** (Poseidon arXiv:2405.19101, DPOT 2403.03542, MPP 2310.02994, PDE-Transformer 2505.24717, CoDA-NO 2403.12553) — finetuning realistic, browser export impossible. **The \"finetune then distill into a small FNO/CNO student\" path is our proposal, not a published result, and must be labelled as such.**\n- **DiffusionPDE** (arXiv:2406.17763; CCS-specific arXiv:2602.12274 reporting 7.7% vs 86.9% at 25% observation) — right shape for sparse boreholes, sampling cost fatal for interactive use.\n- **Transolver / AB-UPT / KANO / SFNO / WNO / TFNO axis** — present SFNO/WNO/LNO/AFNO/TFNO/HNO as **one axis with a selection rule**, not five methods. Presenting them as five would be padding, which is exactly what arXiv:2606.24851 does properly.\n\n---\n\n## 3. New cases\n\n### 3.1 `gw-theis-pumping` — transient groundwater with a pumping well **[highest applied priority]**\n\n**Engineering question.** A dewatering well is switched on next to a pit wall. How far does the drawdown cone reach, and how fast, and can a surrogate answer that for any pumping rate without a new solve each time?\n\n**PDE.** Confined transient groundwater flow:\n$$S\\frac{\\partial h}{\\partial t} = T\\nabla^2 h, \\qquad 2\\pi r_w T\\left.\\frac{\\partial h}{\\partial r}\\right|_{r=r_w} = Q$$\n\n**Parameters, with units and justification.**\n\n| Symbol | Value | Unit | Note |\n|---|---|---|---|\n| $T$ | 100 | m²/day | transmissivity, $=K b$ with $K=10$ m/day, $b=10$ m |\n| $S$ | $1\\times10^{-3}$ | – | storativity, low-end confined/semiconfined |\n| $Q$ | 500 | m³/day | single production well |\n| $r_w$ | 0.15 | m | finite well radius (no point-sink singularity) |\n| domain | 10,000 x 10,000 | m | well at centre |\n| $t$ | 0 to 2 | days | |\n| $h_0$ | 100 | m | initial head |\n\n**Why these and not the ones in the public post.** The post's $S=0.2$ is a **specific yield**, i.e. an *unconfined* parameter, for which confined Theis is the wrong analytic solution (it needs Boulton/Neuman, or at minimum the Dupuit correction $s'=s-s^2/2b$). Using $S=0.2$ with Theis would produce a plausible-looking, wrong anchor. Chosen $T/S = 10^5$ m²/day gives radius of influence $\\sqrt{2.25Tt/S}=671$ m at $t=2$ d, comfortably inside the 5,000 m half-width, so the domain is effectively infinite over the validation window and Theis genuinely applies.\n\n**Validation reference — analytic.** Theis:\n$$s(r,t)=\\frac{Q}{4\\pi T}W(u), \\qquad u=\\frac{r^2S}{4Tt}, \\qquad W(u)=\\int_u^\\infty \\frac{e^{-\\xi}}{\\xi}d\\xi = E_1(u)$$\nSpot check: $r=50$ m, $t=1$ d gives $u=6.25\\times10^{-3}$, $W\\approx4.50$, $s\\approx1.79$ m. Compare on the annulus $r\\in[5,500]$ m where the finite-radius and point-sink solutions agree and the boundary has not yet reached.\n\n**Label.** `synthetic` with `validation_anchor: \"theis-analytic\"` — a closed-form anchor, the strongest class in the catalogue.\n\n**Tiers exercised.** Tier 0 (PINO), Tier 1.1 (conformal band on the drawdown), Tier 2.1 (Fourier continuation: this domain is emphatically non-periodic).\n\n**`system_type`:** `time-evol-2d`.\n\n---\n\n### 3.2 `gw-transmissivity-inverse` — recover $T(x)$ from sparse observation wells\n\n**Question.** Given drawdown records from 6 monitoring wells, where is the low-transmissivity barrier?\n\n**PDE.** Same as 3.1, with $T(x)$ heterogeneous and unknown. Inverse problem over $T$.\n\n**Parameters.** $T(x)\\in[10,300]$ m²/day as a thresholded GRF (reusing `_make_coeff`'s machinery), 6 observation wells, 2% Gaussian noise on the head records.\n\n**Validation reference.** The synthetic truth $T(x)$ used to generate the data (recovery error), plus a classical adjoint-based inversion as the compute-matched baseline.\n\n**Label.** `synthetic`.\n\n**Tiers.** Tier 3.2 (DINO/DIFNO) is the whole point — this is the case that justifies Jacobian supervision, and the metric is inversion error at fixed compute, not forward L2.\n\n---\n\n### 3.3 `bench-darcy-conformal` — how wrong can this surrogate be on the next field?\n\n**Question.** The operator reports 5.5% mean error. What is the *bound* on the instance I am about to feed it?\n\n**PDE.** Unchanged from `bench-darcy-operator`: $-\\nabla\\cdot(a\\nabla u)=1$, $u|_{\\partial\\Omega}=0$.\n\n**Parameters.** Existing family ($a\\in\\{3,12\\}$, $\\sigma=3$ GRF, 32x32) plus a 64-instance calibration split and a shift arm at $\\sigma=1.5$.\n\n**Validation reference.** Empirical coverage vs nominal $1-\\alpha$ on a third held-out split. Not a physics anchor — a *statistical* one, which is the correct anchor for this method.\n\n**Label.** `synthetic`.\n\n**Tiers.** Tier 1.1. Variants = $\\alpha\\in\\{0.2,0.1,0.05\\}$ plus the distribution-shift arm, so the UI shows coverage degrading under shift rather than asserting the caveat in prose.\n\n**`system_type`:** `uq-bayesian` (existing `UQBandKit`).\n\n---\n\n### 3.4 `dyn-kepler-orbit` — the clean conservation anchor\n\n**Question.** Does a learned dynamics model conserve what the system conserves, or does it just look right for 20 seconds?\n\n**ODE.** Two-body Kepler: $\\ddot{\\mathbf r}=-\\mu\\mathbf r/\\|\\mathbf r\\|^3$, with $H=\\tfrac12\\|\\mathbf p\\|^2-\\mu/\\|\\mathbf r\\|$ and angular momentum $L=r\\times p$ both exactly conserved.\n\n**Parameters.** $\\mu=1$ (non-dimensional), $e\\in\\{0.0,0.3,0.6,0.9\\}$ as the variant axis, $t\\in[0,50]$ (≈8 orbits at $a=1$).\n\n**Validation reference — analytic.** Kepler's equation $M=E-e\\sin E$ solved to machine precision gives the exact orbit; energy and angular momentum are exactly constant by construction.\n\n**Label.** `synthetic`, `validation_anchor: \"kepler-analytic\"`.\n\n**Tiers.** Tier 1.3. **Why this and not only the double pendulum:** the double pendulum is chaotic, so trajectory divergence conflates model error with Lyapunov amplification and a trajectory metric cannot separate them. Kepler is non-chaotic and integrable, so energy drift is attributable entirely to the model. Ship both: Kepler proves the mechanism, the double pendulum shows it under chaos.\n\n**`system_type`:** `ode-dynamical`.\n\n---\n\n### 3.5 `mine-twophase-injection` — displacement front in a porous medium\n\n**Question.** Injecting process water into a porous deposit, where is the displacement front after 30 days, and how does that move with viscosity ratio?\n\n**PDE.** Two-phase immiscible flow, fractional-flow form:\n$$\\phi\\frac{\\partial S_w}{\\partial t} + u_t\\frac{\\partial f_w(S_w)}{\\partial x}=0, \\qquad f_w=\\frac{\\lambda_w}{\\lambda_w+\\lambda_n}, \\quad \\lambda_i=\\frac{k_{ri}(S_w)}{\\mu_i}$$\nwith Corey relative permeabilities $k_{rw}=S_e^{n_w}$, $k_{rn}=(1-S_e)^{n_n}$.\n\n**Parameters.** $\\phi=0.25$; $u_t=0.5$ m/day; $\\mu_w=1$ cP, $\\mu_n\\in\\{0.5, 2, 10\\}$ cP (the variant axis, wired via **FiLM**, not retraining); $S_{wc}=0.15$, $S_{nr}=0.20$; $n_w=n_n=2$; $L=100$ m; $t\\in[0,30]$ days.\n\n**Validation reference — analytic.** Buckley-Leverett with the Welge tangent construction: the shock saturation $S_{wf}$ solves $f_w'(S_{wf})=\\big(f_w(S_{wf})-f_w(S_{wc})\\big)/(S_{wf}-S_{wc})$, and the front position is $x_f=u_t f_w'(S_{wf})t/\\phi$. **A genuine closed form for a sharp front** — precisely the feature that discriminates U-FNO from plain FNO.\n\n**Label.** `synthetic`, `validation_anchor: \"buckley-leverett-analytic\"`.\n\n**Tiers.** Tier 3.1 (U-FNO + FiLM), Tier 1.1 (conformal band).\n\n**Honest scope note for the doc.** This is the 1D fractional-flow model, not a reservoir simulation. It is the analogue of the CO2-storage line (arXiv:2109.03697), it is not that line.\n\n---\n\n### 3.6 `bench-basis-selection` — does the Green's-function rule hold on our problems?\n\n**Question.** Which spectral basis should an operator use, and is that a principled choice or folklore?\n\n**PDE.** Two arms held identical apart from the basis: **elliptic** ($-\\nabla\\cdot(a\\nabla u)=1$, symmetric Green's function) and **time-dependent** (`bench-heat1d`'s $u_t=\\alpha u_{xx}$).\n\n**Parameters.** Existing case parameters, unchanged, so the comparison is clean. Variants = {FNO-elliptic, HNO-elliptic, FNO-time, HNO-time}.\n\n**Validation reference.** Held-out relative L2 in each arm, at matched parameter count.\n\n**Label.** `synthetic`.\n\n**Tiers.** Tier 2.3.\n\n**The deliverable is the rule, not a winner.** arXiv:2606.24851 predicts HNO favourable on the elliptic arm and FNO on the time-dependent arm. If our numbers do not reproduce that, **the case ships the refutation**, consistent with how `CAOS_SEISMIC` treats null results.\n\n---\n\n### Ladder extensions (not new cases)\n\n- `dyn-double-pendulum`: add the Hamiltonian arm beside the shipped residual PINN; metric = energy drift + leave-time vs RK45.\n- `bench-darcy-operator`: add the local-kernel arm (Tier 1.2) and the conformal band (Tier 1.1).\n- `bench-burgers1d` / `bench-allencahn`: add the HO-FNO arm (Tier 3.3).\n\n---\n\n## 4. The PINO unit, in full\n\n**Framing correction.** This is a **repair-and-extend** unit, not a build. The engine is sound and verified; the experiment around it is not, the unit never landed in docs or web, and the repo currently makes a claim its own artifact refutes. Nothing else in this plan ships before this is fixed.\n\n### 4.1 What exists and is trustworthy\n\n`model/pino.py::darcy_residual_fd` — the finite-volume divergence form with harmonic-mean face conductivities, matching `datasets/darcy.py::_solve_darcy` exactly. Verified in both directions before any case was built on it:\n\n| Input | rms residual | Expected |\n|---|---|---|\n| the reference solution $u$ | **3.6e-14** | ~0 (machine precision) |\n| $1.10\\,u$ | **1.0e-01** | ~10%, the operator being linear |\n| $u=0$ | **1.000** | exactly $f=1$ |\n\nThe second and third rows matter as much as the first: a residual that is accidentally zero for everything passes row 1 and fails rows 2 and 3. Also trustworthy: the hard boundary mask $16x(1-x)y(1-y)$ (`boundary_mask`), the fixed `U_SCALE = 0.02` derived from the problem statement rather than from label statistics (which is what makes a zero-label run genuinely zero-label), and `test_time_optimize` with the anchor loss.\n\n### 4.2 What is broken\n\n1. **λ selection.** `balance_lambda` sets $\\hat\\lambda = \\|\\nabla\\mathcal{L}_{\\text{data}}\\|/\\|\\nabla\\mathcal{L}_{\\text{pde}}\\|$, which *equalises gradient norms*. At $n=8$ it chose $\\lambda=29.3$, leaving the data term ~3% of the update, so 8 labels contributed almost nothing and the physics term alone (which reaches only 0.590 in the zero-label run) dominated. Result: PINO 0.344 vs FNO 0.152. **No validation split ever checked this choice.** The heuristic was introduced to fix the zero-label regime and silently broke the mid-label regime.\n2. **The n=0 comparison is vacuous.** A data-only FNO with zero labels is an untrained random net; its 1.074 is not a baseline, and \"PINO wins by 47%\" against it is not a result.\n3. **No plain-PINN arm exists anywhere.** The commenter's question was about PINNs training slowly and blowing up gradients. Without a single-instance PINN arm, we cannot answer it.\n4. **The unit never landed:** no case doc, no web page.\n\n### 4.3 Pre-training scheme (paper Section 3, Algorithm 1)\n\nPhase 1 minimises\n$$\\mathcal{L}=\\mathcal{L}_{\\text{data}} + \\lambda\\,\\mathbb{E}_{a'\\sim\\mu}\\big\\|\\mathcal{R}\\big(a',\\mathcal{G}_\\theta(a')\\big)\\big\\|^2$$\nwhere $\\mathcal{L}_{\\text{data}}$ is relative-L2 over $n_{\\text{labels}}$ solved instances and the second expectation runs over **virtual instances** $a'\\sim\\mu$ drawn fresh every step. Generating $a'$ costs a Gaussian filter and a threshold (`_make_coeff`, $\\sigma=3$, threshold to $\\{3,12\\}$) with **no solve** — the paper's \"we have access to the unlimited dataset by sampling new $a_j$ in each iteration\". This is what makes the label budget the free variable and it is already implemented (`bench_darcy_pino.py:181-185, 214-221`).\n\nThe boundary condition is **hard**, not penalised (`to_physical` multiplies by `boundary_mask`). This is load-bearing and the reason must go in the doc: the residual is evaluated on the interior only, and the Darcy solution is unique only together with its boundary condition, so with a merely soft penalty a large physics weight drives the interior residual to zero around the *wrong* boundary values and converges to a different member of the solution family.\n\n**Fix to λ.** Replace the single auto-balanced λ with a **swept λ selected on a validation split**, per budget: $\\lambda\\in\\{0.1,1,3,10,30\\}$, 8 validation instances held out from the labelled pool (and, at $n=0$, from a small dedicated set). Report the full sweep in the case doc, not only the winner, and keep `balance_lambda` as a *reported diagnostic* rather than the selector. **The measured $\\|\\nabla\\mathcal{L}_{\\text{data}}\\|=3.28$ vs $\\|\\nabla\\mathcal{L}_{\\text{pde}}\\|=0.085$ imbalance (a factor of 38.6) is the best gradient-pathology teaching datum in the repo — surface it as the failure-mode axis (Gap table row 5), not as a solved problem.**\n\n### 4.4 The PDE loss on a grid output: spectral vs finite difference, explicitly\n\nAn operator emits $u$ on a grid, so there is no autograd in $x$. The paper: \"it is not straightforward to write out the solution function in the neural operator which directly outputs the numerical solution $u=\\mathcal{G}_\\theta(a)$ on a grid, especially for FNO which uses FFT.\" It offers finite difference ($O(n)$), Fourier ($O(n\\log n)$), or a query-function autograd, and warns \"finite difference methods require a fine-resolution uniform grid; spectral methods require smoothness and uniform grids.\"\n\n**We use finite difference as the primary, and we measured why.** Darcy here is non-periodic ($u=0$ on the boundary) with a piecewise-constant coefficient. On a reference field the naive spectral Laplacian route gives rms $|-\\bar a\\nabla^2u - f| = 6.0$, i.e. **six times the source term** — the periodicity assumption failing exactly as warned. Against the FD route's 3.6e-14, that is not a close call.\n\n$$\\mathcal{R} = \\frac{1}{h^2}\\Big[(a_E+a_W+a_N+a_S)u_C - a_Eu_E - a_Wu_W - a_Nu_N - a_Su_S\\Big] - f$$\n\nwith harmonic-mean faces $a_E = 2a_Ca_{E'}/(a_C+a_{E'})$ — the correct face average for a discontinuous coefficient, because it is what keeps the normal flux continuous across a material interface (an arithmetic mean over-conducts across a jump). Residual normalised by $f$ so the term is dimensionless.\n\n**The upgrade (Tier 2.1).** Add `spectral_laplacian_fc` using Fourier continuation, which is the documented fix for the non-periodic case (FC-PINO, arXiv:2211.15960; `FCLegendre`/`FourierDiff` ship in neuraloperator 2.0.0). Then the case can show **three** residual routes measured on the same reference field — FD (3.6e-14), naive spectral (6.0), FC-spectral (to be measured) — which turns an asserted trade-off into a table. **The FC machinery lives in the loss and never enters the exported graph, so its FFTs are irrelevant to any deployment constraint.** This resolves an apparent tension in the research findings, which treated FFT-in-the-loss and FFT-in-the-model as the same problem. They are not.\n\n### 4.5 Test-time optimization with the anchor loss (paper Section 3.2, phase 2)\n\nImplemented in `test_time_optimize`. Deep-copy the pre-trained net, freeze $u_0=\\mathcal{G}_{\\theta_0}(a)$, then for the queried instance minimise\n$$\\mathcal{L}_{\\text{pde}} + \\alpha\\,\\mathcal{L}_{\\text{op}}, \\qquad \\mathcal{L}_{\\text{op}}=\\big\\|\\mathcal{G}_{\\theta_i}(a)-\\mathcal{G}_{\\theta_0}(a)\\big\\|^2$$\nAdam, lr 1e-4, grad-norm clip 1.0, 120 steps, returning a per-step history so the app can **show** the residual falling rather than assert it. The anchor is what makes a fine grid usable: \"using a higher resolution and finer grid will reduce the truncation error. However, it may make the optimization unstable. Using hard constraints such as the anchor loss $\\mathcal{L}_{op}$ relieves such a problem.\"\n\n**Currently the TTO result is computed and stored in `_STATE` but never surfaced** — `tto_l2` and `tto_hist` do not reach `extra_metrics`, so they are absent from the manifest and invisible in the app. Fix: emit both, and render the residual-vs-step history as a chart in the case's web page. Add an $\\alpha$ sweep $\\{0, 0.1, 1, 10\\}$ including $\\alpha=0$ so the anchor's contribution is measured, not assumed.\n\n### 4.6 What to measure to show PINO beating both a plain PINN and a data-only FNO\n\nFour arms, same Darcy family, same seed, same held-out test set of 32 instances, same 32x32 grid.\n\n| Arm | What it is | Why it is in the table |\n|---|---|---|\n| **A. Plain PINN** | coordinate MLP $(x,y)\\to u$ for **one** instance, hard BC via the same mask, autograd residual | the commenter's baseline; the thing PINO is claimed to beat |\n| **B. Data-only FNO** | existing lane, $\\mathcal{L}_{\\text{data}}$ only | the operator baseline that currently wins at $n=8,32$ |\n| **C. PINO** | $\\mathcal{L}_{\\text{data}}+\\lambda\\mathcal{L}_{\\text{pde}}$ + virtual instances, λ swept on validation | the method |\n| **D. PINO + TTO** | C, then test-time optimization with the anchor on the queried instance | the full paper recipe |\n\n**Metrics.**\n\n1. **Data-efficiency curve.** Held-out relative L2 vs $n_{\\text{labels}}\\in\\{0,4,8,16,32,64,128\\}$ for B, C, D. The honest headline is the **crossing point**: at what budget does C match B at $2n$. The current spike claim was \"roughly 1.5-2x labels at the low end\"; the built case shows *no* advantage at 8 or 32. Whatever the repaired sweep gives is what ships.\n2. **Arm A is compared on a different axis, because comparing it on L2-vs-labels is category-confused.** A plain PINN uses zero labels by construction and solves one instance. The correct comparison is **wall-clock to reach a target relative L2 (say 5%) on ONE queried instance**: PINN-from-scratch vs PINO-pretrained-plus-TTO (arm D). That is the amortisation argument stated as a measurement.\n3. **Training cost, reported, because \"PINNs are slow\" is the whole critique.** Already measured: PINO 26.7 / 84.7 / 229.9 / 791.4 s against FNO 0.3 / 28.5 / 89.0 / 396.9 s at $n=0/8/32/128$. The PINO lane roughly doubles training time. **Report it; do not bury it.**\n4. **Amortisation crossover, measured, not asserted.** Measured in-session: classical solve 42.4 ms at 32x32, ONNX inference 35.05 ms. **At our grid size the operator does not beat the solver per query at all**, so the crossover is at infinity and training cost is never repaid. Repeat the measurement at 64x64 (solve 163 ms, measured) and 128x128, where the FFT-based operator's scaling should start to win. **Publish the grid size at which the operator first wins, or publish that it never did within our range.**\n5. **TTO effect.** Interior residual and relative L2 before vs after, plus wall-clock, plus the $\\alpha$ sweep.\n6. **OOD.** Test on a shifted family ($\\sigma=1.5$, and contrast $a\\in\\{1,50\\}$). The physics term should help most here, since the equation still holds off-distribution while the data prior does not.\n\n### 4.7 The honest caveat, stated in the case doc and the app\n\n- **The FD residual uses the same stencil as the reference solver.** Driving it to zero *is* solving the reference's discrete system. So \"PINO matches the reference\" is **not** a discretisation-accuracy result. What the physics term can buy, and all we may claim, is **data efficiency measured on our own grid**.\n- **32x32, a two-value thresholded GRF, one PDE family, 32 test instances.** A small-scale reproduction can show a *mechanism* and a *direction*. It cannot establish a rate, a scaling law, or a SOTA claim, and it cannot generalise to another PDE family.\n- **The paper's headline numbers are the paper's** (20x error and 25x speedup vs PINN on Kolmogorov flow, 400x vs a GPU pseudo-spectral solver, 3000x on Darcy inversion, arXiv:2111.03794). Cite them as theirs. Report ours separately, adjacent, with our grid size next to them.\n- **At $n=0$ the data-only comparator is an untrained network.** Say so, and drop the \"47% better\" framing.\n- **For one well-posed forward instance a classical sparse direct solve wins**, and at our grid size it wins on inference too (42.4 ms vs 35.05 ms is a tie against a deliberately slow Python-loop assembly). The operator's argument is many-query amortisation at grid sizes we have not yet run.\n\n### 4.8 Landing checklist for the unit\n\n`docs/cases/bench-darcy-pino.md` (theory, the three-way residual verification table, the λ sweep, all four arms, the caveats above, KaTeX equations, real DOIs, a theme-aware SVG of the two-phase scheme) · `frontend/src/content/cases/DarcyPinoContext.tsx` + registry entry, EN/ES · correct `docs/methods/operator-learning.md:163` (\"far less data hunger\") and line 208 to match the measured result · surface `tto_l2`/`tto_hist` in `extra_metrics` · add the failure-mode axis doc carrying the 38.6x gradient-imbalance datum · update `wip/beyond-sota/plan-2026-07-15.md` §B.4 to record that the spike did not reproduce in the built case, and why.\n\n---\n\n## 5. What we must not claim\n\n**On PINO.**\n- Not \"PINO beats the data-only FNO at every label budget.\" The shipped artifact shows it losing 2.3x at $n=8$ and 1.7x at $n=32$. Until the repaired sweep says otherwise, the honest statement is that the physics term helps at the extremes and hurt in the middle under a bad λ.\n- Not the paper's 20x / 25x / 400x / 3000x as ours.\n- Not \"PINO matches the reference solver\" as an accuracy result, while the residual shares the reference's stencil.\n- Not \"physics-informed means no data needed.\" Our own zero-label run reaches 0.590 relative L2, which is not a usable field.\n\n**On operators generally.**\n- Not \"the neural operator is faster than the solver.\" At 32x32, measured: 35.05 ms inference vs 42.4 ms solve, against a deliberately slow assembly. **No speed claim without a grid size and a measured crossover.**\n- Not Geo-FNO's \"$10^5\\times$ faster\", GINO's \"26,000x\" or Nested FNO's \"700,000x\" as anything resembling ours. GINO's figure is for a **scalar drag coefficient**, not a full field; Nested FNO's is a ratio against a full reservoir-simulator forecast, not a per-timestep number.\n- Not \"FNO is resolution-invariant\" without the truncation caveat: train-resolution mode growth toward Nyquist breaks super-resolution.\n\n**On the new methods.**\n- Not \"HNO beats FNO.\" arXiv:2606.24851 reports **no** speed or accuracy improvement numbers; its contribution is a selection rule, and it expects FNO to remain preferable for time-dependent operators.\n- Not a CNO percentage without opening the PDF — the abstract carries none.\n- Not a WNO figure at all; that abstract contains no quantitative comparison.\n- Not KANO's ~2500x. It is a quantum Hamiltonian learning result and does not transfer to subsurface flow. Also note arXiv:2504.11397: KANs beat MLPs mainly in shallow settings, \"not consistently better in deep configurations\".\n- Not conformal prediction as a *conditional* guarantee. Coverage is **marginal**, degrades under distribution shift, and exchangeability breaks across autoregressive rollout horizons unless calibrated per horizon.\n- Not invariant projection outside conservative systems. Groundwater with a pumping well is forced and dissipative; only mass is available there, not energy.\n- Not the DINO speedup figures as read from tables — per the findings they are abstract-level from the arXiv listing and need a full-text check before quotation.\n- Not \"distill a foundation model into a browser-sized operator\" as published. **That is our proposal and must be labelled as such.**\n- Not a Transolver or AB-UPT advantage at our scale. Their wins are at $10^5$-$10^8$ mesh cells; at tens of thousands of points on a mapped uniform grid an FNO is competitive and cheaper.\n\n**Where a classical solver still wins, stated plainly in the docs.**\n- **One well-posed forward instance.** A sparse direct solve is exact to machine precision, needs no training, and at our grid size is not slower.\n- **Any instance outside the training family.** The surrogate has no error control off-distribution; the solver does not care.\n- **When a certified error bound is required.** FEM/FV come with convergence theory and a posteriori estimators. A conformal band is a distribution-free *marginal* guarantee under exchangeability, which is a materially weaker statement, and we must not let it read as the stronger one.\n- **Small domains, few queries.** The operator's entire argument is amortisation over many queries. Below the crossover — which for us is currently *not reached at all* at 32x32 — the training cost is never repaid.\n- **Theory backing for the above:** Kovachki, Lanthaler & Stuart, arXiv:2402.15715. The curse of parametric complexity says that for general Lipschitz operators on infinite-dimensional spaces the required parameter count scales like $\\exp(\\epsilon^{-\\gamma})$, so universality alone buys nothing; every practical success rests on additional structure a specific PDE family happens to have. That is the citation that lets the catalogue state its limits with authority instead of hedging, and it should be added to `docs/methods/operator-learning.md`.\n\n---\n\n### Build order\n\n1. **Tier 0** — repair and land the PINO unit (§4). Blocks everything.\n2. **Tier 1.1** conformal (`bench-darcy-conformal`) — hours, not days.\n3. **Tier 1.3** structure-preserving (`dyn-kepler-orbit` + double-pendulum ladder).\n4. **Tier 2.1** Fourier continuation, folded back into the PINO unit as its third residual route.\n5. **`gw-theis-pumping`** — the applied case with the analytic anchor.\n6. **Tier 1.2** local kernels; **Tier 2.4** the live-operator product decision.\n7. **Tier 2.2 / 2.3** CNO and the basis-selection study.\n8. **Tier 3** U-FNO + two-phase, DINO + inverse, HO-FNO.\n9. **Tier 4** catalogue-only method docs, explicitly labelled not implemented.\n\nEach unit lands complete — pipeline, case, ladder, deep docs, web page — in its own commit before the next starts.",
    "methodCount": 101,
    "sources": [
      "https://arxiv.org/abs/2512.01421 - Fourier Neural Operators Explained: A Practical Perspective (Duruisseaux, Kossaifi, Anandkumar), 1 Dec 2025 rev 22 Jan 2026, NeuralOperator 2.0.0",
      "https://arxiv.org/abs/2606.00677 - On the Limits of Resolution Equivariance in Fourier Neural Operators (Colagrande, Caillon, Feillet, Allauzen), 30 May 2026",
      "https://arxiv.org/abs/2508.11119 - Are Fourier Neural Operators Really Faster for Time-Domain Wave Propagation? (Voytan, Li), 14 Aug 2025",
      "https://arxiv.org/abs/2606.24851 - Real vs. Complex Spectral Bases for Neural Operators: The Role of Green's Function Alignment / Hartley Neural Operator (Sulskis, Ravi), 23 June 2026",
      "https://arxiv.org/abs/2606.28122 - Higher-Order Fourier Neural Operator: Explicit Mode Mixer for Nonlinear PDEs (Colagrande, Caillon, Feillet, Allauzen), 26 June 2026",
      "https://arxiv.org/abs/2606.14913 - Structure-Informed Neural Operators for Long-Time Prediction of Parametric Hamiltonian PDEs (Obieke, Chukwuemeka, Oguadimma), 12 June 2026",
      "https://arxiv.org/abs/2606.06164 - On the training of physics-informed neural operators for solving parametric PDEs (Chen, Cui, Chen, Wang, Ma), 4 June 2026",
      "https://arxiv.org/abs/2602.19265 - Spectral bias in physics-informed and operator learning: Analysis and mitigation guidelines (Khodakarami, Oommen, Ahmadi Daryakenari, Beekenkamp, Karniadakis), 22 Feb 2026",
      "https://arxiv.org/abs/2601.14517 - Learning PDE Solvers with Physics and Data: A Unifying View of PINNs and Neural Operators (Dai, Chen, Wang, Jia, Xie, Kumar, Yu), 20 Jan 2026",
      "https://arxiv.org/abs/2509.16825 - KANO: Kolmogorov-Arnold Neural Operator (Lee, Liu, Yu, Wang, Jeong, Niu, Zhang), 20 Sept 2025 rev 24 Feb 2026",
      "https://arxiv.org/abs/2206.10745 - Derivative-Informed Neural Operator (O'Leary-Roseberry, Chen, Villa, Ghattas), 21 June 2022",
      "https://arxiv.org/abs/2512.14086 - Derivative-Informed Fourier Neural Operator: Universal Approximation and Applications to PDE-Constrained Optimization (Yao, Luo, Cao, Kovachki, O'Leary-Roseberry, Ghattas), 16 Dec 2025",
      "https://arxiv.org/abs/2305.20053 - Efficient PDE-Constrained optimization under high-dimensional uncertainty using derivative-informed neural operators (Luo, O'Leary-Roseberry, Chen, Ghattas)",
      "https://arxiv.org/abs/2312.14810 - Accurate, scalable, and efficient Bayesian optimal experimental design with derivative-informed neural operators (Go, Chen)",
      "https://arxiv.org/abs/2403.08220 - Derivative-informed neural operator acceleration of geometric MCMC (Cao, O'Leary-Roseberry, Ghattas)",
      "https://arxiv.org/abs/2509.13620 - DeFINO: a reduced-order derivative-informed neural operator for subsurface fluid-flow (Park, Bruer, Erdinc, Gahlot, Herrmann)",
      "https://arxiv.org/abs/2302.01178 - Convolutional Neural Operators for robust and accurate learning of PDEs (Raonic, Molinaro, De Ryck, Rohner, Bartolucci, Alaifari, Mishra, de Bezenac), NeurIPS 2023",
      "https://arxiv.org/abs/2402.16845 - Neural Operators with Localized Integral and Differential Kernels (Liu-Schiaffini, Berner, Bonev, Kurth, Azizzadenesheli, Anandkumar), ICML 2024",
      "https://arxiv.org/abs/2211.15188 - Incremental Spatial and Spectral Learning of Neural Operators for Solving Large-Scale PDEs (George, Zhao, Kossaifi, Li, Anandkumar)",
      "https://arxiv.org/abs/2402.02366 - Transolver: A Fast Transformer Solver for PDEs on General Geometries (Wu, Luo, Wang, Wang, Long), ICML 2024",
      "https://arxiv.org/abs/2502.02414 - Transolver++: An Accurate Neural Solver for PDEs on Million-Scale Geometries",
      "https://arxiv.org/abs/2505.24717 - PDE-Transformer: Efficient and Versatile Transformers for Physics Simulations (Holzschuh, Liu, Kohl, Thuerey), ICML 2025",
      "https://arxiv.org/abs/2405.19101 - Poseidon: Efficient Foundation Models for PDEs, NeurIPS 2024",
      "https://arxiv.org/abs/2403.03542 - DPOT: Auto-Regressive Denoising Operator Transformer for Large-Scale PDE Pre-Training, ICML 2024",
      "https://arxiv.org/abs/2310.02994 - Multiple Physics Pretraining for Physical Surrogate Models (McCabe, Regaldo-Saint Blancard, Parker et al.)",
      "https://arxiv.org/abs/2403.12553 - Pretraining Codomain Attention Neural Operators for Solving Multiphysics PDEs (CoDA-NO)",
      "https://arxiv.org/abs/2406.17763 - DiffusionPDE: Generative PDE-Solving Under Partial Observation (Huang, Yang, Wang, Park), NeurIPS 2024",
      "https://arxiv.org/abs/2304.07993 - In-Context Operator Learning with Data Prompts for Differential Equation Problems (Yang, Liu, Meng, Osher), PNAS",
      "https://arxiv.org/abs/2502.09692 - AB-UPT: Scaling Neural CFD Surrogates for High-Fidelity Automotive Aerodynamics (Alkin, Bleeker, Kurle, Kronlachner, Sonnleitner, Dorfer, Brandstetter)",
      "https://arxiv.org/abs/2402.15715 - Operator learning: algorithms and analysis (Kovachki, Lanthaler, Stuart)",
      "https://arxiv.org/abs/2306.03838 - Spherical Fourier Neural Operators: Learning Stable Dynamics on the Sphere (Bonev, Kurth, Hundt, Pathak, Baust, Kashinath, Anandkumar)",
      "https://arxiv.org/abs/2205.02191 - Wavelet neural operator (Tripura, Chakraborty)",
      "https://arxiv.org/abs/2111.13802 - Factorized Fourier Neural Operators (Tran, Mathews, Xie, Ong), ICLR 2023",
      "https://arxiv.org/abs/2207.05209 - Fourier Neural Operator with Learned Deformations for PDEs on General Geometries / Geo-FNO (Li, Huang, Liu, Anandkumar), JMLR 24 (2023)",
      "https://arxiv.org/abs/2309.00583 - Geometry-Informed Neural Operator for Large-Scale 3D PDEs / GINO (Li, Kovachki, Choy et al.)",
      "https://arxiv.org/abs/2507.20065 - Geometric Operator Learning with Optimal Transport / OTNO (Li, Li, Kovachki, Anandkumar), 26 July 2025",
      "https://arxiv.org/abs/2109.03697 - U-FNO: an enhanced Fourier neural operator-based deep-learning model for multiphase flow (Wen, Li, Azizzadenesheli, Anandkumar, Benson), Advances in Water Resources 2022",
      "https://arxiv.org/abs/2210.17051 - Real-time high-resolution CO2 geological storage prediction using nested Fourier neural operators (Wen, Li, Long, Azizzadenesheli, Anandkumar, Benson), Energy & Environmental Science 16(4) 1732-1741 (2023)",
      "https://arxiv.org/abs/2503.11031 - Fourier Neural Operator based surrogates for CO2 storage in realistic geologies (Chandra, Koch, Pawar, Panda, Azizzadenesheli, Snippe, Alpak, Hariri, Etienam, Devarakota, Anandkumar, Hohl), Mar 2025",
      "https://arxiv.org/abs/2305.17289 - Fourier-DeepONet for full waveform inversion (Zhu, Feng, Lin, Lu)",
      "https://arxiv.org/abs/2103.10974 - Learning the solution operator of parametric PDEs with physics-informed DeepONets (Wang, Wang, Perdikaris)",
      "https://arxiv.org/abs/2111.03794 - Physics-Informed Neural Operator for Learning Partial Differential Equations (Li, Zheng, Kovachki, Jin, Chen, Liu, Azizzadenesheli, Anandkumar), full text via ar5iv",
      "https://arxiv.org/abs/2010.08895 - Fourier Neural Operator for Parametric Partial Differential Equations (Li et al.), ICLR 2021",
      "https://raw.githubusercontent.com/microsoft/onnxruntime/main/js/web/docs/webgpu-operators.md - ONNX Runtime WebGPU EP operator support table (DFT absent, Einsum/MatMul/Conv/Resize/GridSample present)",
      "https://neuraloperator.github.io/dev/auto_examples/models/plot_FNO_darcy.html - neuraloperator FNO-on-Darcy example, 17.489 s / 191,881 params",
      "https://neuraloperator.github.io/dev/auto_examples/layers/plot_fourier_continuation.html - FCLegendre / FCGram Fourier continuation layers",
      "https://neuraloperator.github.io/dev/auto_examples/layers/plot_fourier_diff.html - FourierDiff spectral differentiation with use_fc",
      "https://neuraloperator.github.io/dev/auto_examples/index.html - neuraloperator example gallery (Divergence-Free Spectral Projection, DISCO Convolutions, Resampling, OTNO on Car CFD, incremental meta-learning, TensorGRaD)",
      "https://docs.nvidia.com/physicsnemo/latest/index.html - NVIDIA PhysicsNeMo docs (no documented ONNX/TensorRT export path for FNO/AFNO/DoMINO; PyTorch-native inference + torch.compile only)",
      "https://arxiv.org/abs/2606.09923 - Conformal Prediction for Neural Operators: Distribution-Free UQ in Physics Simulation",
      "https://arxiv.org/abs/2606.08654 - Operator learning for 2D incompressible Navier-Stokes: a conformal prediction approach in the data-scarce regime",
      "https://arxiv.org/abs/2606.17513 - Geometry-Aware Post-Hoc Uncertainty Quantification in Operator Learning (REEF-GP)",
      "https://arxiv.org/abs/2606.29440 - Randomized neural operator for parametric PDEs with fast training and conformal uncertainty quantification",
      "https://arxiv.org/abs/2607.17297 - Multi-Granularity Conformal Prediction for Reliable Neural-Operator Automotive Aerodynamic Surrogates",
      "https://arxiv.org/abs/2602.08215 - Distribution-Free Robust Predict-Then-Optimize in Function Spaces (conformal in infinite-dimensional Sobolev spaces)",
      "https://arxiv.org/abs/2310.00057 - Multi-fidelity DeepONet for real-time settlement prediction during tunnel construction (the ONE on-topic geotechnical/mining operator-learning paper found)",
      "https://arxiv.org/abs/2509.20238 - Velocity model building from seismic images using a Convolutional Neural Operator",
      "https://arxiv.org/abs/2602.12274 - Function-Space Decoupled Diffusion for Forward and Inverse Modeling in CCS",
      "https://arxiv.org/abs/2511.20543 - Feature-Modulated UFNO (FiLM) for multiphase flow / subsurface CO2 storage",
      "https://arxiv.org/abs/2604.01802 - Real-Time Sensing of Inaccessible Physical Fields via an Edge-Deployable Hardware-Portable Graph Neural Operator (VIRSO)",
      "https://arxiv.org/abs/2507.22959 - Scientific Machine Learning with Kolmogorov-Arnold Networks (review, Faroughi et al.)",
      "https://arxiv.org/abs/2601.07760 - Free-RBF-KAN (spline-free KAN, relevant to exporting KAN layers to ONNX)",
      "arXiv:2202.04836 - Gruver, Finzi, Stanton, Wilson, Deconstructing the Inductive Biases of Hamiltonian Neural Networks, ICLR 2022 (energy bound and the symplecticity refutation, verified via ar5iv full text)",
      "arXiv:1906.01563 - Greydanus, Dzamba, Yosinski, Hamiltonian Neural Networks, NeurIPS 2019 (Table 1 fully re-verified including the omitted real-pendulum row)",
      "arXiv:2003.04630 - Cranmer, Greydanus, Hoyer, Battaglia, Spergel, Ho, Lagrangian Neural Networks (0.4% vs 8% claim and its 40x100-step protocol verified)",
      "arXiv:1907.04490 - Lutter, Ritter, Peters, Deep Lagrangian Networks, ICLR 2019 (500Hz/200Hz, Barrett WAM, analytic-derivative requirement verified)",
      "arXiv:2001.03750 - Jin, Zhang, Zhu, Tang, Karniadakis, SympNets; Neural Networks 132:166-179, DOI 10.1016/j.neunet.2020.08.017 (Tables 1 and 3 verified exactly)",
      "arXiv:1909.13334 - Chen, Zhang, Arjovsky, Bottou, Symplectic Recurrent Neural Networks, ICLR 2020 (Tables 1, 2, 3 verified; two first-pass numbers corrected)",
      "arXiv:2010.13581 - Finzi, Wang, Wilson, Simplifying Hamiltonian and Lagrangian Neural Networks via Explicit Constraints, NeurIPS 2020; repo github.com/mfinzi/constrained-hamiltonian-neural-networks",
      "arXiv:1909.12077 - Zhong, Dey, Chakraborty, Symplectic ODE-Net: Learning Hamiltonian Dynamics with Control, ICLR 2020",
      "arXiv:2002.08860 - Zhong, Dey, Chakraborty, Dissipative SymODEN, ICLR 2020 DeepDiffEq workshop",
      "arXiv:2206.02660 - Eidnes, Stasik, Sterud, Bohn, Riemer-Sorensen, Pseudo-Hamiltonian Neural Networks with State-Dependent External Forces; Physica D, DOI 10.1016/j.physd.2023.133673",
      "arXiv:2502.02480 - Roth, Klein, Kannapinn, Peters, Weeger, Stable Port-Hamiltonian Neural Networks, NeurIPS 2025",
      "arXiv:2602.17998 - Bhardwaj, Bajaj, PHAST: Port-Hamiltonian Architecture for Structured Temporal Dynamics Forecasting (2026)",
      "arXiv:2505.20370 - Hansen, Celledoni, Tapley, Learning mechanical systems from real-world data using discrete forced Lagrangian dynamics (2025)",
      "arXiv:2109.00092 - Zhang, Shin, Karniadakis, GFINNs; Phil. Trans. R. Soc. A, DOI 10.1098/rsta.2021.0207",
      "arXiv:2203.01874 - Hernandez, Badias, Chinesta, Cueto, Thermodynamics-informed graph neural networks; IEEE Trans. AI 5(3):967-976",
      "arXiv:2211.01873 - Hernandez, Badias, Chinesta, Cueto, Port-metriplectic neural networks; Computational Mechanics, DOI 10.1007/s00466-023-02296-w",
      "arXiv:2405.13093 - Tierz, Alfaro, Gonzalez, Chinesta, Cueto, Graph neural networks informed locally by thermodynamics; Engineering Applications of AI (2025)",
      "arXiv:2605.09058 - Votruba, He, Qiu, Reina, Pavelka, Nonlinear GENERIC Informed Neural Networks (N-GINNs) (2026)",
      "Chen, Matsubara, Yaguchi, Neural Symplectic Form: Learning Hamiltonian Equations on General Coordinate Systems, NeurIPS 2021 (proceedings 8b519f198dd26772e3e82874826b04aa)",
      "Sipka, Pavelka, Esen, Grmela, Direct Poisson neural networks, J. Phys. A 56(49), DOI 10.1088/1751-8121/ad0803 (2023)",
      "Lie-Poisson Neural Networks (LPNets), Neural Networks 173 (2024), DOI 10.1016/j.neunet.2024.106162; CLPNets, Neural Networks (2025), ScienceDirect S089360802500320X - CITED FROM INDEX LISTINGS, NOT FETCHED",
      "arXiv:2412.16787 - Canizares, Murari, Schonlieb, Sherry, Shumaylov, Symplectic Neural Flows (SympFlow), Dec 2024 rev. Mar 2026",
      "arXiv:2502.20212 - Cheng, Wang, Cao, Chen, Pseudo-symplectic neural network; J. Comput. Phys., ScienceDirect S0021999125009118",
      "arXiv:2210.01741 - Richter-Powell, Lipman, Chen, Neural Conservation Laws: A Divergence-Free Perspective, NeurIPS 2022",
      "arXiv:2309.00583 - Li, Kovachki, Choy, Li, Kossaifi, Otta, Nabian, Stadler, Hundt, Azizzadenesheli, Anandkumar, Geometry-Informed Neural Operator (GINO), NeurIPS 2023",
      "arXiv:2010.03409 - Pfaff, Fortunato, Sanchez-Gonzalez, Battaglia, Learning Mesh-Based Simulation with Graph Networks (MeshGraphNets), ICLR 2021",
      "arXiv:2101.03164 - Batzner et al., NequIP; Nature Communications 13:2453, DOI 10.1038/s41467-022-29939-5. MACE: Batatia, Kovacs, Simm, Ortner, Csanyi, NeurIPS 2022, OpenReview YPpSngE-ZU; repo github.com/ACEsuit/mace",
      "arXiv:2202.07643 - Brandstetter, Welling, Worrall, Lie Point Symmetry Data Augmentation for Neural PDE Solvers, ICML 2022",
      "arXiv:2603.14734 - Cheng, Gauge-Equivariant Intrinsic Neural Operators for Geometry-Consistent Learning of Elliptic PDE Maps (2026)",
      "arXiv:2603.20474 - Ray, From Data to Laws: Neural Discovery of Conservation Laws Without False Positives (2026)",
      "arXiv:2603.29496 - Oprisa, Toth, Metriplector: From Field Theory to Neural Architecture (2026) - INCLUDED AS AN ANTI-RECOMMENDATION",
      "arXiv:2604.00277 - Betteti, Laurenti, Hybrid Energy-Based Models for Physical AI: Provably Stable Identification of Port-Hamiltonian Dynamics (2026)",
      "arXiv:2604.13297 - Structure- and Stability-Preserving Learning of Port-Hamiltonian Systems (2026) - SURFACED IN SEARCH, NOT FETCHED",
      "arXiv:2604.02601 - WGFINNs: Weak formulation-based GENERIC formalism informed neural networks (2026) - SURFACED IN SEARCH, NOT FETCHED",
      "Springer, A survey of geometric graph neural networks: data structures, models and applications, Frontiers of Computer Science, DOI 10.1007/s11704-025-41426-w (2025) - PAYWALLED, REDIRECT BLOCKED",
      "github.com/pytorch/pytorch/issues/105838 - aten::grad ONNX export failure, confirmed error string and Done status",
      "github.com/greydanus/hamiltonian-nn, github.com/milutter/deep_lagrangian_networks - reference implementations",
      "arXiv:2408.11104 - ConFIG: Towards Conflict-free Training of Physics Informed Neural Networks (Liu, Chu, Thuerey), ICLR 2025 Spotlight - https://arxiv.org/abs/2408.11104",
      "arXiv:2409.18426 - Dual Cone Gradient Descent for Training Physics-Informed Neural Networks (Hwang, Lim), NeurIPS 2024 - https://arxiv.org/abs/2409.18426",
      "arXiv:2502.00604 - Gradient Alignment in Physics-informed Neural Networks: A Second-Order Optimization Perspective (Wang, Bhartari, Li, Perdikaris), NeurIPS 2025 - https://arxiv.org/abs/2502.00604",
      "arXiv:2501.16371 - Optimizing the Optimizer for Physics-Informed Neural Networks and Kolmogorov-Arnold Networks (Kiyani, Shukla, Urban, Darbon, Karniadakis), v6 Feb 2026 - https://arxiv.org/abs/2501.16371",
      "arXiv:2302.13163 - Achieving High Accuracy with PINNs via Energy Natural Gradients (Muller, Zeinhofer), ICML 2023 - https://arxiv.org/abs/2302.13163",
      "arXiv:2505.12149 - Improving Energy Natural Gradient Descent through Woodbury, Momentum, and Randomization (Guzman-Cordero, Dangel, Goldshlager, Zeinhofer) - https://arxiv.org/abs/2505.12149",
      "arXiv:2505.21404 - Dual Natural Gradient Descent for Scalable Training of Physics-Informed Neural Networks (Jnini, Vella) - https://arxiv.org/abs/2505.21404",
      "arXiv:2405.15603 - Kronecker-Factored Approximate Curvature for Physics-Informed Neural Networks (Dangel, Muller, Zeinhofer), NeurIPS 2024 - https://arxiv.org/abs/2405.15603",
      "arXiv:2402.01868 - Challenges in Training PINNs: A Loss Landscape Perspective (Rathore, Lei, Frangella, Lu, Udell), ICML 2024 Oral - https://arxiv.org/abs/2402.01868",
      "arXiv:2604.05230 - Curvature-Aware Optimization for High-Accuracy Physics-Informed Neural Networks (Jnini, Kiyani, Shukla, Urban, Ahmadi Daryakenari, Muller, Zeinhofer, Karniadakis), 2026 - https://arxiv.org/abs/2604.05230",
      "arXiv:2605.23391 - Coupling-Robust Accuracy in Multiphysics PINNs via Kronecker-Preconditioned Optimization (Park, Kim, Hong), 2026 - https://arxiv.org/abs/2605.23391",
      "arXiv:2402.00326 - PirateNets: Physics-informed Deep Learning with Residual Adaptive Networks (Wang, Li, Chen, Perdikaris), JMLR 25 (2024) - https://arxiv.org/abs/2402.00326",
      "arXiv:2308.08468 - An Expert's Guide to Training Physics-informed Neural Networks (Wang, Sankaran, Wang, Perdikaris) - https://arxiv.org/abs/2308.08468",
      "arXiv:2306.15969 - Separable Physics-Informed Neural Networks (Cho, Nam, Yang, Yun, Hong, Park), NeurIPS 2023 - https://arxiv.org/abs/2306.15969",
      "arXiv:2107.07871 - Finite Basis Physics-Informed Neural Networks (FBPINNs) (Moseley, Markham, Nissen-Meyer), Adv. Comput. Math. 49:62 (2023) - https://arxiv.org/abs/2107.07871",
      "arXiv:2104.08426 - Exact imposition of boundary conditions with distance functions in physics-informed deep neural networks (Sukumar, Srivastava), CMAME 2022 - https://arxiv.org/abs/2104.08426",
      "arXiv:2207.10289 - A comprehensive study of non-adaptive and residual-based adaptive sampling for PINNs (Wu, Zhu, Tan, Kartha, Lu), CMAME 403:115671 (2023) - https://arxiv.org/abs/2207.10289",
      "arXiv:2405.14369 - RoPINN: Region Optimized Physics-Informed Neural Networks (Wu, Luo, Ma, Wang, Long), NeurIPS 2024 - https://arxiv.org/abs/2405.14369",
      "arXiv:2111.02801 - Gradient-enhanced physics-informed neural networks for forward and inverse PDE problems (Yu, Lu, Meng, Karniadakis), CMAME 2022 - https://arxiv.org/abs/2111.02801",
      "arXiv:2505.10949 - FP64 is All You Need: Rethinking Failure Modes in Physics-Informed Neural Networks (Xu, Liu, Nassereldine, Xiong) - https://arxiv.org/abs/2505.10949",
      "arXiv:2605.30910 - PINNs Failure Modes are Overfitting (Andersen, Matsubara), May 2026 - https://arxiv.org/abs/2605.30910",
      "arXiv:2306.08827 - PINNacle: A Comprehensive Benchmark of Physics-Informed Neural Networks for Solving PDEs (Hao et al.), NeurIPS 2024 Datasets and Benchmarks Track - https://arxiv.org/abs/2306.08827",
      "arXiv:2410.13228 - From PINNs to PIKANs: Recent Advances in Physics-Informed Machine Learning (Toscano, Oommen, Varghese, Zou, Ahmadi Daryakenari, Wu, Karniadakis) - https://arxiv.org/abs/2410.13228",
      "arXiv:2411.06286 - SPIKANs: Separable Physics-Informed Kolmogorov-Arnold Networks (Jacob, Howard, Stinis) - https://arxiv.org/abs/2411.06286",
      "arXiv:2407.17611 - Adaptive Training of Grid-Dependent Physics-Informed Kolmogorov-Arnold Networks (Rigas, Papachristou, Papadopoulos, Anagnostopoulos, Alexandridis) - https://arxiv.org/abs/2407.17611",
      "arXiv:2509.13620 - A reduced-order derivative-informed neural operator for subsurface fluid-flow (Park, Bruer, Erdinc, Gahlot, Herrmann) - https://arxiv.org/abs/2509.13620",
      "arXiv:2509.08967 - Physics-informed waveform inversion using pretrained wavefield neural operators (Huang, Wang, Alkhalifah) - https://arxiv.org/abs/2509.08967",
      "arXiv:2501.13271 - Hybrid Two-Stage Reconstruction of Multiscale Subsurface Flow with Physics-informed Residual Connected Neural Operator (Li, Chen) - https://arxiv.org/abs/2501.13271",
      "arXiv:2001.04536 - Understanding and mitigating gradient pathologies in physics-informed neural networks (Wang, Teng, Perdikaris), SIAM J. Sci. Comput. 43(5):A3055-A3081 (2021) [VERIFICATION of first-pass claim] - https://arxiv.org/abs/2001.04536",
      "arXiv:2307.00379 - Residual-based attention and connection to information bottleneck theory in PINNs (Anagnostopoulos, Toscano, Stergiopulos, Karniadakis), CMAME 2024 [VERIFICATION of first-pass numbers] - https://arxiv.org/abs/2307.00379",
      "arXiv:2203.07404 - Respecting causality for training physics-informed neural networks (Wang, Sankaran, Perdikaris), CMAME 2024 [VERIFICATION of first-pass Table 1] - https://arxiv.org/abs/2203.07404",
      "Code: https://github.com/tum-pbs/ConFIG (pip install conflictfree), https://github.com/youngsikhwang/Dual-Cone-Gradient-Descent, https://github.com/PredictiveIntelligenceLab/jaxpi/tree/pirate, https://github.com/benmoseley/FBPINNs, https://github.com/lu-group/pinn-sampling, https://github.com/thuml/RoPINN, https://github.com/i207m/PINNacle",
      "https://arxiv.org/abs/2606.12735 — Reyna & Tartakovsky (2026), PINNs and RBFs for PDEs with Dirac Delta Sources; full PDF read (28 pp.), submitted to CMAME",
      "https://arxiv.org/abs/2507.09330 and https://arxiv.org/pdf/2507.09330v1 — Walter, Kong, Hanson-Hedgecock & Vilarrasa, WellPINN; full PDF read for verification of every headline number",
      "https://github.com/linuswalter/WellPINN — verified live, public, GPL-3.0, populated with 01_WellPINN_model.py, 02_WellPINN_postprocessing.py, 01_groundtruth/ (groundtruth_data_over_space.csv, groundtruth_data_well.csv, OGS_Modell/), 14 commits",
      "https://www.osti.gov/servlets/purl/1976802 — Zhang, Zhu, Wang, Ju, Qian, Ye & Yang (2022), GW-PINN, Advances in Water Resources 165, 104243; accepted manuscript read for verification",
      "https://arxiv.org/abs/2607.12271 and https://arxiv.org/html/2607.12271v1 — Rao, Hamacher & Halilovic (2026), parametric PINN for heterogeneous soil thermal problems with BHEs as singular sources",
      "https://arxiv.org/abs/2602.08419 — Radial Muntz-Szasz Networks (2026)",
      "https://arxiv.org/abs/2512.22222 — Muntz-Szasz Networks (2025)",
      "https://arxiv.org/abs/2303.04778 — Jiang, Zhu & Lu, Fourier-MIONet for geological carbon sequestration",
      "https://arxiv.org/abs/2108.08481 — Kovachki et al., Neural Operator: Learning Maps Between Function Spaces",
      "https://arxiv.org/abs/2409.16572 — Efficient and generalizable nested Fourier-DeepONet for 3D geological carbon sequestration",
      "https://arxiv.org/abs/2410.20118 — GeoFUSE: U-FNO surrogate for seawater intrusion",
      "https://arxiv.org/abs/2503.11031 — FNO surrogates for CO2 storage in realistic geologies",
      "https://arxiv.org/abs/2509.21485 — Neural operators for transient fluid flow in subsurface reservoir systems (TFNO-opt)",
      "https://arxiv.org/abs/2606.07215 — Comparative study of deep learning models for geological carbon sequestration (U-Net/V-Net/TCN/FNO/U-FNO)",
      "https://arxiv.org/abs/2107.07871 — Moseley, Markham & Nissen-Meyer, FBPINNs",
      "https://arxiv.org/abs/2211.05560 — FBPINNs as a Schwarz domain decomposition method",
      "https://arxiv.org/abs/2306.05486 — Multilevel domain decomposition-based architectures for PINNs",
      "https://arxiv.org/abs/2409.01949 — ELM-FBPINNs: an efficient multilevel random feature method",
      "https://arxiv.org/abs/2506.17626 — Local feature filtering for domain-decomposed random feature methods",
      "https://arxiv.org/abs/2407.03372 — PINNs for heterogeneous poroelastic media (CoNN + I-PINNs)",
      "https://arxiv.org/abs/2404.13909 — PINNs with curriculum training for poroelastic flow and deformation",
      "https://arxiv.org/abs/2110.03049 — Haghighat, Amini & Juanes, PINN for multiphase poroelasticity with stress-split sequential training (CMAME 397, 115141)",
      "https://arxiv.org/abs/2407.15887 — Separable DeepONet, benchmarked on Biot consolidation",
      "https://arxiv.org/abs/2604.13291 — Physics-informed reservoir characterization with a differentiable simulator",
      "https://arxiv.org/abs/2509.13620 — DeFINO: reduced-order derivative-informed neural operator for subsurface fluid flow",
      "https://arxiv.org/abs/2601.11193 — A Machine-Learned Near-Well Model in OPM Flow (learned Peaceman-like well index)",
      "https://arxiv.org/abs/2501.15908 — Evidential Physics-Informed Neural Networks",
      "https://arxiv.org/abs/2511.18515 — RRaPINNs: Residual Risk-Aware Physics Informed Neural Networks (CVaR)",
      "https://arxiv.org/abs/2202.11274 — Forward and inverse modeling of unsaturated flow (Richardson-Richards) with PINNs and domain decomposition",
      "https://gw-project.org/books/analysis-and-evaluation-of-pumping-test-data/ — Kruseman, de Ridder & Verweij, Analysis and Evaluation of Pumping Test Data, 2nd ed., free download confirmed (compendium of analytic well-test solutions; individual equations NOT verified in this session)",
      "Barker, J.A. (1988) A generalized radial flow model for hydraulic tests in fractured rock, Water Resources Research 24(10), 1796-1804, DOI 10.1029/WR024i010p01796 — CITED, NOT VERIFIED in session",
      "Peaceman, D.W. (1978) SPE Journal 18(3), 183-194, SPE-6893-PA, DOI 10.2118/6893-PA and (1983) SPE J 23(3), 531-543, DOI 10.2118/10528-PA — CITED, NOT VERIFIED in session (paywalled)",
      "Shen, C. et al. (2023) Differentiable modelling to unify machine learning and physical models for geosciences, Nature Reviews Earth & Environment 4(8), 552-567, DOI 10.1038/s43017-023-00450-9 — cited by WellPINN's reference list",
      "https://www.ci2ma.udec.cl/pdf/pre-publicaciones/2026/pp26-23.pdf",
      "https://www.ci2ma.udec.cl/pdf/pre-publicaciones/2024/pp24-08.pdf",
      "https://www.ci2ma.udec.cl/publicaciones/prepublicaciones/",
      "https://arxiv.org/abs/2603.24819",
      "https://arxiv.org/abs/2606.13695",
      "https://arxiv.org/abs/2408.15267",
      "https://arxiv.org/abs/2410.19661",
      "https://arxiv.org/abs/2503.04225",
      "https://arxiv.org/abs/2601.14529",
      "https://arxiv.org/abs/2507.13878",
      "https://arxiv.org/abs/2404.06802",
      "https://arxiv.org/abs/2602.11621",
      "https://arxiv.org/abs/2607.19296",
      "https://arxiv.org/abs/2602.15068",
      "https://arxiv.org/abs/2606.21945",
      "https://arxiv.org/abs/2202.04679",
      "https://arxiv.org/abs/2509.02091",
      "https://arxiv.org/abs/2510.18266",
      "https://arxiv.org/abs/2511.04490",
      "https://arxiv.org/abs/2501.01587",
      "https://arxiv.org/abs/2505.19036",
      "https://arxiv.org/abs/2508.16032",
      "https://arxiv.org/abs/2602.09988",
      "https://arxiv.org/abs/2512.12074",
      "https://arxiv.org/abs/2602.19265",
      "https://arxiv.org/abs/2503.11031",
      "https://arxiv.org/abs/2410.20118",
      "https://arxiv.org/abs/2412.05576",
      "https://arxiv.org/abs/2311.15288",
      "https://arxiv.org/abs/2303.04778",
      "https://arxiv.org/abs/2606.07215",
      "https://arxiv.org/abs/2512.01977",
      "https://arxiv.org/abs/2407.06216",
      "https://arxiv.org/abs/2205.13102",
      "https://arxiv.org/abs/2104.09111",
      "https://arxiv.org/abs/1704.06533",
      "https://arxiv.org/abs/2601.06635",
      "https://arxiv.org/abs/2602.03360",
      "https://arxiv.org/abs/2607.09684",
      "http://export.arxiv.org/api/query?search_query=abs:%22discontinuous+flux%22",
      "https://doi.org/10.14356/kona.2017017",
      "https://doi.org/10.5277/ppmp170128",
      "https://arxiv.org/abs/2003.06097 and https://ar5iv.labs.arxiv.org/html/2003.06097 (B-PINN; Tables 1 and 2 re-verified)",
      "https://arxiv.org/abs/2509.13717 and https://ar5iv.labs.arxiv.org/html/2509.13717 (conformal PINN; Tables 1-3 re-verified)",
      "https://arxiv.org/abs/2503.19333 and https://arxiv.org/html/2503.19333v2 (E-PINNs; v1 vs v2 abstracts compared, all coverage tables read)",
      "https://arxiv.org/abs/2201.07766 (Psaros, Meng, Zou, Guo, Karniadakis, UQ in SciML survey, DOI 10.1016/j.jcp.2022.111902)",
      "https://arxiv.org/abs/2208.11866 (NeuralUQ library) and https://github.com/Crunch-UQ4MI",
      "https://arxiv.org/abs/1905.07488 (APT / SNPE-C)",
      "https://arxiv.org/abs/2003.06281 (BayesFlow)",
      "https://arxiv.org/abs/2305.17161 (Flow Matching Posterior Estimation, NeurIPS 2023)",
      "https://arxiv.org/abs/2007.09114 (sbi toolkit) and https://github.com/sbi-dev/sbi",
      "https://arxiv.org/abs/2401.06230 (WISE, DOI 10.1190/geo2023-0744.1)",
      "https://arxiv.org/abs/2207.11640 (Siahkoohi et al., physics-based latent distribution correction)",
      "https://arxiv.org/abs/1808.03620 (Kovachki & Stuart, EKI, DOI 10.1088/1361-6420/ab1c3a)",
      "https://arxiv.org/abs/2303.07392 (Pensoneault & Zhu, EKI for B-PINNs)",
      "https://arxiv.org/abs/2402.06110 (SH-ESMDA / SH-RML for Geological Carbon Storage, DOI 10.1016/j.ijggc.2024.104190)",
      "https://arxiv.org/abs/2107.08924 (Osband et al., Epistemic Neural Networks / epinet)",
      "https://arxiv.org/abs/1910.02600 (Amini et al., Deep Evidential Regression, NeurIPS 2020)",
      "https://arxiv.org/abs/2501.15908 (Tan, Wang, McBeth, Evidential PINNs, SCML 2025)",
      "https://arxiv.org/abs/2205.10060 (Meinert et al., critique of DER, AAAI 2023, DOI 10.1609/aaai.v37i8.26096)",
      "https://arxiv.org/abs/2106.14806 (Daxberger et al., Laplace Redux, NeurIPS 2021) and https://github.com/aleximmer/Laplace",
      "https://arxiv.org/abs/2510.03278 (Landgren, per-constraint Hessian decomposition for B-PINNs)",
      "https://arxiv.org/abs/2209.14687 (Chung et al., Diffusion Posterior Sampling, ICLR 2023) and https://github.com/DPS2022/diffusion-posterior-sampling",
      "https://arxiv.org/abs/2406.17763 (DiffusionPDE)",
      "https://arxiv.org/abs/2306.10574 (Rozet & Louppe, Score-based Data Assimilation)",
      "https://arxiv.org/abs/2603.01231 (velocity model building/editing with guided DDIM, 2026)",
      "https://arxiv.org/abs/2602.12274 (function-space decoupled diffusion for carbon capture and storage, 2026)",
      "https://arxiv.org/abs/2606.26592 (latent diffusion posterior sampling with surrogate likelihood guidance for PDE inverse problems, 2026)",
      "https://arxiv.org/abs/2106.05863 (Meng et al., Learning Functional Priors and Posteriors, DOI 10.1016/j.jcp.2022.111073)",
      "https://arxiv.org/abs/2302.11002 (Hansen et al., ProbConserv, Physica D 457:133952, DOI 10.1016/j.physd.2023.133952)",
      "https://arxiv.org/abs/2403.10642 (Mouli et al., DiverseNO + Operator-ProbConserv, ICML 2024)",
      "https://arxiv.org/abs/1905.03222 (Romano, Patterson, Candes, Conformalized Quantile Regression)",
      "https://arxiv.org/abs/2208.02814 (Angelopoulos et al., Conformal Risk Control)",
      "https://arxiv.org/abs/1904.06019 (Tibshirani, Barber, Candes, Ramdas, Conformal Prediction Under Covariate Shift)",
      "https://arxiv.org/abs/2402.01960 (Ma, Azizzadenesheli, Anandkumar, risk-controlling quantile neural operator)",
      "https://arxiv.org/abs/2402.15406 (Moya et al., Conformalized-DeepONet)",
      "https://arxiv.org/abs/2504.15240 (Mollaali et al., Conformalized-KANs)",
      "https://arxiv.org/abs/2405.08111 (Podina, Torabi Rad, Kohandel, Conformalized PINNs)",
      "https://arxiv.org/abs/2606.09923 (Chin, Conformal Prediction for Neural Operators, 2026)",
      "https://arxiv.org/abs/1804.06788 (Talts, Betancourt, Simpson, Vehtari, Gelman, Simulation-Based Calibration)",
      "https://arxiv.org/abs/2302.03026 (Lemos et al., TARP, ICML 2023)",
      "https://arxiv.org/abs/2106.11642 (D'Angelo & Fortuin, Repulsive Deep Ensembles are Bayesian, NeurIPS 2021)",
      "https://arxiv.org/abs/2203.12351 (Gou, Zhang, Zhu, Gao, B-PINN Eikonal subsurface tomography with SVGD, DOI 10.1109/TGRS.2023.3286438)",
      "https://arxiv.org/abs/2103.12959 (Chen, Hosseini, Owhadi, Stuart, Solving and Learning Nonlinear PDEs with Gaussian Processes)",
      "https://arxiv.org/abs/2208.01565 (Magnani et al., Approximate Bayesian Neural Operators)",
      "https://arxiv.org/abs/2110.13330 (Bajaj et al., GP-smoothed PINNs, DOI 10.1088/2632-2153/acb416)",
      "https://arxiv.org/abs/2606.17513 (geometry-aware post-hoc GP UQ in operator learning, 2026)",
      "https://arxiv.org/abs/2310.10776 (Zou, Meng, Karniadakis, Correcting model misspecification in PINNs)",
      "https://arxiv.org/abs/2301.07609 (Alberts & Bilionis, Physics-informed Information Field Theory, DOI 10.1016/j.jcp.2023.112100)",
      "https://arxiv.org/abs/2502.12902 (Buelte, Scholl, Kutyniok, Probabilistic Neural Operators, TMLR 2025)",
      "https://arxiv.org/abs/2408.09340 (Hou, Li, Wu, Wang, MBPINN; HMC convergence-failure quote re-verified)",
      "https://arxiv.org/abs/2203.03048 (Yang, Kissas, Perdikaris, UQDeepONet randomized priors, DOI 10.1016/j.cma.2022.115399; abstract re-verified)"
    ],
    "quantitative": [
      {
        "claim": "PINO on Kolmogorov flow Re=500, T=0.5: PINNs 18.7% solution error in 4577 s; PINO 0.9% in 608 s with 0 data and 0 PDE instances; 0.9% in 536 s with 0.4k data; 0.9% in 473 s with 0.4k data + 160k PDE instances. Paper states '20x smaller error and 25x speedup'. CONFIRMS the first pass exactly.",
        "source": "arXiv:2111.03794 Table 4 (Li, Zheng, Kovachki, Jin, Chen, Liu, Azizzadenesheli, Anandkumar), read via ar5iv full text"
      },
      {
        "claim": "PINO on Darcy: DeepONet with data 6.97 +/- 0.09%, PINO with data 1.22 +/- 0.03%, PINO WITHOUT data 1.50 +/- 0.03%. The no-data row was missing from the first pass and is the more interesting number: physics-only training gets within 0.28 points of physics-plus-data.",
        "source": "arXiv:2111.03794 Table 2, read via ar5iv full text"
      },
      {
        "claim": "PINO on Kolmogorov T=0.125 at 256x256x65: 6.22 +/- 0.11% with 0 data / 2200 PDE instances, 6.01 +/- 0.12% with 0.8k data, 5.04 +/- 0.11% with 2200 data. Entirely absent from the first pass; shows the data-free advantage shrinks on the harder setting.",
        "source": "arXiv:2111.03794 Table 3, read via ar5iv full text"
      },
      {
        "claim": "Tucker-tensorized FNO is SLOWER than optimized GPU finite-difference by 1.4x to 80x for time-domain wave propagation at matched 10 m grids, across isotropic acoustic, TTI-acoustic, isotropic elastic and VTI-elastic. Only with a 2.5x coarser grid do small architectures (4-8 layers, 12-24 channels) reach up to 10x faster; large ones (12 layers, 36 channels) are 1.09-1.30x faster on elastic and 1.41-5.9x SLOWER on acoustic. Surface/boundary-only prediction reaches up to 4e3x faster.",
        "source": "arXiv:2508.11119, Voytan & Li, 'Are Fourier Neural Operators Really Faster for Time-Domain Wave Propagation?', 14 Aug 2025"
      },
      {
        "claim": "Zero-shot super-resolution refuted as a reliable property: on Darcy, going from training resolution s to test S > s, 'direct fine-grid inference is not reliably beneficial and can be worse than the low-grid-plus-upsampling baseline'. Mechanism: under Fourier truncation intermediate representations concentrate energy in low frequencies, with high-frequency output produced mainly by late nonlinear/decoder stages; nonlinear aliasing is the key obstacle.",
        "source": "arXiv:2606.00677, Colagrande, Caillon, Feillet, Allauzen, 'On the Limits of Resolution Equivariance in Fourier Neural Operators', 30 May 2026"
      },
      {
        "claim": "ONNX Runtime WebGPU EP: DFT is ABSENT. Present with these opset ranges: Einsum ai.onnx(12+), MatMul ai.onnx(1-12,13+), Conv ai.onnx(1-10,11+), Gelu ai.onnx(20+), LayerNormalization ai.onnx(1-16,17+), Resize ai.onnx(10,11-12,13-17,18,19+), GridSample ai.onnx(16-19), ScatterND ai.onnx(11-12,13-15,16-17,18+), Slice, Concat, Transpose, Pad. Confirms the first pass's blocker AND confirms its proposed cos/sin matmul rewrite is fully WebGPU-viable.",
        "source": "microsoft/onnxruntime, main branch, js/web/docs/webgpu-operators.md, fetched directly"
      },
      {
        "claim": "neuraloperator FNO-on-Darcy tutorial: 'Total running time of the script: 17.489 seconds', 191,881 parameters, 1000 training samples at 16x16 (test sets of 100 and 50 at 16x16 and 32x32), 15 epochs, AdamW lr 0.01 wd 1e-4, CosineAnnealingLR, H1 training loss with L2 evaluation. CONFIRMS the first pass exactly.",
        "source": "neuraloperator.github.io/dev/auto_examples/models/plot_FNO_darcy.html"
      },
      {
        "claim": "Adding localized integral and differential kernel layers to FNOs reduces relative L2 error by 34-72%, on turbulent 2D Navier-Stokes and spherical shallow water. Conv-only, so ONNX and WebGPU clean. Cheapest accuracy upgrade available.",
        "source": "arXiv:2402.16845, Liu-Schiaffini, Berner, Bonev, Kurth, Azizzadenesheli, Anandkumar, ICML 2024"
      },
      {
        "claim": "Incremental FNO: 10% reduction in testing error, 20% fewer frequency modes than standard FNO, 30% faster training. Already shipped as a neuraloperator training example.",
        "source": "arXiv:2211.15188, George, Zhao, Kossaifi, Li, Anandkumar"
      },
      {
        "claim": "F-FNO error reductions the first pass omitted: 83% on Navier-Stokes, 57% on airfoil flow, 60% on plastic forging, 31% on elasticity; order-of-magnitude speedup over pseudo-spectral methods at equal solution quality, with time steps an order of magnitude larger.",
        "source": "arXiv:2111.13802, Tran, Mathews, Xie, Ong, ICLR 2023"
      },
      {
        "claim": "GINO: 26,000x speedup computing drag coefficients versus optimized GPU-based CFD; one-fourth reduction in error rate versus deep neural network approaches on unseen geometry-boundary-condition combinations; trained on only 500 data points; Reynolds numbers up to 5 million.",
        "source": "arXiv:2309.00583, Li, Kovachki, Choy, Li, Kossaifi, Otta, Nabian, Stadler, Hundt, Azizzadenesheli, Anandkumar"
      },
      {
        "claim": "Geo-FNO: '10^5 times faster than the standard numerical solvers' and 'twice more accurate compared to direct interpolation on existing ML-based PDE solvers such as the standard FNO', over elasticity, plasticity, Euler and Navier-Stokes.",
        "source": "arXiv:2207.05209, Li, Huang, Liu, Anandkumar, JMLR 24 (2023)"
      },
      {
        "claim": "Nested FNO 'speeds up flow prediction nearly 700,000 times compared to existing methods' for high-resolution dynamic 3D CO2 geological storage at basin scale. U-FNO separately requires 'only a third of the training data to achieve the equivalent accuracy as CNN'. Shell/NVIDIA follow-up on realistic 3D geologies reports 'O(10^5) computational acceleration with minimal sacrifice in prediction accuracy'.",
        "source": "arXiv:2210.17051 (Energy & Environmental Science 16(4) 1732-1741, 2023); arXiv:2109.03697 (Advances in Water Resources 2022); arXiv:2503.11031 (Chandra et al., Mar 2025)"
      },
      {
        "claim": "Transolver: 22% relative gain across six standard benchmarks with linear complexity. Transolver++: million-scale points on a single GPU for the first time, 13% relative improvement on six standard benchmarks, over 20% gain on million-scale industrial simulations 100x larger than previous benchmarks.",
        "source": "arXiv:2402.02366 (Wu, Luo, Wang, Wang, Long, ICML 2024) and arXiv:2502.02414"
      },
      {
        "claim": "KANO versus FNO on quantum Hamiltonian learning: roughly 6e-6 state infidelity from projective measurement data versus FNO's roughly 1.5e-2 given ideal full-wave-function data, i.e. about 2500x more accurate. Theoretical claim: FNO 'stays practical only for spectrally sparse operators and strictly imposes a fast-decaying input Fourier tail', while KANO remains expressive over generic variable-coefficient PDEs.",
        "source": "arXiv:2509.16825, Lee, Liu, Yu, Wang, Jeong, Niu, Zhang, 20 Sept 2025, latest version 24 Feb 2026"
      },
      {
        "claim": "HO-FNO: 'a single HO-FNO layer outperforms FNO models with up to 16 layers' on Poisson equations with polynomial forcing, with consistent improvement over other spectral neural operators and performance matching or exceeding transformers and state-space models.",
        "source": "arXiv:2606.28122, Colagrande, Caillon, Feillet, Allauzen, 26 June 2026"
      },
      {
        "claim": "Hartley Neural Operator is iso-parametric with FNO at equal width (one real weight where FNO carries a complex pair) while retaining twice as many frequency corners, because the real Hartley spectrum has no conjugate-symmetry halving. The paper offers a selection rule, not a performance claim: Hartley basis for elliptic operators, Fourier basis for time-dependent ones.",
        "source": "arXiv:2606.24851, Sulskis & Ravi, 'Real vs. Complex Spectral Bases for Neural Operators: The Role of Green's Function Alignment', 23 June 2026"
      },
      {
        "claim": "Derivative-informed operator learning speedups: 10^3-10^7x execution time for PDE-constrained optimization under high-dimensional uncertainty with solutions over 10x more cost-efficient after construction; over 1000x for Bayesian optimal experimental design including offline construction; 3-9x faster effective posterior samples in geometric MCMC breaking even after 10-25 samples; 3-8 orders of magnitude for shape-derivative state and gradient evaluations with 1-2 orders fewer PDE solves.",
        "source": "arXiv:2305.20053; arXiv:2312.14810; arXiv:2403.08220; arXiv:2603.03211 (all abstract-level, retrieved via arXiv API)"
      },
      {
        "claim": "Conformal UQ on operators: split conformal on an FNO reports 89.1% empirical coverage at alpha=0.1 with a 68%/32% epistemic/aleatoric split; multi-granularity conformal on an automotive aero surrogate reduces mean interval width 22.68% for pressure fields and coverage standard deviation from 10.41 to 3.10 percentage points. Cost at inference is one stored scalar quantile.",
        "source": "arXiv:2606.09923; arXiv:2607.17297 (abstract-level, retrieved via arXiv API)"
      },
      {
        "claim": "CoDA-NO outperforms existing methods by over 36% on few-shot downstream tasks (fluid flow, fluid-structure interaction, Rayleigh-Benard convection). DPOT scales to 0.5B parameters over 100,000+ trajectories across 10+ PDE datasets. PDE-Transformer trains on 16 different PDE types. Poseidon reports significant gains in sample efficiency and accuracy across 15 downstream tasks and generalizes to unseen physics.",
        "source": "arXiv:2403.12553; arXiv:2403.03542; arXiv:2505.24717 (ICML 2025); arXiv:2405.19101 (NeurIPS 2024)"
      },
      {
        "claim": "SFNO: 'stable auto-regressive rollouts for a year of simulated time (1,460 steps)' while 'retaining physically plausible dynamics'. The strongest published long-rollout stability claim in the FNO family, and the reason the invariant-projection idea deserves a look for a rollout lab.",
        "source": "arXiv:2306.03838, Bonev, Kurth, Hundt, Pathak, Baust, Kashinath, Anandkumar, 6 June 2023"
      },
      {
        "claim": "AB-UPT: volumetric meshes up to 100 million cells (tested 33,000 to 150 million), 'models can be trained on a single GPU in less than a day', 'predict industry-standard surface and volume fields within seconds', with hard divergence-free constraints enforced 'without degradation in performance'.",
        "source": "arXiv:2502.09692, Alkin, Bleeker, Kurle, Kronlachner, Sonnleitner, Dorfer, Brandstetter, v4 Oct 2025"
      },
      {
        "claim": "Function-space decoupled diffusion for CCS reports 7.7% relative error for CO2 flow with 25% sparse observations versus 86.9% for baselines, which is the size of gap that justifies diffusion sampling cost for sparse-sensor subsurface inversion.",
        "source": "arXiv:2602.12274 (abstract-level, retrieved via arXiv API)"
      },
      {
        "claim": "CORRECTED. SRNN 20-mass spring chain, noiseless: recurrent leapfrog H-NET 2.37 +/- 0.87 versus single-step LEAPFROG H-NET 3.36 +/- 0.67 (like-for-like, about 1.4x). The 7.24 +/- 0.64 quoted by the first pass is the single-step H-NET trained AND tested with EULER integration, so the '3x' figure credits the recurrence with the integrator's benefit.",
        "source": "arXiv:1909.13334 Table 1 (Chen, Zhang, Arjovsky, Bottou, ICLR 2020)"
      },
      {
        "claim": "CORRECTED. SRNN three-body at dt=1: SRNN 0.26 +/- 0.07, single-step LEAPFROG H-NET 0.35 +/- 0.09, ground-truth leapfrog integration of the TRUE equations 0.47 +/- 0.18. The '0.79 to 1.76' HNN baselines are the EULER-TRAINED variants (0.79 single-step, 1.76 recurrent). Beating the true equations is confirmed, but the non-recurrent 0.35 also beats them, so recurrence is not the cause.",
        "source": "arXiv:1909.13334 Tables 1 and 3"
      },
      {
        "claim": "OMITTED AND DECISIVE. HNN on the REAL pendulum (the paper's only real-world data): train loss (x1e3) baseline 2.7 +/- 0.2 versus HNN 9.2 +/- 0.5; test loss (x1e3) baseline 2.2 +/- 0.3 versus HNN 6.0 +/- 0.6. HNN fits real dissipative data about 2.7x WORSE than the black-box baseline, winning only on energy MSE (14 +/- 5 versus 390 +/- 7).",
        "source": "arXiv:1906.01563 Table 1, row 3 (Greydanus, Dzamba, Yosinski, NeurIPS 2019)"
      },
      {
        "claim": "CORRECTED FRAMING. LNN double pendulum: 'the mean energy discrepancy between the true total energy and predicted was 8% and 0.4% of the max potential energy for the baseline and LNN models respectively' - measured over 40 random initial conditions with 100 time steps EACH, not a single 4,000-step rollout. The per-rollout horizon is 100 steps.",
        "source": "arXiv:2003.04630 (Cranmer, Greydanus, Hoyer, Battaglia, Spergel, Ho)"
      },
      {
        "claim": "CONFIRMED VERBATIM. Gruver et al. energy bound: 'Given bounded dynamics error ||e(z)|| < delta in a bounded state region, |H(z_hat_T) - H(z_T)| < T*delta*sup||grad H||'. Linear growth in T, contrasted with state error which can grow exponentially via Lyapunov effects.",
        "source": "arXiv:2202.04836 Appendix B.1 (Gruver, Finzi, Stanton, Wilson, ICLR 2022)"
      },
      {
        "claim": "CONFIRMED VERBATIM AND UNDER-USED BY THE FIRST PASS. 'The improved generalization of HNNs is the result of modeling acceleration directly and avoiding artificial complexity from the coordinate system, rather than symplectic structure or energy conservation.' Plus: 'Surprisingly HNNs underperform NODEs on all the systems we consider.'",
        "source": "arXiv:2202.04836 (Gruver, Finzi, Stanton, Wilson, ICLR 2022)"
      },
      {
        "claim": "CONFIRMED EXACTLY. SympNets pendulum Table 3, log10 test MSE: h=0.1 LA-SympNet -7.3 +/- 0.1, G-SympNet -3.0 +/- 0.2, S-HNN -1.6 +/- 0.6; h=3 LA-SympNet -6.7 +/- 0.4, G-SympNet -5.2 +/- 0.5, S-HNN N/A; irregular data LA -4.4 +/- 0.4, G -4.1 +/- 0.5, S-HNN -3.2 +/- 0.2. Valid prediction time (log10): h=0.1 LA 3.5 +/- 0.3 vs S-HNN 0.3 +/- 0.3. Parameters (Table 1): pendulum LA-SympNet 14, G-SympNet 0.5K, S-HNN 2K; double pendulum LA 0.2K, G 2K, S-HNN 5K.",
        "source": "arXiv:2001.03750 Tables 1 and 3; Neural Networks 132:166-179, DOI 10.1016/j.neunet.2020.08.017"
      },
      {
        "claim": "CONFIRMED EXACTLY. HNN Table 1 remaining rows: mass-spring energy (x1e3) 170 +/- 20 baseline vs 0.38 +/- 0.1 HNN; ideal pendulum energy 42 +/- 10 vs 25 +/- 5; two-body energy (x1e6) 6.3e4 +/- 3e4 vs 39 +/- 5 and test loss 30 +/- 0.1 vs 2.8 +/- 0.1; pixel pendulum energy 9.3 +/- 1 vs 0.15 +/- 0.01. Energy metric is 'MSE between the true and predicted total energies'.",
        "source": "arXiv:1906.01563 Table 1"
      },
      {
        "claim": "CONFIRMED. DeLaN real-time performance: 'control frequency is set to 500Hz' with desired joint state at 'fd = 200Hz' and the constraint 'T <= 1/200s', on a physical 7-DOF Barrett WAM with cable drives plus a simulated 2-DOF PyBullet robot. Positive-definiteness by construction: '0 < x^T L_hat L_hat^T x for all x'. And verbatim: 'the derivatives contained in f_hat^-1 must be computed analytically' because 'automatic differentiation do not allow the backpropagation of the gradient through the computed derivatives'.",
        "source": "arXiv:1907.04490 (Lutter, Ritter, Peters, ICLR 2019)"
      },
      {
        "claim": "CORRECTED SCOPE. CHNN/CLNN: 'our model can be 100 times more data efficient or 260 times more accurate' applies to the GYROSCOPE system specifically (Figure 1 caption). The general across-suite claim is the weaker '10 to 100 times more accurate than HNNs and DeLaNs'. The paper does NOT analyse the cost or conditioning of the constraint-matrix inverse.",
        "source": "arXiv:2010.13581 (Finzi, Wang, Wilson, NeurIPS 2020)"
      },
      {
        "claim": "Thermodynamics-informed GNN: relative L2 error below 3% on all three cases (viscoelastic Couette flow across position/velocity/energy/conformation tensor; viscoelastic bending beam across position/velocity/stress; flow past cylinder for velocity and pressure). Datasets 100x101 nodes, 52x756 nodes, 30x~1200 nodes. Hidden dims F_h = 10 to 128.",
        "source": "arXiv:2203.01874, IEEE Trans. AI 5(3):967-976 (Hernandez, Badias, Chinesta, Cueto)"
      },
      {
        "claim": "GINO: '26,000x speed-up compared to optimized GPU-based computational fluid dynamics (CFD) simulators' for drag-coefficient computation, and 'one-fourth reduction in error rate compared to deep neural network approaches' on new geometries. Trained on 500 data points, 3D vehicle geometries at Reynolds numbers up to five million. QUOTE WITH THE SCOPE ATTACHED: it is a single-quantity result on external automotive aerodynamics, not a general speedup.",
        "source": "arXiv:2309.00583 (Li, Kovachki, Choy et al., NeurIPS 2023)"
      },
      {
        "claim": "MeshGraphNets: 'running 1-2 orders of magnitude faster than the simulation on which it is trained' (10-100x), on aerodynamics, structural mechanics and cloth.",
        "source": "arXiv:2010.03409 (Pfaff, Fortunato, Sanchez-Gonzalez, Battaglia, ICLR 2021)"
      },
      {
        "claim": "NequIP: 'outperforms existing models with up to three orders of magnitude fewer training data' via E(3)-equivariant convolutions on geometric tensors. Note the relevance caveat: this is an atomistic interatomic potential, not a plant-scale method.",
        "source": "arXiv:2101.03164, Nature Communications 13:2453, DOI 10.1038/s41467-022-29939-5 (Batzner et al.)"
      },
      {
        "claim": "Lie point symmetry data augmentation: 'can easily be deployed to improve neural PDE solver sample complexity by an order of magnitude' (about 10x less training data), architecture-agnostic and with zero ONNX implications since it is a data-pipeline change.",
        "source": "arXiv:2202.07643 (Brandstetter, Welling, Worrall, ICML 2022)"
      },
      {
        "claim": "Direct Poisson NN: rigid body IJ dM = 1.7e-3, SJ dM = 4.6e-4; particle in 2D both ~3e-3; heavy top SJ dM = 1.6e-2 and 4.6e-3. Networks are one hidden layer of 64 softplus units. On NON-Hamiltonian systems the ordering reverses (WJ best), which turns the variant comparison into a Hamiltonian/non-Hamiltonian classifier.",
        "source": "J. Phys. A 56(49), DOI 10.1088/1751-8121/ad0803 (Sipka, Pavelka, Esen, Grmela, 2023)"
      },
      {
        "claim": "Dissipative SymODEN, Task 2 (pendulum with embedded angles): test error 0.13 +/- 0.25 versus 0.33 +/- 1.22 for the geometric baseline, 0.16M parameters. Honest counterweight: prediction error 1.04 +/- 1.3 versus 0.81 +/- 0.68, i.e. WORSE on that metric. The paper does NOT demonstrate energy-based control, only states it 'paves the way'.",
        "source": "arXiv:2002.08860 (Zhong, Dey, Chakraborty)"
      },
      {
        "claim": "NGCG conservation-law discovery: Discovery Rate 1.0, False Discovery Rate 0.0, F1 1.0 on systems with true conservation laws; constancy 'two to three orders of magnitude lower than the best baseline'; robust at noise sigma = 0.1 with 50-100 trajectories in under one minute per system. Covers Hamiltonian AND dissipative ODEs.",
        "source": "arXiv:2603.20474 (Ray, 2026)"
      },
      {
        "claim": "ONNX export blocker confirmed: 'Exporting the operator aten::grad to ONNX opset version 18 is not supported.' Issue status shows Done on the ONNX project board with no explicit in-graph resolution or workaround documented; the dynamo exporter is not discussed in the thread. The general lesson is broader than autodiff: any forward-pass op outside the standard opset blocks export, which includes general matrix inversion (LNN's Hessian inverse, CHNN's projection inverse).",
        "source": "github.com/pytorch/pytorch/issues/105838"
      },
      {
        "claim": "GFINN scaling limit stated by the authors: parameters from the skew-symmetric matrices number 2Kd(d-1), growing as O(Kd^2) in state dimension d. Sparse parameterization is mentioned as mitigation but not benchmarked, and no timing benchmarks are given. The 'GFINNs outperform existing architectures' claim versus SPNNs, GNODEs and SDENets is QUALITATIVE in the accessible text.",
        "source": "arXiv:2109.00092, Phil. Trans. R. Soc. A, DOI 10.1098/rsta.2021.0207 (Zhang, Shin, Karniadakis)"
      },
      {
        "claim": "CONFIRMED VERBATIM. Wang, Teng & Perdikaris: developments \"consistently improve the predictive accuracy of physics-informed neural networks by a factor of 50-100x across a range of problems in computational physics.\" The first pass quoted this correctly.",
        "source": "arXiv:2001.04536 abstract, fetched directly"
      },
      {
        "claim": "CONFIRMED VERBATIM. RBA: \"this general method consistently achieves a relative L2 error of the order of 10^-5 using standard optimizers on typical benchmark cases of the literature.\" The first pass quoted this correctly.",
        "source": "arXiv:2307.00379 abstract, fetched directly"
      },
      {
        "claim": "CONFIRMED, full table. RBA Allen-Cahn comparison: RBA 5.7e-5; DASA-PINNs 8.57e-5; causal training 1.39e-4; time marching 1.68e-2; self-adaptive weights 2.1e-2; vanilla PINN 4.98e-1. Every number the first pass cited matches.",
        "source": "arXiv:2307.00379 (ar5iv full text), Allen-Cahn comparison table"
      },
      {
        "claim": "CONFIRMED, full table, plus two rows the first pass omitted. RBA 2D Helmholtz: RBA(Fourier) 1.46e-5; DASA-PINNs 5.35e-5; RBA(ADF) 8.04e-5; LEARNING-RATE ANNEALING 1.49e-3; SELF-ADAPTIVE WEIGHTS 3.20e-3; vanilla PINN 8.14e-2. The two omitted rows matter: they show the grad-norm annealing of method #1 is 100x WORSE than RBA on this problem.",
        "source": "arXiv:2307.00379 (ar5iv full text), Helmholtz comparison table"
      },
      {
        "claim": "CONFIRMED EXACTLY. RBA per-iteration training time is identical to vanilla PINN: 17 ms/iter for BOTH on Allen-Cahn, 13.8 ms/iter for BOTH on Helmholtz. Also recovered the default hyperparameters the first pass did not give: eta* = 0.01, gamma = 0.999, which by the bound lambda in (0, eta*/(1-gamma)] gives a weight ceiling of exactly 10.",
        "source": "arXiv:2307.00379 (ar5iv full text), timing table and boundedness result"
      },
      {
        "claim": "CONFIRMED, every row. Causal training Allen-Cahn Table 1: original Raissi formulation 4.98e-1; adaptive time sampling 2.33e-2; self-attention 2.10e-2; time marching 1.68e-2; causal training (MLP) 1.43e-3; causal + modified MLP 1.39e-4. The first pass reproduced this table correctly.",
        "source": "arXiv:2203.07404 (ar5iv full text), Table 1"
      },
      {
        "claim": "CONFIRMED with an ATTRIBUTION CORRECTION. The \"10-100x improvements in accuracy compared to competing approaches\" claim is REAL but is NOT in the abstract, which says only \"significant accuracy improvements.\" It appears in the body, scoped to the Allen-Cahn result: \"outperforms the best reported result of competing approaches by a factor of ~10-100x.\" Cite it as a body claim about one benchmark, not as a general abstract-level claim.",
        "source": "arXiv:2203.07404, abstract vs body text (both fetched)"
      },
      {
        "claim": "CONFIRMED. Causal training chaotic-system results: Lorenz relative L2 x 1.139e-2, y 1.656e-2, z 7.038e-3; Kuramoto-Sivashinsky regular 3.49e-4, chaotic 2.46e-2; Navier-Stokes u 3.90e-2, v 2.61e-2, w (vorticity) 3.53e-2. Also confirmed verbatim: \"To the best of our knowledge, this is the first time that PINNs have been successful in simulating such systems.\"",
        "source": "arXiv:2203.07404 (ar5iv full text) and abstract"
      },
      {
        "claim": "CONFIRMED, both values. PINNacle Table 3: on Poisson2d-CG (complex geometry) PINN-NTK gets 1.43E-2 vs vanilla PINN 6.36E-1, a 44x improvement. On Heat2d-MS (multi-scale) PINN-NTK gets 4.40E-2 vs vanilla PINN 6.21E-2. CAVEAT the first pass should have flagged: on the multi-scale problem the gain over vanilla is only about 1.4x, so calling NTK weighting a multi-scale fix on this evidence overstates it. The complex-geometry result is the strong one.",
        "source": "arXiv:2306.08827 (ar5iv full text), Table 3"
      },
      {
        "claim": "NEW, and the most quotable number for an applied audience. In PINNacle, a vanilla PINN solves only 10 OUT OF 22 TASKS at a 10% relative-error threshold. The benchmark also concludes that \"few methods can adequately address nonlinear problems\" and names high-dimensional and nonlinear problems as the pressing open challenge.",
        "source": "arXiv:2306.08827 (ar5iv full text)"
      },
      {
        "claim": "CORRECTION. PINNacle's venue is NeurIPS 2024 Datasets and Benchmarks Track (arXiv preprint June 2023), and it benchmarks 11 methods, not \"about 10\": PINN, PINN(Adam+L-BFGS), PINN-LRA, PINN-NTK, RAR, MultiAdam, gPINN, hp-VPINN, LAAF, GAAF, FBPINN.",
        "source": "https://github.com/i207m/PINNacle README and NeurIPS 2024 proceedings listing"
      },
      {
        "claim": "NEW, and the largest verified gains in the field. Burgers with PINNs, double precision: Adam+L-BFGS(Wolfe) 2.05e-3, Adam+BFGS(Wolfe) 1.50e-5, Adam+SSBFGS(Wolfe) 9.62e-8, Adam+SSBroyden(Wolfe) 7.57e-8. That is about 27,000x better than L-BFGS and about 198x better than BFGS, with NO adaptive weighting.",
        "source": "arXiv:2501.16371v6 (Kiyani, Shukla, Urban, Darbon, Karniadakis), results tables"
      },
      {
        "claim": "NEW. Allen-Cahn with PINNs, double precision: Adam+BFGS(Wolfe) 9.84e-4 vs Adam+SSBroyden(Wolfe) 9.43e-7, about 1,043x. Compare against the best weighting-based Allen-Cahn result anywhere in the first-pass list, RBA at 5.7e-5: the optimizer alone beats it by about 60x with a vanilla architecture.",
        "source": "arXiv:2501.16371v6, results tables"
      },
      {
        "claim": "NEW. Same paper on PIKANs (Chebyshev degree 3), double precision: Burgers BFGS 5.38e-6 -> SSBroyden 1.79e-8 (about 301x), Allen-Cahn BFGS 1.23e-3 -> SSBroyden 9.01e-6 (about 137x). Read together with the PINN rows, this says the OPTIMIZER dominates the architecture: PIKAN wins Burgers marginally, PINN wins Allen-Cahn by 10x, but SSBroyden wins by 100-1000x on both.",
        "source": "arXiv:2501.16371v6, results tables"
      },
      {
        "claim": "NEW, verbatim. SOAP: \"state-of-the-art results on 10 challenging PDE benchmarks, including the first successful application to turbulent flows with Reynolds numbers up to 10,000, with 2-10x accuracy improvements over existing methods.\"",
        "source": "arXiv:2502.00604 abstract (Wang, Bhartari, Li, Perdikaris), NeurIPS 2025"
      },
      {
        "claim": "NEW. PirateNets vs prior SOTA, relative L2: Allen-Cahn 2.24e-5 (prior 5.37e-5); Korteweg-de Vries 4.27e-4 (prior 2.45e-3); Grey-Scott 3.61e-3 (prior 6.13, i.e. the prior method DIVERGED); Ginzburg-Landau 1.49e-2 (prior 3.20e-2); lid-driven cavity Re=3200 4.21e-2 (prior 1.58e-1). Grey-Scott and the cavity are qualitative rescues, not incremental speedups.",
        "source": "arXiv:2402.00326 (ar5iv full text), JMLR 25 (2024)"
      },
      {
        "claim": "NEW, verbatim. SPINN: \"drastically reduced computational costs (62x in wall-clock time, 1,394x in FLOPs given the same number of collocation points)\", enables >10^7 collocation points on one commodity GPU, and solves a chaotic (2+1)-D Navier-Stokes in 9 MINUTES vs 10 HOURS for the best prior method on a single GPU at maintained accuracy.",
        "source": "arXiv:2306.15969 abstract (Cho, Nam, Yang, Yun, Hong, Park), NeurIPS 2023"
      },
      {
        "claim": "NEW, verbatim. FP64: \"with standard FP32, the LBFGS optimizer prematurely satisfies its convergence test, freezing the network in a spurious failure phase. Simply upgrading to FP64 rescues optimization, enabling vanilla PINNs to solve PDEs without any failure modes.\" This reframes failure modes as precision-induced stalls, not local minima, and exposes a three-stage dynamic (unconverged, failure, success) whose boundaries move with precision.",
        "source": "arXiv:2505.10949 abstract (Xu, Liu, Nassereldine, Xiong)"
      },
      {
        "claim": "NEW, verbatim, and the newest result in the field. Overfitting reframing: \"failure modes are the result of overfitting: the loss is minimized on the collocation points, but not elsewhere. Applying regularization causes the failure modes to vanish... we extend double backpropagation over the full set of residuals, and use it to achieve state-of-the-art performance on four standard failure mode equations with up to 23x fewer collocation points and a vanilla architecture.\"",
        "source": "arXiv:2605.30910 abstract (Andersen & Matsubara), submitted 29 May 2026"
      },
      {
        "claim": "NEW. D-NGD scales second-order PINN optimization to networks with up to 12.8 MILLION parameters on a SINGLE GPU, delivering \"one- to three-order-of-magnitude lower final error L2 than first-order methods (Adam, SGD) and quasi-Newton methods.\" This removes the standard objection that natural-gradient methods cap out at a few thousand parameters.",
        "source": "arXiv:2505.21404v2 abstract (Jnini & Vella)"
      },
      {
        "claim": "NEW. Energy natural gradient descent with Woodbury, momentum and randomization achieves \"the same L2 error as the original ENGD up to 75x faster.\" Baseline ENGD claim, verbatim: \"errors several orders of magnitude smaller than what is obtained when training PINNs with standard optimizers like gradient descent or Adam, even when those are allowed significantly more computation time.\"",
        "source": "arXiv:2505.12149 abstract (Guzman-Cordero, Dangel, Goldshlager, Zeinhofer); arXiv:2302.13163 abstract (Muller & Zeinhofer, ICML 2023)"
      },
      {
        "claim": "NEW, and the number that frames the entire second-order line. KFAC-for-PINNs abstract: natural-gradient and Gauss-Newton methods improve \"the accuracy achieved by first-order methods by several orders of magnitude\" but \"only scale to networks with a few thousand parameters due to the high computational cost to evaluate, store, and invert the curvature matrix.\" KFAC and D-NGD are the two answers to that scaling limit.",
        "source": "arXiv:2405.15603 abstract (Dangel, Muller, Zeinhofer), NeurIPS 2024"
      },
      {
        "claim": "NEW. DCGD's empirical measurement of how common the problem is: conflicting gradients (negative inner product between the residual-loss and boundary-loss gradients) occur in ABOUT HALF of all training iterations. This is the quantitative justification for treating direction, not just magnitude, as a first-class failure.",
        "source": "arXiv:2409.18426 (Hwang & Lim), NeurIPS 2024"
      },
      {
        "claim": "NEW. M-ConFIG (momentum variant) runs at approximately 0.56x the per-iteration computational cost of standard three-loss PINN training, so the conflict-free update is cheaper per step than the baseline it improves on, not just more accurate.",
        "source": "arXiv:2408.11104v3 (ar5iv/HTML full text), ICLR 2025 Spotlight"
      },
      {
        "claim": "NEW. FBPINN achieves the smallest error of all 11 benchmarked methods on the chaotic Gray-Scott equation in PINNacle, at 7.99% L2RE. This is the strongest independent evidence for domain decomposition on hard, large, multi-scale problems, the regime that matters most for subsurface work.",
        "source": "arXiv:2306.08827 / PINNacle NeurIPS 2024 results"
      },
      {
        "claim": "NEW. RAD/RAR-D adaptive sampling conclusions are drawn from MORE THAN 6,000 PINN simulations across four forward and two inverse problems, comparing 10 sampling schemes. Verdict: RAD and RAR-D \"significantly improve the accuracy of PINNs with fewer residual points.\" This is the largest controlled empirical study in the area and the first pass omits the sampling axis entirely.",
        "source": "arXiv:2207.10289 abstract (Wu, Zhu, Tan, Kartha, Lu), CMAME 403:115671 (2023)"
      },
      {
        "claim": "NEW. RoPINN improves diverse PINN backbones (PINN, KAN, PINNsFormer) across 19 tasks \"without extra backpropagation or gradient calculation\", i.e. at genuinely zero additional per-step gradient cost, via one-sample Monte Carlo region optimization with a variance-calibrated trust region.",
        "source": "arXiv:2405.14369 abstract (Wu, Luo, Ma, Wang, Long), NeurIPS 2024"
      },
      {
        "claim": "NEW. Adaptive grid-dependent PIKANs in JAX report up to 84x FASTER TRAINING and up to 43.02% L2 error reduction from the adaptive features, at far fewer parameters than larger fixed architectures. Useful as the counterweight to the claim that KANs are simply slower than MLPs.",
        "source": "arXiv:2407.17611 (Rigas, Papachristou, Papadopoulos, Anagnostopoulos, Alexandridis)"
      },
      {
        "claim": "NEW, corroborating SOAP from an independent group. Across 222 experiments on coupled multiphysics PINNs, SOAP+GradNorm is the configuration that maintains final-epoch L2 accuracy as inter-equation coupling strength grows. NTK analysis shows the standard NTK spectral radius grows as Omega(gamma^2) with coupling strength gamma, while block-diagonal Gauss-Newton preconditioning yields a bounded spectral radius INDEPENDENT of coupling. Directly relevant to any multiphysics industrial model.",
        "source": "arXiv:2605.23391v2 (Park, Kim, Hong), 2026"
      },
      {
        "claim": "RBF-RLS achieves global RMSE 1.20e-3 versus PINN 1.34e-2 (11x better) on the diffusion equation with point-source forcing — explicitly described as the hydraulic head in a 1D confined aquifer under constant pumping rate Q — while using 7,056 RBF coefficients against the PINN's 17,025 trainable parameters",
        "source": "Reyna & Tartakovsky, arXiv:2606.12735v1, Section 3.3 and Figure 4b: 'Despite using fewer than half as many learned parameters, RBF-RLS achieves errors that are an order of magnitude smaller.'"
      },
      {
        "claim": "Gaussian mollification of a Dirac source costs about 14x in accuracy versus analytic integration: eps_x(t=1) = 3.6e-3 with the delta integrated analytically, versus 5.0e-2 for the best Gaussian mollifier (sigma_0 = 0.5 h_x) and 2e-1 at sigma_0 = 1.0 h_x",
        "source": "Reyna & Tartakovsky, arXiv:2606.12735v1, Figure 1d legend and Section 3.1.1"
      },
      {
        "claim": "'The PINN method failed to converge for this problem' (ADE with point-source initial condition) and 'As was the case for the ADE with a point source, and for similar reasons, we couldn't obtain reasonable solutions using PINNs' (ADE with mobile-immobile exchange)",
        "source": "Reyna & Tartakovsky, arXiv:2606.12735v1, Sections 3.1.1 and 3.2"
      },
      {
        "claim": "NTK result: with a Dirac source the residual 'does not converge to zero. Instead, it decreases linearly in training time, while all other residuals converge to nonzero constants' and 'increasing network width ... does not remove this coupling', so 'errors in PINNs with integrated Dirac deltas cannot be reduced by increasing the complexity of the Neural Network'",
        "source": "Reyna & Tartakovsky, arXiv:2606.12735v1, Section 3.3.1 and Appendix D"
      },
      {
        "claim": "RBF-RLS inverse robustness: across 30 random initializations every run converged to the same optimum, recovering v within 0.022% and D within 0.228% of reference, with a mean of only 11 forward solves",
        "source": "Reyna & Tartakovsky, arXiv:2606.12735v1, Section 3.1.1"
      },
      {
        "claim": "CONFIRMED — WellPINN max absolute error on the well falls from AE_max = 0.53 (single domain D1) to 0.11 (three domains D3); MAE reduced to 1.02e-2; max absolute residual from 1.1e1 to 3.1e-1; per-stage losses 1e-1->1e-4, 7e-1->1e-3, 3e1->9e-2",
        "source": "Walter et al., arXiv:2507.09330v1, Section 3 'Modeling Results', p. 9 — verified verbatim"
      },
      {
        "claim": "CONFIRMED — WellPINN training times 696 s, 1317 s and 1801 s (total 3814 s, about 63 minutes) on a single NVIDIA RTX A4500; 24,000 epochs per stage decomposing as Adam 20,000 + L-BFGS 4,000",
        "source": "Walter et al., arXiv:2507.09330v1, Section 3 p. 9 and Table 1 p. 18"
      },
      {
        "claim": "CONFIRMED verbatim — prior PINN well representations 'underestimate the well pressure p_w by up to 30% during late injection times and completely miss early-time diffusion', and typically require an equivalent well radius of 'at least 10% of the domain size'",
        "source": "Walter et al., arXiv:2507.09330v1, Introduction p. 3"
      },
      {
        "claim": "CONFIRMED — optimal equivalent-well-radius ratio: 'All three metrics show optimal accuracy for 0.1 < b < 0.17' from a parameter study over 0.04 < b < 0.22 with five realizations per value; b = 0.17 is the specific value chosen for the three-subdomain case",
        "source": "Walter et al., arXiv:2507.09330v1, Section 4 Discussion p. 11 and Figure 4"
      },
      {
        "claim": "CORRECTION — GW-PINN's activation function is sin, NOT tanh: 'We always use 4 hidden layers (i.e., D = 4) and 40 neurons per each hidden layer ... and the activation function is chosen as the sin function'",
        "source": "Zhang et al., GW-PINN accepted manuscript (OSTI 1976802), Section 5, line 465"
      },
      {
        "claim": "OMITTED BY FIRST PASS — GW-PINN's single-well confined aquifer case (Test Case 2) gives MAE 0.43 m and RRMSE 0.59% at 20 days, by far the worst of the five cases; the 0.010 m / 0.449% figure is Test Case 1, which contains NO pumping well (two parallel canals, precipitation only), and the 0.052 m / 0.058% L-shape figure is at t = 1 day, not 20 days",
        "source": "Zhang et al., GW-PINN accepted manuscript (OSTI 1976802), Sections 5.1.3, 5.2, 5.3 (lines 613, 631-632, 641, 650)"
      },
      {
        "claim": "CONFIRMED — GW-PINN soft constraint fails outright: 'the use of soft constraint fails to train GW-PINN successfully and almost nothing is learned in this case while the use of hard constraint is able to train GW-PINN very well' (Test Case 1), and 'The soft constraint again fails to train GW-PINN' (Test Case 2). Mollifier s = 30 confirmed; parameterized-Q surrogate uses 1250 spatial x 50 temporal x 50 Q points and trains in about 60 min",
        "source": "Zhang et al., GW-PINN accepted manuscript (OSTI 1976802), lines 492-495, 522-523, 469, 681-683"
      },
      {
        "claim": "Hybrid analytical-PINN cuts relative L2 error from 11.4% (analytical alone) to 0.63% (18x) for the infinite line source at month 180; FLS 11.5% -> 2.64%; MFLS 3.90% -> 1.42%. Network trained on a single source deploys to 25-source arrays by superposition with no retraining",
        "source": "Rao, Hamacher & Halilovic, arXiv:2607.12271 (arXiv:2607.12271v1 HTML), Scenario 1 results"
      },
      {
        "claim": "Fourier-MIONet trains about 3.5x faster with 90% fewer parameters than U-FNO, using under 15% of the CPU memory and under 35% of the GPU memory, at similar accuracy, and predicts 30 years of CO2 storage evolution from only 6 time snapshots",
        "source": "Jiang, Zhu & Lu, 'Fourier-MIONet', arXiv:2303.04778"
      },
      {
        "claim": "Neural-operator speedups for subsurface flow: GeoFUSE (U-FNO + PCA + ESMDA) reports approximately 360,000x for seawater intrusion; FNO for CO2 plume migration in realistic 3D geologies reports O(1e5); TFNO-opt for underground gas storage reports six orders of magnitude",
        "source": "arXiv:2410.20118 (GeoFUSE); arXiv:2503.11031; arXiv:2509.21485"
      },
      {
        "claim": "Composite Neural Network + Interface-PINNs for heterogeneous poroelastic media: RMSE 100x better than a standard PINN, 40x faster than a single-network approach, and one order of magnitude better than XPINNs. Curriculum training cuts poroelastic PINN training time by roughly 50%",
        "source": "arXiv:2407.03372 (CoNN + I-PINNs); arXiv:2404.13909 (curriculum training)"
      },
      {
        "claim": "Radial Muntz-Szasz Networks achieve 1.5x to 51x lower RMSE than MLPs and 10x to 100x lower than SIREN across 10 benchmarks in 2D/3D, using 27 parameters versus 33,537 (MLP) and 8,577 (SIREN); RMN-MC recovers source centres to below 1e-4",
        "source": "arXiv:2602.08419, 'Radial Muntz-Szasz Networks'"
      },
      {
        "claim": "ELM-FBPINN follow-up (QR-based local feature filtering and preconditioning) reports condition-number reductions of up to eleven orders of magnitude and LSQR speedups of 10-1000x",
        "source": "arXiv:2506.17626, 'Local Feature Filtering for Scalable and Well-Conditioned Domain-Decomposed Random Feature Methods'"
      },
      {
        "claim": "Differentiable-simulator reservoir characterization halved the pressure inference error versus a data-driven baseline while maintaining accuracy on extreme pressure events",
        "source": "arXiv:2604.13291, 'Physics-informed reservoir characterization from bulk and extreme pressure events with a differentiable simulator'"
      },
      {
        "claim": "Flotation column verified parameter set, reproducible exactly: phi_c = 0.74, n_S = 0.46, z_U = 0 m, z_F = 0.33 m, z_E = 1 m, phi_F = 0.3, psi_F = 0.2, Q_F = 8.85e-5 m3/s, Q_W = 2e-6 m3/s; Q_U = 5.94, 5.96, 5.975, 5.977 e-5 m3/s gives effluent phi_E = 0.8537, 0.8592, 0.8634, 0.8640 via phi_E = Q_F*phi_F/(Q_W+Q_F-Q_U). Transient runs use N = 800 cells.",
        "source": "Betancourt, Bürger, Diehl, Martí, Vásquez, CI2MA Preprint 2024-08, pp. 6-7 and Figures 2, 4, 7. Read directly from https://www.ci2ma.udec.cl/pdf/pre-publicaciones/2024/pp24-08.pdf . ALL FIRST-PASS NUMBERS CONFIRMED."
      },
      {
        "claim": "The feasible operating window for the flotation column is roughly Q_U in 5.4e-5 to 6.1e-5 m3/s at Q_F = 8.85e-5: at Q_U = 5.4e-5 the froth washes out entirely, at 6.1e-5 the tank fills with froth. That is about a +/-6% window, which quantifies the 'narrow wedge' the first pass described only qualitatively.",
        "source": "Same preprint, Section 7 and Figure 3(b); the desired steady state is at (Q_U, Q_F) = (5.9, 8.85)e-5. Read directly."
      },
      {
        "claim": "Korzhinskii-Net mineral prospectivity PINN: mean PR-AUC 0.708 versus 0.235 for the strongest classical baseline (SVM), and mean fractional rank 0.036 versus 0.475, across six ore provinces and three commodity classes under leakage-controlled 5-fold cross-validation with hard ring-shaped negatives.",
        "source": "Kriuk B., arXiv:2606.13695v2 (2026), abstract fetched verbatim from https://arxiv.org/abs/2606.13695 . Caveat: single-author, apparently not peer reviewed."
      },
      {
        "claim": "Economic MPC for froth flotation improved mineral recovery from 9% to 29% under feed flowrate disturbances while maintaining a minimum concentrate grade of 20%.",
        "source": "Quintanilla, Navia, Neethling, Brito-Parada, arXiv:2410.19661 (2024), abstract fetched verbatim. CRITICAL CAVEAT, verified from the same abstract: this is a 30-LITRE LABORATORY-SCALE cell, not a plant."
      },
      {
        "claim": "SAG mill digital twin: trained on 68 hours of industrial operational data with 8 hours for validation, predicting within a 2.5-minute horizon at 30-second intervals with errors smaller than 5%.",
        "source": "Quintanilla, Fernández, Mancilla, Rojas, Navia, arXiv:2503.04225 (2025), published Minerals Engineering 2025. Abstract fetched from https://arxiv.org/abs/2503.04225 ."
      },
      {
        "claim": "Bubble-particle collision model validated by DNS at bubble Stokes numbers 2.8 and 6.3, particle Stokes numbers 0.01 to 2, Taylor Reynolds number 64, with good model-simulation agreement only for Froude number Fr <= 0.25. Smaller bubbles, larger particles and stronger turbulence increase collision rates.",
        "source": "Chan, Jiang, Krug, arXiv:2507.13878v2 (2025); Chemical Engineering Science 321 (2026). Abstract fetched from https://arxiv.org/abs/2507.13878 ."
      },
      {
        "claim": "Corrected closed forms for the distributed-rate-constant flotation family: exponential gives R = R_inf[1 - 1/(1 + t/lambda)], NOT R_inf(1 - 1/(t/lambda)) as printed; triangular gives R = R_inf[1 - (1-exp(-bt))^2/(bt)^2], i.e. the coefficient must be -2exp(-bt), not -exp(-bt).",
        "source": "Derived, not cited: R(t) = R_inf[1 - L_f(t)] with L_f the Laplace transform of f(K). Exponential is Gamma with p=1; a triangular density on [0,2b] is the sum of two Uniform[0,b] variables so its transform is the square of the rectangular case. Certain by construction; corrects Bu et al. (2017), DOI 10.5277/ppmp170128, as transcribed by the first pass."
      },
      {
        "claim": "Countercurrent decantation model supplies an exact invariant for scoring a learned surrogate in physical units: global volume conservation q_w + q_in = q_e + q_u (Lemma 2.1), with feasibility requiring all effluent bulk velocities q_{e,i} >= 0 via a backward recursion. Numerical examples use N = 3 clarifier-thickeners.",
        "source": "Barajas-Calonge, Bürger, Diehl, Villada, CI2MA Preprint 2026-23 (20 July 2026), Sections 1.1, 2.1-2.2 and Lemma 2.1. Read directly from https://www.ci2ma.udec.cl/pdf/pre-publicaciones/2026/pp26-23.pdf ."
      },
      {
        "claim": "Contradictory PIKAN evidence, both directions worth quoting: one 2026 matched-parameter benchmark reports PIKANs consistently more accurate with fewer iterations, while another reports KANs underperform MLPs on oscillatory systems with hyperparameter fragility, and a third reports MLP-based PINNs solving inverse problems roughly 1,000 times FASTER than PIKANs at comparable accuracy.",
        "source": "arXiv:2602.15068 (abstract fetched, no numbers given); arXiv:2602.09988 and arXiv:2512.12074 (arXiv API listing metadata only, UNVERIFIED at abstract level)."
      },
      {
        "claim": "Neural-operator speedups reported in adjacent subsurface domains, all UNVERIFIED listing metadata: FNO for CO2 storage O(10^5) acceleration (arXiv:2503.11031); GeoFUSE U-FNO 360,000x versus PFLOTRAN (arXiv:2410.20118); STONet ~100x for solute transport (arXiv:2412.05576); U-DeepONet 18x faster training than U-FNO (arXiv:2311.15288).",
        "source": "arXiv API listing summaries; individual abstracts NOT opened. Treat as indicative only. Note these compare a forward pass against a cold simulator run and exclude training cost, and the 1-D mining models solve in milliseconds anyway, so speed is the wrong argument for a surrogate here."
      },
      {
        "claim": "Sensitivity attribution in tailings dam-breach flows is spatially structured: breach parameters dominate near the dam, tailings yield stress dominates farther downstream.",
        "source": "Taglieri Sáo, de Freitas Maciel, Cardoso Eleutério, arXiv:2607.19296v1 (21 July 2026), abstract fetched. No numerical Sobol indices are given in the abstract."
      },
      {
        "claim": "The arXiv 'discontinuous flux' corpus (30 most recent papers) contains ZERO machine-learning or neural-network papers and zero clarifier-thickener, sedimentation or flotation applications; applications are dominated by road traffic (LWR/ARZ models). This is the direct evidence for the white space.",
        "source": "arXiv API query on abs:\"discontinuous flux\", 30 results sorted by submission date, retrieved 22 July 2026. Explicit negative result reported in the returned listing."
      },
      {
        "claim": "CORRECTION. B-PINN 2D inverse problem, noise level 0.1, true k=1.0: HMC recovers 0.978 with sd 4.98e-2 (2.2% relative error); VI recovers 1.116 with sd 3.45e-2 (11.6% relative error). The first pass's 'HMC relative errors on k >5%' is wrong; the true figure is 2.2%.",
        "source": "ar5iv.labs.arxiv.org/html/2003.06097 Table 2 (Yang, Meng, Karniadakis, J. Comput. Phys. 425:109913, DOI 10.1016/j.jcp.2020.109913)"
      },
      {
        "claim": "CORRECTION. B-PINN/HMC empirical coverage at 95% nominal is 0.51, on 1D NONLINEAR Poisson at noise setting rho=0.30. At rho=0.10 the same method covers 1.00. The first pass reported '0.54 ... at sigma_f=0.1', wrong in both value and condition.",
        "source": "arxiv.org/html/2503.19333v2 Table 2 (Jacob, Nair, Howard, Drgona, Stinis, 'E-PINNs')"
      },
      {
        "claim": "UNVERIFIED, RECOMMEND DROPPING. The claim '17.5-21.7 HMC iterations/s vs 95-135 it/s' does not appear in arXiv:2503.19333. The paper reports only total wall-clock seconds: 'Time (s) is total wall-clock elapsed time as follows: E-PINN totals base and epinet training time, Dropout-PINN sums training and Monte Carlo inference time, and B-PINN reports Hamiltonian Monte Carlo sampling plus inference time.'",
        "source": "arxiv.org/html/2503.19333v2, full-text check of all tables and the cost discussion"
      },
      {
        "claim": "CORRECTED READING. At 95% nominal in the E-PINN study, E-PINN, Dropout-5% and Dropout-10% report coverage 1.00 in essentially every cell of Tables 1 (1D Poisson), 2 (1D nonlinear Poisson) and 4 (2D nonlinear Poisson), across all noise levels. B-PINN reports 1.00 except for 0.51 at rho=0.30 (Table 2) and 0.99 on the 2D problem. The comparison is over-coverage versus one catastrophic under-coverage, not 'calibrated versus uncalibrated'.",
        "source": "arxiv.org/html/2503.19333v2 Tables 1, 2 and 4"
      },
      {
        "claim": "CONFIRMED. 2D Allen-Cahn at 0.95 nominal, coverage BEFORE conformal calibration: geometric distance 1.00, latent distance 0.84, dropout 0.42, VI 0.99, HMC 0.76. After calibration all methods land in 0.94-0.97.",
        "source": "ar5iv.labs.arxiv.org/html/2509.13717 Table 1 (Yu, Ho, Wang), independently re-fetched"
      },
      {
        "claim": "CONFIRMED. 3D Helmholtz at 0.95 nominal, before -> after conformal: geometric distance 1.00 -> 0.95, latent distance 0.12 -> 0.92, dropout 0.99 -> 0.95.",
        "source": "ar5iv.labs.arxiv.org/html/2509.13717 Table 2"
      },
      {
        "claim": "PARTIALLY CONFIRMED. 1D damped harmonic oscillator, LOCAL conformal prediction: 0.96 coverage with ACD 0.0183 over the global domain, and 0.94 coverage with ACD 0.0669 restricted to the heteroskedastic regions. The first pass's companion figures for STANDARD CP in those regions (0.76 coverage, ACD 0.2925) were not returned by my fetch of the table and remain unconfirmed.",
        "source": "ar5iv.labs.arxiv.org/html/2509.13717 Table 3"
      },
      {
        "claim": "NUANCE THE FIRST PASS OMITTED. In the same 1D inverse table, dropout's standard deviations are 6.508e-3 (1% rate) and 6.45e-3 (5% rate) against HMC's 5.63e-2. Dropout produces bands roughly 9x NARROWER than HMC while being more biased (0.746 and 0.633 versus true 0.7), which is why it collapses to 0.42 coverage in the conformal study. Narrowness is the pathology, not the merit.",
        "source": "ar5iv.labs.arxiv.org/html/2003.06097 Table 1, cross-read against arXiv:2509.13717 Table 1"
      },
      {
        "claim": "Surrogate-based hybrid ES-MDA for Geological Carbon Storage, using FNO and Transformer-UNet surrogates, 'has the potential to make the standard ESMDA process at least 50% faster or more, depending on the number of assimilation steps'; and 'SH-RML offers better uncertainty quantification compared to conventional ESMDA for the case study'.",
        "source": "arXiv:2402.06110 (Seabra, Muecke, Silva, Voskov, Vossepoel), Int. J. Greenhouse Gas Control, DOI 10.1016/j.ijggc.2024.104190"
      },
      {
        "claim": "Flow matching posterior estimation on gravitational-wave inference outperforms comparable discrete flows, 'reducing training time by 30% with substantially improved accuracy'.",
        "source": "arXiv:2305.17161 (Dax, Wildberger, Buchholz, Green, Macke, Schoelkopf), NeurIPS 2023"
      },
      {
        "claim": "Amortized variational inference with a conditional normalizing flow plus physics-based latent correction delivers a seismic image and its uncertainty 'at approximately the same cost as five reverse-time migrations'.",
        "source": "arXiv:2207.11640 (Siahkoohi, Rizzuti, Orozco, Herrmann)"
      },
      {
        "claim": "Risk-controlling quantile neural operator on a 3D car-surface pressure task: 'our method is the only one that meets the target calibration percentage (percentage of test samples for which the uncertainty estimates are calibrated) of 98%.'",
        "source": "arXiv:2402.01960 (Ma, Azizzadenesheli, Anandkumar)"
      },
      {
        "claim": "Split conformal on an FNO (33.7M parameters, 800 training samples, 5 ensemble members, NVIDIA V100) for steady-state heat conduction: 89.1% empirical coverage at target alpha=0.1, with uncertainty decomposed as 68% epistemic / 32% aleatoric. Note this UNDER-covers the 90% target, so it is a useful counterexample to 'conformal always lands on nominal' when the calibration set is small or non-exchangeable.",
        "source": "arXiv:2606.09923 (Chin, 'Conformal Prediction for Neural Operators')"
      },
      {
        "claim": "GP/kernel collocation for nonlinear PDEs and inverse problems converges 'in a small number of iterations (2 to 10), for a wide range of PDEs', and unlike traditional inverse-problem approaches 'solves for both parameter and PDE solution simultaneously'. Validated on permeability identification in Darcy flow.",
        "source": "arXiv:2103.12959 (Chen, Hosseini, Owhadi, Stuart), J. Comput. Phys."
      },
      {
        "claim": "Epinets: 'With an epinet, conventional neural networks outperform very large ensembles, consisting of hundreds or more particles, with orders of magnitude less computation.' The authors also state the epinet 'does not fit the traditional framework of Bayesian neural networks.'",
        "source": "arXiv:2107.08924 (Osband, Wen, Asghari, Dwaracherla, Ibrahimi, Lu, Van Roy)"
      },
      {
        "claim": "EKI as the B-PINN posterior estimator 'can achieve inference results with informative uncertainty estimates comparable to Hamiltonian Monte Carlo (HMC)-based B-PINNs with a much reduced computational cost', and needs no autodiff because 'derivatives are replaced by differences from within the ensemble'.",
        "source": "arXiv:2303.07392 (Pensoneault & Zhu) and arXiv:1808.03620 (Kovachki & Stuart, DOI 10.1088/1361-6420/ab1c3a)"
      },
      {
        "claim": "Deep evidential regression is not an exact posterior: 'we detail the theoretical shortcomings and analyze the performance on synthetic and real-world data sets, showing that Deep Evidential Regression is a heuristic rather than an exact uncertainty quantification.'",
        "source": "arXiv:2205.10060 (Meinert, Gawlikowski, Lavin), AAAI 37(8):9134-9142, DOI 10.1609/aaai.v37i8.26096"
      },
      {
        "claim": "Repulsive ensembles are provably Bayesian: adding a kernelized repulsion term 'transforms the maximum a posteriori inference into proper Bayesian inference. Namely, we show that the training dynamics of our proposed repulsive ensembles follow a Wasserstein gradient flow of the KL divergence with the true posterior.' This directly repairs the first pass's 'IT IS NOT A POSTERIOR' objection to deep ensembles.",
        "source": "arXiv:2106.11642 (D'Angelo & Fortuin), NeurIPS 2021"
      },
      {
        "claim": "TARP coverage testing is 'necessary and sufficient to show that a posterior estimator is accurate', requires only posterior SAMPLES rather than density evaluations (so it applies to diffusion and GAN samplers), and 'can detect inaccurate inferences in cases where existing methods fail'.",
        "source": "arXiv:2302.03026 (Lemos, Coogan, Hezaveh, Perreault-Levasseur), ICML 2023"
      },
      {
        "claim": "Physics-informed information field theory carries a self-diagnosis for model-form error: 'the method is equipped with a metric which allows the posterior to automatically quantify model-form uncertainty' and 'correctly identifies when the physics cannot be trusted, in which case it automatically treats learning the field as a regression problem'.",
        "source": "arXiv:2301.07609 (Alberts & Bilionis), J. Comput. Phys., DOI 10.1016/j.jcp.2023.112100"
      }
    ]
  },
  "workflowProgress": [
    {
      "type": "workflow_phase",
      "index": 1,
      "title": "Survey"
    },
    {
      "type": "workflow_phase",
      "index": 2,
      "title": "Deepen"
    },
    {
      "type": "workflow_phase",
      "index": 3,
      "title": "Synthesize"
    },
    {
      "type": "workflow_agent",
      "index": 1,
      "label": "survey:operators",
      "phaseIndex": 1,
      "phaseTitle": "Survey",
      "agentId": "ac8dba7fc0200f246",
      "model": "claude-opus-4-8[1m]",
      "state": "done",
      "startedAt": 1784719798247,
      "queuedAt": 1784719798209,
      "attempt": 1,
      "lastToolName": "StructuredOutput",
      "promptPreview": "Research NEURAL OPERATORS for PDEs, 2020-2026, focused on what is IMPLEMENTABLE in a small offline PyTorch pipeline that exports to ONNX for browser replay.\n\nCover: Fourier Neural Operator (FNO, arXiv:2010.08895), Physics-Informed Neural Operator (PINO, arXiv:2111.03794), DeepONet + physics-informed DeepONet, Geo-FNO (irregular geometries), U-FNO, F-FNO (factorized), GINO, Convolutional Neural Ope…",
      "lastProgressAt": 1784723748324,
      "tokens": 189505,
      "toolCalls": 87,
      "durationMs": 3950075,
      "resultPreview": "{\"keyFindings\":[\"ONNX EXPORT IS SOLVED, BUT ONLY ONE WAY (empirically verified on this machine, torch 2.12.1+cpu / onnx 1.22.0 / onnxruntime 1.27.0 / onnxscript 0.7.0, opset 20). The legacy TorchScript exporter (dynamo=False) FAILS: `UnsupportedOperatorError: Exporting the operator 'aten::fft_rfft2' to ONNX opset version 20 is not supported`. The dynamo exporter (dynamo=True) SUCCEEDS and lowers t…"
    },
    {
      "type": "workflow_agent",
      "index": 2,
      "label": "survey:geometric",
      "phaseIndex": 1,
      "phaseTitle": "Survey",
      "agentId": "a199a8bd431363af1",
      "model": "claude-opus-4-8[1m]",
      "state": "done",
      "startedAt": 1784719798249,
      "queuedAt": 1784719798209,
      "attempt": 1,
      "lastToolName": "StructuredOutput",
      "promptPreview": "Research GEOMETRIC and STRUCTURE-PRESERVING physics-informed machine learning, 2019-2026. Context: the RSS 2026 workshop \"The Geometry of Motion: Physics-Informed Structures for Learning and Control\" (speakers Arne Sachtler, Jake Welde, Thomas Cohn, Rodrigo Perez Dattari) lists these topics: Riemannian geometry in robotics; Lie groups, symmetries and invariances for learning and control; geometric…",
      "lastProgressAt": 1784723081327,
      "tokens": 123650,
      "toolCalls": 76,
      "durationMs": 3283077,
      "resultPreview": "{\"methods\":[{\"name\":\"Hamiltonian Neural Networks (HNN)\",\"whatItIs\":\"An MLP outputs a single scalar H_theta(q,p); the vector field is the symplectic gradient of that scalar rather than a directly regressed derivative. Trained on (q,p,dq/dt,dp/dt) tuples, unsupervised w.r.t. energy (energy is never labelled).\",\"formulation\":\"Dynamics: $\\\\dot q = \\\\partial\\\\mathcal H/\\\\partial p,\\\\; \\\\dot p = -\\\\part…"
    },
    {
      "type": "workflow_agent",
      "index": 3,
      "label": "survey:pinnfixes",
      "phaseIndex": 1,
      "phaseTitle": "Survey",
      "agentId": "a24b61e13db1219eb",
      "model": "claude-opus-4-8[1m]",
      "state": "done",
      "startedAt": 1784719798250,
      "queuedAt": 1784719798209,
      "attempt": 1,
      "lastToolName": "StructuredOutput",
      "promptPreview": "Research the KNOWN FAILURE MODES of Physics-Informed Neural Networks and the state-of-the-art FIXES, 2020-2026. A domain expert publicly criticized a PINN catalogue with exactly this: \"PINNs are usually very good at solving smooth systems; the issue is PINNs tend to have very long training times and gradient explosions in their structures.\"\n\nI need the rigorous, citable version of that critique an…",
      "lastProgressAt": 1784722505695,
      "tokens": 133543,
      "toolCalls": 84,
      "durationMs": 2707444,
      "resultPreview": "{\"methods\":[{\"name\":\"Gradient pathology diagnosis + learning-rate annealing (grad-norm loss balancing)\",\"whatItIs\":\"The founding diagnosis of the critic's complaint. Wang, Teng & Perdikaris show the composite PINN loss L = L_r + sum_i lambda_i L_i is numerically stiff: the back-propagated gradients of the residual term and of the boundary/initial terms have wildly different magnitudes, so gradient…"
    },
    {
      "type": "workflow_agent",
      "index": 4,
      "label": "survey:applied-subsurface",
      "phaseIndex": 1,
      "phaseTitle": "Survey",
      "agentId": "a035f9326003235f7",
      "model": "claude-opus-4-8[1m]",
      "state": "done",
      "startedAt": 1784719798251,
      "queuedAt": 1784719798209,
      "attempt": 1,
      "lastToolName": "StructuredOutput",
      "promptPreview": "Research APPLIED physics-informed ML problems in SUBSURFACE / GROUNDWATER / geotechnical engineering, with an emphasis on problems that can be posed cleanly and validated against a classical numerical reference.\n\nSpecifically:\n- transient groundwater flow (confined aquifer): the governing PDE S dh/dt = div(K grad h) + recharge - pumping; the Theis solution and its assumptions; typical parameter ra…",
      "lastProgressAt": 1784723285317,
      "tokens": 220001,
      "toolCalls": 91,
      "durationMs": 3487066,
      "resultPreview": "{\"methods\":[{\"name\":\"Theis (1935) transient confined-aquifer drawdown — the canonical analytic anchor\",\"whatItIs\":\"Exact closed-form solution of radial transient flow to a fully-penetrating well of infinitesimal diameter in a homogeneous, isotropic, infinite, confined aquifer of constant thickness. This is THE clean validation anchor for any groundwater PINN: it is an exact solution of the same PD…"
    },
    {
      "type": "workflow_agent",
      "index": 5,
      "label": "survey:applied-mining",
      "phaseIndex": 1,
      "phaseTitle": "Survey",
      "agentId": "a821fce3e620e6916",
      "model": "claude-opus-4-8[1m]",
      "state": "done",
      "startedAt": 1784719798252,
      "queuedAt": 1784719798210,
      "attempt": 1,
      "lastToolName": "StructuredOutput",
      "promptPreview": "Research APPLIED physics-informed machine learning and mechanistic modelling in MINING and MINERAL PROCESSING, 2020-2026, that goes BEYOND simple first-order kinetics.\n\nCover, with equations and citations:\n- flotation: beyond first-order kinetics - distributed rate constants (gamma/rectangular distribution of k), froth recovery and the collection/froth split, bubble-particle attachment models, pop…",
      "lastProgressAt": 1784723537490,
      "tokens": 256346,
      "toolCalls": 92,
      "durationMs": 3739237,
      "resultPreview": "{\"methods\":[{\"name\":\"Distributed flotation rate constant: rectangular (Klimpel) and gamma models\",\"whatItIs\":\"Replaces the single first-order rate constant with a continuous distribution f(K) over a heterogeneous feed. Integrating the first-order kinetics over f(K) gives closed-form recovery-time curves that fit real batch tests far better than a single exponential, without any PDE solve.\",\"formul…"
    },
    {
      "type": "workflow_agent",
      "index": 6,
      "label": "survey:uq-inverse (retry 1)",
      "phaseIndex": 1,
      "phaseTitle": "Survey",
      "agentId": "a3c8a4fa5c444ad7c",
      "model": "claude-opus-4-8[1m]",
      "state": "done",
      "startedAt": 1784719798252,
      "queuedAt": 1784719798210,
      "attempt": 2,
      "lastAttemptReason": "stalled",
      "lastToolName": "StructuredOutput",
      "promptPreview": "Research the SOTA in UNCERTAINTY QUANTIFICATION and INVERSE PROBLEMS with physics-informed machine learning and neural operators, 2021-2026.\n\nCover with citations:\n- Bayesian PINNs (B-PINN, Yang/Meng/Karniadakis), deep ensembles vs HMC vs variational inference for PINN UQ, and what each actually calibrates\n- conformal prediction applied to PINN/neural-operator outputs (distribution-free coverage g…",
      "lastProgressAt": 1784724604659,
      "tokens": 287738,
      "toolCalls": 209,
      "durationMs": 4806406,
      "resultPreview": "{\"methods\":[{\"name\":\"B-PINN (Bayesian Physics-Informed Neural Network) with HMC or VI posterior estimation\",\"whatItIs\":\"A BNN whose likelihood contains BOTH measurement terms and PDE-residual terms; the PINN acts as the prior structure, HMC or mean-field VI as the posterior estimator. Unknown PDE parameters (e.g. diffusivity k) are appended to the parameter vector and sampled jointly with the weig…"
    },
    {
      "type": "workflow_agent",
      "index": 7,
      "label": "deepen:pinnfixes (retry 1)",
      "phaseIndex": 2,
      "phaseTitle": "Deepen",
      "agentId": "ab4b46fcc6aa5b950",
      "model": "claude-opus-4-8[1m]",
      "state": "done",
      "startedAt": 1784722505704,
      "queuedAt": 1784722505701,
      "attempt": 2,
      "lastAttemptReason": "stalled",
      "lastToolName": "StructuredOutput",
      "promptPreview": "You are reviewing a research survey on \"pinnfixes\" for a physics-informed ML product that must go BEYOND the state of the art.\n\nHere is what the first pass found:\n{\n \"methods\": [\n  {\n   \"name\": \"Gradient pathology diagnosis + learning-rate annealing (grad-norm loss balancing)\",\n   \"whatItIs\": \"The founding diagnosis of the critic's complaint. Wang, Teng & Perdikaris show the composite PINN loss L …",
      "lastProgressAt": 1784724604370,
      "tokens": 154405,
      "toolCalls": 65,
      "durationMs": 2098664,
      "resultPreview": "{\"methods\":[{\"name\":\"ConFIG (Conflict-Free Inverse Gradients)\",\"whatItIs\":\"The first-pass list balances gradient MAGNITUDES only. ConFIG attacks the orthogonal failure: the residual gradient and the BC/IC gradient point in conflicting DIRECTIONS (negative inner product), so any magnitude rescaling still produces an update that increases one of the losses. ConFIG computes an update direction guaran…"
    },
    {
      "type": "workflow_agent",
      "index": 8,
      "label": "deepen:geometric",
      "phaseIndex": 2,
      "phaseTitle": "Deepen",
      "agentId": "ae686d3c5ac7221f8",
      "model": "claude-opus-4-8[1m]",
      "state": "done",
      "startedAt": 1784723081348,
      "queuedAt": 1784723081345,
      "attempt": 1,
      "lastToolName": "StructuredOutput",
      "promptPreview": "You are reviewing a research survey on \"geometric\" for a physics-informed ML product that must go BEYOND the state of the art.\n\nHere is what the first pass found:\n{\n \"methods\": [\n  {\n   \"name\": \"Hamiltonian Neural Networks (HNN)\",\n   \"whatItIs\": \"An MLP outputs a single scalar H_theta(q,p); the vector field is the symplectic gradient of that scalar rather than a directly regressed derivative. Trai…",
      "lastProgressAt": 1784724604576,
      "tokens": 96658,
      "toolCalls": 54,
      "durationMs": 1523226,
      "resultPreview": "{\"keyFindings\":[\"THE BIGGEST MISS IS DISSIPATION. Every method in the first pass (HNN, LNN, DeLaN, SympNets, SRNN, VIN) models a CONSERVATIVE or a frictionless rigid-body system. No mining, industrial or subsurface process is conservative: comminution, flotation, conveyor friction, viscoplastic rock, groundwater flow and heat transfer are all irreversible. The entire port-Hamiltonian branch (SymOD…"
    },
    {
      "type": "workflow_agent",
      "index": 9,
      "label": "deepen:applied-subsurface",
      "phaseIndex": 2,
      "phaseTitle": "Deepen",
      "agentId": "aea693e6969f85516",
      "model": "claude-opus-4-8[1m]",
      "state": "done",
      "startedAt": 1784723285321,
      "queuedAt": 1784723285320,
      "attempt": 1,
      "lastToolName": "StructuredOutput",
      "promptPreview": "You are reviewing a research survey on \"applied-subsurface\" for a physics-informed ML product that must go BEYOND the state of the art.\n\nHere is what the first pass found:\n{\n \"methods\": [\n  {\n   \"name\": \"Theis (1935) transient confined-aquifer drawdown — the canonical analytic anchor\",\n   \"whatItIs\": \"Exact closed-form solution of radial transient flow to a fully-penetrating well of infinitesimal …",
      "lastProgressAt": 1784724604431,
      "tokens": 186375,
      "toolCalls": 33,
      "durationMs": 1319109,
      "resultPreview": "{\"methods\":[{\"name\":\"RBF-RLS with analytically integrated Dirac source (Reyna & Tartakovsky 2026) — refutes the mollifier\",\"whatItIs\":\"The single most important omission. Tartakovsky's group (the top authority in subsurface PINNs) shows that (a) the Dirac source can be integrated out ANALYTICALLY at the residual level, needing no Gaussian mollifier at all, and (b) once you do that, a PINN provably…"
    },
    {
      "type": "workflow_agent",
      "index": 10,
      "label": "deepen:applied-mining",
      "phaseIndex": 2,
      "phaseTitle": "Deepen",
      "agentId": "aad40c4cf3cc63eed",
      "model": "claude-opus-4-8[1m]",
      "state": "done",
      "startedAt": 1784723537493,
      "queuedAt": 1784723537492,
      "attempt": 1,
      "lastToolName": "StructuredOutput",
      "promptPreview": "You are reviewing a research survey on \"applied-mining\" for a physics-informed ML product that must go BEYOND the state of the art.\n\nHere is what the first pass found:\n{\n \"methods\": [\n  {\n   \"name\": \"Distributed flotation rate constant: rectangular (Klimpel) and gamma models\",\n   \"whatItIs\": \"Replaces the single first-order rate constant with a continuous distribution f(K) over a heterogeneous fee…",
      "lastProgressAt": 1784728601723,
      "tokens": 129594,
      "toolCalls": 44,
      "durationMs": 5064230,
      "resultPreview": "{\"keyFindings\":[\"VERIFICATION PASSED on the flotation column (CI2MA pp24-08). I read the PDF directly and every headline number the first pass quoted is correct verbatim. Confirmed on p.6: phi_c = 0.74 '(cf. Eq. (21) of Neethling and Cilliers, 2003)' and n_S = 0.46 '(Stevenson et al., 2008)'. Confirmed on p.7: z_U = 0 m, z_F = 0.33 m, z_E = 1 m, and the Figure 2 caption gives exactly phi_F=0.3, ps…"
    },
    {
      "type": "workflow_agent",
      "index": 11,
      "label": "deepen:operators",
      "phaseIndex": 2,
      "phaseTitle": "Deepen",
      "agentId": "a6579a1c59027ca70",
      "model": "claude-opus-4-8[1m]",
      "state": "done",
      "startedAt": 1784723748328,
      "queuedAt": 1784723748326,
      "attempt": 1,
      "lastToolName": "StructuredOutput",
      "promptPreview": "You are reviewing a research survey on \"operators\" for a physics-informed ML product that must go BEYOND the state of the art.\n\nHere is what the first pass found:\n{\n \"keyFindings\": [\n  \"ONNX EXPORT IS SOLVED, BUT ONLY ONE WAY (empirically verified on this machine, torch 2.12.1+cpu / onnx 1.22.0 / onnxruntime 1.27.0 / onnxscript 0.7.0, opset 20). The legacy TorchScript exporter (dynamo=False) FAILS…",
      "lastProgressAt": 1784727455451,
      "tokens": 109559,
      "toolCalls": 64,
      "durationMs": 3707123,
      "resultPreview": "{\"keyFindings\":[\"CORRECTION, THE MOST IMPORTANT ONE: THE ZERO-SHOT SUPER-RESOLUTION DEMO IS EXPORTABLE BUT SCIENTIFICALLY UNRELIABLE. The first pass verified the *mechanics* (dynamic_shapes keeps one graph running at 64/96/128 and matching PyTorch to ~6e-08) and that is correct and useful. But it then framed resolution invariance as the flagship demo. arXiv:2606.00677 (Colagrande, Caillon, Feillet…"
    },
    {
      "type": "workflow_agent",
      "index": 12,
      "label": "deepen:uq-inverse",
      "phaseIndex": 2,
      "phaseTitle": "Deepen",
      "agentId": "ab6b2a4e00d09e410",
      "model": "claude-opus-4-8[1m]",
      "state": "done",
      "startedAt": 1784724604705,
      "queuedAt": 1784724604700,
      "attempt": 1,
      "lastToolName": "StructuredOutput",
      "promptPreview": "You are reviewing a research survey on \"uq-inverse\" for a physics-informed ML product that must go BEYOND the state of the art.\n\nHere is what the first pass found:\n{\n \"methods\": [\n  {\n   \"name\": \"B-PINN (Bayesian Physics-Informed Neural Network) with HMC or VI posterior estimation\",\n   \"whatItIs\": \"A BNN whose likelihood contains BOTH measurement terms and PDE-residual terms; the PINN acts as the …",
      "lastProgressAt": 1784727340050,
      "tokens": 95246,
      "toolCalls": 25,
      "durationMs": 2735344,
      "resultPreview": "{\"methods\":[{\"name\":\"Amortized posterior estimation with conditional normalizing flows (NPE / APT-SNPE-C / BayesFlow / FMPE), incl. the geophysical instantiation WISE\",\"whatItIs\":\"THE LARGEST GAP IN THE FIRST PASS: it contains no amortized-inference entry at all. Instead of sampling a posterior per dataset (HMC) you TRAIN a conditional density estimator q_phi(theta | y) on simulated (theta, y) pai…"
    },
    {
      "type": "workflow_agent",
      "index": 13,
      "label": "synthesize:plan (retry 1)",
      "phaseIndex": 3,
      "phaseTitle": "Synthesize",
      "agentId": "ac405fdba05655fa2",
      "model": "claude-opus-4-8[1m]",
      "state": "done",
      "startedAt": 1784728601743,
      "queuedAt": 1784728601737,
      "attempt": 2,
      "lastAttemptReason": "stalled",
      "lastToolName": "Bash",
      "lastToolSummary": "cd \"D:/_Repos/Research_Caos/CAOS_PINNLAB/data-pipeline\" && …",
      "promptPreview": "You are the research lead for PINN-Lab, an open catalogue of physics-informed machine learning cases.\n\nCURRENT STATE (verified from the repo, do not assume otherwise). 21 cases, one method each:\n- hard-constraints (poisson2d, ctrl-zero-source), hard-constraints-rar (allencahn, burgers1d),\n  time-dependent-hard-constraints (heat1d), siren-hard-constraints (wave1d),\n  multioutput-loss-weighting (nav…",
      "lastProgressAt": 1784730521968,
      "tokens": 219031,
      "toolCalls": 43,
      "durationMs": 1920222,
      "resultPreview": "I verified the repo state directly; the brief's premise is out of date in one decisive way and the shipped artifact contradicts the plan it was built from. Both facts drive this plan.\n\n---\n\n# PINN-Lab: implementation plan, operator & structure-preserving expansion\n\n**Verified against the working tree** (`D:\\_Repos\\Research_Caos\\CAOS_PINNLAB`, branch `develop`, HEAD `a0ac5d5`) on 2026-07-22. Every …"
    }
  ],
  "totalTokens": 2201651,
  "totalToolCalls": 967
}