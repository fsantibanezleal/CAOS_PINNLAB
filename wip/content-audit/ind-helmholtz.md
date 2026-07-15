# Content audit: ind-helmholtz (in-app vs authoritative doc)

Date: 2026-07-15
Auditor: content-audit subagent (honest depth + coherence pass; not a rubber-stamp)

Ground truth: `docs/cases/ind-helmholtz.md`
In-app files audited:
- Deep context: `frontend/src/content/cases/HelmholtzContext.tsx` (EN + ES)
- Scenario: `frontend/src/content/scenarios.ts` (`"ind-helmholtz"`)
- Results: `frontend/src/content/results.ts` (`"ind-helmholtz"`)
- Constraints: `frontend/src/content/constraints.ts` (`"ind-helmholtz"`)

## Verdict

**Severity 2 (real gaps, one minor contradiction). Broadly coherent, genuinely deep Context, but not complete.**

The Context is one of the richer ones in the catalogue: it carries the governing equation, a correct MMS
residual derivation, the spectral-bias citation (Rahaman 2019), the Fourier-feature map with `B~N(0,σ²)` and
the two scales, the frozen/seeded `B`, ONNX traceability, and a strong scope/out-of-scope section (sweep,
resonance, Sommerfeld, heterogeneous medium). The scenario/results/constraints short-content is substantive
and its method numbers (9.3%, 121%, 100x weight, k₀=6π, MMS+FDM anchor) match the doc.

Two problems keep this off a clean pass:
1. **One factual contradiction**: the Context's lobe-count phrase "three per side, nine by nine in all"
   (ES: "tres por lado, nueve por nueve en total") is wrong and internally inconsistent. The field
   `sin(6πx)sin(6πy)` has 6 sign-lobes per axis (36 total), not "nine by nine" (81).
2. **Depth gaps**: the Context never states the case's HEADLINE measured numbers or the doc's central
   honesty caveat (the ~10% relative-L2 is CPU-limited, not 1e-3, would tighten with GPU + frequency
   annealing), never states the network architecture / optimizer, and its "how to read the viz" section
   omits the Compare / Diagnostics / Training views that carry the case's strongest evidence.

None of this is hollow or fabricated. The fixes are additive plus one correction.

---

## Contradictions (inApp vs docSays)

### C1. Field lobe count is wrong and self-inconsistent  (HelmholtzContext.tsx, EN line ~180-181, ES line ~85-86)

- **inApp (EN):** "a regular grid of lobes alternating between +1 and −1 (three per side, **nine by nine in all**)"
- **inApp (ES):** "una cuadrícula regular de lóbulos que alternan entre +1 y −1 (tres por lado, **nueve por nueve en total**)"
- **docSays:** "$k_0=6\pi$, i.e. three full standing-wave periods per axis" and "three full wavelengths per side."
- **Why it's wrong:** `sin(6πx)` completes 3 full periods on [0,1], so it has 6 sign-alternating lobes per
  axis (3 positive + 3 negative), giving **6×6 = 36** lobes in 2D. The phrase is also internally
  inconsistent: "three per side" implies 3×3 = 9 total, but the text says "nine by nine" = 81. Neither 9 nor
  81 matches the true 36. **Fix:** either "six lobes per side, thirty-six in all" (counting sign lobes), or,
  to stay aligned with the doc's wavelength framing, "three full wavelengths per side, a 6x6 checkerboard of
  36 alternating lobes."

### C2. "perfectly reflecting walls" is a loose/misleading gloss for the Dirichlet BC  (results.ts, assumptions_en/es)

- **inApp:** assumptions list `"perfectly reflecting walls"` (ES `"paredes perfectamente reflectantes"`).
- **docSays:** "homogeneous Dirichlet walls ... $u|_{\partial\Omega}=0$"; the Context frames this as "a
  clamped membrane patch" / fixed edge.
- **Why it's a nuance, not a clean match:** `u=0` is a pressure-release (mechanically, clamped/fixed) boundary.
  It is fully reflecting in the energy sense, but "perfectly reflecting walls" reads to an acoustician as a
  RIGID (Neumann, ∂u/∂n=0) wall, which is the opposite BC. Low severity, but it should say "fixed / clamped
  (pressure-release) walls, u=0" to match the doc's Dirichlet framing rather than imply a rigid wall.

---

## Depth gaps (real doc content the app omits)

### G1. The headline measured result and the honesty caveat are absent from the Context

The doc's Result + Honesty sections are the crux of the case, and the deep Context never states them:

- relative-L2 vs analytic u* = **0.1026 (~10%)**, max abs error = **0.158**, ONNX parity (max abs) = **4.77e-07**,
  live lane = **340 KB ONNX, 18.5 ms infer**, measured at seed 42.
- The doc's honesty point: "The ~10% relative-L2 is **honestly CPU-limited** ... at $k_0=6\pi$ on the CPU lane
  the standing pattern is resolved to ~10%, not to 1e-3 ... GPU training plus **frequency annealing** would
  tighten it further; this is not dressed up."

The Context's closing viz paragraph says only "compare its output against the exact pattern to see the PINN's
residual error" without ever quoting the residual or the CPU-limited caveat. This is the single biggest gap.

### G2. Network architecture and optimizer are omitted

Doc Method: "a **256-dim embedding** feeding a **4x128 tanh FNN**"; "Optimization is **Adam (25 000) -> L-BFGS**";
collocation "~12 points per wavelength per axis." The Context describes the Fourier map in detail but never
states the network size or the two-stage optimizer. These are concrete, transcribable method facts.

### G3. The "how to read the viz" guidance omits Compare / Diagnostics / Training

The doc devotes three sections to these views with real numbers:
- **Compare:** naive lane **120.8%** relative-L2 vs standard; Fourier-feature fix **9.3%**.
- **Diagnostics:** the **wavenumber sweep** (naive ~3% at n=1 rising to ~100% at n>=2, Fourier stays low) and
  the **radial spectral energy** (the high-|k| band the naive lane cannot reach).
- **Training:** the naive tanh lane never leaves **~100%** relative-L2 at ANY checkpoint (spectral bias is a
  training-time pathology, not a capacity limit), while Fourier converges to **~9%**.

The results.ts verdict does mention "watch Training" and quotes 121%/9.3%, so the short content partly covers
this. But the deep Context's viz section only lists heatmap, line-cuts, and Live: it never points the reader
to the Compare/Diagnostics/Training views that hold the case's strongest evidence (the sweep and the
never-converging naive training curve).

### G4. (minor) The app never surfaces the primary anchor number 10.26%

The doc's validation anchor is `analytic`, and the metric it scores is relative-L2 vs analytic u* = **0.1026
(10.26%)**. The in-app results.ts verdict reports "Global error 9.3%", which is the doc's *Compare-view*
figure (Fourier vs the FDM standard), a different comparison. Both are real doc numbers, but the app presents
the vs-standard figure as the global error and never shows the 10.26% vs-analytic headline that
`validation_anchor = analytic` actually scores. Worth reconciling so the two numbers are not silently
conflated.

---

## Concrete proposed enrichments (grounded in doc quotes; no invented numbers)

**E1 (fix C1) - `HelmholtzContext.tsx`, "What the benchmark shows" paragraph (EN line ~180, ES line ~85).**
Replace "(three per side, nine by nine in all)" / "(tres por lado, nueve por nueve en total)" with the
correct geometry. EN: "three full wavelengths per side, a 6x6 checkerboard of 36 alternating lobes."
ES: "tres longitudes de onda completas por lado, un tablero de 6x6 con 36 lóbulos alternados."

**E2 (fill G1) - `HelmholtzContext.tsx`, add a short "Result and honesty" paragraph after the Fourier-map
paragraph (before Scope), transcribed from the doc.** Suggested EN:
"Measured at seed 42, the Fourier-feature PINN reaches a relative-L2 of 0.1026 (~10%) against the analytic
field, with a max absolute error of 0.158; the ONNX export matches the trained net to 4.77e-07 (max abs) and
runs live at 340 KB / 18.5 ms. The ~10% is honestly CPU-limited: the Fourier map lifts the spectral-bias
plateau that would leave a vanilla MLP an order of magnitude worse, but at k0=6π on the CPU lane the standing
pattern resolves to ~10%, not to 1e-3. GPU training plus frequency annealing would tighten it further; this
is not dressed up." (Mirror in ES.)

**E3 (fill G2) - `HelmholtzContext.tsx`, extend the Fourier-map paragraph.** Add, per the doc:
"The 256-dim embedding feeds a 4x128 tanh FNN, optimized Adam (25 000 steps) then L-BFGS; collocation uses
~12 points per wavelength per axis." (Mirror in ES.)

**E4 (fill G3) - `HelmholtzContext.tsx`, "How to read and use the viz" paragraph.** Add a sentence steering
to the ladder views, transcribed from the doc: "The Compare view puts the classical finite-difference solve,
the naive plain-tanh PINN (120.8% relative-L2), and the Fourier-feature PINN (9.3%) on one grid; Diagnostics
shows the wavenumber sweep (naive ~3% at n=1 climbing to ~100% at n>=2 while Fourier stays low) and the
radial spectral energy; Training replays real checkpoints where the naive lane never leaves ~100% and Fourier
converges to ~9%." (Mirror in ES.)

**E5 (fix C2) - `results.ts`, assumptions_en/es.** Change `"perfectly reflecting walls"` /
`"paredes perfectamente reflectantes"` to `"fixed (clamped, pressure-release) walls: u = 0"` /
`"paredes fijas (empotradas, de presión nula): u = 0"` so it matches the doc's Dirichlet framing instead of
implying a rigid Neumann wall.

**E6 (reconcile G4) - `results.ts`, verdict_en/es.** Optionally distinguish the two doc figures so they are
not conflated: keep "9.3% vs the classical standard" but add "(~10% relative-L2 against the exact analytic
field, the case's validation anchor)". This surfaces the 0.1026 headline the app currently never shows.

---

## What is already coherent (no change needed)

- Governing equation, MMS solution, source f = k0² sin·sin, and the residual derivation in the Context all
  match the doc and are mathematically correct.
- Soft Dirichlet BC "u = 0 on ∂Ω (soft, 100x weight)" (constraints.ts) matches `loss_weights=[1,100]`.
- Wavenumber "k0 = 6π (n=3)" and "~3 wavelengths across the box" match the doc.
- Anchor "analytic MMS + classical FDM standard" (constraints.ts) matches the doc.
- Receiver reading 0.990 "within 1% of the exact 1.000" (results.ts) is consistent: u*(0.25,0.25) =
  sin(1.5π)² = 1.000 exactly, so 0.990 is a valid ~1% local reading (an app-level readout, not a doc number,
  and not contradicting anything).
- Spectral-bias framing and the "watch it in Training" pointer (scenarios.ts, results.ts) match the doc's
  Training-view narrative.
