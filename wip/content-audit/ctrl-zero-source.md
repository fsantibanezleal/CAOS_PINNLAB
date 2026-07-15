# Content audit: ctrl-zero-source (in-app vs authoritative doc)

- Case: `ctrl-zero-source` — "MMS verification family, parametric source amplitude a (contains the degenerate control)"
- Authoritative doc: `docs/cases/ctrl-zero-source.md`
- In-app deep context: `frontend/src/content/cases/ZeroSourceContext.tsx`
- In-app short content: `frontend/src/content/scenarios.ts`, `frontend/src/content/results.ts`, `frontend/src/content/constraints.ts`
- Date: 2026-07-15

## Verdict

**Severity 3 (contradiction + hollow short content).** The deep Context component is genuinely
rich and fully coherent with the doc (correct governing equations, manufactured solution, source
derivation, hard-constraint ansatz, honest scope). `constraints.ts` is also already reconciled to
the new parametric framing. **But `scenarios.ts` and `results.ts` are STALE**: they describe the
OLD degenerate-only null control ("the only physical answer is exactly zero everywhere"), which the
doc explicitly reframed into a parametric MMS family ("This generalises the old degenerate-only
control into an honest parametric family whose a=0 limit *is* the control"). This produces a real
contradiction against both the doc AND the app's own Live viz (which sweeps a and renders a two-mode
structured field at a>0). On top of that, every surface OMITS the doc's headline measured numbers
(relative-L2 <= 0.15% for a>=0.2, ONNX parity 4.2e-7), and the one number the app does cite
(RMS 5e-4 / max 0.001) does not match the doc's stated a=0 field-norm figure (2.1%).

- coherentWithDoc: **false** (scenarios + results contradict the doc's reframing)
- contextDepth: **rich** (ZeroSourceContext is theory + equation + honest scope, coherent)
- scenarioResultsDepth: **thin** (null-only framing, omits the family and the real numbers, misstates the answer)

## Contradictions (inApp vs docSays)

### 1. results.ts `calculates_en` claims the answer is zero EVERYWHERE
- **inApp** (`results.ts`, `ctrl-zero-source.calculates_en`): "The field SHOULD be nothing: with
  zero load and zero boundary the only physical answer is exactly zero everywhere. This case exists
  to prove the machinery does not invent structure."
- **docSays**: the case is a parametric family, not a null-only problem. "at a=1 a two-mode field
  (a dominant fundamental lobe + a finer second-mode ripple)"; the field is zero ONLY at the a=0
  limit ("**at a=0 this is the archetype's degenerate negative control**"). The Live tab "sweeps a
  and the field fades to flat zero" — i.e. it is non-zero for a>0. The app's own Context correctly
  says "up to a=1, where the two-mode structure appears."
- Impact: the Results panel tells the user the answer is "exactly zero everywhere" while the Live
  viz on the same case renders structured lobes. Directly false for a>0.

### 2. scenarios.ts `measured_en` describes only the null problem
- **inApp** (`scenarios.ts`, `ctrl-zero-source.measured_en`): "Zero source, zero boundary: the only
  solution is u = 0. The machinery must find it (RMS 5e-4, computed)."
- **docSays**: the validation anchor is "the manufactured u*(x,y;a)" over "Six variants
  (a=0,0.2,0.4,0.6,0.8,1.0)"; only the a=0 member is zero-source. Measured relative-L2 is
  "<= 0.15% for a>=0.2". The "only solution is u=0" statement is true only at a=0.

### 3. Number mismatch: a=0 field magnitude (5e-4 / 0.001 vs 2.1%)
- **inApp**: `scenarios.ts` says "RMS 5e-4, computed"; `results.ts` `answer_en` says "typical value
  0.0005, largest anywhere 0.001".
- **docSays** (Result table, measured seed 42): "a=0 -> ||pred|| = 2.1% (the degenerate control:
  u*≡0, so the metric is the field norm — essentially flat zero)".
- Note: these may be different metrics (absolute RMS field value vs a normalized field-norm
  percentage), so this is a reconciliation flag rather than a certain arithmetic error. Either way
  the app cites a number (5e-4 / 0.001) that appears nowhere in the authoritative Result, and omits
  the doc's stated 2.1%. One must be made to match the committed artifact; the app should not present
  a different a=0 figure than the doc's measured Result.

### 4. results.ts `assumptions_en` lists only "the null problem"
- **inApp** (`results.ts`, `ctrl-zero-source.assumptions_en`): ["zero source, zero boundary: the null problem"].
- **docSays** / **constraints.ts** (which IS already correct): the assumptions are a hard zero
  Dirichlet boundary (exact by construction), a two-mode manufactured solution u* = a·g with source
  f = -∇²u*, and amplitude a in [0,1] as a network input whose a=0 limit is the degenerate control.
- Impact: internal incoherence — `constraints.ts` and `ZeroSourceContext.tsx` describe the MMS
  family; `results.ts` assumptions and `scenarios.ts` still describe the pre-reframe null case.

## Depth gaps (real doc content the app omits)

- **Measured accuracy for the structured members**: "relative-L2 vs u* <= 0.15% for a>=0.2". This is
  the actual verification result of the case and it appears on NO in-app surface (not scenarios, not
  results, not even the deep Context, which says only "the error stays tiny"). This is the headline
  number and should be quoted.
- **ONNX parity (max abs) = 4.2e-7**: the browser-vs-trained-net fidelity figure. Omitted everywhere.
  Relevant because the Live tab re-evaluates in-browser via onnxruntime-web.
- **The six variants and seed**: "a=0,0.2,0.4,0.6,0.8,1.0", "measured, seed 42". Short content gives
  no sense of the sweep grid; the Context describes the sweep only qualitatively.
- **The a=0 metric caveat**: the doc explains WHY a=0 uses a different metric ("u*≡0, so the metric
  is the field norm"). The Context describes the collapse to ||u_hat|| but never gives the 2.1%; the
  short content gives an unexplained 5e-4.
- **Net / training detail** (lower priority, optional for depth): "[3,48,48,48,48,1] tanh (DeepXDE),
  Hypercube (x,y,a), Adam (12000, lr 1e-3) -> L-BFGS, anchor via solution=analytic". The Context
  describes the method qualitatively but omits these concrete specifics.

## Concrete proposed enrichments (grounded in doc quotes; faithful, no invented numbers)

Reconcile the RMS-5e-4 vs 2.1% discrepancy FIRST against the committed artifact; the text below uses
the doc's stated figures. If the artifact confirms 5e-4 is the true a=0 field RMS, keep it but also
add the doc's 2.1% field-norm figure and label each metric — do not silently drop the doc's number.

### File: `frontend/src/content/results.ts` (entry `ctrl-zero-source`)

- `calculates_en` -> reframe from "zero everywhere" to the family:
  > "The field is a manufactured Poisson solution swept by an amplitude a: at a=0 the true answer is
  > exactly nothing (the degenerate control, proving the machinery invents no structure); as a rises
  > a two-mode field switches on: a dominant central lobe with a finer ripple superimposed. Both axes
  > are position, color = u."
- `assumptions_en` -> match `constraints.ts` and the doc:
  > ["hard zero boundary, exact by construction (no BC loss)", "two-mode manufactured solution
  > u* = a·g, imposed source f = -∇²u*", "amplitude a in [0,1] is a network input; a=0 is the
  > degenerate null control"]
- `answer_en` -> add the real measured numbers from the Result table:
  > "Across a = 0 to 1: relative-L2 vs the exact u* stays <= 0.15% for a >= 0.2; at a=0 the field
  > collapses to essentially flat zero (field norm 2.1%, no invented structure). Browser ONNX matches
  > the trained net to 4.2e-7."
- `verdict_en` -> keep the null-test point but promote it to the family verification the doc states:
  > "MMS verification passed across the amplitude family, and its a=0 limit reproduces the mandatory
  > degenerate null control: an estimator that fabricates signal from nothing cannot be trusted, and
  > this one does not. Exact-by-construction error; synthetic, not fit to data."
- Mirror all four into the `_es` fields.

### File: `frontend/src/content/scenarios.ts` (entry `ctrl-zero-source`)

- `measured_en` -> replace the null-only line with the family + real anchor:
  > "The truth is the exact manufactured solution u*(x,y;a) over six amplitudes (a = 0 to 1). Only
  > a=0 is the zero-source null control (field essentially flat zero); for a >= 0.2 the network
  > recovers the two-mode field to relative-L2 <= 0.15% (seed 42)."
- Consider adjusting `situation_en` so the null-test hook does not imply the WHOLE case is a null
  test: e.g. append "and this null test is the a=0 limit of a manufactured Poisson family that
  verifies the solver across a range of loads." (Optional; the current hook is fine narratively but
  reinforces the null-only misread when the results/measured fields also stay null-only.)
- Mirror into `_es`.

### File: `frontend/src/content/cases/ZeroSourceContext.tsx` (deep Context)

Context is rich and coherent; the only gap is that it never states the measured numbers. In the
"What each variant shows" paragraph, replace the qualitative "the error stays tiny across the whole
family" with the concrete doc figures:
  > "Each variant reports its relative-L2 against the exact manufactured solution: measured (seed 42)
  > it stays <= 0.15% for a >= 0.2, and at a=0 the metric collapses to the field norm (||pred|| =
  > 2.1%, essentially flat zero). The in-browser ONNX matches the trained network to 4.2e-7."
Mirror into the `es` branch. Optionally add the net/training spec ([3,48,48,48,48,1] tanh, DeepXDE,
Adam 12000 lr 1e-3 -> L-BFGS) to the Method section for parity with the doc.

### File: `frontend/src/content/constraints.ts` (entry `ctrl-zero-source`)

No change needed. Already coherent with the doc: bc "u = 0 on the boundary (hard)", param
"amplitude a (a=0 is the degenerate control: u ≡ 0)", anchor "manufactured u* (any a)".
