# Content audit — bench-heat1d (1D transient heat / diffusion, parametric diffusivity)

Date: 2026-07-15
Ground truth: `docs/cases/bench-heat1d.md`
In-app surfaces audited:
- Deep context: `frontend/src/content/cases/Heat1dContext.tsx` (EN + ES)
- Scenario: `frontend/src/content/scenarios.ts` (`"bench-heat1d"`)
- Results: `frontend/src/content/results.ts` (`"bench-heat1d"`)
- Constraints: `frontend/src/content/constraints.ts` (`"bench-heat1d"`)
- Registry mapping: `frontend/src/content/cases/registry.tsx` (`"bench-heat1d": Heat1dContext`)

## Verdict

The deep Context is **genuinely rich and almost entirely coherent** with the doc: it carries the
governing parabolic PDE, the closed-form anchor `u* = e^(-απ²t) sin(πx)`, the derivation of the decay
constant `απ²`, the exact hard-constraint output transform `û = t·x(1-x)·N + sin(πx)` (verbatim from the
doc), the correct domain/grid (`161×101`, `x,t ∈ [0,1]`), the α range `[0.1,1.0]`, and an honest
scope/out-of-scope section. constraints.ts matches the doc on all four pins. scenarios.ts and results.ts
are substantive, not filler: results.ts derives the cooling half-life `ln2/(απ²)` and checks it
(0.70 vs 0.702, 0.07 vs 0.070) — I re-derived these and they are correct.

But this is NOT a clean pass. There are **two real contradictions** and **several depth omissions**:

1. results.ts reports the field error as **0.2%**, which is worse than and disagrees with the doc's
   measured headline metric **relative-L2 ≤ 0.15%** across all 6 variants.
2. The Context and results.ts **contradict each other on the heatmap's time axis orientation** (Context:
   hot start at TOP, time increases downward; results.ts: hot start at BOTTOM, time increases upward).
   At least one misdescribes the actual rendered viz.

`coherentWithDoc = false`. Severity 3 (genuine contradictions), though both are modest in magnitude and
concentrated in results.ts; the Context prose itself is strong.

---

## Contradictions (inApp vs docSays)

### C1 — Field error number: app says 0.2%, doc measured ≤ 0.15%
- **inApp** (`results.ts`, verdict_en/es): "overall field error 0.2% vs exact" / "error global del campo 0.2% vs exacta".
- **docSays** (Result table): "relative-L2 vs analytic **≤ 0.15 %** across all 6 variants."
- The app's 0.2% is *higher* than the doc's stated worst-case bound, so it overstates the error. If 0.2%
  came from a different aggregate metric, that is undisclosed and still reads as a discrepancy against the
  doc's headline number. The doc's ≤0.15% (relative-L2, seed 42) is the number to cite.
- **Fix:** in `results.ts` change "overall field error 0.2% vs exact" to **"overall field error ≤ 0.15%
  (relative-L2) vs exact"** (ES: "error global del campo ≤ 0.15% (L2 relativo) vs exacta").

### C2 — Heatmap time-axis orientation contradicts between Context and results.ts
- **inApp A** (`Heat1dContext.tsx`, EN): "The heatmap shows u(x,t) ... a bright band **at the top (small t)**
  fading **downward (large t)**." ES: "una banda brillante **arriba (t pequeño)** ... hacia **abajo (t grande)**."
  → time increases downward, hot start at top.
- **inApp B** (`results.ts`, calculates_en): "vertical axis = time flowing **upward** ... The bright band
  **at the bottom is the hot start**; it fades **upward** as the bar cools." → time increases upward, hot start at bottom.
- **docSays:** the doc does not pin the axis orientation, so the doc cannot arbitrate — but the two in-app
  surfaces are mutually exclusive, so at least one is wrong about the rendered field. This will confuse a
  reader who opens both panels.
- **Fix:** screenshot-verify the actual heatmap orientation, then make both files agree. (Per the repo's
  own web-viz lesson: screenshot in both themes; do not trust the prose over the pixels.) Align the losing
  surface to the rendered truth; do not guess.

---

## Depth gaps (real doc content the app omits)

### G1 — ONNX parity number (~5e-7) is nowhere in the app
- Doc Result: "ONNX parity (max abs) **~5e-7**." The Context mentions "the hard constraint survives the
  ONNX export, which is why the Live tab re-evaluates the exact field" but gives no parity figure, and
  results.ts omits it entirely.
- **Enrich:** append to the Context's ONNX sentence (`Heat1dContext.tsx`, Scope paragraph): EN "...which is
  why the Live tab re-evaluates the field faithfully (ONNX parity ~5e-7 max abs vs the trained net)." ES
  equivalently "...por eso el tab Live re-evalúa el campo fielmente (paridad ONNX ~5e-7 máx. abs. vs la red entrenada)."

### G2 — Network architecture and optimizer schedule are absent
- Doc Method: "Net **[3,48,48,48,48,1] tanh (DeepXDE), Adam → L-BFGS**." The Context describes the method
  and hard constraint but never states the concrete training recipe.
- **Enrich:** add one sentence to the Formalization section of `Heat1dContext.tsx`: EN "The raw net
  `N_θ(x,t,α)` is a `[3,48,48,48,48,1]` tanh MLP (DeepXDE), trained Adam then polished with L-BFGS on the
  bare PDE residual." ES "La red cruda `N_θ(x,t,α)` es un MLP tanh `[3,48,48,48,48,1]` (DeepXDE), entrenado
  con Adam y pulido con L-BFGS sobre el residual puro de la EDP."

### G3 — Doc's headline accuracy metric (≤0.15% relative-L2, seed 42, 6 variants) is not cited as such
- Beyond C1, the app never states the doc's actual validation result the way the doc frames it: relative-L2
  ≤ 0.15% across the six variants `α = 0.1, 0.2, 0.4, 0.6, 0.8, 1.0` at seed 42. The Context lists the six α
  values but attaches no measured accuracy to them.
- **Enrich:** in the Context "What each variant shows" paragraph, close with EN "Across all six variants the
  reconstruction stays within **0.15% relative-L2** of the closed form." ES "En las seis variantes la
  reconstrucción se mantiene dentro del **0.15% de L2 relativo** respecto de la forma cerrada." (This also
  resolves C1 by making the app cite the doc's own number.)

### G4 — Honesty caveat is under-reinforced in prose (badge exists, prose does not)
- The provenance badge `synthetic` IS surfaced on the case card (via `lib/contract.ts` `DATA_LABELS`), so
  provenance is not fully hidden. But the doc's Honesty note is stronger than a one-word badge: "The truth
  is the **closed-form separable solution** (exact for every α), not a measured dataset — a **textbook
  benchmark** chosen to validate the method. The network only ever sees the PDE residual." Meanwhile the
  scenario dresses this as a real steel-plant cooling problem, which could lead a reader to think there is
  measured data. The Context prose never states "this is a textbook benchmark; the anchor is analytic, not
  data."
- **Enrich:** add to the Scope & assumptions paragraph of `Heat1dContext.tsx`: EN "This is a **textbook
  benchmark**: the ground truth is the exact separable solution, not a measured dataset, and the network is
  graded against that closed form." ES "Es un **benchmark de libro de texto**: la verdad de referencia es la
  solución separable exacta, no un dataset medido, y la red se califica contra esa forma cerrada."

---

## Minor notes (severity ≤ 1, optional)

- **Scenario IC realism (`scenarios.ts`).** The quench framing ("a quenched steel bar leaves the bath") is
  fine as an illustrative stake, but the actual IC is a single fundamental mode `sin(πx)` (zero at both
  surfaces), which is not what a bar "leaving a bath" physically looks like. The scenario says only "the
  initial heat profile is known" and never discloses the specific sinusoidal single-mode IC. Not a
  contradiction (scenarios are declared illustrative), but a half-sentence noting the profile is a single
  sine mode would keep the framing honest. Low priority.
- The Context's out-of-scope line references "the reactive-transport and diffusion-barrier cases" (ES:
  "transporte reactivo y barrera de difusión"). Not verified against the live case registry in this audit;
  worth a quick check that those sibling cases exist so the cross-reference is not dangling.

## Things that are CORRECT (verified, not to be touched)

- Governing PDE `u_t = α u_xx`, IC `u(x,0)=sin(πx)`, Dirichlet BCs, domain, `161×101` grid, α ∈ [0.1,1.0]:
  all match the doc.
- Closed form `u* = e^(-απ²t) sin(πx)` and decay-constant derivation `απ²`: match the doc.
- Hard-constraint transform `û = t·x(1-x)·N + sin(πx)` and the t=0 / x=0,1 vanishing argument: verbatim-faithful to the doc.
- constraints.ts pins (IC hard, BC hard, param α∈[0.1,1], anchor `e^(-απ²t) sin(πx)`): all correct.
- results.ts cooling half-lives `ln2/(απ²)`: re-derived independently — 0.7023 (α=0.1) and 0.0702 (α=1.0),
  matching the app's "0.70 vs 0.702 ... 0.07 vs 0.070". Correct, and deeper than the doc (the doc does not
  give half-lives).
