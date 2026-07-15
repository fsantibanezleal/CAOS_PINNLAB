# Content audit: `mine-flotation-kinetics`

Date: 2026-07-15
Auditor: content-coherence pass (in-app vs authoritative doc)

Authoritative doc: `docs/cases/mine-flotation-kinetics.md`
In-app files audited:
- Deep context: `frontend/src/content/cases/FlotationContext.tsx` (EN + ES)
- Scenario: `frontend/src/content/scenarios.ts` (`mine-flotation-kinetics`)
- Results: `frontend/src/content/results.ts` (`mine-flotation-kinetics`)
- Constraints: `frontend/src/content/constraints.ts` (`mine-flotation-kinetics`)

---

## Verdict

**Coherent, but the deep Context is quantitatively hollow at the one place the doc is richest: the Result.**

No contradictions were found. Every method name, mechanism, domain, IC, ansatz, honesty caveat and
short-content number in the app agrees with the doc, and the two design read-offs in `results.ts`
(k=0.5 -> 39% at end of cell / never reaches t90 in-window; k=5 -> t90 = 0.46) were re-derived from the
closed form and match exactly. The scenario/results/constraints layer is substantive and honest.

The real defect is a **depth gap in the Context**: `FlotationContext.tsx` presents the problem, the
governing equation, the hard-IC ansatz and an honest scope, but it **never states a single measured
result**. The doc's entire "Result (measured, seed 42)" table (relative-L2 **7.6e-4 / 0.076 %**, max abs
**2.22e-3**, ONNX parity **7.15e-7**, live lane **19.4 KB / 0.77 ms**) is absent from the deep context,
and the network architecture / optimizer recipe is absent too. The app tells the reader what the PINN is
asked to do and never how well it actually did it. Separately, the `results.ts` verdict under-sells the
precision by citing a loose "within 2%" when the doc's measured error is an order of magnitude tighter.

Severity: **2 (real gap)** — no contradiction, not filler, but the deep context omits the doc's measured
outcome, which is load-bearing content.

---

## Coherence check (contradictions)

**None found.** Cross-checks performed:

| Claim (in-app) | Doc | Status |
|---|---|---|
| `\dot C=-kC`, `C(k,0)=1`, `C*=e^{-kt}`, `R=1-C` (Context Formalization) | identical governing eq. + anchor (doc Problem, l.12) | agrees |
| Hard IC ansatz `C_θ = 1 + t·N_θ(k,t)`, collapses to 1 at t=0 (Context) | `y -> 1 + t·N(k,t)`, "IC never a soft penalty", `num_boundary=0` (doc Method) | agrees |
| Domain `k∈[0.5,5]`, `t∈[0,1]`, `81×81` grid (Context + constraints) | `k∈[0.5,5]`, `t∈[0,1]`, `81×81` (doc Problem) | agrees |
| `k` as second network input, "whole family in one net" (Context/scenarios/constraints) | parametric input, "solution operator over the whole k-family" (doc Method) | agrees |
| Anchor `exact C* = e^{-kt}` (constraints) | validation anchor = analytic, exact (doc Result) | agrees |
| Illustrative-synthetic, not fit to plant/lab, public datasets are 0-D with no rate-constant axis (Context Scope) | `real_or_synthetic = synthetic-illustrative`; Kaggle iron-ore is 0-D, no k axis (doc Honesty) | agrees |
| Out of scope: distributed-k spectra, entrainment, second-order, froth dynamics (Context) | "real circuits add size-by-size k distributions" (doc verdict echo); consistent framing | agrees |
| k=0.5 -> 39% recovery, never reaches t90 in-window (results answer) | closed form: R(0.5,1)=0.3935; t90(0.5)=ln10/0.5=4.61 ≫ 1 | agrees (re-derived) |
| k=5 -> t90 = 0.46 (results answer) | closed form: t90(5)=ln10/5=0.4605 | agrees (re-derived) |
| `t90 = ln10/k`, `R = 1-e^{-kt}` (results verdict) | closed form of the doc's anchor | agrees |

Soft note (not a contradiction): the `results.ts` verdict says read-off values "match the exact kinetics
closed form **within 2%**". The doc's measured net error is relative-L2 **0.076 %** (max abs 2.22e-3),
i.e. an order of magnitude tighter. "Within 2%" is defensible for a grid/interpolated read-off but it
under-represents the actual precision the doc reports and never surfaces the real number. Treated below as
a depth gap, not a contradiction.

---

## Depth gaps (real doc content the app omits)

### G1 (primary) — Context states no measured result at all
`FlotationContext.tsx` ends its "Formalization" at the analytic anchor and never reports the trained net's
accuracy. The doc's whole Result section is missing from the deep context:
- relative-L2 vs analytic `C*=e^{-kt}` = **7.6e-4 (0.076 %)**
- max absolute error = **2.22e-3**
- clears the case's own band (`< 5e-3`) **by an order of magnitude**, so **no accuracy caveat is needed**
- ONNX parity (max abs) = **7.15e-7**; live lane **19.4 KB**, infer **0.77 ms**

A reader of the deep context cannot tell how faithfully the PINN reproduces the anchor. This is the single
biggest gap.

### G2 — Context omits the network architecture and optimizer recipe
The doc's Method gives: FNN `[2] -> [32]×3 -> [1]`, `tanh`, Glorot-normal init; **10 000 Adam @ lr=1e-3**
over **2000 domain + 200 initial** collocation points, then an **L-BFGS** polish; scored on **4000 test
points** via l2 relative error. The Context describes the residual and ansatz but names none of the
architecture/optimizer. Peer cases in this app surface the `Adam -> L-BFGS` recipe and the ansatz in their
context/constraints; this one does not.

### G3 — Live/ONNX justification is qualitative only
The Context says the parametric net "re-evaluates the full map in your browser (onnxruntime-web)" but never
gives the doc's numbers that justify the live lane: a single **19.4 KB** ONNX reproducing the trained net
to **7.15e-7** and inferring in **0.77 ms**. Those are exactly why "the App can sweep k interactively and
read C(k,t) and R(k,t) live" (doc Result).

### G4 — `results.ts` verdict under-cites the achieved precision
See soft note above: verdict cites "within 2%" while the doc's measured net error is **0.076 %** relative-L2
(max abs 2.22e-3). The design-read-off tolerance and the net error are two different things; the verdict can
keep the read-off statement but should also cite the net's real accuracy from the doc.

---

## Concrete proposed enrichments (faithful to the doc; no invented numbers)

### E1 — Add a "Result (measured)" paragraph to `FlotationContext.tsx` (both EN and ES)
Insert after the Formalization paragraph (after the `C_θ = 1 + t·N_θ(k,t)` sentence), before "Scope".

EN (grounded in doc l.34-47):
> **Result (measured, seed 42).** The trained parametric net matches the analytic anchor `C*=e^{-kt}` to a
> relative-L2 of **0.076 %** (7.6e-4), max absolute error **2.22e-3** — clearing this case's own band
> (< 5e-3) by an order of magnitude. This is a smooth, low-dimensional field with an exact analytic anchor,
> so the CPU lane resolves it essentially to optimizer precision and no accuracy caveat is needed. A single
> **19.4 KB** ONNX reproduces the net to **7.15e-7** and infers in **0.77 ms**, so the App sweeps `k`
> interactively and reads `C(k,t)` and `R=1-C` live.

ES (mirror):
> **Resultado (medido, semilla 42).** La red paramétrica entrenada calza con el ancla analítica `C*=e^{-kt}`
> a un L2 relativo de **0,076 %** (7,6e-4), error absoluto máximo **2,22e-3**: supera la banda propia del
> caso (< 5e-3) por un orden de magnitud. Es un campo suave y de baja dimensión con ancla analítica exacta,
> así que la corrida en CPU lo resuelve esencialmente a precisión de optimizador y no se necesita salvedad de
> exactitud. Un único ONNX de **19,4 KB** reproduce la red a **7,15e-7** e infiere en **0,77 ms**, así que la
> App barre `k` de forma interactiva y lee `C(k,t)` y `R=1-C` en vivo.

### E2 — Add the architecture + optimizer to `FlotationContext.tsx` (both langs)
Append one sentence to the Formalization paragraph (grounded in doc l.21, l.29-30):

EN:
> The net is a small FNN `[2] -> [32]×3 -> [1]` with `tanh` activations; because the IC is hard, the loss is
> PDE-residual-only (`num_boundary=0`), trained by 10 000 Adam steps (lr=1e-3) over 2000 domain + 200 initial
> collocation points, then an L-BFGS polish, and scored against the closed form on 4000 test points.

ES:
> La red es una FNN pequeña `[2] -> [32]×3 -> [1]` con activaciones `tanh`; como la CI es dura, la pérdida es
> solo el residual de la EDO (`num_boundary=0`), entrenada con 10 000 pasos Adam (lr=1e-3) sobre 2000 puntos
> de dominio + 200 iniciales, luego un pulido L-BFGS, y evaluada contra la forma cerrada en 4000 puntos de
> prueba.

### E3 — Tighten the `results.ts` verdict to cite the real net error (both langs)
File: `frontend/src/content/results.ts`, key `mine-flotation-kinetics`.
Current EN: `"The read-off values match the exact kinetics closed form within 2% ... the design chart is numerically sound."`
Proposed EN (keep the design read-off claim, add the doc's measured metric, doc l.38-44):
> "The read-off values match the exact kinetics closed form (recovery 1-e^(-kt), t90 = ln10/k), and the
> underlying net itself hits the analytic anchor to 0.076% relative-L2 (max abs 2.2e-3), clearing its band
> by an order of magnitude: the design chart is numerically sound. First-order kinetics is the standard first
> approximation: real circuits add size-by-size k distributions."

Proposed ES (mirror):
> "Los valores leídos calzan con la forma cerrada de la cinética (recuperación 1-e^(-kt), t90 = ln10/k), y la
> red misma alcanza el ancla analítica a 0,076% de L2 relativo (abs máx 2,2e-3), superando su banda por un
> orden de magnitud: la carta de diseño es numéricamente sólida. La cinética de primer orden es la primera
> aproximación estándar: los circuitos reales agregan distribuciones de k por tamaño."

### E4 (optional) — add the ONNX size/parity to `constraints.ts` anchor line
File: `frontend/src/content/constraints.ts`, key `mine-flotation-kinetics`. The `anchor` chip could read
`exact C* = e^(-kt) (net matches to 7.6e-4 rel-L2)` to carry the measured fit alongside the anchor, keeping
the chip honest about the achieved accuracy rather than only the target.

---

## Scope-clean confirmations (no action needed)
- Scenario `situation`/`measured`, results `calculates`/`assumptions`/`answer`, constraints all agree with
  the doc and each other. The honesty framing (synthetic-illustrative, not plant/lab-fit, distributed-k out
  of scope) is preserved consistently across doc, Context, and results verdict.
- No fabricated numbers were found in any in-app file; the Context's `kt=const` contour statement is a valid
  consequence of `C=e^{-kt}`, not an invention.
