# Content audit: poll-ocean-transport (in-app vs authoritative doc)

Date: 2026-07-15
Auditor: content-coherence sub-agent
Authoritative doc: `docs/cases/poll-ocean-transport.md`
In-app files reviewed:
- `frontend/src/content/cases/OceanTransportContext.tsx` (deep Context, EN + ES)
- `frontend/src/content/scenarios.ts` (situation/measured)
- `frontend/src/content/results.ts` (calculates/assumptions/answer/verdict)
- `frontend/src/content/constraints.ts` (ic/param/anchor chips)
- Cross-check source: `data-pipeline/pinnlab/cases/poll_ocean_transport.py` (exact params + solution)

## Verdict

**Coherent with the doc, no contradictions, but with one real depth gap.** The Context is genuinely
rich (governing equation, exact closed-form solution, center/variance/peak decomposition, Péclet,
domain/grid, an honest scope-and-assumptions block, and a viz-reading guide), and every physical
claim in Context/constraints/scenarios matches the doc and the pipeline (`v=(0.45,0.35)`, `D=0.01`,
`c*` Green's function, soft IC/BC, exact-not-manufactured anchor). No in-app number contradicts the
doc.

The one substantive weakness is the **measured-accuracy headline**: the in-app `results.ts` verdict
reports "**Under 1% field error**", which is the loose registry gate (`expected_band: relative-L2 <
1e-2`), NOT the doc's actual measured result of **relative-L2 <= 0.19% across all 6 snapshots (0.06%
at t=0, 0.19% at t=1)**. The claim is technically true (0.19% is under 1%) so it is not a
contradiction, but it understates the real measured precision by ~5x and hides the per-snapshot
result the doc treats as the headline. Two smaller depth gaps (training config; ONNX parity) round
out the list.

Severity: **2 (real gap)** — coherent, no contradictions, but a material understatement of the
measured result plus omitted doc detail worth surfacing.

## Contradictions (in-app claim vs doc)

None found. Every method name, mechanism, equation, coefficient, and error statement in the app is
consistent with `docs/cases/poll-ocean-transport.md` and the pipeline. The "under 1%" verdict is
listed below as a depth gap (an understatement), not a contradiction, because it does not conflict
with the doc's number — it merely fails to report it.

## Depth gaps (real doc content the app omits or dilutes)

### GAP 1 (primary) — measured accuracy is diluted to the loose gate, not the real number
- **In-app** (`results.ts`, `poll-ocean-transport` verdict_en):
  "Under 1% field error vs the exact drifting-spreading solution: arrival times are trustworthy..."
  (verdict_es: "Menos de 1% de error vs la solución exacta...").
- **Doc says** (Result table): "relative-L2 vs exact **<= 0.19 %** across all 6 snapshots
  (t=0 -> 0.06 %; t=1 -> 0.19 %)".
- **Why it matters:** the app cites the registry `expected_band` (`relative-L2 vs exact < 1e-2`),
  the pass/fail gate, in place of the actual measured headline. The measured result is ~5x tighter
  and grows monotonically with spread (0.06% -> 0.19%), which is itself a teaching point (error
  accumulates as the patch broadens). The app leaves the strongest, most honest number on the table.

### GAP 2 — the concrete PINN training recipe is absent from Context
- **In-app** (`OceanTransportContext.tsx`): describes soft, weighted IC/BC ("impuestas de forma
  blanda y ponderada" / "imposed softly and weighted") but gives no architecture, optimizer, or
  loss weights.
- **Doc says** (Method): "Net **[3,64,64,64,64,1] tanh (DeepXDE)**, **Adam (18000, lr 1e-3) ->
  L-BFGS**, **loss weights [1,10,50] for [pde,bc,ic]**." (These match the pipeline `CASE.train`.)
- **Why it matters:** the loss weights `[1,10,50]` are the mechanism behind "reported L2 is the true
  PINN error" — the BC/IC penalties are soft precisely so the interior is genuinely learned. Naming
  the 4x64 tanh net and the Adam->L-BFGS schedule adds real depth for near-zero cost and is already
  a fact on disk.

### GAP 3 — ONNX parity is never surfaced in-app
- **Doc says** (Result table): "ONNX parity (max abs) **4.8e-7**", lane **live** (one shared ONNX;
  Live = time scrubber).
- **In-app:** the Context/results mention the browser ONNX runtime ("onnxruntime-web") but never the
  parity figure. For a case whose whole point is a single shared ONNX scrubbed live, the 4.8e-7
  export-fidelity number is the evidence that the in-browser scrubber equals the trained net; it
  belongs somewhere user-visible.

### GAP 4 (minor) — the scenario checkpoint answer has no stated location
- **In-app** (`results.ts` answer_en): "The spill reaches the checkpoint at **t = 0.44** (first
  significant arrival), still rising to **0.077** at the window's end."
- **Observation:** these numbers are plausible and consistent with the exact solution
  (params `x0,y0=(0.25,0.30)`, `s0^2=0.008`, `v=(0.45,0.35)`, `D=0.01`; center at t=1 is (0.70,0.65),
  peak 0.286, so c=0.077 corresponds to a checkpoint ~0.27 downstream of the center). They are NOT in
  the doc (the doc has no checkpoint scenario), so they cannot be cross-verified against ground truth,
  and the app never states WHERE the checkpoint is. Not a contradiction; a transparency gap. Either
  disclose the checkpoint coordinate or add it to the doc so the answer is gradeable like the rest.

## Strengths (coherent, keep as-is)

- Governing equation and exact `c*` Green's function are transcribed verbatim and correctly in
  Context (`Equation` block matches the doc formula exactly).
- The center/variance/peak decomposition ("center moves with v; variance grows linearly; peak decays
  as s0^2/s^2 because mass is conserved") mirrors the doc's paragraph precisely.
- The honesty block is faithful: "illustrative-synthetic ... NOT fit to a real spill or a real
  ocean-current product ... Out of scope: rotating/time-varying currents..." matches the doc's
  Honesty section, including "the open boundary lets mass leave, as at a coast."
- The "reported L2 is the true PINN error" nuance (from soft IC/BC) is preserved in Context.
- constraints chips (`Gaussian patch released at (x0,y0)`, `current v=(0.45,0.35)`, `2-D Green's
  function (exact, not manufactured)`) all match doc + pipeline.

## Concrete proposed enrichments (faithful to the doc, no invented numbers)

### E1 — fix the accuracy headline in `results.ts` (addresses GAP 1)
Replace verdict_en for `poll-ocean-transport`:
> Current: "Under 1% field error vs the exact drifting-spreading solution: arrival times are
> trustworthy under the stated current. If the real current shifts, re-ask the network (time is a
> live input): that is the point of carrying the whole window in one model."
>
> Proposed: "Relative-L2 <= 0.19% vs the exact drifting-spreading solution across all six snapshots
> (0.06% at release, rising to 0.19% at t=1 as the patch broadens): arrival times are trustworthy
> under the stated current. If the real current shifts, re-ask the network (time is a live input):
> that is the point of carrying the whole window in one model."

verdict_es (mirror):
> "Relative-L2 <= 0.19% vs la solucion exacta de deriva y esparcimiento en las seis instantaneas
> (0.06% en el vertido, subiendo a 0.19% en t=1 al ensancharse el parche): los tiempos de llegada son
> confiables bajo la corriente declarada. Si la corriente real cambia, re-pregunta a la red (el
> tiempo es entrada viva): ese es el punto de llevar toda la ventana en un modelo."

(Numbers quoted directly from doc Result table: "relative-L2 vs exact <= 0.19 % across all 6
snapshots (t=0 -> 0.06 %; t=1 -> 0.19 %)".)

### E2 — add the training recipe to Context "Formalization" (addresses GAP 2)
In `OceanTransportContext.tsx`, extend the Formalization paragraph (both EN and ES) after "...the
reported L2 is the true PINN error." Append (EN):
> "Concretely: a 4x64 tanh network with (x,y,t) inputs (DeepXDE), trained by Adam (18000 steps,
> lr 1e-3) then polished with L-BFGS, with loss weights [1,10,50] on [PDE, BC, IC]: the heavier IC
> weight anchors the release without hard-clamping the interior."

ES:
> "En concreto: una red tanh de 4x64 con entradas (x,y,t) (DeepXDE), entrenada con Adam (18000 pasos,
> lr 1e-3) y pulida con L-BFGS, con pesos de perdida [1,10,50] sobre [EDP, BC, IC]: el peso mayor en
> la IC ancla el vertido sin fijar duramente el interior."

(All values from doc Method + pipeline `CASE.train`: net `[3,64,64,64,64,1]`, Adam 18000 lr 1e-3,
L-BFGS, loss_weights `[1,10,50]`.)

### E3 — surface ONNX parity (addresses GAP 3)
In the Context viz-reading paragraph, where it already says the scrubber runs "live in your browser
(onnxruntime-web)", append (EN): "the browser ONNX matches the trained network to 4.8e-7 (max abs),
so the live scrubber is the same solver, not an approximation." ES mirror with "4.8e-7 (max abs)".
(From doc Result table "ONNX parity (max abs) 4.8e-7".) Alternatively expose it as a metric chip in
the results panel.

### E4 — disclose the checkpoint (addresses GAP 4)
Either (a) add the checkpoint coordinate to `results.ts` answer text (e.g. "at the coastal intake at
(xc, yc)") using the exact coordinate the scenario probe uses, or (b) add a one-line checkpoint
definition to `docs/cases/poll-ocean-transport.md` so t=0.44 / c=0.077 become gradeable against the
exact solution like every other number in the case. Do not invent the coordinate: read it from the
scenario/probe definition that produced 0.44 and 0.077.

## Files to edit
- `frontend/src/content/results.ts` (E1, and optionally E4)
- `frontend/src/content/cases/OceanTransportContext.tsx` (E2, E3)
- `docs/cases/poll-ocean-transport.md` (optional E4b: add checkpoint definition)
