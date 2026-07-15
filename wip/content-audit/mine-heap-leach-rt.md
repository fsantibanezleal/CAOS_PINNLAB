# Content audit: mine-heap-leach-rt (in-app vs authoritative doc)

Date: 2026-07-15
Doc (ground truth): `docs/cases/mine-heap-leach-rt.md`
In-app sources audited:
- `frontend/src/content/cases/HeapLeachContext.tsx` (deep Context, EN+ES)
- `frontend/src/content/scenarios.ts` (short "situation/measured")
- `frontend/src/content/results.ts` (short "calculates/assumptions/answer/verdict")
- `frontend/src/content/constraints.ts` (chips)

## Verdict

**Not coherent. Severity 3.** The deep `HeapLeachContext.tsx` is genuinely rich and faithful to the
doc (theory, governing PDE, the exact MMS pair with the `+1.5` offset, term-by-term residual derivation,
the "why time not Peclet" argument, and an honest scope section that matches the doc's Honesty section).
But the SHORT content in `scenarios.ts` and `results.ts` invents a **percolation "breakthrough at the
base at t = 0.60"** narrative that the doc never states and that CONTRADICTS the actual manufactured
field, and it reports a **"0.4% field error"** that is 20-40x worse than the doc's measured result
(cA rel-L2 <= 1e-4, cB <= 2e-4). The short content is well-written prose bolted onto a fabricated
mechanism and a wrong number. The Context itself even says the opposite ("relax toward the +1.5
baseline"), so scenarios/results contradict the app's own deep panel as well as the doc.

Context depth: rich. Scenario/results depth: adequate word-count, but incoherent (fabricated mechanism).

---

## Contradictions (inApp vs docSays)

### C1 - Fabricated "breakthrough at the base at t = 0.60" (severity 3)

- **inApp** (`results.ts` -> `mine-heap-leach-rt.answer_en`): "The reagent front breaks through at the
  base at t = 0.60 (half of its total change there): before that, irrigation is filling the heap, not
  leaching the bottom." (ES mirror: "El frente de reactivo rompe en la base en t = 0.60...").
- **inApp** (`results.ts` -> `calculates_en`): "The moment the base lights up is breakthrough: when
  leaching actually starts paying."
- **inApp** (`scenarios.ts` -> `situation_en`): "The schedule hangs on the breakthrough time at the base."
  and `measured_en`: "The 2-D reactive-transport field over time gives the breakthrough curve at any depth."
- **docSays**: The plotted primary species is `c_A^* = e^{-t} sin(pi x) cos(pi z) + 1.5`. This is a
  standing sine-cosine pattern whose amplitude decays exponentially from t=0; both species "relax toward
  the +1.5 baseline". At the base (z=1, z points down), `cos(pi) = -1`, so `c_A(base) = 1.5 - e^{-t} sin(pi x)`:
  it is NONZERO and at MAXIMAL contrast at t=0, then fades toward 1.5. There is no front percolating down and
  arriving at the base; the base never "lights up" from nothing. The doc Result section reports ONLY per-snapshot
  L2 errors and contains no breakthrough time. The doc's Honesty section is explicit that the single bimolecular
  `k_f c_A c_B` term is "a deliberate, well-posed teaching simplification" and not a plant percolation model.
- Cross-check inside the app: `HeapLeachContext.tsx` itself states the honest behavior ("t=0 is the manufactured
  initial state (maximal contrast between species) ... both relax toward the +1.5 baseline with c_A nearly flat").
  So the "breakthrough" story in scenarios/results contradicts BOTH the doc and the app's own Context.
- Note on the number: "half of its total change" does not yield t=0.60 for c_A at the base either (relaxation to
  the 1.5 asymptote reaches half-change at t = ln2 ~ 0.69; relative to t in [0,1] it is ~0.38). t=0.60 is simply
  a valid snapshot label, not a computed breakthrough.

### C2 - Wrong error magnitude: "0.4%" vs measured <= 1e-4 / <= 2e-4 (severity 3)

- **inApp** (`results.ts` -> `verdict_en`): "Field error 0.4% vs the reference for this closure: the
  breakthrough timing is sound within the model."
- **docSays** (Result, measured, seed 42): c_A relative-L2 vs MMS analytic "<= 1e-4 across all 6 snapshots";
  c_B relative-L2 "<= 2e-4 across all 6 snapshots"; and "Both species reach relative-L2 below 2e-4 at every
  snapshot, comfortably inside the case's `< 2e-2 per species` target band, so this is a genuinely
  well-converged result, not a CPU-limited compromise." ONNX parity max abs = 1.19e-06.
- 0.4% (= 4e-3) is 20-40x worse than the doc's actually measured errors (1e-4 / 2e-4). The value appears to be
  a generic filler copied across mining cases (`mine-comminution-pbe` and `mine-thickener-settling` also say
  "0.4%" / "sub-percent"). The app is under-selling a genuinely excellent, exactly-anchored result and quoting a
  number the doc does not support.

### C3 - Imprecise assumptions vs the actual model (severity 1-2)

- **inApp** (`results.ts` -> `assumptions_en`): ["constant irrigation at the top", "homogeneous heap (one
  permeability, one reaction rate)", "two-species reaction-transport closure"].
- **docSays**: The boundary/initial conditions are SOFT Dirichlet set to `c^*` on the whole boundary and IC
  (MMS-derived), not a "constant irrigation at the top". The transport is a fixed downward Darcy velocity
  `v=(0,1)` with isotropic dispersion `D=0.05` and bimolecular rate `k_f=1` - the doc never parameterizes it by a
  "permeability". "one reaction rate" (k_f=1) is fine; "one permeability" and "constant irrigation at the top" are
  loose paraphrases that reinforce the incorrect percolation-breakthrough framing.

---

## Depth gaps (real doc content the Context/short-content omits)

The Context is theory-rich but **result-number-thin**: it explains the method and MMS exactly, yet never quotes a
single measured value. It should carry the doc's Result numbers.

- G1 - **Measured result numbers.** The Context says only "el L2 reportado por especie es el error real del PINN"
  but never states it. Doc: c_A rel-L2 <= 1e-4, c_B <= 2e-4 across all 6 snapshots, target band `< 2e-2 per
  species`, ONNX parity 1.19e-06. None of these appear in any in-app surface (and results.ts states a wrong
  number instead, see C2).
- G2 - **Architecture.** Doc: single shared FNN `[3] -> [40]x4 -> [2]` tanh, one body predicting both c_A and c_B
  so the `k_f c_A c_B` coupling is differentiated through a single autodiff graph. Context says "una sola red /
  single network" but omits the concrete `[3]->[40]x4->[2]` shape and the "one autodiff graph" rationale.
- G3 - **Loss weighting.** Doc: loss weighted `[1, 1, 10, 10, 10, 10]` over `[eqA, eqB, bcA, bcB, icA, icB]`
  (BC/IC up-weighted 10x). Context says only "BC ... e IC blandas y ponderadas / soft, weighted" without the 10x.
- G4 - **Optimizer.** Doc: Adam (20,000 steps, lr 1e-3) then L-BFGS. Context omits it entirely.

(Constraints chips are accurate: "MMS-derived boundary/forcing per species", "time t is the Live scrubber",
"manufactured 2-species (cA*, cB*)". No fix needed there; optionally add the `v=(0,1), D=0.05, k_f=1` params.)

---

## Concrete proposed enrichments (grounded in doc quotes; no invented numbers)

### Fix `frontend/src/content/results.ts` -> `mine-heap-leach-rt`

Replace the fabricated breakthrough story with the actual manufactured-decay behavior and the doc's measured
numbers.

- `calculates_en` (remove "the base lights up is breakthrough"):
  "The animated field is the plotted reactant c_A over a vertical heap section (both axes are position, z points
  downward); color = concentration. It is a manufactured sine-cosine pattern that starts at maximal contrast at
  t=0 and, frame by frame, relaxes toward the +1.5 baseline as e^{-t}."
- `answer_en` (remove t=0.60 breakthrough):
  "c_A decays as e^{-t}, faster than c_B's e^{-t/2}: by t~0.6 the c_A field is nearly flat toward the +1.5
  baseline while c_B is still modulated. That unequal decay is what the time scrubber shows; there is no
  percolation 'breakthrough' in this manufactured solution."
- `verdict_en` (replace 0.4%):
  "c_A matches the analytic MMS to relative-L2 <= 1e-4 and c_B to <= 2e-4 at every one of the 6 snapshots,
  comfortably inside the < 2e-2 per-species target band (ONNX parity 1.19e-06). A genuinely well-converged
  coupled two-species solve validated against an exact manufactured truth, not a fit to data. Synthetic-
  illustrative: a real heap adds a shrinking-core sink, dual porosity and channeling: not the calendar."
- `assumptions_en`: ["uniform downward Darcy percolation v=(0,1)", "isotropic dispersion D=0.05, bimolecular
  reaction k_f=1", "manufactured (MMS) 2-species truth; soft Dirichlet BC/IC = c*"].
- Mirror all four in the `_es` fields.

### Fix `frontend/src/content/scenarios.ts` -> `mine-heap-leach-rt`

Keep the "why heap leaching matters" hook, but stop claiming the case delivers a breakthrough time/curve
(the doc/Context do not).

- `situation_en`: keep the operational motivation, then state what the case actually demonstrates, e.g. append/
  replace tail: "This case is the catalogue's coupled-PINN stress test: one network solving two nonlinear
  advection-diffusion-reaction PDEs at once, validated against an exact manufactured (MMS) solution."
- `measured_en`: replace "gives the breakthrough curve at any depth" with "the 2-D two-species reactive-transport
  field over time c(x,z;t), scored against an exact manufactured (MMS) reference per snapshot."
- Mirror in `_es`.

### Enrich `frontend/src/content/cases/HeapLeachContext.tsx`

Add a short "Result (measured, seed 42)" paragraph after the Formalization block (both EN and ES), transcribed
from the doc:
  "Measured (seed 42): c_A relative-L2 <= 1e-4 and c_B <= 2e-4 against the analytic MMS across all 6 snapshots
  (target < 2e-2 per species), ONNX parity 1.19e-06. Architecture: one shared [3] -> [40]x4 -> [2] tanh network
  predicting both c_A and c_B through a single autodiff graph; loss weighted [1,1,10,10,10,10] over
  [eqA,eqB,bcA,bcB,icA,icB] (BC/IC up-weighted 10x); Adam 20,000 steps at lr 1e-3, then L-BFGS."

This closes G1-G4 and makes the deep panel quote the same numbers the corrected results.ts will show.
