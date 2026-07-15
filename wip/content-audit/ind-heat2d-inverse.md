# Content audit: ind-heat2d-inverse (in-app vs authoritative doc)

Date: 2026-07-15
Auditor scope: coherence + depth + gaps of the IN-APP content against `docs/cases/ind-heat2d-inverse.md`.

Sources compared:
- Ground truth doc: `docs/cases/ind-heat2d-inverse.md`
- In-app deep context: `frontend/src/content/cases/Heat2dInverseContext.tsx` (EN + ES)
- In-app short content: `frontend/src/content/scenarios.ts`, `results.ts`, `constraints.ts` (keyed `ind-heat2d-inverse`)

## Verdict

**Coherent with the doc: YES. No contradictions found.** The in-app content is genuinely deep, not
hollow, and every method name, mechanism, equation, and the two headline numbers it states (`4%`
recovery, `356%` without data) match the doc. Severity is a **real gap (2)**, not a contradiction: the
app omits one of the doc's two headline measured results (the forward-`T` accuracy, `0.77%` / "under 1%")
and the quantified honesty caveat (max-abs error `0.44` in `k`). These are real, load-bearing numbers the
doc emphasizes and that appear NOWHERE in any of the four in-app sources (grep-verified: `0.44`, `0.77`,
`12.6`, `17.3` are absent from `frontend/src/content`).

- Context depth: **rich** (governing PDE, MMS triple, product-rule residual, weighted-obs loss, all four
  method design choices, honest low-|∇T| scope caveat, viz-reading guidance). This is not filler.
- Scenario + results + constraints depth: **rich** (they carry `4%`, `356%`, the "data makes the inverse
  solvable" thesis, and the diminishing-returns verdict). Substantive, not stubs.

## Contradictions (in-app claim vs doc)

None. Every number and mechanism checked is consistent:

| in-app | doc | status |
|---|---|---|
| recovered `k` to `4%` (results.answer, results.verdict, constraints.data) | `4.0%` (0.040295) | OK |
| `356%` without data (scenarios.measured, results.verdict, constraints.unknown) | `356%` at n=0 | OK |
| PDE `∇·(k∇T)=q`, `T=0` on ∂Ω (Context, constraints.bc) | identical | OK |
| MMS triple `T*=sinπx sinπy`, `k*=1+½ sinπx sinπy`, `q=∇·(k*∇T*)` (Context) | identical | OK |
| product-rule residual `k(Txx+Tyy)+kx Tx+ky Ty−q` (Context) | identical | OK |
| `81×81` grid, `σ=0.01`, `~100` sensors, `λ=100`, dome height `≈1.5` (Context) | identical | OK |
| softplus positivity, `x(1-x)y(1-y)` hard boundary, PFNN two-output (Context) | identical | OK |
| error largest where `|∇T|` small / near boundary / at T's peak (Context, results) | identical framing | OK |

One wording imprecision worth tightening (not a contradiction): `results.ts` assumptions say
`sensor noise ~1%`, while the doc states `σ=0.01` (absolute Gaussian on `T`). That is ~1% only relative to
the peak `T≈1`; per-reading it is much larger near the boundary where `T≈0`. Recommend the precise gloss
"`σ=0.01` absolute (~1% of peak T)".

Soft framing note (not a doc contradiction): `scenarios.situation` motivates the case as defect mapping
("a delamination, a void, moisture ... a MAP of the defect"). The doc's manufactured truth `k*` is a smooth
central conductive dome, not a defect. This is legitimate motivation and the app stays honest because
`results.calculates`/`answer` describe the actual "conductive dome and its insulating edges". No change
required; flagged only for transparency.

## Depth gaps (real doc content the app omits)

### GAP 1 (primary, severity 2): forward-`T` reconstruction accuracy `0.77%` / "under 1%"
The doc's Result table reports `forward T relative-L2 vs T* = 0.77% (0.007664)` and the prose stresses the
dual result: "the temperature field is reconstructed to under 1%, and the conductivity to a few percent."
The app states only the `4%` `k` recovery; the `T` accuracy is absent from all four sources. This is half
of the doc's headline result. The Context even introduces `T` as the "auxiliary state" and discusses where
its error grows, but never gives the number.

### GAP 2 (severity 1-2): the quantified honesty caveat, max-abs error `0.44` in `k`
The doc's Honesty/Result: "max abs error in k = 0.439" and "the max-abs error of 0.44 sits where `|∇T|` is
small ... because there `k` barely influences the residual and is therefore weakly identifiable." The
Context gives the qualitative caveat ("error is naturally larger where `|∇T|` is small") but drops the
`0.44` magnitude that makes it concrete and honest.

### GAP 3 (severity 1): identifiability-sweep numbers + the training-budget caveat
The doc gives the full ladder: `356%` (n=0), `17.3%` (n=10), `16.3%` (n=25), `13.6%` (n=50), `12.6%` (n=100)
under "one fixed fast training budget", and explains the fully-trained n=100 reaches `4.0%`. The app
(`results.verdict`, `constraints.data`) references "the cliff ... ANY sensors beat none; returns diminish
past ~10" but surfaces no numbers and, more importantly, never explains why the Diagnostics chart's n=100
point (`12.6%`) differs from the `4.0%` headline. A user reading both can be confused. The prose should note
the sweep uses a fast fixed budget while `4.0%` is the fully-trained run.

### GAP 4 (severity 0-1, optional): implementation specifics
Doc Method lists: `PFNN([2,[40,40],[40,40],[40,40],2])`, `k` = FIRST output, softplus `ε=1e-3`, training
`Adam 2e4 steps @ lr 1e-3 then L-BFGS`, sensors added as `anchors`. The Context names PFNN and softplus
generically (`ε`) but omits the width, the `ε=1e-3` value, and the training recipe. Optional depth; not a
gap in correctness.

## Concrete proposed enrichments (faithful to doc, no invented numbers)

### E1 - add the forward-T result (fixes GAP 1). File: `frontend/src/content/cases/Heat2dInverseContext.tsx`
In the "What the benchmark shows" paragraph (EN block ~line 178, ES block ~line 82), append a sentence
grounded in doc lines 50, 55-56:

> EN: "Quantitatively the recovered conductivity lands at 4.0% relative-L2 of `k*`, and the temperature
> field it implies is reconstructed to under 1% (0.77%): the physics recovers both fields, the conductivity
> to a few percent and the temperature to sub-percent."
> ES: "Cuantitativamente, la conductividad recuperada queda al 4.0% relativo-L2 de `k*`, y el campo de
> temperatura implicado se reconstruye a menos del 1% (0.77%): la física recupera ambos campos, la
> conductividad a un pequeño porcentaje y la temperatura a sub-porcentaje."

Optionally also extend `results.ts` `answer_en`/`answer_es` (line 107-108) to append "; the temperature
field it implies is reconstructed to under 1%."

### E2 - quantify the honesty caveat (fixes GAP 2). File: `Heat2dInverseContext.tsx`, Scope paragraph
In the "Scope & assumptions" paragraph (EN ~line 171-175, ES ~line 74-78), extend the low-|∇T| sentence
grounded in doc lines 48, 56-58:

> EN: "... so the data inform conductivity less: the worst-case error (max-abs ≈ 0.44 in `k`) concentrates
> exactly there, at the boundary and the field's peak: an honest limitation set by the physics, not a
> tuning failure."
> ES: "... así que los datos informan menos sobre la conductividad: el error de peor caso (máx-abs ≈ 0.44 en
> `k`) se concentra justo ahí, en el borde y en el pico del campo: una limitación honesta impuesta por la
> física, no una falla de ajuste."

### E3 - surface the sweep ladder + budget caveat (fixes GAP 3). File: `frontend/src/content/results.ts`
Extend `verdict_en`/`verdict_es` (lines 109-110) grounded in doc lines 86-90:

> EN: append " The sweep runs one fixed fast training budget (356% at 0 sensors, ~17% at 10, ~13% at 100);
> the fully-trained run reaches the 4% headline: that is why the Diagnostics curve bottoms near 13%, not 4%."
> ES: append " El barrido usa un presupuesto de entrenamiento rapido fijo (356% con 0 sensores, ~17% con 10,
> ~13% con 100); la corrida completamente entrenada llega al 4% del titular: por eso la curva de Diagnostico
> toca fondo cerca del 13%, no del 4%."

(Numbers taken verbatim from doc line 87-89: 356% n=0, 17.3% n=10, 12.6% n=100, 4.0% fully-trained.)

### E4 - optional implementation depth (fixes GAP 4). File: `Heat2dInverseContext.tsx`, method bullets
In the "Two outputs" bullet (EN ~line 155, ES ~line 58) name the architecture and training, grounded in doc
lines 28-30, 39-40:

> EN: "a parallel-branch network `PFNN([2,[40,40]x3,2])` emits `(k,T)` with `k` as its first output; trained
> Adam (2e4 steps, lr 1e-3) then L-BFGS."
> ES: analogous.

And in the "Hard positivity" bullet, make `ε` explicit: `k = softplus(·)+1e-3`.

### E5 - precise noise gloss. File: `frontend/src/content/results.ts`
Change `assumptions_en` item `"sensor noise ~1%"` (line 105) to `"sensor noise sigma=0.01 (~1% of peak T)"`
and the ES analog, grounded in doc line 22 / 66-67.

## Structured summary
- coherentWithDoc: true (no contradictions)
- contextDepth: rich; scenarioResultsDepth: rich
- primary gap: forward-T `0.77%` result absent in-app (doc headline)
- secondary gaps: max-abs `0.44` caveat number; sweep ladder + budget caveat
- severity: 2 (real gap, honest content omitted; app is not hollow and does not contradict the doc)
