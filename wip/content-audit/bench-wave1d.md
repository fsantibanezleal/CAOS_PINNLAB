# Content audit: bench-wave1d (in-app vs authoritative doc)

Date: 2026-07-15
Doc audited: `docs/cases/bench-wave1d.md`
In-app files audited:
- `frontend/src/content/cases/Wave1dContext.tsx` (deep Context, EN + ES)
- `frontend/src/content/scenarios.ts` ("bench-wave1d" entry)
- `frontend/src/content/results.ts` ("bench-wave1d" entry)
- `frontend/src/content/constraints.ts` ("bench-wave1d" entry)

## Verdict

**Coherent with the doc, but the app under-tells the doc's most interesting content.** No numeric
or mechanistic CONTRADICTION was found: the governing equation, the exact anchor
`u* = sin(pi x) cos(c pi t)`, the hard-constraint transform, the domain/grid, the parameter range
`c in [0.5, 2]`, and the period relation `T = 2/c` are all correct and mutually consistent across
Context, scenarios, results, and constraints. The field-error figure "0.3%" is a faithful rounding of
the doc's measured `<= 0.32%`.

Where the app falls short is DEPTH on the RESULT and METHOD-engineering side. The doc's headline honest
finding (the parametric oscillatory family needed net CAPACITY; a bigger SIREN nails the full 4x speed
family including the `c=2` corner where a smaller net STALLED), the measured ONNX parity (`~7e-7`), and
the training recipe (net `[3,96,96,96,96,96,1]`, Adam 22000 then L-BFGS, `num_boundary=0`) are omitted or
only half-present. The Context prose on the PROBLEM/PHYSICS is genuinely rich; the RESULT framing is
adequate but reframes the doc's measured metric (relative-L2) as "measured periods" and mildly overstates
what can be measured within the `t in [0,1]` window.

- coherentWithDoc: **true**
- contextDepth: **rich** (theory + governing eq + exact anchor + hard-constraint derivation + honest scope)
- scenarioResultsDepth: **adequate** (substantive, but omits the capacity/c=2 finding and the ONNX parity)
- severity: **2** (real gap, no contradiction)

## Contradictions (in-app vs doc)

**None hard.** Two items are framing/precision caveats, not contradictions:

1. **"field error 0.3%" (results.ts) vs doc "<= 0.32% across all 6 variants".**
   Not a contradiction (0.32 rounds to 0.3), but the app drops the doc's precise framing: 0.32% is the
   WORST case across all six speeds, INCLUDING the 4x extremes. Presenting a bare "0.3%" loses "this is
   the ceiling over the whole family". Recommend restoring the doc's exact figure and its scope.

2. **"Measured periods equal the exact 2/c at every speed in the window" (results.ts verdict).**
   `T = 2/c` is correct and the field is accurate enough that a read-off is essentially exact, so this is
   not false. But it is a mild overstatement: at slow speeds the full period exceeds the window
   (`c=0.5 -> T=4`, `c=0.75 -> T=2.67`, both `> 1`), so the period there is INFERRED from the visible
   fraction, not read off a complete oscillation. Also note the doc's actual MEASURED metric is
   relative-L2, not a period; "measured periods" is a derived reframe. Recommend softening to "the
   reconstructed motion reproduces the exact period relation T = 2/c".

## Depth gaps (real doc content the app omits)

**GAP 1 (significant) - the capacity / c=2-corner honest finding is entirely absent.**
Doc Method: "The oscillatory parametric family needed capacity: net [3,96,96,96,96,96,1] sin (DeepXDE),
Adam (22000) -> L-BFGS." Doc Result: "The bigger SIREN net nails the full 4x speed family, including the
fast c=2 corner where a smaller net stalled." This is the single most instructive engineering nugget in
the doc (SIREN alone is not enough; you also need enough capacity, and the failure mode is the fast
corner). Neither the Context nor results.ts mentions it. The Context only contrasts SIREN vs tanh
("SIREN handles it where a tanh would stall"), never SIREN-capacity vs a smaller SIREN.

**GAP 2 (moderate) - measured ONNX parity `~7e-7` omitted.**
Doc Result table: "ONNX parity (max abs) ~7e-7". The Context's "How to read the viz" advertises the Live
tab recomputing "in your browser (onnxruntime-web)" but never anchors that the deployed ONNX matches the
trained net to `~7e-7`. This is a measured number that directly backs the Live claim.

**GAP 3 (moderate) - training recipe thin in Context.**
Doc Method gives net architecture (`[3,96,96,96,96,96,1]` sin, DeepXDE), optimizer schedule
(Adam 22000 -> L-BFGS), and `num_boundary=0`. The Context captures the ESSENCE of `num_boundary=0`
("With no IC/BC loss terms, training is stable") but omits the architecture and optimizer schedule. For a
Method-level Context this is a real omission vs the doc's Method section.

**GAP 4 (minor) - synthetic/anchor honesty not made crisp.**
Doc Honesty: "The anchor is the closed-form standing-wave solution (exact for every c), not a measured
dataset: the relative-L2 is a genuine solution-accuracy number, not a fit proxy." The Context does call
`u*` "our anchor" but never states there is NO measured data and that the error is therefore a genuine
solution-accuracy number rather than a fit-to-data proxy.

## Concrete proposed enrichments (grounded in doc quotes; no invented numbers)

### E1 - add the capacity finding to `Wave1dContext.tsx` (Scope & assumptions paragraph)
Append to the EN "Scope & assumptions" paragraph (mirror in ES), grounded in doc Method + Result:

> "The oscillatory parametric family also needs CAPACITY: a larger SIREN (five hidden layers of 96 units,
> trained Adam for 22000 steps then L-BFGS) nails the full 4x speed family, including the fast c=2 corner
> where a smaller net stalled."

ES:
> "La familia oscilatoria paramétrica tambien exige CAPACIDAD: una SIREN mayor (cinco capas ocultas de 96
> unidades, Adam por 22000 pasos y luego L-BFGS) domina toda la familia de velocidad 4x, incluida la
> esquina rapida c=2 donde una red menor se estancaba."

### E2 - enrich `results.ts` verdict for "bench-wave1d"
Replace the current `verdict_en` with the doc-exact field figure + the c=2 honest point + ONNX parity, and
soften the period claim:

> verdict_en: "The reconstructed motion reproduces the exact period relation T = 2/c at every speed
> (2.00 vs 2.00, 1.60 vs 1.60, ...); field error <= 0.32% across all six speeds, including the fast c=2
> corner where a smaller net stalled, and the deployed ONNX matches the trained net to ~7e-7. Safe for
> resonance screening across the whole speed family."

ES mirror:
> verdict_es: "El movimiento reconstruido reproduce la relacion exacta de periodo T = 2/c en cada
> velocidad (2.00 vs 2.00, 1.60 vs 1.60, ...); error de campo <= 0.32% en las seis velocidades, incluida
> la esquina rapida c=2 donde una red menor se estancaba, y el ONNX desplegado coincide con la red
> entrenada a ~7e-7. Seguro para cribado de resonancia en toda la familia de velocidades."

(All figures are doc-sourced: `<= 0.32%`, `~7e-7`, the c=2/smaller-net-stalled finding, T=2/c.)

### E3 - anchor the Live claim with ONNX parity in `Wave1dContext.tsx` ("How to read the viz")
In the onnxruntime-web sentence, add: "the exported ONNX matches the trained net to about 7e-7, so the
Live readout is the same solution you see in the baked chips." (doc: "ONNX parity (max abs) ~7e-7").

### E4 - make the synthetic honesty explicit in `Wave1dContext.tsx` ("Scope & assumptions")
Add one sentence (doc Honesty): "There is no measured dataset here: the error is graded against the exact
standing wave u*(x,t;c), so the reported error is a genuine solution-accuracy number, not a fit-to-data
proxy." ES mirror.

## Items confirmed COHERENT (no action needed)
- Governing PDE `u_tt = c^2 u_xx`, ICs `u(x,0)=sin(pi x)`, `u_t(x,0)=0`, BCs `u(0,t)=u(1,t)=0`: match doc.
- Exact anchor `u* = sin(pi x) cos(c pi t)`: match doc (Context, constraints).
- Hard constraint `u_theta = sin(pi x) + t^2 x(1-x) N(x,t,c)` and its IC/BC reasoning: match doc.
- Domain `x,t in [0,1]`, `161x161` grid, `c in [0.5, 2]`: match doc.
- Period fractions (c=0.5 quarter, c=1 half with sign flip at t=1, c=2 full): match doc and are
  mathematically correct.
- constraints.ts pins (ic/bc/param/anchor) all correct vs doc.
- scenarios.ts situation/measured (resonance framing, c from tension/density) coherent with doc physics.
