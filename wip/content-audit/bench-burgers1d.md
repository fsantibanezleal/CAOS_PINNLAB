# Content audit — bench-burgers1d (in-app vs authoritative doc)

Date: 2026-07-15
Doc (ground truth): `docs/cases/bench-burgers1d.md`
In-app files audited:
- `frontend/src/content/cases/Burgers1dContext.tsx` (Burgers1dContext, EN + ES)
- `frontend/src/content/scenarios.ts` (`bench-burgers1d`)
- `frontend/src/content/results.ts` (`bench-burgers1d`)
- `frontend/src/content/constraints.ts` (`bench-burgers1d`)

## Verdict

This is one of the STRONGER cases. The Context is genuinely rich and coherent with the doc: it
carries the governing PDE, the exact traveling-wave closed form with `k = Δ/4ν`, the hard-constraint
output transform, RAR with the correct citation (Wu et al., CMAME 2023), the left/right states with
`s = 1/2`, `x0 = -0.4`, the domain, the `~4ν` thickness (with the arithmetic done: 0.08 at ν=0.02,
0.32 at ν=0.08), the honest out-of-scope list, and the Raissi sine-IC caveat. Nothing in the Context
flatly contradicts the doc. The scenario, constraints, and results entries are all coherent framings.

The one real weakness is in `results.ts`: the verdict reports a single **"field error 0.1%"**, which
is the BEST-case (sharpest) variant only. The doc's honest headline is a **range, ≤ 1.2% across all 6
variants** (0.08% at ν=0.02 up to 1.2% at ν=0.08). The app quotes the most flattering number and drops
both the worst case and the doc's counterintuitive honesty insight ("the sharpest viscosity is the MOST
accurate, because the hard constraint encodes the front so the net only learns the small interior
translation"). The measured ONNX parity (6.5e-7) appears nowhere. These are understatement + omission,
not fabrication. Severity 2 (real gap), not a hollow-content or flat-contradiction case.

- contextDepth: **rich**
- coherentWithDoc: **true** (Context coherent; results.ts understates but does not state a false number)
- scenarioResultsDepth: **adequate**
- severity: **2**

## Contradictions / coherence issues

### C1 — results.ts verdict cherry-picks the best-case field error (understatement)
- inApp (`results.ts` → `bench-burgers1d.verdict_en`): "Arrival read within 2 time steps of the exact
  0.80; **field error 0.1%**. Reliable warning-time estimates for the whole viscosity family from one
  network."
- docSays (`Result` table): "relative-L2 vs analytic **≤ 1.2 % across all 6 variants** (sharpest ν=0.02
  → 0.08%; ν=0.08 → 1.2%)" and "all variants inside the `< 2e-2` band."
- Why it matters: 0.1% corresponds only to the sharpest variant (doc: 0.08%). The most diffuse variant
  is 1.2%, ~12x higher. Presenting "field error 0.1%" as THE field error understates the honest worst
  case. Every other case's results verdict (heat1d 0.2%, wave1d 0.3%, allencahn 0.4%) reads as an
  overall figure, so a reader takes 0.1% as representative when it is the floor, not the ceiling.

## Depth gaps (real doc content the app omits)

### G1 — The doc's key honesty insight is missing everywhere
The doc states: "The hard-constraint baseline encodes the front, so the net only learns the small
interior translation — the *sharpest* viscosity is the *most* accurate." This is a genuinely
non-obvious result (sharper fronts are normally HARDER, but here the closed-form front is baked into
`g(x;ν)`, so the residual task shrinks as ν shrinks). The Context's method section says "The network
only learns the interior translation" but never draws the accuracy conclusion; `results.ts` does not
mention it. This insight is the whole payoff of the hard-constraint design and should be surfaced.

### G2 — No measured accuracy numbers anywhere except the single 0.1%
The doc's `Result` section reports: relative-L2 range 0.08%–1.2% across six baked ν
(0.02, 0.03, 0.04, 0.05, 0.06, 0.08), ONNX parity max abs **6.5e-7**, seed 42, `< 2e-2` band. The
Context enumerates the six variants qualitatively (widths ~0.08 to ~0.32) but quotes zero measured
error and no ONNX parity. `results.ts` carries only the cherry-picked 0.1%. The measured L2 range and
the ONNX parity (evidence the exported model matches the trained net) are real doc content the app
drops.

### G3 — "< 2e-2 band" framing dropped
The doc frames all six variants as landing inside a `< 2e-2` acceptance band ("Honest CPU-trained
accuracy, all variants inside the `< 2e-2` band"). This band is the pass/fail context for the numbers
and is absent from the app.

## Concrete proposed enrichments (grounded in doc quotes; no invented numbers)

### E1 — `frontend/src/content/results.ts`, `bench-burgers1d.verdict_en/_es`
Replace the single "field error 0.1%" with the honest range + the insight. EN:

> "Arrival read within 2 time steps of the exact 0.80. Field error rises with viscosity, from 0.08% at
> the sharpest front (ν=0.02) to 1.2% at the most diffuse (ν=0.08), every variant inside the < 2e-2
> band: the hard constraint bakes in the exact front, so the SHARPEST viscosity is the most accurate.
> Reliable warning-time estimates for the whole viscosity family from one network."

ES (mirror):

> "Llegada leída a menos de 2 pasos temporales del 0.80 exacto. El error de campo crece con la
> viscosidad, de 0.08% en el frente más agudo (ν=0.02) a 1.2% en el más difuso (ν=0.08), cada variante
> dentro de la banda < 2e-2: la restricción dura codifica el frente exacto, por eso la viscosidad MÁS
> aguda es la más precisa. Estimaciones confiables de tiempo de aviso para toda la familia desde una
> red."

Doc anchor: "≤ 1.2 % across all 6 variants (sharpest ν=0.02 → 0.08%; ν=0.08 → 1.2%)" and "the sharpest
viscosity is the most accurate ... all variants inside the `< 2e-2` band."

### E2 — `frontend/src/content/cases/Burgers1dContext.tsx`, "The method: hard constraints + RAR" section (EN + ES)
Append one sentence connecting the transform to the accuracy result and citing ONNX parity:

> EN: "Because the base fit `g(x;ν)` already carries the exact front, the network only has to learn the
> small interior translation, so accuracy is BEST at the sharpest front: measured relative-L2 is 0.08%
> at ν=0.02 and rises to 1.2% at ν=0.08 (seed 42), and the ONNX export matches the trained net to
> 6.5e-7 max abs, which is why the Live tab reproduces the field exactly."
> ES: "Como el ajuste base `g(x;ν)` ya carga el frente exacto, la red solo aprende la pequeña traslación
> interior, por lo que la precisión es MEJOR en el frente más agudo: la L2 relativa medida es 0.08% en
> ν=0.02 y sube a 1.2% en ν=0.08 (semilla 42), y la exportación ONNX coincide con la red entrenada a
> 6.5e-7 en máximo absoluto, por eso el tab Live reproduce el campo con exactitud."

Doc anchor: "The hard-constraint baseline encodes the front, so the net only learns the small interior
translation — the sharpest viscosity is the most accurate"; "ONNX parity (max abs) 6.5e-7."

### E3 (optional, low priority) — Context "What each variant shows" paragraph
The six ν values are already named; optionally attach their measured L2 (0.08% at ν=0.02 ... 1.2% at
ν=0.08) so the qualitative sweep is anchored to numbers. Doc anchor: same Result table. Only add if the
team wants measured numbers in-Context; otherwise E1 + E2 already close the gap.

## Items checked and found COHERENT (no action)
- Governing PDE `u_t + u u_x = ν u_xx`: Context matches doc.
- Closed form `u* = s - (Δ/2) tanh(k(x - x0 - s t))`, `k = Δ/4ν`: Context matches doc verbatim.
- Hard-constraint transform `u_θ = g(x;ν) + t(1-x²)N_θ`: Context matches doc.
- `s = 1/2`, `Δ = 1`, `x0 = -0.4`, ν ∈ [0.02,0.08], thickness ~4ν: all match; Context adds the correct
  Rankine-Hugoniot label and the 0.08/0.32 width arithmetic.
- results.ts arrival t=0.80 at x=0: physically exact (x0 + s·t = -0.4 + 0.5·0.80 = 0), self-consistent.
- constraints.ts pins (IC front at x0=-0.4 hard, BC saturated states hard, ν param sets width, Whitham
  anchor): all coherent with doc.
- scenarios.ts framing (racing front / arrival-time / warning-time): illustrative, consistent with the
  physics; not contradicted by doc.
- Raissi sine-IC out-of-scope note: Context matches doc's Honesty section.
