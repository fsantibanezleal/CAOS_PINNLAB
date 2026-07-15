# Content audit: bench-allencahn (in-app vs authoritative doc)

Date: 2026-07-15
Doc audited: `docs/cases/bench-allencahn.md`
In-app files audited:
- `frontend/src/content/cases/AllenCahnContext.tsx` (deep Context, EN + ES)
- `frontend/src/content/scenarios.ts` (situation/measured)
- `frontend/src/content/results.ts` (calculates/assumptions/answer/verdict)
- `frontend/src/content/constraints.ts` (pins)

## Verdict

**Coherent with the doc, no contradictions found. Real depth gap: the deep Context omits every measured result.**

The Context prose is genuinely rich on THEORY: it carries the governing equation, the double-well
potential `W(u)=5/4(1-u^2)^2`, the L2 gradient-flow / Ginzburg-Landau energy functional, the
interface-width scaling `l ~ sqrt(d/5) ~ 0.014`, the exact hard-constraint ansatz, and why the naive
soft PINN collapses. On several of these it is DEEPER than the doc (the energy functional and the
`0.014` interface width are not even in the doc, and both are correct). So depth of theory = rich.

The failure is on the RESULT axis. The deep Context never states a single measured number: not the
`0.41 %` relative-L2, not the `95.4 %` naive collapse, not the ONNX parity, not the live lane, not
the seed, and it describes the RAR loop only as "several rounds" where the doc gives an exact,
reproducible recipe (4 rounds, 100k pool, top-600, 5000 Adam iters, single final L-BFGS) and a full
architecture. A reader of the Context learns the physics and the mechanism but never learns HOW WELL
the recipe worked. The short `results.ts` verdict does carry the rounded `0.4 %` and `95 %`, so the
numbers exist in the app; they are simply absent from the deep chapter that teaches the method.

Severity: 2 (real gap, not a contradiction and not hollow).

## Coherence check (contradictions)

None. Every in-app claim that overlaps the doc matches it:

| In-app claim | Location | Doc | Status |
|---|---|---|---|
| `u_t = d u_xx + 5(u-u^3)`, `d=0.001` | Context | same | OK |
| domain `x in [-1,1]`, `t in [0,1]`, grid `201x101` | Context | same 201x101 | OK |
| IC `u(x,0)=x^2 cos(pi x)` (hard ansatz) | Context, constraints | same | OK |
| ansatz `u = x^2 cos(pi x) + t(1-x^2) N(x,t)` | Context | identical | OK |
| endpoint-matched at `x=±1` (hard) | Context, constraints | "endpoint-matched" | OK |
| SOTA ceiling PirateNets ~2e-5 (jaxpi), cited not claimed | Context | same | OK |
| fixed-parameter benchmark, symmetric front stationary | Context | same | OK |
| `0.4 %` error vs spectral; naive `95 %` collapse | results.verdict | `0.41 %` / `95.4 %` | OK (rounded) |
| walls trustworthy to ~one grid cell (~0.01) | results.verdict | 201 pts over width 2 => 0.01 spacing | OK (consistent) |

Derived-physics claims the app ADDS beyond the doc, checked and correct (not contradictions, they are
good enrichments): `W(u)=5/4(1-u^2)^2` gives `-dW/du = 5(u-u^3)` exactly; `l ~ sqrt(0.001/5) = 0.0141`.

One app-only claim not verifiable from the doc (not a contradiction, flagged for provenance): the
`results.ts` answer states the walls end at `x = ±0.49`. The doc gives no wall positions; this is an
app-measured readout. Coherent with the symmetric-front picture; leave as is, but it is the app's own
measurement, not backed by the doc.

## Depth gaps (real doc content the in-app content omits)

1. **The whole measured Result is missing from the deep Context.** The doc's Result table gives
   relative-L2 `0.41 % (0.004106)`, ONNX parity `1.16e-06` max abs, and lane `live (49 KB ONNX,
   opset 18)`, at `seed 42`. The Context states none of these. It asserts "the recipe that works"
   without ever saying how well it worked.

2. **RAR recipe is vague where the doc is exact.** Context: "collocation points are added where the
   residual is largest ... over several rounds." Doc: "4 rounds of greedy residual refinement: each
   round draws a 100 000-point pool, adds the top-600 highest-residual points as anchors, then
   re-fits (5 000 Adam iters); a single final L-BFGS polishes after the loop."

3. **Architecture omitted.** Doc: "`[2, 64, 64, 64, 1]` tanh FNN, 8 000 domain / 400 boundary / 800
   initial points, Adam (lr 1e-3, 20 000 iters) then L-BFGS, then the RAR loop." Context gives no
   network size, point budget, or optimiser schedule.

4. **Naive-collapse number absent at the point the Context makes the claim.** The Context says the
   soft PINN "collapses" and "gets stuck" but gives no number; the doc's `95.4 %` (and the Compare
   view's side-by-side error maps) quantify exactly how bad. Only `results.ts` carries the rounded 95 %.

5. **Trimmed-RAR honesty note omitted.** Doc: "The trimmed RAR (4 rounds + one final L-BFGS instead
   of 6 per-round L-BFGS) bakes far faster for the same honest sub-1 % accuracy." This is a genuine
   engineering caveat the deep chapter should carry.

6. **The honest "clears the band" framing is absent.** Doc: "0.41 % ... clears the expected band
   (`< 1e-2`) by ~2.4x." The Context never anchors the result against an expected band.

## Concrete proposed enrichments (grounded in doc quotes)

All text below is transcribed/adapted from `docs/cases/bench-allencahn.md`. No invented numbers.
Mirror each EN edit in the ES branch of the same file.

### Edit A: add a "Measured result" block to `AllenCahnContext.tsx`

Insert after the SOTA-ceiling paragraph (EN branch, after line 139; mirror in ES after line 58),
before "Scope & assumptions":

> **Measured result (seed 42).** Validated against the spectral reference `Allen_Cahn.npz`
> (DeepXDE/Raissi) on the 201x101 (x,t) grid, the hard-constraint + RAR network reaches
> **0.41 % relative-L2** (0.004106): sub-1 %, clearing the expected `< 1e-2` band by about 2.4x, so
> the recipe genuinely resolves the transition layers rather than collapsing to the metastable state.
> The naive soft PINN, by contrast, smears the sharp +/-1 layers at **95.4 %** relative-L2: the
> Compare and Training views show the two side by side with their error maps. The trained model ships
> **live** (49 KB ONNX, opset 18; ONNX parity 1.16e-06 max abs).

(ES mirror: "Resultado medido (semilla 42). Validado contra la referencia espectral `Allen_Cahn.npz`
(DeepXDE/Raissi) en la grilla 201x101 (x,t), la red con restriccion dura + RAR alcanza 0.41 % de
L2-relativo (0.004106): sub-1 %, superando la banda esperada `< 1e-2` por ~2.4x ... el PINN suave
ingenuo, en cambio, difumina las capas +/-1 con 95.4 % de L2-relativo ... modelo en vivo (ONNX 49 KB,
opset 18; paridad ONNX 1.16e-06 max abs).")

### Edit B: make the RAR bullet in `AllenCahnContext.tsx` concrete

Replace the RAR bullet (EN line 134 "after the base fit, collocation points are added where the
residual is largest: right on the moving interfaces: over several rounds, chasing the front.") with:

> **RAR (residual-based adaptive refinement):** after the Adam->L-BFGS base fit, **4 rounds** of
> greedy refinement: each round draws a 100 000-point pool, adds the **top-600** highest-residual
> points as anchors, and re-fits (5 000 Adam iters); a **single final L-BFGS** polishes after the
> loop, concentrating collocation density exactly on the thin moving layer.

### Edit C: add the architecture + trimmed-RAR note

Append to the method section (EN, after line 139; ES mirror):

> Architecture: a `[2, 64, 64, 64, 1]` tanh FNN with 8 000 domain / 400 boundary / 800 initial
> points, Adam (lr 1e-3, 20 000 iters) then L-BFGS before the RAR loop. The trimmed RAR (4 rounds +
> one final L-BFGS instead of 6 per-round L-BFGS) bakes far faster for the same honest sub-1 %
> accuracy.

### Edit D (optional, small): quantify the naive collapse where the Context first claims it

In the "why the naive PINN fails" paragraph, add the number so the claim is not asserted numberless:
"... a metastable state has small residual and the optimiser gets stuck there (**95.4 % relative-L2**
against the spectral reference: the Compare/Training views show it)."

## Notes on the short content (no edits required)

- `scenarios.ts` bench-allencahn: substantive and coherent (grain-size framing, the metastable
  trap "a lazy solver settles into the WRONG one while looking converged"). Matches the doc's honesty.
- `results.ts` bench-allencahn: substantive; carries `0.4 %`, `95 %`, wall positions `±0.49`, and the
  one-grid-cell (~0.01) trust bound. This is where the app's measured numbers live; the fix is to lift
  them into the deep Context, not to change these.
- `constraints.ts` bench-allencahn: correct pins (hard IC ansatz, endpoint-matched BC, spectral Raissi
  npz anchor + RAR). Coherent.
