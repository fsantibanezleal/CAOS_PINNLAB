# Content audit: `poll-soil-barrier` (in-app vs authoritative doc)

Date: 2026-07-15
Auditor: honest depth+coherence pass (not a rubber-stamp).

Ground truth compared against:
- Doc: `docs/cases/poll-soil-barrier.md`
- Pipeline (code of record): `data-pipeline/pinnlab/cases/poll_soil_barrier.py`
- Baked manifest (measured): `frontend/public/data/derived/manifests/poll-soil-barrier.json`

In-app content audited:
- Deep Context: `frontend/src/content/cases/SoilBarrierContext.tsx` (EN + ES)
- Scenario: `frontend/src/content/scenarios.ts` (`poll-soil-barrier`)
- Results: `frontend/src/content/results.ts` (`poll-soil-barrier`)
- Constraints: `frontend/src/content/constraints.ts` (`poll-soil-barrier`)

---

## Verdict

**Coherent with the doc: NO** (one hard, confirmed numeric contradiction; one minor mischaracterization).
**Context depth: RICH** and faithful. **Scenario+Results depth: RICH** and grounded (readouts are computed
from the real baked trace, not invented).

The single serious issue is a **confirmed factual contradiction in `constraints.ts`**: it states the barrier
diffusivity **jumps 100x**, while the doc, the pipeline code, the Context, the manifest variant, and the
results verdict all agree the contrast is **10x**. Worse, 100x is the exact value the case documents itself as
deliberately NOT using (a 100x jump is called out in both the doc and the code comment as too severe for this
lane). This is a self-contradiction that any reader cross-checking the chips against the Context will catch.

Everything else is coherent: the Context is a genuinely deep derivation (governing PDE, the series-resistance
MMS with a full constant-flux proof, the kink mechanism `c_x = -1/(D R(L))`, the FBPINN partition-of-unity
blend, the hard-constraint output transform, honest scope, and the ADR-0016 §9.A justification for shipping a
single variant). The results numbers (19% error, t=0.39 half-rise, 0.157 end level) are all grounded in the
baked field.

Severity: **3** (a quantitative contradiction, not merely a gap).

---

## Contradictions (inApp vs docSays)

### C1 (severity 3, CONFIRMED) — the diffusion contrast is stated as 100x, not 10x
- **inApp** (`constraints.ts`, `poll-soil-barrier`, the `param` chip):
  - EN: `"D(x) jumps 100x inside the barrier (the kink)"`
  - ES: `"D(x) salta 100x dentro de la barrera (el quiebre)"`
- **docSays**: the contrast is **10x**. Doc Problem: `D_soil = 1.0`, `D_barrier = 0.1`, "whose diffusivity is
  **10x lower**". Doc Result/Honesty is explicit that 100x was avoided on purpose:
  > "A 10x contrast was used precisely because a 100x jump makes the kink too severe for this lane".
  Pipeline confirms: `poll_soil_barrier.py` lines 23-24 `D_SOIL = 1.0`, `D_BARR = 1.0e-1  # 10x diffusion
  contrast (a 100x jump makes the kink too severe for a 2-channel net on CPU)`; variant id `barrier10x`,
  label "Barrier (10x contrast)". Manifest variant metrics carry the same `barrier10x`. The Context itself
  says "a 10x contrast" (both EN and ES: `un contraste de 10x`).
- Note the wording is doubly wrong: the coefficient does not "jump up" 100x, it **drops** ~10x inside the
  barrier (that low D is the dominant resistance). The chip should read as a 10x DROP.

### C2 (severity 1, minor) — the upstream source is called "constant" but it is a rising source
- **inApp**:
  - `results.ts` assumptions: `"constant source upstream"` / `"fuente constante aguas arriba"`.
  - `scenarios.ts` is looser but compatible ("the source history").
- **docSays**: the inlet is a **rising** source, not constant. Doc Method + equation:
  `c(0,t) = 1 - e^{-t}`; Context (EN) calls it "inlet ... (a rising source)". `1 - e^{-t}` rises from 0 and
  only asymptotes to 1. Calling it a "constant source" mischaracterizes the boundary condition the whole MMS
  time-term `f = e^{-t} Psi(x)` is built around.

No other contradictions found. In particular the results verdict "~19% error ... the domain-split variant does
no better: shown honestly in Compare" is coherent with the doc Comparison ("both land near 19%; the FBPINN's
advantage at the barrier kink is subtle here ... with no fabricated gap"). The measured chips are exact:
manifest `l2_relative = 0.191872` (doc 0.192), `max_abs_error = 0.134161` (doc 0.134), `onnx_parity =
8.9e-08` (doc 8.9e-08), lane `live` (doc live). The results readouts `t = 0.39` and `0.157` are computed by
`build_estimates.py` as the half-rise time and end value at x=0.75 from the real baked trace, and are
physically consistent with the analytic (`1-e^{-t}` reaching half of its t=1 value at t≈0.38).

---

## Depth gaps (real doc content the app omits)

The Context is rich, so these are enrichment opportunities, not signs of a hollow panel.

- **G1 — beta value.** Doc gives the partition sharpness explicitly: `(beta = 40)`. The Context writes
  `w_left = sigma(beta (x_c - x))` but never states beta = 40 (code: `BETA = 40.0`).
- **G2 — the third anchor column (the center seam).** Doc Method: "Three extra anchor columns are seeded at
  `x = A_B, B_B, x_c` to pin the faces." Code seeds columns at AB, BB, **and X_C = 0.5** (lines 117-121). The
  Context only says "anchor points are seeded on the two barrier faces", omitting the center-seam anchor
  where the two sub-nets hand off, exactly the seam the partition-of-unity blend needs pinned.
- **G3 — the exact measured metrics.** The Context mentions `~2e-1` relative L2 but never gives the concrete
  scored numbers the doc Result table carries: relative-L2 **0.192 (19.2%)**, max abs error **0.134**, ONNX
  parity **8.9e-08**, lane **live**. A rich Context should name at least the 19.2% and 0.134.
- **G4 — the training recipe.** Doc Method: "Trained Adam (18k) -> L-BFGS in DeepXDE." Code: `[2]+[64]*4+[2]`,
  `adam: 18000, lbfgs: True`. The Context omits the optimizer schedule and net size entirely.
- **G5 — the honest path-forward note.** Doc Result/Honesty: "the strict per-subdomain-normalized FBPINN plus
  a GPU lane tighten it further" with a pointer to `docs/methods/domain-decomposition.md`. The Context conveys
  the CPU-limited framing but omits that this is a deliberately simplified 2-channel blend and that the strict
  per-subdomain-normalized FBPINN + GPU is what closes the gap.

---

## Concrete proposed enrichments (grounded in doc quotes)

### Fix F1 (BLOCKING) — correct the 100x chip in `frontend/src/content/constraints.ts`
Under `"poll-soil-barrier"`, the `param` entry. Replace:
```
{ kind: "param", en: "D(x) jumps 100x inside the barrier (the kink)", es: "D(x) salta 100x dentro de la barrera (el quiebre)" },
```
with (10x, phrased as a drop, matching doc `D_soil=1.0`, `D_barrier=0.1`):
```
{ kind: "param", en: "D(x) drops 10x inside the barrier (the kink)", es: "D(x) baja 10x dentro de la barrera (el quiebre)" },
```

### Fix F2 (minor) — call the source "rising", not "constant", in `frontend/src/content/results.ts`
Under `"poll-soil-barrier"`, `assumptions_en` / `assumptions_es`. Replace `"constant source upstream"` /
`"fuente constante aguas arriba"` with a phrasing that matches the doc's `c(0,t) = 1 - e^{-t}`:
```
en: "rising upstream source c(0,t)=1-e^{-t}", es: "fuente ascendente aguas arriba c(0,t)=1-e^{-t}"
```
(Optional: mirror the same wording in `scenarios.ts` `measured_*`, which currently says "the source history".)

### Fix F3 — add beta = 40 in `frontend/src/content/cases/SoilBarrierContext.tsx`
In the FBPINN paragraph, after the partition-of-unity equation, state the value. Grounded in doc `(beta = 40)`.
Suggested EN addition after "that switch across the barrier centre": "with window sharpness `beta = 40`". ES:
"con nitidez de ventana `beta = 40`".

### Fix F4 — correct the anchor sentence in `SoilBarrierContext.tsx` (add the center seam)
Doc: "Three extra anchor columns are seeded at `x = A_B, B_B, x_c`." Replace (EN) "anchor points are seeded on
the two barrier faces so the kink is resolved well" with:
```
three anchor columns are seeded, on the two barrier faces x=0.45, 0.55 and on the seam x_c=0.5 where the sub-nets hand off, so the kink is resolved well
```
Mirror in ES ("se siembran tres columnas de anclaje sobre las dos caras x=0.45, 0.55 y sobre la costura x_c=0.5 donde las sub-redes se relevan").

### Fix F5 — add a short measured-result + honesty line to `SoilBarrierContext.tsx`
The Context ends on viz-reading; add one sentence with the scored numbers and the honest path forward, grounded
in the doc Result table and Honesty section. Suggested EN (append near the "why single variant" paragraph or
the viz paragraph):
```
Scored against the closed-form MMS anchor this 2-channel CPU lane lands at 19.2% relative-L2 (max abs error 0.134, ONNX parity 8.9e-08, live lane): the coefficient-jump kink is honestly the hard part, and the strict per-subdomain-normalized FBPINN plus a GPU lane tighten it further (see the domain-decomposition method page).
```
ES equivalent, preserving the numbers verbatim. Optionally add the training recipe (Adam 18k -> L-BFGS,
net [2]+[64]x4+[2]) for G4.

---

## Summary table

| item | status |
|---|---|
| C1 constraints "100x" vs doc/code/manifest "10x" | CONTRADICTION (confirmed, blocking) |
| C2 "constant source" vs doc "rising 1-e^{-t}" | minor contradiction |
| Context 19.2%/0.134/8.9e-08 coherence | coherent (metrics match manifest) |
| results t=0.39 / 0.157 readouts | grounded in baked trace (correct) |
| Context depth | rich, faithful derivation |
| G1 beta=40 | omitted |
| G2 third anchor column (seam x_c) | omitted / understated |
| G3 exact metrics in Context | omitted |
| G4 training recipe | omitted |
| G5 GPU / strict-FBPINN path forward | omitted |
