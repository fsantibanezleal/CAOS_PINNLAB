# Content audit — `poll-tailings-seepage`

Date: 2026-07-15
Scope: in-app content (Context + scenario + results + constraints) vs the authoritative doc
`docs/cases/poll-tailings-seepage.md`. Honest depth + coherence audit.

## Verdict

Severity **3** (a contradiction present). `coherentWithDoc = false`.

The deep Context component (`TailingsSeepageContext.tsx`) is genuinely **rich** and faithful to the doc:
it carries the Richards equation, the Gardner closure, the expanded divergence form, the Kirchhoff transform,
the exact closed-form anchor `ψ*(z,t;α)` with the dispersion relation `λ(α)`, every constant
(`θs=0.43, θr=0.078, Ks=0.25, κ=0.9`), the `α∈[1.0,2.5]` range, and an honest scope/assumptions block that
matches the doc's Honesty section almost verbatim (no open unsaturated-zone dataset; van Genuchten-Mualem and the
saturated Darcy inverse as extensions). Context depth is not the problem.

The problem is in **`results.ts`**: one direct physics contradiction (the α to soil-texture mapping is inverted vs
the doc and vs the app's own Context), and one imprecise/stale characterization of the validation anchor. The
scenario/constraints entries are coherent. The measured numbers from the doc's Result table (`≤0.26%`, `3.6e-7`)
are omitted in favor of vaguer wording.

---

## Contradictions (inApp vs docSays)

### C1 — α to soil-texture mapping is INVERTED (severity 3)

- **inApp** — `frontend/src/content/results.ts`, `poll-tailings-seepage.answer_en`:
  > "the coarser the material (higher α), the weaker the suction that remains"

  and `answer_es`:
  > "a material más grueso (α mayor), succión restante más débil"

- **docSays** — `docs/cases/poll-tailings-seepage.md`, Problem:
  > "α∈[1.0,2.5]: smaller α (broader pores) → deeper, more stratified suction."

  The doc maps **smaller α = broader/coarser pores** and (by implication) **larger α = finer pores**. The app's own
  `TailingsSeepageContext.tsx` says the same thing explicitly: *"small α (coarse, sandy pores) ... deep, stratified
  suction profiles; large α (fine, clayey pores) retains water and gives shallow suction."*

- **Why it is a real contradiction, not a wording quibble:** `results.ts` labels **higher α as "coarser"**, but the
  doc and Context both label **higher α as finer**. The *direction of the suction result* ("higher α → weaker
  suction") is correct per the doc; only the parenthetical texture label is flipped. A dam-safety reader taking the
  Results tab at face value would read the material family backwards relative to every other surface in the product.
  (Note for the record: in real soil physics Gardner's α is actually large for coarse sand, so `results.ts` happens
  to align with field convention while the doc/Context invert it. That is a doc-internal question and out of scope
  here; against the stated ground-truth doc, `results.ts` is the one that contradicts.)

- **Fix (faithful to doc):** change the parenthetical so texture follows the doc's mapping.
  - `answer_en`: "the **finer** the material (higher α), the weaker the suction that remains: the screening chart in
    one network."
  - `answer_es`: "a material **más fino** (α mayor), succión restante más débil: la carta de cribado en una red."

### C2 — validation anchor mischaracterized as an approximate "linearized model per regime" (severity 2, PLAUSIBLE)

- **inApp** — `frontend/src/content/results.ts`, `poll-tailings-seepage.verdict_en`:
  > "Validated against the analytic solution of the linearized model per regime (sub-percent field error)"

  `verdict_es`: "Validado contra la solución analítica del modelo linealizado por régimen (error de campo bajo el 1%)".

- **docSays** — `docs/cases/poll-tailings-seepage.md`, Problem + Result:
  > "The Kirchhoff transform m=e^{αψ} linearises the nonlinear operator *exactly* into a constant-coefficient
  > advection-diffusion in m, which admits an exact separable mode. The head solution, valid for **any** α, is the
  > anchor."
  > "Validation anchor: the **exact Kirchhoff family** ψ*(z,t;α)."

  The Context reinforces this: *"which is why it is an exact solution, not one propped up by an added source"* and
  *"our validation anchor in closed form for any α"*.

- **Why it is a real coherence issue:** the phrase "the **linearized model**" reads as validation against an
  *approximate* linearized surrogate, whereas the whole point of the case is that the Kirchhoff transform linearizes
  the **genuine nonlinear** operator *exactly*, so `ψ*` is an exact solution of the full nonlinear Richards+Gardner
  PDE. "per regime" further implies a piecewise/per-α solution, whereas the doc stresses it is **one** closed form
  valid for **any** α. This looks like stale phrasing from before the case was reframed (the doc explicitly notes it
  "replaces the earlier α-independent manufactured-source MMS").

- **Fix (faithful to doc):**
  - `verdict_en`: "Validated against the **exact Kirchhoff-transform solution** ψ*(z,t;α), the closed form the
    genuine nonlinear operator admits for any α (relative-L2 ≤ 0.26% across all six variants): correct for RANKING
    materials in screening. ..."
  - `verdict_es`: "Validado contra la **solución exacta por transformada de Kirchhoff** ψ*(z,t;α), la forma cerrada
    que el operador no lineal genuino admite para cualquier α (L2 relativo ≤ 0.26% en las seis variantes): correcto
    para ORDENAR materiales en cribado. ..."

---

## Depth gaps (real doc content the app omits)

### G1 — precise measured numbers replaced by "sub-percent"

The doc's Result table gives hard figures the app never states:

| metric | doc value |
|---|---|
| relative-L2 vs exact | **≤ 0.26 %** across all 6 variants |
| ONNX parity (max abs) | **3.6e-7** |

`results.ts` verdict says only "sub-percent field error" / "error de campo bajo el 1%"; `TailingsSeepageContext.tsx`
says only "the reported L2 is the true PINN error against the exact solution" with no number. Every other case's
verdict cites its baked number inline (e.g. bench-heat1d "0.2% vs exact"); this one should too.

- **Proposed enrichment (results.ts verdict, folds into C2 fix):** cite `relative-L2 ≤ 0.26% across all six
  variants (α = 1.0 ... 2.5)`. Optionally add the ONNX parity `3.6e-7` in Context or the Live/Implementation copy
  since the case is the shared-ONNX `live` lane.

### G2 — the ψ<0 invariant is asserted but its verification is dropped

- **docSays** (Problem): the strictly-unsaturated invariant is "verified, plus a finite-difference residual ≤10^-6".
- **inApp:** Context asserts `ψ<0` strictly everywhere and for every α (good) but omits that this was checked and
  that the FD residual is `≤1e-6`. That FD residual is independent evidence the closed form actually solves the PDE.
- **Proposed enrichment (`TailingsSeepageContext.tsx`, Formalization or Scope):** add, faithful to the doc, e.g.
  "(the ψ<0 invariant is verified across the cube, and a finite-difference residual of the closed form is ≤1e-6)".

### G3 — the concrete training recipe is absent

- **docSays** (Method): "Net [3,48,48,48,48,1] tanh (DeepXDE), Adam (18000, lr 1e-3) → L-BFGS, loss weights [1,10]".
- **inApp:** Context describes the method conceptually ("minimises the Richards residual at collocation points, with
  the IC and boundary conditions ... imposed softly") but gives no architecture, optimizer schedule, or loss
  weights. This is standard depth in the doc that the Context could carry (or the Implementation page could).
- **Proposed enrichment (`TailingsSeepageContext.tsx`, end of Formalization):** one sentence transcribing the recipe:
  "The net is [3,48,48,48,48,1] tanh (DeepXDE), trained Adam (18000 steps, lr 1e-3) then L-BFGS, PDE:anchor loss
  weights [1,10]; α enters as the third network input so one trained net spans the family."

### G4 — the doc's "genuine nonlinear operator, source-free" residual detail

- **docSays** (Method): the PDE residual evaluates the genuine nonlinear Gardner operator
  `C(ψ)ψ_t − Kψ_zz − K'(ψ_z^2 + ψ_z)`, **source-free**, on the network output.
- **inApp:** Context does write this exact operator in the "Formalization" Equation block (good — this is present),
  and the constraints anchor entry says "Kirchhoff-transform exact (α-dependent)". So G4 is largely covered; note
  only that the *residual is source-free* (a point the doc stresses as the honesty of the family) could be surfaced
  once more in the Results verdict where C2 currently muddies it. Low priority; C2's fix already restores it.

---

## Entries that are coherent (no action)

- `scenarios.ts` situation/measured: coherent framing (family not one case; α is the parametric axis one net carries).
- `constraints.ts`: `ic` "exact separable profile at t=0", `bc` "head profile pinned at the column ends",
  `param` "Gardner sorptive number α (network input)", `anchor` "Kirchhoff-transform exact (α-dependent)" — all
  match the doc's Method (soft Dirichlet anchor on the (z,t,α) cube boundary incl. the t=0 IC face).
- `results.ts` calculates/assumptions: coherent (Gardner's exponential law; suction = negative pressure; per-material
  α). The `(values below)` in the answer resolves to baked per-variant estimate items rendered under the answer
  (per the results.ts header contract), so those numbers are wired, not missing.

---

## Priority for the fix pass

1. **C1** (severity 3) — flip "coarser" to "finer" in `results.ts` answer_en/answer_es. One-line, high-impact.
2. **C2 + G1** (severity 2) — rewrite the `results.ts` verdict to name the exact Kirchhoff solution and cite
   `≤ 0.26%` across the six variants; drop "linearized model per regime".
3. **G2 / G3** (severity 1) — optional Context enrichment (FD residual ≤1e-6; training recipe) for depth parity
   with the doc.
