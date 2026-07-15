# Content audit — `mine-thickener-settling`

In-app content vs authoritative doc `docs/cases/mine-thickener-settling.md`.
Date: 2026-07-15.

## Verdict

**NOT coherent with the doc. Severity 3 (a direct method contradiction).**

The deep Context prose is genuinely rich in every other respect (governing PDE, the exact
MMS tanh-front anchor, the closed-form derivatives, the degenerate gel switch, real
literature parameters, an honest synthetic-illustrative scope), and the short
scenario/results/constraints entries are numerically consistent with the doc. But the
**entire "Method" section of `ThickenerContext.tsx` describes the wrong algorithm**: it
presents **RAR (residual-based adaptive refinement)** as the method used, in the section
heading and a full paragraph, in both the EN and ES branches. The authoritative doc says the
exact opposite: **RAR was DROPPED for this case** because it destabilised the stiff
degenerate-diffusion residual, and the real recipe is a widened front + Adam→L-BFGS. This is
not a nuance; it is the headline method claim of the case, and it is false against ground
truth. It must be fixed before this case is called done.

Because the false method occupies the section that should carry the true recipe, the doc's
actual training details (net architecture, optimizer schedule, loss weights) are also
**absent** from the app, so the error is a contradiction and a gap at the same time.

---

## Contradictions (inApp vs docSays)

### C1 — The method is RAR (FALSE). Doc: RAR was dropped; method is Adam→L-BFGS. [severity 3]

**File:** `frontend/src/content/cases/ThickenerContext.tsx` — the `<h3>` method heading and
the paragraph under it, BOTH language branches (ES lines 57-67, EN lines 154-165).

**inApp (EN, lines 154-165):**
> "**The method: the sharp front + RAR.** The challenge is the thin front (W=0.10) that also
> moves ... On top of the base fit we apply **RAR** (residual-based adaptive refinement, Wu
> et al., CMAME 2023): the residual is evaluated over a large pool of points and the
> highest-error ones ... are **added, over several rounds**, closing with one L-BFGS polish."

**inApp (ES, lines 57-67):**
> "**El método: el frente afilado + RAR.** ... Sobre el ajuste base aplicamos **RAR** ... se
> añaden los de mayor error ... varias rondas, cerrando con un L-BFGS."

**docSays (Method, lines 27-31):**
> "**Hard-front MMS family on the true nonlinear operator.** Net [3,64,64,64,64,1] tanh
> (DeepXDE), Adam (24000, lr 1e-3) → L-BFGS, loss weights [1,10] ... **(RAR was dropped here
> — it de-stabilised the stiff degenerate-diffusion residual; widening the front to W=0.10 +
> Adam→L-BFGS converges cleanly instead.)**"

The app tells the reader RAR is the technique that makes this case work, over "several
rounds". The doc says RAR was tried and **removed** precisely because it broke on this
degenerate residual, and the fix that actually converged was widening the front to W=0.10 and
running plain Adam→L-BFGS. The app's own "closing with one L-BFGS polish" is the only part
that survives; the RAR framing around it is wrong.

Note this contradiction is **local to `ThickenerContext.tsx`**: `scenarios.ts`,
`results.ts`, and `constraints.ts` do not mention RAR, so a reader skimming the short
content would not catch it. That is exactly the "consistent across the app but false vs the
doc" trap: the method claim is only stated once, and that one statement is wrong.

---

## Coherent items (verified, no action needed)

These were checked against the doc and hold:

- **Governing PDE** `φ_t + ∂_z f_bk(φ) = ∂_z(D(φ) φ_z)` — matches doc line 12.
- **MMS anchor** `φ*(z,t;R) = φ_lo + (φ_hi−φ_lo)·½(1−tanh((z−s)/W))`, `s(t)=z_0−R·t` —
  matches doc line 19. The closed-form derivatives (φ*_z, φ*_t, φ*_zz) are a correct,
  value-adding expansion the doc does not spell out.
- **Flux params** `φ_max=0.66`, `C=5`, hindered Richardson-Zaki — match doc line 15.
- **Gel / degenerate switch** `φ_c=0.23`, tanh-regularized, C¹ residual — match doc lines 16-17.
- **Front width** `W=0.10` and **rate range** `R∈[0.3,0.9]` — match doc lines 21-22.
- **Six variants** R=0.30/0.42/0.54/0.66/0.78/0.90 — match doc line 35.
- **Soft IC/BC** imposed equal to φ* so the reported L2 is the true PINN error — matches doc
  lines 28-29 ("soft-imposed on the whole (z,t,R) cube boundary incl. the t=0 IC").
- **Honesty / scope** synthetic-illustrative, no public (z,t,φ) thickener field, not a plant
  fit — matches doc lines 46-51.
- **results.ts answer** "t = 0.44 at the strongest dose" and "not within the window at the
  weakest" — arithmetically correct: mid-height z=0.5 with z_0=0.9 gives t=0.4/R, so R=0.9→
  t=0.444 and R=0.3→t=1.33 (>1, outside t∈[0,1]). Consistent.
- **results.ts verdict** "sub-percent error vs the reference per regime" — consistent with the
  doc's measured relative-L2 ≤ 0.41 % (line 39).
- **constraints.ts** (MMS descending-front IC; R network input; MMS exact through genuine
  nonlinear flux + degenerate D) — faithful to the doc, no RAR claim.

---

## Depth gaps (real doc content the app omits)

### G1 — The true training recipe is missing. [ties to C1, high value]

**File:** `frontend/src/content/cases/ThickenerContext.tsx`, method section.

The doc's Method line names concrete, checkable details the app never states because the RAR
paragraph occupies that slot:

- Net architecture **[3,64,64,64,64,1]**, tanh, DeepXDE.
- **Adam 24000 steps, lr 1e-3, then L-BFGS**.
- **Loss weights [1,10]**.
- The design rationale: **RAR was dropped, front widened to W=0.10** so plain Adam→L-BFGS
  converges on the stiff degenerate residual.

Replacing the false RAR paragraph with this real recipe fixes C1 and closes G1 in one edit.

### G2 — The measured numbers are never surfaced in the deep Context. [minor]

The doc's Result table (lines 37-44) gives **relative-L2 ≤ 0.41 % across all 6 variants** and
**ONNX parity max-abs 5.1e-7**. The Context says "the reported L2 is the true PINN error" but
never states what that error is. `results.ts` only says "sub-percent". Citing ≤ 0.41 % (and
optionally the ONNX parity, since the case runs live via onnxruntime-web) would ground the
accuracy claim the Context alludes to.

### G3 — `v_0 = −1` and `z_0 = 0.9` are not given numerically. [minor]

The Context writes `f_bk = v_0 φ (1−φ/φ_max)^C` and `s(t) = z_0 − R·t` but lists only φ_max and
C, and never pins v_0 or z_0. The doc gives **v_0 = −1** (line 15) and **z_0 = 0.9** (line 21).
z_0 in particular is load-bearing: it is why the `results.ts` mid-height time is t=0.44 at
R=0.9. Stating "front starts at z_0=0.9" makes the results answer self-consistent for a reader.

---

## Concrete proposed enrichments (faithful to the doc; no invented numbers)

### E1 (fixes C1 + G1) — Rewrite the method section. `ThickenerContext.tsx`

Replace the heading `The method: the sharp front + RAR` (ES: `El método: el frente afilado +
RAR`) and its paragraph. Do NOT mention RAR as the method used. Proposed EN body, grounded in
doc lines 27-31:

> **The method: a widened front + Adam→L-BFGS.** The challenge is the thin, moving front
> (W=0.10), and adding R as a network input grows the problem by a dimension. The net is
> [3,64,64,64,64,1] with tanh activations (DeepXDE); training is Adam for 24000 steps at
> lr 1e-3, then an L-BFGS polish, with loss weights [1,10] on (residual, boundary). Residual
> adaptive refinement (RAR) was tried and **dropped here**: it de-stabilised the stiff
> degenerate-diffusion residual. What converged cleanly instead was widening the front to
> W=0.10 so a plain Adam→L-BFGS schedule resolves the moving interface across the whole R
> family. The hindered flux f_bk and the degenerate diffusion D(φ) with the gel switch are the
> genuine Bürger-Concha terms; the MMS only supplies an exact truth to measure against.

ES equivalent (mirror wording, keep KaTeX InlineMath for W, R, f_bk, D). Drop the "Wu et al.,
CMAME 2023" RAR citation from this case, since RAR is not used.

### E2 (closes G2) — Cite the measured error in the Context. `ThickenerContext.tsx`

In the formalization paragraph where it says "the reported L2 is the true PINN error", append
(doc lines 39, 43): "and it stays **≤ 0.41 % relative-L2 across all six R variants**, well
inside the < 2e-2 acceptance band." Optionally add to `results.ts` verdict: swap "sub-percent
error" for "relative-L2 ≤ 0.41 % across the six variants".

### E3 (closes G3) — Add v_0 and z_0. `ThickenerContext.tsx`

In the Kynch-flux bullet, add `v_0=-1` alongside φ_max=0.66, C=5 (doc line 15). In the
formalization paragraph, state the front starts at `z_0=0.9` (doc line 21) so the mid-height
timing in results.ts (t=0.44 at R=0.9) is self-explanatory.

---

## Files reviewed

- Doc (ground truth): `docs/cases/mine-thickener-settling.md`
- Deep Context: `frontend/src/content/cases/ThickenerContext.tsx` (registry name
  `ThickenerContext`, key `mine-thickener-settling`)
- Short content: `frontend/src/content/scenarios.ts`, `results.ts`, `constraints.ts`
  (entries keyed `mine-thickener-settling`)
