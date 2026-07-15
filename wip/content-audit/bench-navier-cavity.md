# Content audit: bench-navier-cavity (in-app vs authoritative doc)

- Case: `bench-navier-cavity` (steady incompressible Navier-Stokes, lid-driven cavity, u-v-p PINN)
- Authoritative doc: `docs/cases/bench-navier-cavity.md`
- In-app deep context: `frontend/src/content/cases/NavierCavityContext.tsx` (Context name resolved from `registry.tsx`)
- In-app short content: `frontend/src/content/scenarios.ts`, `results.ts`, `constraints.ts` (keyed `bench-navier-cavity`)
- Audit date: 2026-07-15

## Verdict

**coherentWithDoc = false. severity = 3 (one hard numeric contradiction).**

The deep Context is genuinely **rich** and, on physics, almost entirely faithful to the doc: it carries the three
coupled residuals, the boundary-condition equation, the weighted loss equation, Re=100 / nu=0.01 / rho=1, the
regularized lid `16 x^2 (1-x)^2`, the `p(0,0)=0` gauge, and a well-argued single-variant justification. The
constraints entry is fully coherent. The scenario is a solid operator framing.

But there is one **material contradiction**: the Context's honesty paragraph states the PINN reaches
"a relative error of **a few percent** against Ghia," whereas the doc's measured, explicitly-headline number is
**relative-L2 = 0.1675 (~17%)**, driven by a coarse v-centerline (**0.2178**). "A few percent" understates the
doc's own reported error by roughly 3-4x, and it inverts the doc's central honesty point ("the gap to Ghia is the
headline number, not a footnote"). The `results.ts` verdict compounds this by surfacing only the flattering
`velocity RMSE 0.05` and reframing the caveat around "fine pressure details" (a caveat the doc does not make),
while omitting the 17% headline and the specific v=0.22 coarseness the doc insists on.

## Contradictions (inApp vs docSays)

### C1 (hard, severity 3) — "a few percent" vs measured ~17% relative-L2
- **inApp** (`NavierCavityContext.tsx`, EN method-honesty para, L204-207; ES L81-85):
  "a soft primitive-variable PINN on CPU reaches **a relative error of a few percent** against Ghia: competent and
  qualitatively correct, but not of spectral accuracy." / ES: "alcanza un **error relativo de unos pocos por
  ciento** frente a Ghia".
- **docSays** (Result table + prose, L45-56): "relative-L2 vs Ghia (mean of u,v centerlines) **0.1675**";
  "u-centerline rel-L2 0.1173", "v-centerline rel-L2 0.2178"; and in prose: "**~17 % relative-L2 against Ghia is
  reduced fidelity, not a publication-grade cavity solve** ... the centerline match, especially v, at 0.22, is
  coarse". The word "relative" in the Context maps to the doc's relative-L2, which is 17%, not "a few percent".
  The Honesty section reinforces: "the gap to Ghia is the headline number, not a footnote."

### C2 (soft / honesty-placement) — flattering RMSE only, doc-unsupported pressure caveat
- **inApp** (`results.ts`, verdict_en, L89): "Validated against the published Ghia et al. benchmark centerlines
  (**velocity RMSE 0.05**): the flow structure is right and the eye location is usable for mixing questions.
  **Fine pressure details are NOT resolved at this budget**: use the classical solver for those."
- **docSays**: The headline is the **relative-L2 0.17**, and the honest coarseness the doc calls out is the
  **velocity centerline (especially v at 0.22)**, not pressure. The doc never states pressure is unresolved; the
  Context itself describes a clean pressure structure ("pressure field shows a minimum at the vortex core and
  maxima where the stream impinges on the walls", L234-235). RMSE 0.05 is itself defensible (doc Validation
  section: u RMSE 0.053, v RMSE 0.029), but selecting only that metric while dropping the 17% headline makes the
  verdict rosier than the doc, and the pressure caveat is an app invention not grounded in the doc.

### Note (not a contradiction) — vortex-eye coordinate is app-introduced
- `results.ts` answer_en (L87): "The vortex eye sits at (x, y) = **(0.61, 0.75)**". The doc gives **no** vortex-eye
  coordinate. The value matches the canonical Ghia Re=100 primary-vortex center (~0.617, 0.734), so it is
  plausible and consistent with the Context's "center shifted toward the upper-right corner". Flag only to
  confirm it is actually read from the pipeline output (as the copy claims: "read from the reconstructed flow's
  circulation") and not hardcoded from the literature.

## Depth gaps (real doc content the app omits)

1. **The headline number itself.** Neither the Context nor `results.ts` states the doc's headline `0.1675 (~17%)`,
   nor the u/v split (`0.1173` / `0.2178`). The Context says only "the relative-L2 error ... is the reported
   metric" (L238) without the value. The doc's most important honesty fact is absent from the deep content.
2. **Which centerline is coarse and why.** The doc's specific, honest mechanism, "the centerline match, especially
   v, at 0.22, is coarse because the CPU lane cannot afford the iteration count a sharp Re=100 cavity needs", is
   omitted. The Context's honesty paragraph gestures at a ceiling but with the wrong magnitude.
3. **Training/architecture specifics from doc Method (L25-35).** The Context says "a single FNN" and "L-BFGS
   polish" but omits: architecture `2 -> [64] x 5 -> 3, tanh, Glorot`; optimizer `Adam 20,000 steps, lr 1e-3`
   then L-BFGS; collocation `2601 interior + 400 boundary`. These are concrete, quotable, and raise depth.
4. **The named GPU alternative.** The doc names the horizon lane explicitly ("A GPU lane (**PhysicsNeMo**) would
   tighten this", L56). The Context says "a GPU pass" generically; naming PhysicsNeMo is stronger and matches the
   doc.
5. **The Diagnostics/Validation provenance detail (doc L73-78).** The doc records that "a rushed finite-difference
   cavity solve diverged and was rejected"; the Ghia table is the anchor precisely because no robust field solver
   was trusted. Optional, but it is real honesty content the app could surface in the results/verdict.

## Concrete proposed enrichments (grounded in doc quotes; no invented numbers)

### E1 - Fix C1 in `NavierCavityContext.tsx` (replace the "a few percent" honesty paragraph, both EN and ES)

EN (replace L204-207 body):
> It is worth being honest about the accuracy ceiling. On the CPU lane this soft primitive-variable PINN scores a
> relative-L2 of **0.1675 (~17%)** against the Ghia centerlines (u-centerline **0.117**, v-centerline **0.218**).
> The network captures the primary vortex and the qualitative corner-eddy structure, but the centerline match,
> **especially the vertical velocity v at 0.22, is coarse**, because the CPU lane cannot afford the iteration
> count a sharp Re=100 cavity needs. This is **reduced fidelity, not a publication-grade cavity solve**, and the
> gap to Ghia is reported as the headline number, not a footnote. Stronger formulations (stream-function-vorticity,
> hard incompressibility constraints, or a GPU pass on **PhysicsNeMo**) would tighten it; they are cited as the
> horizon, not claimed here.

ES (mirror, replace L81-85 body): keep the same numbers `0.1675 (~17%)`, `u 0.117`, `v 0.218`, wording
"fidelidad reducida, no una solucion de precision de publicacion", naming PhysicsNeMo.

### E2 - Fix C2 in `results.ts` (verdict_en / verdict_es for `bench-navier-cavity`)

EN:
> Validated against the published Ghia et al. Re=100 benchmark centerlines: **relative-L2 0.17** (velocity RMSE
> ~0.05), the honest headline for the CPU lane. The primary vortex and corner-eddy structure are right and the eye
> location is usable for mixing questions, but the centerline match, **especially the vertical velocity v (0.22),
> is coarse: reduced fidelity, not a publication-grade solve**. For sharper detail use the classical solver or a
> GPU lane.

(Drop the doc-unsupported "fine pressure details are NOT resolved" claim, or demote it below the velocity-fidelity
headline, since the doc's stated coarseness is the v-centerline, not pressure.)

### E3 - Add training specifics to the Context method section (doc L25-35)

Add one bullet after the "Coupled outputs" bullet, both langs:
> **Architecture and training:** a `2 -> [64] x 5 -> 3` tanh/Glorot network, optimized with Adam (20,000 steps,
> lr 1e-3) then an L-BFGS polish, over 2601 interior + 400 boundary collocation points.

### E4 - Optional: name the loss-weight vector explicitly (doc L31)

In the "Loss weighting" bullet, the doc's exact vector is `loss_weights = [1,1,1,10,10,10,10,10]` (3 residuals at
1; lid-u, lid-v, wall-u, wall-v, pressure-gauge at 10). Stating the vector makes the "x10" concrete and matches
the doc verbatim.

## What is already coherent (no change needed)

- Governing equations, BC equation, weighted-loss equation: match doc L14-16, L44-48, L73.
- Re=100, nu=0.01, rho=1, regularized lid `16 x^2 (1-x)^2`, `p(0,0)=0` gauge, 101x101 grid: all match doc.
- `constraints.ts`: "no-slip walls + regularized moving lid (soft, 10x weight)", "pressure gauge p(0,0)=0",
  "Ghia 1982 Re=100 centerlines" -> fully coherent with doc Method loss weights and Result anchor.
- `scenarios.ts`: operator framing (stirred tank / recirculation / Ghia-graded) is substantive and accurate.
- Single-variant justification (no Reynolds sweep because no analytic anchor exists; Ghia is discrete) is a strong,
  faithful elaboration of the doc's honesty stance.
