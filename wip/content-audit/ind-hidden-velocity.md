# Content audit: ind-hidden-velocity (in-app content vs authoritative doc)

Date: 2026-07-15
Auditor: content-audit sub-agent
Scope: honest depth + coherence audit of the IN-APP content for PINN-Lab case `ind-hidden-velocity`
against its authoritative doc and its baked measured results. Not a rubber-stamp.

## Sources compared

- AUTHORITATIVE DOC (ground truth): `docs/cases/ind-hidden-velocity.md`
- MEASURED RESULTS (shipped numbers): `data/derived/manifests/ind-hidden-velocity.json`
- IN-APP deep context: `frontend/src/content/cases/HiddenVelocityContext.tsx` (EN + ES branches)
- IN-APP short content: `frontend/src/content/scenarios.ts`, `frontend/src/content/results.ts`,
  `frontend/src/content/constraints.ts` (entries keyed `ind-hidden-velocity`)

## Verdict

**Coherent-with-doc: NO. Severity 3 (a numeric contradiction against the baked manifest, plus an internal
term-count incoherence).** The in-app content is otherwise genuinely deep: `HiddenVelocityContext.tsx`
carries real theory (closed-form flow truth, the transport PDE + the full loss functional, the FD
stability checks with the Péclet number, the load-bearing steady-flow residuals, the swept-mask
identifiability honesty, scope/limits). scenarios/results/constraints are substantive and mostly faithful
to the measured numbers. But two coherence defects and several depth gaps must be fixed.

Ratings: contextDepth = **rich**; scenarioResultsDepth = **rich** (but with one contradicted number);
severity = **3**.

## Contradictions (in-app text vs doc / baked manifest)

### C1 (numeric, must fix) — held-out dye error stated as 0.8%, baked value is ~0.5%
- IN-APP (`frontend/src/content/results.ts`, line 119 `verdict_en` and line 120 `verdict_es`):
  "the reconstructed dye checks out on 160 never-trained samples **(0.8% error)**" /
  "el tinte reconstruido valida en 160 muestras nunca entrenadas **(0.8% de error)**".
- DOC / MANIFEST says: the shipped held-out metric is `dye_holdout_rmse = 0.004751`
  (`data/derived/manifests/ind-hidden-velocity.json`; the doc's Result section names this exact key,
  "the held-out dye RMSE (`dye_holdout_rmse`, from the 160 never-trained samples via the exported ONNX)").
  0.004751 of the unit-max dye is ~0.48%, i.e. ~0.5% — and it lands right at the 0.5%-of-max noise floor.
  There is no 0.8% anywhere in the manifest.
- Why it matters: this is the one number in results.ts that does not trace to the baked manifest. It reads
  as a stale or mistyped value. Fix to ~0.5% (or state the raw RMSE 0.0048) so the app matches the shipped
  artifact.

### C2 (claim / internal incoherence, must fix) — "the loss couples three terms" contradicts both the doc and the equation printed one line below it
- IN-APP (`frontend/src/content/cases/HiddenVelocityContext.tsx`, line 104 EN / line 34 ES):
  "the loss couples **three terms**: the transport residual, the incompressibility residual, and the dye
  data" / "la pérdida acopla **tres términos**: el residual de transporte, el residual de incompresibilidad
  y los datos de tinte".
- DOC says: "The loss couples **five terms** (`loss_weights = [1, 1, 1, 1, 60]`)" and prints
  L = transport + incompressibility + ||u_t||^2 + ||v_t||^2 + lambda*data.
- Internal incoherence: the very next line in the Context renders the five-term equation
  (`...+\big\|u_t\big\|^2+\big\|v_t\big\|^2+\lambda\sum...`). Calling it "three terms" undercounts by two
  and, worse, hides the `u_t = v_t = 0` steadiness residuals that this case's own narrative calls
  "load-bearing" (the ones that move the recovered current from 38-60% off to 16%). The prose undersells the
  exact mechanism the case is built to demonstrate. Present in both EN and ES.

## Depth gaps (real doc content the app omits and should include)

### G1 (moderate) — the primary full-grid metric value (27.8%) is never shown
The doc calls the full-grid `l2_relative` "the primary metric" that "honestly includes both" regions.
The app shows the two split numbers (16.4% swept, 37.5% dead) but never the full-grid 27.8%
(`l2_relative = 0.278`). The honesty framing in the Context ("The global L2 is published as-is even though
the dead zones inflate it") states that a number exists but withholds its value. Give the number.

### G2 (moderate) — the soft-vs-hard incompressibility engineering caveat is dropped
Doc: "Incompressibility is soft rather than a stream-function hard constraint because the exported ONNX must
emit (u, v, c) directly (a psi-derivative formulation does not export cleanly)." This is an honest
export/engineering trade-off. Neither the Context nor results.ts mentions it. It belongs in the Context
formalization or scope note.

### G3 (minor) — swept-mask threshold value omitted
Doc: the mask is "where max_t |grad c_FD| exceeds **5% of its max**". The Context (EN line 124 / ES line 54)
only says "exceeds **a threshold**". State the 5% so the honesty claim is concrete and reproducible.

### G4 (minor-moderate) — network + training recipe absent
Doc gives: FNN `[3, 64, 64, 64, 64, 3]` mapping (x,y,t) -> (u,v,c); `loss_weights = [1,1,1,1,60]`
(so lambda = 60); Adam 1.5e4 steps then L-BFGS; seed 42. The Context writes lambda symbolically but never
states lambda = 60, the architecture, or the optimizer schedule. One sentence would add real reproducibility
depth.

### G5 (minor) — the central-vs-upwind FD justification is truncated
Doc: central differencing is legitimate because cell Peclet ~0.59 < 2, and an upwind scheme would inject
numerical diffusion of order `A*dx/2 ~ 0.006`, comparable to D = 0.02 itself, corrupting the reference. The
Context cites the Peclet number but drops the "why not upwind" consequence, which is the actual reason the
scheme choice matters.

### G6 (minor) — dye patch variance and the "~3/4 ring" motion omitted
Doc: Gaussian patch has `s0^2 = 0.006` and "sweeps roughly three quarters of the ring around the vortex
center by t=1". The Context gives the center (0.5, 0.2) but not the variance or the sweep extent (the sweep
extent is what motivates the ~2/3 swept-area fraction and the dead zones).

Note on non-issues (checked, NOT contradictions):
- Swept fraction "~2/3" (Context) vs 67.3% (manifest) vs "~67%" (doc): consistent approximations.
- "16%" (results) vs 16.4% (manifest): rounding, fine. "37%" (results) vs 37.5% (manifest estimate):
  rounding, fine. Center (0.47, 0.50) / "3% off": matches manifest exactly.
- The full 5-term loss EQUATION in the Context matches the doc verbatim; only the "three terms" PROSE label
  is wrong (see C2).
- scenarios.ts (`ind-hidden-velocity`): "~640 sparse noisy dye samples... NO velocity data anywhere, no
  boundary or initial conditions assumed": faithful to the doc's selling point. No issue.

## Concrete proposed enrichments (grounded in doc quotes; no invented numbers)

### E1 — fix C1 in `frontend/src/content/results.ts` (verdict_en line 119, verdict_es line 120)
Replace "(0.8% error)" / "(0.8% de error)" with the baked value:
- EN: "...the reconstructed dye checks out on 160 never-trained samples (**~0.5% error, at the noise floor**)."
- ES: "...el tinte reconstruido valida en 160 muestras nunca entrenadas (**~0.5% de error, en el piso de
  ruido**)."
(Source: `dye_holdout_rmse = 0.004751`; noise is "0.5% of max" per the doc's Problem section. If a bare
number is preferred, use "0.48%".)

### E2 — fix C2 in `frontend/src/content/cases/HiddenVelocityContext.tsx` (EN line 104, ES line 34)
Replace "the loss couples three terms: the transport residual, the incompressibility residual, and the dye
data" with a five-term statement that names the steadiness residuals:
- EN: "the loss couples **five terms** (weights `[1, 1, 1, 1, 60]`): the transport residual, the
  incompressibility residual, the two steadiness residuals `u_t = v_t = 0`, and the dye-data term:"
- ES: "la pérdida acopla **cinco términos** (pesos `[1, 1, 1, 1, 60]`): el residual de transporte, el de
  incompresibilidad, los dos residuales de estacionariedad `u_t = v_t = 0`, y el término de datos de tinte:"
(Source doc: "The loss couples five terms (`loss_weights = [1, 1, 1, 1, 60]`)". This also supplies lambda = 60
for G4 and makes the following equation and the "load-bearing" paragraph self-consistent.)

### E3 — G1, add the primary full-grid number, in `HiddenVelocityContext.tsx` honesty section
(EN after line 126 / ES after line 56), append to the "global L2 is published as-is" sentence:
- EN: "...never dressed up: the full-grid current error is **27.8%**, versus **16.4% inside the swept region**
  and **37.5%** in the dead zones."
- ES: "...nunca se maquilla: el error de corriente en toda la grilla es **27.8%**, frente a **16.4% dentro de
  la región barrida** y **37.5%** en las zonas muertas."
(Source: `l2_relative = 0.278`, `speed_rel_rmse_swept = 0.1644`, `speed_rel_rmse_dead = 0.3743`.)

### E4 — G2, add the soft-incompressibility caveat, in `HiddenVelocityContext.tsx` formalization
(EN after the loss equation ~line 107 / ES ~line 37):
- EN: "Incompressibility is imposed **softly** (a residual) rather than as a hard stream-function constraint,
  because the exported ONNX must emit (u, v, c) directly: a psi-derivative formulation does not export
  cleanly."
- ES: "La incompresibilidad se impone **suave** (un residual) y no como restriccion dura de funcion de
  corriente, porque la ONNX exportada debe emitir (u, v, c) directamente: una formulacion en derivadas de psi
  no exporta limpio."
(Source doc, Method section, verbatim rationale.)

### E5 — G3, state the mask threshold, in `HiddenVelocityContext.tsx` (EN line 124 / ES line 54)
Change "exceeds a threshold" / "supera un umbral" to "exceeds **5% of its max**" / "supera el **5% de su
maximo**". (Source doc, "Honest identifiability" section.)

### E6 — G4/G5, add one reproducibility sentence to the Context formalization
- EN: "Concretely: an FNN `[3, 64, 64, 64, 64, 3]`, trained with Adam (1.5e4 steps) then L-BFGS, seed 42.
  The dye truth uses central differencing (legitimate because cell Peclet `A*dx/D ~ 0.59 < 2`; an upwind
  scheme would inject numerical diffusion `~ A*dx/2 ~ 0.006`, comparable to D itself)."
- ES analog. (Source doc, Method + "dye truth" sections.)

## Files that need edits
- `frontend/src/content/results.ts` — C1 (verdict_en/verdict_es holdout number).
- `frontend/src/content/cases/HiddenVelocityContext.tsx` — C2 (term count, both branches) + G1/G2/G3/G4/G5
  enrichments (both branches).
- No edit needed to `scenarios.ts` or `constraints.ts` (faithful as-is).
