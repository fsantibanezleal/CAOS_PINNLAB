# Content audit: `mine-comminution-pbe` (in-app vs authoritative doc)

Date: 2026-07-15. Auditor: subagent (honest depth+coherence pass, not a rubber-stamp).

Sources compared:
- Ground truth doc: `docs/cases/mine-comminution-pbe.md`
- Measured artifact (seed 42): `data/derived/manifests/mine-comminution-pbe.json`
- Pipeline source: `data-pipeline/pinnlab/cases/mine_comminution_pbe.py`
- In-app deep Context: `frontend/src/content/cases/ComminutionContext.tsx` (EN + ES)
- In-app short content: `frontend/src/content/scenarios.ts`, `results.ts`, `constraints.ts`

## Verdict

**Severity 3 (contradiction).** `coherentWithDoc = false`.

The Context is genuinely **rich** prose (governing PDE, exact Green's-function anchor, variable table, full PBE integral, Fokker-Planck reduction, scope, per-variant walk, viz guide) but it **contradicts the doc on the defining method mechanism**: it says the initial and boundary conditions are "imposed softly and weighted", whereas the case actually uses a **hard-IC output transform** (`u = g0(s) + t*N`, IC exact by construction) with **no boundary loss at all** (`num_boundary=0`, `bcs=[]`). It also omits the output-transform equation entirely. Secondary: the `results.ts` verdict headlines a flat "0.4% field error" and drops the doc's honest worst-corner caveat (g=0.6 at ~2%, Pe~50), and two short-content entries mislabel the reduced surrogate as the full "breakage model / breakage law".

The short-content numbers themselves are NOT fabricated: the passing-fraction curve (0.5% -> 63%) and the 0.4% all trace to the manifest (`estimate.items` and `comparison.summary`). The problem is selective framing, not invented numbers.

---

## Contradictions (inApp vs docSays)

### C1 (severity 3) - Method: IC is HARD (output transform), not "soft/weighted"; and there is NO BC term
- **inApp** (`ComminutionContext.tsx`, Formalization para, EN L136-139 / ES L48-51):
  "The PINN minimises the transport residual at collocation points, with the IC (the feed) and BC (n* on the boundary) **imposed softly and weighted**: so the network genuinely learns the interior field and the reported L2 is the true PINN error."
- **docSays** (`docs/.../mine-comminution-pbe.md` Method): "**Drift-diffusion PINN with a hard-IC output transform** (the burgers/flotation pattern): u_theta = g0(s) + t N_theta(s,t,g) where g0(s)=n*(s,0) is the (grind-rate-independent) initial Gaussian, **so the IC is exact** and the net learns the interior evolution."
- **Confirmed by pipeline** (`mine_comminution_pbe.py`): `output_transform` returns `g0 + tt * n` (IC exact by construction), and `dde.data.PDE(..., num_boundary=0, ...)` with an empty BC list, `solution=analytic`. There is no soft/weighted IC loss and no BC loss whatsoever. The Context invents a soft weighted BC term that does not exist.
- Impact: this is the single most important architectural fact of the case (the burgers/flotation hard-IC pattern), and the app states its opposite in BOTH languages.

### C2 (severity 2) - Verdict headlines "0.4%" and drops the honest worst-case band
- **inApp** (`results.ts`, `verdict_en` L209): "**0.4% field error** vs the reference; the passing curve is monotone and consistent across the whole g family."
- **docSays** (Result + Honesty): "relative-L2 vs exact **<= 2.0% across all 6 variants** ... The high-grind corner **g=0.6 is advection-leaning (Peclet ~50) and sits at ~2%** ... the lower-grind variants are well under 1%."
- **Manifest per-variant l2_relative**: g00 0.41%, g12 0.53%, g24 0.89%, g36 1.33%, g48 1.79%, **g60 2.02%**. The "0.4%" equals only the best corner (g=0, and the `comparison.summary.adapted_vs_std = 0.0041`). Presenting a single "0.4%" for "the whole g family" understates the family (up to 2.02%) and erases the doc's foregrounded honest edge.
- Not a fabricated number (0.4% traces to `comparison.json`), but a coherence/honesty gap: the app's flagship verdict number is the most optimistic corner with the caveat removed.

### C3 (severity 1) - "breakage law" / "breakage model" overstates the reduced surrogate
- **inApp**: `scenarios.ts` `measured_en` (L128): "The feed size distribution and **the breakage law** with grind rate g as the family axis ..."; `results.ts` `assumptions_en` (L205): first assumption is "**population-balance breakage model**".
- **docSays** (Honesty): "The drift-diffusion is a **deliberate reduction** of the comminution PBE ... but **drops the selection/breakage integral**." The pipeline is a Fokker-Planck drift-diffusion, not the breakage kernel.
- Impact: both short-content entries frame the case as if it carries the breakage law/model, which is exactly what the doc says it does NOT. Minor but it undercuts the honesty framing that the rest of the app maintains.

---

## Depth gaps (real doc content the app omits)

- **G1 - The output-transform equation** `u_theta = g0(s) + t N_theta(s,t,g)`, `g0(s)=n*(s,0)`. This is the method's core and is completely absent from the Context (worse, contradicted per C1). Doc states it verbatim.
- **G2 - The accuracy-vs-grind honest caveat.** The Context's otherwise-rich "Scope & assumptions" and "What each variant shows" never say the error grows with g, nor name the g=0.6 / Peclet~50 worst corner (~2%). The doc foregrounds it; the manifest confirms the monotone 0.41% -> 2.02% climb. This is the honest content the app most needs and currently has nowhere (Context omits it; results.ts verdict buries it under a flat 0.4%).
- **G3 - Training/architecture specifics.** Net `[3,48,48,48,48,1]` tanh (DeepXDE), Hypercube `(s,t,g)`, Adam 15000 @ lr 1e-3 -> L-BFGS, `num_domain=5000 / num_test=8000`. Doc + manifest carry these; Context's method section names none of them (lower priority, but the "reduced population balance" section could name net + optimizer).
- **G4 - ONNX parity 9.5e-7.** A measured Result row in the doc. Context mentions `onnxruntime-web` (the runtime) but not the parity figure that certifies the in-browser net matches the trained model. Minor.

---

## Concrete proposed enrichments (faithful to doc; no invented numbers)

### E1 - Fix C1 + close G1: rewrite the Formalization closing sentence (`ComminutionContext.tsx`)
Edit BOTH branches. Replace the "IC ... and BC ... imposed softly and weighted" sentence.

EN (replace L136-139 tail):
> The PINN uses a **hard-IC output transform** (the burgers/flotation pattern): `u_theta = g0(s) + t * N_theta(s,t,g)` with `g0(s) = n*(s,0)` the grind-rate-independent initial Gaussian, so **the initial condition is satisfied exactly by construction** and the network only has to learn the interior evolution. It minimises the transport residual at collocation points and is anchored to the exact field (`solution = analytic`), with **no separate boundary-loss term**, so the reported relative-L2 is the true PINN error.

ES (replace L48-51 tail), same content: "**transformada de salida con IC dura** ... `u = g0(s) + t * N(s,t,g)`, con `g0(s)=n*(s,0)` ... la condicion inicial se cumple de forma **exacta por construccion** ... sin un termino de perdida de frontera aparte ...".

Grounded in doc: "hard-IC output transform ... u_theta = g0(s) + t N_theta(s,t,g) where g0(s)=n*(s,0) ... so the IC is exact"; and pipeline `num_boundary=0`, `solution=analytic`.

### E2 - Fix C2: rewrite `results.ts` `verdict_en` / `verdict_es`
> **verdict_en:** "0.4% field error at g=0, rising to ~2% at the hardest grind (g=0.6, advection-leaning, Peclet~50); relative-L2 stays <=2% across all six grind rates. The passing curve is monotone and consistent across the whole g family. Valid as the operating-chart shape; real mills add energy-size laws (Bond/JK) on top."

Grounded in doc ("<= 2.0% across all 6 variants ... g=0.6 ... Peclet ~50 ... sits at ~2% ... lower-grind variants well under 1%") and manifest per-variant l2 (0.0041 -> 0.0202). Mirror in `verdict_es`.

### E3 - Close G2 in the Context: add one sentence to "What each variant shows" (`ComminutionContext.tsx`)
Append to the EN L165-171 para (and ES equivalent):
> Accuracy is not uniform across the sweep: the low-grind variants sit well under 1% relative-L2, while the high-grind corner g=0.6 becomes advection-leaning (Peclet ~ 50) and reaches ~2%, the honest edge of the expected band.

Grounded in doc Result/Honesty + manifest.

### E4 - Fix C3: `results.ts` `assumptions_en` first item and `scenarios.ts` `measured_en`
- `assumptions_en[0]`: "population-balance breakage model" -> "**drift-diffusion (Fokker-Planck) reduction of the PBE breakage operator**". (`assumptions_es`: "reduccion de deriva-difusion (Fokker-Planck) del operador de rotura de la PBE".)
- `scenarios.ts` `measured_en`: "... and the breakage law with grind rate g ..." -> "... and the **size-transport (drift-diffusion) reduction of the breakage operator** with grind rate g ...". Mirror `measured_es`.

Grounded in doc Honesty ("deliberate reduction ... drops the selection/breakage integral") and Method ("the Fokker-Planck reduction of the breakage operator").

### E5 (optional, close G3/G4): one clause in Context "reduced population balance" section
Add: "the net is a small tanh MLP `[3,48,48,48,48,1]` (DeepXDE) trained Adam->L-BFGS on a `(s,t,g)` hypercube; the exported ONNX matches the trained model to 9.5e-7 (max abs)." Grounded in doc Method + Result and manifest `onnx.parity_max_abs`.

---

## What is already coherent (do not touch)
- `constraints.ts` entry (ic Gaussian at s0 / param grind rate g network input / anchor advected-diffused Gaussian exact any g): fully faithful.
- Context PDE `n_t + (-g) n_s = D n_ss`, exact solution, s0=0.8, sigma0^2=0.01, D=0.012, g in [0,0.6], domain, 81x81 grid, mass-conservation peak decay, sigma^2=sigma0^2+2Dt: all match doc + pipeline. (Context's use of sigma0^2 for the initial variance is a clarity improvement over the doc's ambiguous "s0^2=0.01".)
- `results.ts` `answer` passing-fraction curve (0.5% -> 63%): grounded in manifest `estimate.items` (g00 0.5% ... g60 63.1%). Real, keep.
- `results.ts` `calculates` field description (size x time, coarse-right to fine-left): matches Context and field_axes.
