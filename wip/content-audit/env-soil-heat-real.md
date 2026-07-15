# Content audit — env-soil-heat-real (in-app vs authoritative doc)

Audited: 2026-07-15
Authoritative doc: `docs/cases/env-soil-heat-real.md`
In-app files:
- Deep context: `frontend/src/content/cases/SoilHeatRealContext.tsx` (EN + ES)
- Short scenario: `frontend/src/content/scenarios.ts` (`env-soil-heat-real`)
- Short results: `frontend/src/content/results.ts` (`env-soil-heat-real`)
- Constraints chips: `frontend/src/content/constraints.ts` (`env-soil-heat-real`)

## Verdict

**Coherent and genuinely deep, with real (not hollow) content — but it omits several distinctive
measured facts from the flagship's own doc, and carries two minor numeric nits.** Severity 2 (real
gaps, no hollow content, no substantive contradiction).

The Context component is rich: it transcribes the governing equation `T_t = alpha T_zz`, the
non-dimensionalisation `alpha = kappa L^2/tau`, the composite PINN loss with its three weighted
terms, the `log kappa` trainable-variable trick, the Adam-only rationale, real point-set Dirichlet
boundaries, the physical diffusivity band 0.2-0.8 mm^2/s, the out-of-sample metric philosophy, an
honest scope/out-of-scope block (moisture-dependent alpha, infiltration advection, freeze/thaw
latent heat), and a viz-reading guide. The scenario/results/constraints entries are substantive:
real motivation (ground-source heat pumps, buried cables, permafrost), the two-boundary / three
held-out split, the recovered alpha, and an honest verdict ("one effective value, not a layered
profile"). Nothing here is filler.

What it is NOT is complete against the doc's *measured record*: the doc's damping-with-depth
amplitude table, the headline relative-L2 (6.9%), the physical explanation of the error pattern
(largest at 10 cm because nearest the noisy surface, smallest at 50 cm because deepest/smoothest),
and the doc's central honesty caveat (the 5 cm boundary carries synoptic weather noise a single-alpha
diffusion cannot resolve, visible in the boundary loss, reported not hidden) are all absent from the
in-app content. Those are exactly the concrete, real-data facts that make this the flagship case, so
their omission is the main finding.

## Contradictions / coherence nits (inApp vs docSays)

1. **Per-depth held-out RMSE — app aligns with only one of the doc's two conflicting lines.**
   - inApp (`results.ts` `verdict_en`/`verdict_es`, and `SoilHeatRealContext` states "~1 degC"):
     `reconstructed to ~1.0 degC (1.24/1.05/0.75)`.
   - docSays: the **Result table** (the headline "measured, seed 42" block) lists
     `10 cm 1.26 . 20 cm 1.06 . 50 cm 0.75`, overall RMSE **1.05 degC**; the doc's own
     **Validation section** lists `RMSE 1.24 / 1.05 / 0.75 degC`.
   - Assessment: the **doc contradicts itself** (1.26/1.06 vs 1.24/1.05). The app matches the
     Validation-section line but not the headline Result table. Not an app-invented number, but the
     app should be reconciled to whichever the baked artifact actually produced, and the doc's two
     lines should be made identical. Minor.

2. **Recovered alpha precision.**
   - inApp (`results.ts` `answer_en`): `0.304 mm^2/s`. (`constraints.ts` and the Context both say
     `0.30 mm^2/s` / `~0.3 mm^2/s`.)
   - docSays: `0.30 mm^2/s` everywhere (Result table and prose).
   - Assessment: 0.304 rounds to 0.30, so no real contradiction, but the app introduces a third
     significant figure that never appears in the doc. Either the doc should adopt 0.304 (if that is
     the baked value) or the app should report 0.30, so all four surfaces agree. Minor.

No substantive contradiction exists on method, mechanism, boundaries, held-out split, the alpha band,
or the out-of-sample framing: those are fully coherent across doc and app.

## Depth gaps (real doc content the app omits)

G1. **The measured damping/phase-lag amplitude table is absent.** The doc quantifies the flagship's
whole physical story with real numbers:

| depth | min ... max (degC) | seasonal amplitude |
|-------|--------------------|--------------------|
| 5 cm  | -2.6 ... 27.0 | ~30 degC (full swing) |
| 50 cm | 1.0 ... 24.2 | ~23 degC (damped) |
| 100 cm | 3.0 ... 21.1 | ~18 degC (more damped + lagged) |

The Context describes damping/phase-lag only qualitatively ("large near surface, flattens at depth").
None of the measured amplitudes appear anywhere in-app. This is the concrete real-data evidence the
flagship should show.

G2. **The headline relative-L2 (6.9%) is never stated in-app.** The doc lists
`held-out relative-L2 vs REAL temps = 6.9%` as a top-line measured metric. The Context mentions "the
relative-L2 and the RMSE in degrees Celsius" as the honest metrics but gives no value; `results.ts`
and `constraints.ts` report only RMSE. The 6.9% figure is missing.

G3. **The physical explanation of the error pattern is missing.** The doc: "The error is largest at
10 cm (nearest the noisy near-surface boundary) and smallest at 50 cm (deepest, smoothest) -
physically exactly what you expect." The app's `results.ts` verdict lists the monotone numbers
1.24/1.05/0.75 but never says *why* they fall that way. That reasoning is the sanity-check that turns
three numbers into a physics argument.

G4. **The doc's central honesty caveat is not reflected.** The doc's Honesty section: "The
near-surface (5 cm) boundary carries synoptic weather noise that a single-alpha diffusion cannot fully
resolve - that residual is visible in the 5 cm boundary loss and is reported, not hidden; it does not
contaminate the held-out interior score." The Context's scope block lists out-of-scope *physics*
(moisture, advection, freeze/thaw) but never states this specific, honest limitation about the
near-surface boundary noise and where it shows up. `results.ts` verdict is honest but only says "one
effective value, not a layered profile". This is the most substantive honesty omission.

G5. **The "recovered alpha lands in the textbook band = independent sanity check" argument is
weakened.** The doc frames the band check as the decisive evidence the inverse "found physics, not a
curve fit". The Context mentions the band (0.2-0.8) and calls an out-of-band value "a red flag", which
is good, but it never closes the loop with the doc's exact framing that the *landing inside* the band
is the independent confirmation. Minor.

G6. **Reproducibility / provenance of the real data is thin in-app.** The doc stresses this is
"real, vendored, reproducible": schema `pinnlab.dataset.uscrn/v1`, one-time fetch then offline,
"0 % missing in 2021", three full seasonal cycles, exercising the ingestion data contract. The
Context names the station and 2019-2021 but omits the vendored/offline/0%-missing angle that is a
selling point of the flagship. Minor (arguably belongs more in the doc than the App context).

## Concrete proposed enrichments (grounded in doc quotes)

E1 (addresses G1) — add the measured amplitudes to the Context "What the benchmark shows" paragraph.
File: `frontend/src/content/cases/SoilHeatRealContext.tsx` (both EN and ES branches).
Append after the damped/phase-lagged sentence, EN:
> "The measured signal is textbook: the 5 cm sensor swings across roughly 30 degC over the year
> (-2.6 to 27.0 degC), the 50 cm sensor only ~23 degC (1.0 to 24.2 degC), and the 100 cm sensor ~18
> degC (3.0 to 21.1 degC) and lagged by weeks. That top-to-bottom shrink in amplitude is what a single
> alpha has to explain."
ES mirror with the same numbers.

E2 (addresses G2) — state the relative-L2 alongside the RMSE.
File: `frontend/src/content/results.ts`, `env-soil-heat-real.verdict_en`/`verdict_es`.
Change the metric clause to include the doc's headline number, e.g. append:
> "...reconstructed to ~1.0 degC (1.24/1.05/0.75), a 6.9% relative-L2 against the real interior
> temperatures."
Grounded in doc: `held-out relative-L2 vs REAL temps = 6.9%`.

E3 (addresses G3) — add the error-pattern reasoning to the results verdict.
File: `frontend/src/content/results.ts`, `env-soil-heat-real.verdict_en`/`verdict_es`.
Append one sentence, EN:
> "The error is largest at 10 cm (nearest the noisy near-surface boundary) and smallest at 50 cm
> (deepest and smoothest): physically exactly what you expect, which is itself a check on the fit."
Verbatim-grounded in the doc's Result prose.

E4 (addresses G4) — add the near-surface-noise honesty caveat to the Context scope block.
File: `frontend/src/content/cases/SoilHeatRealContext.tsx`, "Scope & assumptions" paragraph (EN + ES).
Add, EN:
> "One honest residual is reported, not hidden: the 5 cm boundary carries synoptic weather noise that
> a single effective alpha cannot fully resolve. It is visible in the 5 cm boundary loss, but it does
> not contaminate the held-out interior score, which is what the case is judged on."
Grounded verbatim in the doc's Honesty section.

E5 (addresses G5) — sharpen the band-check framing in the Context "What the benchmark shows"
paragraph. File: `SoilHeatRealContext.tsx` (EN + ES). Add:
> "That the recovered alpha lands inside the textbook range for moist mineral soil is the independent
> sanity check that the inverse found physics, not a curve fit."
Grounded verbatim in the doc.

E6 (addresses G6, optional) — add one clause on provenance to the Context "Components & variables" or
"The problem" paragraph. File: `SoilHeatRealContext.tsx`. E.g.:
> "The data is vendored once from NOAA's open archive and used offline (schema
> pinnlab.dataset.uscrn/v1; 0% missing in 2021), so training is reproducible and CI needs no network."
Grounded in the doc's "The data (real, vendored, reproducible)" section.

R1 (addresses the two nits) — reconcile numbers across all four surfaces and the doc:
- Decide the true baked per-depth RMSE (1.24/1.05 vs 1.26/1.06) and make the doc's Result table and
  Validation section identical, then set `results.ts` to match.
- Decide 0.30 vs 0.304 for alpha; use one value consistently in `results.ts`, `constraints.ts`, the
  Context, and the doc.

## Notes for the maintainer

- The doc's own internal inconsistency (Result table 1.26/1.06 vs Validation 1.24/1.05) should be
  fixed in `docs/cases/env-soil-heat-real.md` regardless of the app; the app merely inherited one of
  the two.
- All proposed enrichment text above is transcribed or paraphrased from the doc; no new numbers were
  invented. Where a number was not in the doc (none here), none was added.
