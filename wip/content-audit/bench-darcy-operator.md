# Content audit: bench-darcy-operator (in-app vs authoritative doc)

Date: 2026-07-15
Auditor: subagent (honest depth + coherence audit, not a rubber-stamp)

## Sources compared

- Authoritative doc (ground truth): `docs/cases/bench-darcy-operator.md`
- In-app deep Context: `frontend/src/content/cases/DarcyOperatorContext.tsx` (EN + ES)
- In-app short content:
  - `frontend/src/content/scenarios.ts` (lines 137-142)
  - `frontend/src/content/results.ts` (lines 222-231)
  - `frontend/src/content/constraints.ts` (lines 58-62)
- Corroborating pages (out of explicit scope, used only to establish app-internal coherence):
  - `frontend/src/pages/Methodology.tsx` (line 117: "held-out test L2 5.6%")
  - `frontend/src/pages/Experiments.tsx` (line 66: "FNO 2.5% (one pass)")

## Verdict

**Not coherent with the doc.** Severity 3 (a measured result number is contradicted).

The Context prose is genuinely deep and, on the physics/method, coherent with the doc: it carries the governing PDE, a weak-form variational characterization, the full FNO operator equation, the harmonic-mean finite-difference anchor, and an honest scope/out-of-scope list. That part is strong.

The problem is the **headline error number**. The doc's authoritative measured result is a **held-out test relative-L2 of 5.5%** (mean over 64 unseen coefficient fields). The app's short content (scenarios + results) instead reports **2.5% as the "average error on maps never seen"**. 2.5% is not the average: the doc uses ~2.5% only for the single Compare-view field (the per-sample range is "mostly 2-10%"). So the app takes a favorable single-field number and presents it as the operator-generalization average, under-reporting the real held-out error by roughly 2x. The app is also internally inconsistent about this number (Methodology 5.6%, Experiments/scenarios/results 2.5%, Context "5-12%"), which is itself evidence the doc's 5.5% headline is being dropped.

## Contradictions (inApp vs docSays)

1. **2.5% presented as the held-out average.**
   - inApp (`results.ts` line 229/230, verdict): "2.5% average error on maps never seen, peaks within ~4% ... 2.5% de error promedio en mapas nunca vistos".
   - inApp (`scenarios.ts` line 140/141, measured): "graded on held-out maps it never saw (2.5% vs finite differences)".
   - docSays (Result table + prose): "held-out test relative-L2 (the operator-generalization metric, 64 unseen a) = **5.5 %**"; "reproduces unseen pressure fields to **~5.5 %** across 64 held-out coefficient fields". The doc uses ~2.5% only in the Comparison section for the Compare view's single held-out field, and states the per-sample spread is "mostly **2-10 %**". Presenting 2.5% as the *average*/held-out grade contradicts the measured 5.5% mean.

2. **Context L2 range "~5-12%" matches neither doc figure.**
   - inApp (`DarcyOperatorContext.tsx` line 85 ES / line 189 EN): "operator-generalization figures (L2 **~5-12%**)".
   - docSays: headline test L2 = **5.5%**; per-sample "mostly **2-10%**". The Context's 5-12% band overshoots the per-sample upper bound (12 vs 10) and never states the actual 5.5% headline. It reads as an invented band rather than the doc's measured numbers.

3. **"peaks within ~4%" is not grounded in the doc.**
   - inApp (`results.ts` line 229/230, verdict): "peaks within ~4% ... picos a ~4%".
   - docSays: no peak-pressure error figure exists in the doc. The only per-sample number is the "2-10%" relative-L2 range. The 4% peak-error claim is unverified/fabricated relative to the doc.

4. (App-internal, supporting) **Methodology says 5.6%, not 5.5%.**
   - inApp (`Methodology.tsx` line 117): "bench-darcy-operator (FNO, held-out test L2 **5.6%**)".
   - docSays: **5.5%**. Minor rounding drift, but it confirms the ~5.5% held-out number is the real one and that scenarios/results' 2.5% is the wrong quantity, not a rounding of it.

## Depth gaps (real doc content the app omits)

1. **The measured headline number 5.5% is absent from all in-app deep/short content for this case.** The Context describes "the held-out test set: the true operator-generalization number" qualitatively but never states its value. scenarios/results substitute 2.5%. The single most important measured result is missing.
2. **ONNX parity value (1.8e-6 max abs) is never stated.** The Context says the baked ONNX is "parity-checked" but gives no number; scenarios/results/constraints omit it entirely.
3. **Dataset size (256 train + 64 test pairs) is never stated.** The Context refers to "the held-out test set" but not the counts; the doc's headline is explicitly "over 64 unseen a".
4. **Concrete FNO hyperparameters are dropped.** Doc: "4 Fourier layers", "keep the lowest 10x10 modes", "Adam, 120 epochs". Context says only "several Fourier layers" and "the lowest Fourier modes". These are depth omissions, not contradictions.
5. **The distinction between the Compare-view single-field number (~2.5%) and the held-out average (5.5%) is collapsed.** The doc keeps them separate on purpose ("The headline metric stays the held-out test-set mean ... each chip also reports its own sample L2"). The app treats 2.5% as if it were the average.

## Concrete proposed enrichments (faithful to the doc; no invented numbers)

Edit `frontend/src/content/results.ts` (bench-darcy-operator, lines 227-230):
- answer_en: replace with something that separates the two numbers, e.g. "For each held-out map: the peak pressure and its location, instantly (below): each within a few percent of the classical solver that would otherwise run per map. The representative Compare field lands at ~2.5% relative-L2." (grounded in doc Comparison "~2.5% relative-L2").
- verdict_en: replace "2.5% average error on maps never seen, peaks within ~4%" with the doc's real held-out average: "Across the 64 held-out maps the operator averages 5.5% relative-L2 (per-map spread mostly 2-10%): good enough to SCREEN thousands of scenarios and send only finalists to the classical solver. OUT of its training family, accuracy degrades: always spot-check out-of-distribution." (grounded in doc: "5.5 %", "mostly 2-10 %"). Drop the unsupported "peaks within ~4%".
- Mirror the same change in verdict_es / answer_es.

Edit `frontend/src/content/scenarios.ts` (bench-darcy-operator, lines 140-141):
- measured_en: replace "(2.5% vs finite differences)" with "(5.5% mean relative-L2 across 64 held-out maps vs finite differences)". Mirror in measured_es. (grounded in doc Result table.)

Edit `frontend/src/content/cases/DarcyOperatorContext.tsx` (lines 85 ES and 189 EN):
- Replace "L2 ~5-12%" with the doc's measured figures: "L2 ~5.5% held-out mean, per-sample mostly 2-10%". Consider stating the value where the Context currently says "the true operator-generalization number" (line 73/178): "the true operator-generalization number (5.5% over the 64 held-out fields)".
- Optionally add, in the Formalization/anchor paragraph, the ONNX parity value and dataset size from the doc: "trained on 256 pairs, graded on 64 held-out; the exported ONNX matches the torch graph to 1.8e-6 (max abs)."
- Optionally tighten "several Fourier layers" to "4 Fourier layers (lowest 10x10 modes)" and "trained ... Adam, 120 epochs" to match the doc's Method section.

Fix `frontend/src/pages/Methodology.tsx` line 117: "5.6%" -> "5.5%" to match the doc's Result table (minor, for consistency).

`frontend/src/content/constraints.ts` (lines 58-62): no change needed. The three entries (u=0 boundary, training PAIRS a->u, finite-difference solve per held-out a) are terse and coherent with the doc.

## Summary of file findings

- Context (`DarcyOperatorContext.tsx`): rich, physics-coherent; one number band ("5-12%") is off vs doc and the 5.5% headline is missing.
- scenarios.ts: substantive framing (groundwater screening, matches doc's precompute rationale) but the measured number (2.5%) contradicts the doc's 5.5% held-out mean.
- results.ts: substantive structure (calculates/assumptions/answer/verdict) but the verdict's "2.5% average" contradicts the doc and "peaks within ~4%" is ungrounded.
- constraints.ts: coherent, no issues.
