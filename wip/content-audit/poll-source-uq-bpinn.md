# Content audit: `poll-source-uq-bpinn`

Date: 2026-07-15
Auditor: content-audit subagent (honest depth + coherence pass, NOT a rubber-stamp)

Authoritative doc: `docs/cases/poll-source-uq-bpinn.md`
In-app files audited:
- Context (deep): `frontend/src/content/cases/SourceUqBpinnContext.tsx`
- Scenario (short): `frontend/src/content/scenarios.ts` key `poll-source-uq-bpinn`
- Result (short): `frontend/src/content/results.ts` key `poll-source-uq-bpinn`
- Constraints legend: `frontend/src/content/constraints.ts` key `poll-source-uq-bpinn`

---

## Verdict

**Severity 3 (contradiction present).** The deep Context component is genuinely **rich** and
coherent with the doc: it carries the governing PDE, the analytic reference field with its
derivation, the hard-BC output transform, the ensemble mean/std equations, the Lakshminarayanan
2017 citation with DOI, and an honest scope section. That part is exemplary.

But the short in-app content contains **one hard physics contradiction** (an "advection + diffusion"
assumption asserted for a case whose governing equation, in both the doc and the Context, is
**pure diffusion**), one **internal-vs-measured incoherence** about where uncertainty peaks, and a
**minor calibration-number mismatch** (99.97% vs the doc's 100%). Separately, the app **never
surfaces the doc's headline accuracy metric (mean relative-L2 = 1.2%)** nor the mean-std / 2.7x
uncertainty-ratio numbers. `coherentWithDoc = false` on the strength of the advection contradiction.

- contextDepth: **rich**
- scenarioResultsDepth: **adequate** (substantive narrative, but carries the advection error and omits the primary metric)
- coherentWithDoc: **false**

---

## Contradictions (in-app vs doc / measured result)

### C1 — "advection + diffusion" contradicts the pure-diffusion governing equation (HARD)
- **inApp** (`results.ts`, `assumptions_en[0]`): `"transport physics known (advection + diffusion)"`
  (ES mirror `assumptions_es[0]`: `"física de transporte conocida (advección + difusión)"`).
- **docSays**: the governing equation is the **heat / pure-diffusion equation**
  `c_t = D c_xx` on `x∈[0,1], t∈[0,1]`, with analytic field `c*(x,t) = e^{-Dπ²t} sin(πx)` (`D=0.1`).
  There is **no advection term**. A standing sine mode that only decays in time
  (`e^{-Dπ²t}`) is the signature of pure diffusion; an advection term `v·c_x` would produce a
  drifting/traveling profile, which this field is not.
- Note this is also an **internal** contradiction: the Context itself states pure diffusion
  ("diffuses ... governed by the heat equation `c_t = D c_xx`", line 109-110). Only `results.ts`
  invents the advection term.
- **Fix**: change the assumption to pure diffusion. Concrete replacement:
  - EN: `"diffusion transport physics known (heat equation c_t = D c_xx, D = 0.1)"`
  - ES: `"física de transporte por difusión conocida (ecuación del calor c_t = D c_xx, D = 0.1)"`

### C2 — where uncertainty peaks: Context "intermediate times" vs measured `t = 0`
- **inApp Context** (`SourceUqBpinnContext.tsx`, EN lines 186-188; ES lines 88-90): the
  uncertainty "grows in the data-sparse regions: **typically the interior band at intermediate
  times**, away from any reading."
- **inApp measured** (`results.ts`, `answer_en`): "the least trustworthy spot in the whole field
  is marked by the widest band (**σ = 0.019 at x = 0.30, t = 0**)."
- **docSays**: the doc pins no location; it only says uncertainty is small near the 24 sensors and
  the `c=0` walls and "grows where the field is unconstrained" (`~2.7x higher in data-sparse
  regions`). It does NOT support "intermediate times".
- The Context's specific claim ("intermediate times") is an unverified specific that the app's own
  measured argmax (`t = 0`) contradicts. This reads as fabricated colour.
- **Fix**: make the Context prose match the trace. Replace "typically the interior band at
  intermediate times, away from any reading" with a claim grounded in the measured argmax, e.g.
  "peaking in the interior away from any sensor (the widest band sits near `x = 0.30`), where
  neither a wall nor a reading pins the field." Keep it faithful: do not assert a time regime the
  trace does not show.

### C3 — calibration number: app 99.97% vs doc 100% (MINOR)
- **inApp** (`results.ts`, `verdict_en`): "**99.97%** of true values fall inside the 2σ band
  (computed calibration)" (ES: `"el 99.97%"`).
- **docSays**: "2σ calibration (truth within mean ± 2 std) = **100 %** (well-calibrated, slightly
  conservative)."
- Numerically near-identical (99.97% vs 100% on a 61x61 = 3721-point grid is ~1 point difference),
  but the two authored surfaces disagree on the exact figure. Reconcile to a single value. If the
  true computed value is 99.97%, update the doc table; if it is 100%, update `results.ts`. Do not
  invent a third number.

---

## Depth gaps (real doc content the app omits)

### G1 — the headline accuracy metric (mean relative-L2 = 1.2%) is nowhere in the app
- **docSays** (Result table): "mean relative-L2 vs `c*` = **1.2 %**" — this is the doc's *primary
  score* ("the primary score is the relative L2 of the mean μ against c*", echoed in the Context).
- The Context states the metric exists but gives no value; `scenarios.ts` and `results.ts` never
  mention accuracy at all (they only discuss calibration/error bars). A reader never learns the
  mean field is actually accurate.
- **Fix (results.ts `verdict_en`)**: prepend the accuracy fact before the calibration sentence,
  e.g. "The mean field is accurate (1.2% relative-L2 vs the analytic mode) AND the error bars are
  honest: ..." — grounded verbatim in the doc's 1.2%.

### G2 — mean ensemble std (0.0068) and the ~2.7x data-sparse ratio are omitted
- **docSays**: "mean / max ensemble std = **0.0068 / 0.0186** (uncertainty **~2.7x higher** in
  data-sparse regions)."
- `results.ts` reports only the max band (σ = 0.019 ≈ 0.0186 rounded); it never gives the mean std
  or the 2.7x ratio that quantifies the whole point of the case (uncertainty concentrates off-data).
- **Fix (results.ts `answer_en` or `verdict_en`)**: add "(mean band σ = 0.0068 across the field;
  the widest band is ~2.7x that, in the data-sparse interior)". Faithful to the doc's numbers.

### G3 — Context does not transcribe any measured result numbers
- The Context is method-rich but numbers-free: the doc's Result block (1.2% L2, 100% calibration,
  0.0068/0.0186 std, live lane 101 KB / 3.8 ms / parity 2.4e-7) never appears in the deep prose.
  For a UQ-benchmark case, at least the two headline outcomes (mean accuracy 1.2% and the
  slightly-conservative ~100% 2σ calibration) belong in the "What the benchmark shows" paragraph.
- **Fix (SourceUqBpinnContext.tsx, "What the benchmark shows" / "Qué muestra el benchmark")**: add
  one sentence transcribed from the doc, e.g. EN: "Measured (seed 42): the mean lands at 1.2%
  relative-L2 of the analytic mode, and the 2σ band is well-calibrated but slightly conservative
  (essentially all truth inside 2σ), with the mean/max ensemble std at 0.0068/0.0186." Do not invent
  beyond these doc figures.

### G4 — the "single honest benchmark, no fabricated regimes" rationale is in the Context but not the short surfaces (minor, acceptable)
- The Context handles the "why one benchmark, not a parametric family" honesty well. `scenarios.ts`
  / `results.ts` don't need it, but the `verdict` could note the field is
  `synthetic-illustrative` (the doc's `real_or_synthetic = synthetic-illustrative`) so the
  compliance framing is not mistaken for a real regulatory dataset. Optional.

---

## What is coherent / good (for balance)

- Governing equation, `D=0.1`, `K=5`, `N=24` sensors, `c=0` walls, analytic `c*` — all match the
  doc across Context and constraints.
- The `[μ, s]` single-ONNX two-output design matches the doc's "one ONNX, two outputs (`[mean,
  std]`)".
- Constraints legend (`bc: c=0 both walls`; `data: sparse noisy sensors, NOT the full IC, hence
  the σ band`; `anchor: analytic diffusion mode + 2σ calibration`) is fully coherent.
- The Lakshminarayanan et al. NeurIPS 2017 citation (arXiv:1612.01474, doi:10.48550/arXiv.1612.01474)
  is present and correct.
- The hard-BC output transform `f = x(1-x)·N(x,t)` and the ensemble mean/std formulas are correctly
  stated (population std, /K) and consistent with "ensemble standard deviation" in the doc.
- Extra detail in the Context not in the doc (sensor noise `σ=0.02`, `61×61` grid) is additive, not
  contradictory.

---

## Concrete edit list (by file)

1. `frontend/src/content/results.ts` — `poll-source-uq-bpinn.assumptions_en[0]` / `assumptions_es[0]`:
   replace "advection + diffusion" / "advección + difusión" with pure-diffusion wording (C1).
2. `frontend/src/content/results.ts` — `verdict_en` / `verdict_es`: prepend the 1.2% mean-accuracy
   fact (G1) and reconcile calibration to the doc's figure (C3); optionally add mean-std/2.7x (G2).
3. `frontend/src/content/cases/SourceUqBpinnContext.tsx` — "What the benchmark shows" paragraph
   (EN ~line 186-188, ES ~line 88-90): fix "intermediate times" to match the measured `t=0` argmax
   (C2), and add one measured-numbers sentence (G3).
