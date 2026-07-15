# Content audit — bench-poisson2d (in-app content vs authoritative doc)

Date: 2026-07-15
Auditor: subagent (honest depth + coherence audit, NOT a rubber-stamp)
Authoritative doc (ground truth): `docs/cases/bench-poisson2d.md`
Ground-truth engine/measured source cross-checked: `data-pipeline/pinnlab/cases/bench_poisson2d.py`, `data/derived/manifests/bench-poisson2d.json`, `data/derived/bench-poisson2d/*.json`

In-app content audited:
- Deep Context: `frontend/src/content/cases/PoissonContext.tsx` (EN + ES)
- Scenario: `frontend/src/content/scenarios.ts` (`bench-poisson2d`)
- Results: `frontend/src/content/results.ts` (`bench-poisson2d`)
- Constraints: `frontend/src/content/constraints.ts` (`bench-poisson2d`)

---

## Verdict

The **deep Context is genuinely rich and coherent** with the doc: it carries the governing equation, the MMS construction `g(t;k)=t(1-t)sin(kπt)`, the closed-form forcing `f=-(g''(x)g(y)+g(x)g''(y))`, the hard-constraint output transform, the spectral-bias discussion, and an honest out-of-scope list. The scenario and constraints are substantive and doc-coherent. This is NOT a hollow chapter.

BUT there is **one real coherence gap the audit exists to catch**: the short **Results verdict quotes a best-case error (0.03%) as THE case error**, which understates the measured worst mode (k=3 = 0.16%) by ~5x and diverges from the doc's honest headline "≤ 0.16% across all 6 variants." A user who loads the k=3 chip or the Charts tab sees ~0.16%, contradicting the verdict's "0.03% ... exact for practical purposes."

Secondary: a numeric contradiction on the **grid size** exists between app (101×101), doc (~121×121) and the actually-served trace (81×81) — here the **doc is the stale/wrong one**, not the app. And the doc's own training figure (Adam 12000) is stale vs the engine (16000); the app correctly omits it.

- coherentWithDoc: **false** (headline error figure diverges from the doc's measured bound and from the app's own per-variant numbers)
- contextDepth: **rich**
- scenarioResultsDepth: **adequate** (scenario rich; results verdict under-reports)
- severity: **2** (real gap: verdict undersells measured spread; plus cross-source grid disagreement)

---

## Contradictions (inApp vs docSays)

### C1 — Results verdict quotes best-case error, not the honest across-family bound  [PRIMARY]
- **inApp** (`results.ts`, verdict_en): "0.03% error vs the closed form: exact for practical purposes."
- **docSays** (Result table, measured, seed 42): "relative-L2 vs analytic **≤ 0.16 %** across all 6 variants."
- **Measured ground truth** (`manifests/bench-poisson2d.json`, per-variant `l2_relative`): k1 0.000347 (0.035%), k1.5 0.000275, k2 0.000257, k2.25 0.000332, k2.5 0.000277, **k3 0.001574 (0.157%)**. The comparison-lane default (`comparison.summary.adapted_vs_std`) is 0.00037 ≈ 0.03%, which is the k=1 figure the app quoted.
- **Why it matters**: 0.03% is the EASIEST mode. The doc deliberately reports the worst-case bound (0.16%, driven by k=3 spectral bias). The app's `Charts` tab and the k=3 chip both surface ~0.16%, so the verdict is internally incoherent with the app's own baked numbers and undersells the measured result 5x. Not fabricated (0.03% is real), but not the honest headline.

### C2 — Field-grid size disagrees across app, doc, and served trace  (here the DOC is wrong)
- **inApp** (`PoissonContext.tsx`, EN line ~95 / ES line ~24): "a **101×101** field grid" / "grilla del campo de **101×101**".
- **docSays** (Problem, line 18): "Field evaluated on a **~121×121** grid over x,y ∈ [0,1]."
- **Ground truth**: case spec `grid={"x":101,"y":101}` (the compute grid) → app matches this. But the actually-served heatmap trace (`data/derived/bench-poisson2d/k1.json`) is decimated to **81×81** axes. So three numbers disagree; the doc's ~121×121 matches NEITHER the spec nor the served trace.
- **Action**: do NOT change the app's 101 blindly (it matches the evaluation grid). The clear fix is to **correct the doc** to 101×101 (or state "computed on 101×101, served decimated to 81×81"). Flagged for coherence; the app is the more-correct side.

### C3 — Context error bound looser than the doc's
- **inApp** (`PoissonContext.tsx`, EN ~138 / ES ~67): "Keeping the L2 **<0.2%** across all of them shows ONE network covers the whole family."
- **docSays**: "≤ 0.16 % across all 6 variants."
- True and coherent (0.157% < 0.2%), but looser than the doc's tighter, measured bound. Minor; tighten to match the doc.

---

## Depth gaps (real doc/measured content the app omits)

### G1 — ONNX parity number (1.2e-7) is never cited, though it underwrites the whole "Live in the browser" claim
The doc's Result table lists "ONNX parity (max abs) | **1.2e-7**" (manifest `onnx.parity_max_abs = 1.23e-07`). The Context asserts the hard constraint "survives the ONNX export intact ... which is why the Live tab re-evaluates the exact field," and Results leans on the Live/parametric story, but **neither cites the parity figure**. This is the one number that makes "Live = the offline solve" a verifiable claim rather than an assertion.

### G2 — The per-variant L2 spread (the spectral-bias signature) is not surfaced
The doc frames the honest bound as a range across 6 variants; the manifest has the full curve (0.035% at k=1 → 0.157% at k=3). The Context talks about spectral bias qualitatively ("k=3 ... the hardest regime for a smooth net") but never gives the measured 0.03%→0.16% climb that PROVES the spectral-bias narrative. Quoting it turns a qualitative claim into evidence.

### G3 — "train → ONNX → web contract hardened on this case" framing is absent
The doc's lead sentence ("This is the case the train → ONNX → web contract was hardened on") is a distinctive honest framing that explains WHY this benchmark exists in the catalogue. Nothing in the app carries it. (Low priority, but it is real doc content the app drops.)

### Minor / no-change-needed
- Training detail: Context says "no boundary loss term" (matches `num_boundary=0`) — good. It omits the iteration budget; do NOT transcribe the doc's "Adam (12000)" because the engine actually uses **Adam 16000** (`train["adam"]=16000`) — the doc is stale here too. If any figure is added, use 16000.
- Lobe nuance (results answer): "mode 1 peaks at the center; higher modes split into symmetric lobes." Measured peaks show k=3 ALSO peaks at center (|u|max=0.0625 at (0.50,0.50)), like k=1 (odd modes have a central lobe); only even/off-integer modes go off-center. The raw estimate items render below the answer so the values are visible; minor, optional to nuance ("odd modes keep a central peak; even modes split into symmetric lobes").

---

## Concrete proposed enrichments (faithful to doc; exact text)

### P1 — `frontend/src/content/results.ts` (`bench-poisson2d`, verdict_en / verdict_es)  [fixes C1, G1, G2]
Replace the single best-case number with the doc's honest across-family bound + the measured spread + the parity figure. Keep the (already good, baked-consistent) honesty tail.

verdict_en (proposed):
> "Relative-L2 stays **≤ 0.16% across all six modes** (0.03% at k=1, climbing to 0.16% at k=3 as spectral bias bites), and the ONNX field matches the trained net to **1.2e-7** (max abs): exact for practical purposes. HONEST: a classical solver is faster and exact for a single case like this; the network's value is answering the WHOLE load family at once. This chapter exists to show where a PINN is NOT the tool."

verdict_es (proposed):
> "La L2 relativa se mantiene **≤ 0.16% en los seis modos** (0.03% en k=1, subiendo a 0.16% en k=3 cuando el sesgo espectral pega), y el campo ONNX coincide con la red entrenada a **1.2e-7** (máx abs): exacto a efectos prácticos. HONESTO: un solucionador clásico es más rápido y exacto para un caso único como este; el valor de la red es responder TODA la familia de cargas a la vez. Este capítulo existe para mostrar dónde una PINN NO es la herramienta."

Grounding: doc Result "≤ 0.16 %"; manifest per-variant l2 (k1 0.000347, k3 0.001574); doc "ONNX parity (max abs) 1.2e-7". No invented numbers.

### P2 — `frontend/src/content/cases/PoissonContext.tsx` (both EN + ES)  [fixes C3, G1, G2]
(a) Tighten the "<0.2%" sentence to the measured spread. EN (from "Keeping the L2 <0.2% across all of them shows ONE network covers the whole family."):
> "Keeping the relative-L2 ≤ 0.16% across all six (0.03% at k=1 rising to 0.16% at the k=3 stress test) shows ONE network covers the whole family."
ES mirror:
> "Que la L2 relativa se mantenga ≤ 0.16% en los seis (0.03% en k=1 subiendo a 0.16% en el test k=3) demuestra que UNA red cubre toda la familia."

(b) In the ONNX/Live sentence, add the parity fact. EN (append to the hard-constraint-survives-export sentence):
> "... which is why the Live tab re-evaluates the exact field in the browser — the ONNX output matches the offline net to 1.2e-7 (max abs)."
ES mirror:
> "... por eso el tab Live re-evalúa el campo exacto en el navegador: la salida ONNX coincide con la red offline a 1.2e-7 (máx abs)."

Grounding: manifest `onnx.parity_max_abs = 1.23e-07` (doc rounds to 1.2e-7); manifest per-variant l2.

### P3 — `docs/cases/bench-poisson2d.md` (DOC fix, not an app fix)  [fixes C2 + stale training figure]
- Line 18: "~121×121 grid" → "101×101 grid" (matches `grid={"x":101,"y":101}`; optionally note the web trace is decimated to 81×81).
- Line 28: "Adam (12000)" → "Adam (16000)" (matches `train["adam"]=16000`).
These reconcile the authoritative doc with the engine; the app is already the more-correct side on both.

---

## Also-worth-noting (outside the 3 scoped short-content files, but in-app + baked)
`data/derived/manifests/bench-poisson2d.json` → `comparison.note_en/es` renders "**Relative-L2 = 0.0%**" (rounds 0.00037 to 0.0%). Even shallower than the results verdict; if the Compare tab shows it, it reads as a hollow "zero error." Recommend "Relative-L2 = 0.04%" (k=1 comparison lane) to stay honest. Regenerated from the pipeline, not hand-edited.
