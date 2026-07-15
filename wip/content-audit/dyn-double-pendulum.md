# Content audit: `dyn-double-pendulum` (in-app vs authoritative doc)

Date: 2026-07-15
Auditor: content-audit sub-agent (honest depth + coherence pass, no rubber-stamp)

## Ground truth used

- Authoritative doc: `docs/cases/dyn-double-pendulum.md`
- Cross-checked against the real artifact (the doc is CORRECT, verified):
  - precompute code `data-pipeline/pinnlab/cases/dyn_double_pendulum.py`
  - committed manifest `data/derived/manifests/dyn-double-pendulum.json`

The doc, the `build()` code, and the manifest agree on every number and on the method: SOFT initial
conditions, `tanh` activation, `[1, 96x4, 2]`, Rel-L2 = 0.093375, leave-time = 1.99 s, lyapunov_est = 0.0205,
ONNX parity 1.669e-6. The doc is the reliable ground truth.

In-app content audited:
- Deep context: `frontend/src/content/cases/DoublePendulumContext.tsx` (EN + ES branches)
- Short content: `frontend/src/content/scenarios.ts`, `results.ts`, `constraints.ts` (keyed `dyn-double-pendulum`)

---

## VERDICT

**Severity 3 (contradiction).** The scenario, results and honesty framing are rich and coherent, but the
**Formalization / method** content is WRONG in a way a green build hides: the App confidently teaches the exact
method the doc says was **tried and REJECTED**. Two hard contradictions, each appearing in more than one file:

1. The Context and `constraints.ts` both claim the initial condition is enforced **exactly by a hard `t^2`
   ansatz**. The doc (and the real `build()`) use **SOFT IC**, and explicitly document the `t^2` ansatz as
   rejected because it kills the gradient at `t=0` (it moved theta_1 the wrong way).
2. The Context claims **SIREN (sinusoidal) activations**. The doc and code use **tanh**; the code comment
   states "SIREN + soft IC was unstable".

The prose is otherwise deep (governing residual, constants, chaos-honesty, scope, viz guide), so this is a
detailed-but-incorrect account, not a hollow one. It must be corrected, not padded.

- coherentWithDoc: **false**
- contextDepth: **rich** (volume), but coherence-defective in the method section
- scenarioResultsDepth: **rich** (scenario + results + verdict are substantive; `results.ts` twin value 2.11 s
  is a real committed number from the manifest `estimate` block)

---

## CONTRADICTIONS (inApp vs docSays)

### C1 - Initial condition: hard `t^2` ansatz (WRONG) vs soft IC (doc)
- **inApp** `DoublePendulumContext.tsx`, EN lines 103-104 (ES lines 36-38):
  "The initial condition is enforced **exactly** by a hard constraint (released from rest):
  `theta_hat_i(t) = theta_i(0) + t^2 * N_theta,i(t)`" followed by the vanishing-derivative argument.
- **docSays** (Method section): "Initial condition = **SOFT** (`dde.icbc.IC` for theta_i(0) + `OperatorBC`
  for theta_i'(0)=0), weighted 100x above the residual. A `t^2` hard-constraint ansatz was **tried first and
  rejected**: it makes `d theta_hat / d params ~ t^2 -> 0` near `t=0`, killing the gradient signal so the net
  never learns the initial acceleration (it moved theta_1 the *wrong way*). Soft IC is well-conditioned and the
  standard DeepXDE IVP approach."
- Confirmed by code `dyn_double_pendulum.py` lines 143-162: `dde.icbc.IC` + `OperatorBC`,
  `loss_weights=[1,1,100,100,100,100]`; comment: "SOFT initial conditions (not a t^2 hard-constraint output
  transform): the hard ansatz ... kills the parameter gradient near t=0".
- The App teaches the FAILED approach as the shipped method. This is the single most important defect.

### C2 - Activation: SIREN (WRONG) vs tanh (doc)
- **inApp** `DoublePendulumContext.tsx`, EN lines 108-109 (ES 42-43): "The network uses **sinusoidal
  activations (SIREN)** for the oscillatory trajectory, and trains Adam -> L-BFGS."
- **docSays**: "Net: `[1, 96x4, 2]`, **tanh**, Adam(25k) -> L-BFGS."
- Confirmed by code line 71: `"activation": "tanh",  # stable for the soft-IC IVP (SIREN + soft IC was
  unstable)`. The App asserts the very configuration the code says was unstable.

### C3 - constraints.ts IC row repeats the rejected ansatz
- **inApp** `constraints.ts` line 69:
  `{ kind: "ic", en: "theta_i(0) = 120 deg, theta_dot_i(0) = 0 (hard, theta_0 + t^2 N ansatz)", ... }`
- **docSays**: IC is SOFT (weighted 100x); the `theta_0 + t^2 N` ansatz was rejected (see C1). The `(hard,
  theta_0 + t^2 N ansatz)` qualifier is false.
- Note: the anchor row on line 70 ("RK45 rtol=atol=1e-10; leave-time is the honest metric") is CORRECT.

Minor framing note (not a contradiction): `results.ts` verdict says the 1.99 s PINN leave-time and the 2.11 s
twin-separation are "two independent measures [that] land on the same value." They are close (both ~2 s) and
2.11 s is genuinely in the manifest `estimate` block, so this is defensible, but the two numbers measure
related-but-different quantities and differ by ~6%. Leave as is or soften to "land in the same ~2 s band".

---

## DEPTH GAPS (real doc content the App omits)

1. **The rejection rationale (the key didactic insight).** The doc's whole method lesson is WHY the natural
   hard `t^2` ansatz fails on this IVP ("`d theta_hat / d params ~ t^2 -> 0` near t=0, killing the gradient ...
   it moved theta_1 the wrong way"). The App not only omits this, it inverts it. This is the case's core
   teaching value and must be restored.

2. **Measured Relative-L2 = 0.0934 (9.3%) over t in [0,3] s.** The doc lists this as part of the headline
   result trio. Neither `results.ts` nor the Context state it. (Manifest: `l2_relative = 0.093375`.)

3. **Twin-IC divergence rate lambda ~ 0.02 1/s (crude largest-Lyapunov).** The Context names the "Lyapunov
   horizon" concept but never gives the measured lambda; `results.ts` frames the twin as "separates by 2.11 s"
   but never states lambda. (Manifest: `lyapunov_est = 0.0205`.)

4. **Soft-IC weighting = 100x above the residual.** Concrete and doc-stated; App omits.

5. **Net architecture `[1, 96x4, 2]`, Adam 25k iterations.** Context says only "Adam -> L-BFGS" with no
   dims/iteration count.

6. **ONNX parity ~1.6e-6.** Doc-stated (manifest 1.669e-6); could ride along in a method note or results.

---

## CONCRETE PROPOSED ENRICHMENTS (grounded in doc quotes; no invented numbers)

### E1 - Fix the Formalization block in `DoublePendulumContext.tsx` (BOTH EN and ES) [fixes C1 + C2 + gaps 1,4,5]

Replace the current "enforced exactly by a hard constraint" paragraph + the `theta_i(0)+t^2 N` displayed
Equation + the SIREN sentence with the SOFT-IC account. Proposed EN text (mirror in ES):

> The initial condition is imposed **softly** (`dde.icbc.IC` for `theta_i(0)` plus an `OperatorBC` for
> `theta_i'(0)=0`), **weighted 100x above the residual** so the IVP is firmly pinned. A `t^2` hard-constraint
> ansatz `theta_hat_i(t) = theta_i(0) + t^2 N_i(t)` was tried first and **rejected**: because the added term
> carries `t^2`, its parameter-gradient `d theta_hat / d params ~ t^2 -> 0` near `t=0`, which kills the signal
> the net needs to learn the initial acceleration (it drove `theta_1` the *wrong way*). Soft IC is
> well-conditioned and the standard DeepXDE IVP approach. The network is a plain `[1, 96x4, 2]` MLP with
> **tanh** activations (SIREN was unstable with soft IC here), trained Adam (25k) -> L-BFGS, exported to ONNX
> (parity ~1.6e-6).

If a displayed Equation is kept, show the residual + soft-IC loss, e.g. the residual line already present plus
`L = (1/N) sum_t (r_1^2 + r_2^2) + 100 (IC + IC')` conceptually; do NOT display the `theta_0 + t^2 N` ansatz as
the method.

### E2 - Fix the IC row in `constraints.ts` (line 69) [fixes C3]

Replace with the soft-IC description, e.g.:

> `{ kind: "ic", en: "theta_i(0)=120 deg, theta_dot_i(0)=0 (soft, weighted 100x; t^2 hard ansatz rejected)",`
> `es: "theta_i(0)=120 deg, theta_dot_i(0)=0 (suave, peso 100x; ansatz duro t^2 descartado)" }`

### E3 - Surface the measured Rel-L2 and Lyapunov in `results.ts` [fixes gaps 2,3]

In the `dyn-double-pendulum` `answer_en/es` or `verdict_en/es`, add the two committed numbers, e.g. append to
the verdict: "Over the full 0-3 s window the trajectory error is **9.3%** (relative-L2), and the twin's
exponential separation gives a crude largest-Lyapunov rate **lambda ~ 0.02 1/s** - both consistent with a ~2 s
predictability horizon." (Numbers from manifest: `l2_relative=0.093375`, `lyapunov_est=0.0205`.)

### E4 - (Optional) Add the rejection story to `constraints.ts` anchor note or a Context callout

The "moved theta_1 the wrong way" failure is the sharpest teaching moment; ensure it survives in the Context
prose (covered by E1) since it is currently absent everywhere in-app.

---

## Files to edit
- `frontend/src/content/cases/DoublePendulumContext.tsx` (Formalization paragraph + Equation + activation line, EN and ES) - C1, C2, gaps 1,4,5
- `frontend/src/content/constraints.ts` (IC row, line 69) - C3
- `frontend/src/content/results.ts` (`dyn-double-pendulum` answer/verdict) - gaps 2,3
