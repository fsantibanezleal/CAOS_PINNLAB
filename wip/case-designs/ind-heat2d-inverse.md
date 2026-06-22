# Case design — `ind-heat2d-inverse`

2D inverse heat conduction: recover the conductivity field `k(x,y)` from ~100 sparse, noisy interior temperature
sensors. Field inverse via a 2-output PINN (the unknown `k` is the second network output, a field — not a `dde.Variable`
scalar), product-rule divergence residual, `PointSetBC` observations, `softplus(k)` positivity, hard-zero boundary on
`T`. Engine: DeepXDE. Validation anchor: an exact MMS pair.

## Decision: SINGLE honest benchmark variant

**Not parametric. Not a discrete sweep. One `default` variant.** Rationale (conservative, adversarially checked):

1. **The parametric pattern requires the swept knob to be a NETWORK INPUT** (in `inputs` + `param_specs`), so one trained
   net + one ONNX sweeps it live. This case's network maps `(x,y) → (k,T)`. The recovered conductivity `k(x,y)` is an
   **output**, not driven by any input scalar. There is no extra physical input axis to sweep.

2. **The plausible "knobs" all require RE-TRAINING, which the architecture forbids.** Noise level `σ`, sensor count
   `N_SENSORS`, and the true-`k` amplitude `A` each change the **observed data and/or the source `q`** — so each needs a
   *fresh inverse solve*. But the pipeline (`pipeline.precompute`) trains **once** per case, then bakes one trace per
   variant by re-evaluating the *same* trained net at each variant's `params` (network-input fills only). There is **no
   per-variant retraining hook**. A noise/sensor sweep is therefore not representable as variants here — each variant
   would silently reuse the one model trained at `σ=0.01, N=100`, which would be a lie.

3. **A parametric-`A` MMS family is genuinely ill-posed and was rejected.** One could imagine a net
   `(x,y,A) → (k,T)` with `k*=1+A·sin πx sin πy`, `T*=sin πx sin πy` (A-independent), `q=q(x,y,A)`. But the inverse
   problem is only well-posed *for the A that generated the observation set*, and `PointSetBC` is a **fixed** point-set
   constraint (it cannot depend on the network-input `A`). So a single observation set cannot anchor the inversion across
   `A`; the swept net would not be a faithful inverse solve. Rejected as dishonest (ADR-0016 §9.A — never fake regimes).

4. **The truth is stationary.** `k*` is a fixed MMS field; there is no traveling/time family (unlike Burgers/ocean) and no
   meaningful physical regime axis. This is exactly the Allen-Cahn situation → ship a single honest benchmark.

Honest label: `synthetic` (MMS ground truth; no open 2D thermal-field inverse dataset exists). The single-variant bake
already validated end-to-end at `l2_relative(k)=0.0403`, `T_l2_relative=0.0077` — within the expected band.

## Analytic anchor + substitution proof

Manufactured solution (MMS) on `(0,1)^2`:

    T*(x,y) = sin(pi x) sin(pi y)
    k*(x,y) = 1 + (1/2) sin(pi x) sin(pi y)
    q(x,y)  = div( k* grad T* )

`T*` satisfies `T*|∂Ω = 0` (each sin factor vanishes at 0 and 1). The source `q` is derived so the triple `(T*,k*,q)`
satisfies `div(k* grad T*) = q` **identically** — verified by symbolic substitution (SymPy):

    q_sympy  = (pi^2/2) ( -4 sin^2(pi x) sin^2(pi y) + sin^2(pi x) - 4 sin(pi x) sin(pi y) + sin^2(pi y) )
    q_build  = (pi^2/2) ( -4 s^2 + sin^2(pi x) + sin^2(pi y) - 4 s ),   s = sin(pi x) sin(pi y)
    simplify(q_sympy - q_build) == 0   ✓   (matches the build's `q_source` exactly)

The PINN recovers `k`; the **primary scored output** is `k` (`outputs[0]`), so `evaluate.run` computes
`l2_relative(k_pred, analytic(XY)=k_true(XY))` — the true recovery error against the exact `k*`. `T` error is reported
separately via `extra_metrics`. This is a genuine analytic anchor, not PINN-on-PINN.

## Exact case `.py` edits (orchestrator applies)

Preserve the existing method/engine entirely. Only two surgical additions: make `field_axes` explicit and add a
single-variant `variants()`. **No changes** to `analytic`, `extra_metrics`, `build`, the product-rule `pde`, the PFNN,
the `softplus(k)` + hard-zero-`T` output transform, the `PointSetBC` observations, or `train`.

### 1. Import `Variant` (extend the existing import)

```python
from .base import CaseSpec, Variant
```

### 2. Add `field_axes` to `CASE` (explicit; the heatmap is over (x,y), and `k` is `outputs[0]` so it is the scored field)

In the `CaseSpec(...)` call, immediately after the `grid={...}` line, add:

```python
    field_axes=("x", "y"),   # explicit: the baked heatmap is k(x,y); no parameter axis (single-variant inverse)
```

(There are no parameter axes — `inputs == field_axes == ("x","y")` — so `param_specs` stays empty and `__post_init__`'s
`param_axes == spec_keys == {}` check passes.)

### 3. Add a single-variant `variants()` (place right after `analytic(...)`, before `extra_metrics`)

```python
def variants() -> list[Variant]:
    """Single honest benchmark variant. The conductivity truth k* is a fixed MMS field; there is no physical knob that
    is a network input, and the only data-side knobs (sensor noise / count / true-k amplitude) would each require a
    fresh inverse solve (the pipeline trains once per case). So this case ships one variant, not a fabricated sweep."""
    return [
        Variant(
            "default",
            "Recovered k(x,y)", "k(x,y) recuperado",
            {},
            "Conductivity field recovered from ~100 sparse noisy T sensors; relative-L2 vs the exact k* ~4e-2.",
            "Campo de conductividad recuperado desde ~100 sensores T dispersos y ruidosos; relativo-L2 vs el k* exacto ~4e-2.",
        )
    ]
```

That is the complete set of edits. After applying, re-bake (`python -m pinnlab.pipeline ind-heat2d-inverse --seed 42`)
to upgrade the manifest from `v1` → `v2` with the explicit `default` variant block.

## Bake recipe (unchanged from the working bake)

- **Net:** `dde.nn.PFNN([2, [40,40], [40,40], [40,40], 2], "tanh", "Glorot uniform")` — parallel branches so `k` and `T`
  have independent subnetworks; `k` = `softplus(branch0)+1e-3 > 0`, `T = x(1-x)y(1-y)·branch1` (hard zero BC).
- **Residual:** product-rule `div(k grad T) = k(T_xx+T_yy)+k_x T_x+k_y T_y`, minus the closed-form `q_source`.
- **Data:** `dde.data.PDE(geom, pde, [observe_T], num_domain=2000, num_boundary=200, anchors=ob_xy, num_test=4000)`;
  `observe_T = PointSetBC(ob_xy, ob_T, component=1)` with `N_SENSORS=100`, `NOISE=0.01`.
- **Optimizer:** Adam `lr=1e-3`, `adam=20000`, `loss_weights=[1, 100]` (PDE, observations), then **L-BFGS** to polish.
  No RAR (the field is smooth; the front-chasing RAR is for sharp-layer cases). No `refine()` hook on this case.
- **Export:** generic coordinate ONNX (input_dim=2), opset 18, dynamic batch axis; parity check ~5e-7 (the softplus +
  polynomial-BC transform are pure tensor ops, so they survive the export).
- **Expected band:** `l2_relative(k)` ≈ **3e-2–5e-2** (loosest near the boundary and at the `T` extrema where `|∇T|` is
  small, so `k` is weakly observable). `T_l2_relative` ≈ **<1e-2**. Reference bake: `k` 0.0403, `T` 0.0077, parity 4.8e-7.
- **Quick smoke first:** `--quick` (300 Adam iters, no L-BFGS) to catch any structural bug (~30 s) before the real
  ~9-min bake.

## Registry line (orchestrator adds to `registry.tsx`)

```
"ind-heat2d-inverse": Heat2dInverseContext,
```

(plus `import { Heat2dInverseContext } from "./Heat2dInverseContext";`)

## Risks / fallback

- **Risk — none structural.** The case already bakes cleanly to a `v1` manifest; the only change is wrapping it in the
  `v2` single-`default` variant shape. Same model, same metrics expected.
- **Risk — boundary error visible.** `k` recovery is loosest near `∂Ω` (small `|∇T|`). This is honest and documented in
  the Context (Scope & assumptions); it is a real property of sparse-data inverse conduction, not a bug. Do not "fix" it
  by shrinking the domain or hiding the edge.
- **Fallback — if the v2 bake regresses** `l2_relative(k)` above ~6e-2, raise `adam` to 25000 or add a short L-BFGS
  second call; do **not** add fake regimes or weaken the noise/sparsity to flatter the number.
- **Do NOT** turn this into a parametric sweep over noise/sensors/amplitude — see Decision §2–3; the architecture trains
  once per case and those knobs need a fresh inverse solve.
