# Case design — `mine-heap-leach-rt` (heap-leach reactive transport)

## Decision: **TIME-SCRUBBER** (variants = time snapshots; the swept parameter is `t`)

`field_axes = (x, z)`, swept parameter `t`, six variants = time snapshots, Live tab = a time scrubber. Same
pattern as `poll-ocean-transport` (an `(x, y, t)` field with a clean exact solution), and exactly the playbook's
"time-scrubber" recipe for `(x, *, t)` cases.

### Why time-scrubber and NOT parametric-in-a-physical-knob

This is an **MMS** (manufactured-solution) case: the source terms `fA`, `fB` are *derived* so that the chosen
`c*` solves the PDE. There is therefore **no closed-form family in a physical knob** (Péclet `Pe = vz·L/D`,
Damköhler `Da = kf·L/vz`, …): changing `D`, `vz` or `kf` would require re-deriving a *different* manufactured
source for the **same** `c*`, which is physically meaningless — the field would not change with the "knob" because
the source absorbs it. Sweeping `Pe`/`Da` here would be a **fabricated regime** (ADR-0016 §9.A), so we do not.

What *is* genuine and exact for the whole history is the **time evolution**: the two species decay at different
rates (`e^{-t}` vs `e^{-t/2}`) while advecting downward and reacting, and the MMS is exact at every `t`. So time is
the honest swept axis: one trained net learns `c(x,z;t)`, and the Live `t` slider scrubs the reacting fronts.

### Why not single-variant

A single snapshot would hide the case's whole point (the *dynamics*: two species relaxing at different rates under
downward percolation + bimolecular consumption). The exact MMS anchor makes every snapshot a *validated* frame, so a
6-frame scrubber is both honest and richer than one static heatmap. (Allen-Cahn shipped single only because its
front is *stationary*; here the field genuinely moves in `t`.)

## Analytic anchor (exact for ALL `t`) + substitution proof

Saturated porous medium; downward Darcy percolation `v = (0, vz)`, `vz = 1`; isotropic dispersion `D = 0.05`;
bimolecular reaction `A + B -> C` at rate `kf·cA·cB`, `kf = 1`. On `(x, z) ∈ (0,1)^2`, `t ∈ (0,1]`:

```
cA_t + v·∇cA = D ∇²cA − kf cA cB + fA
cB_t + v·∇cB = D ∇²cB − kf cA cB + fB
```

Manufactured solution (the `+1.5` offset keeps both concentrations strictly positive, as physical concentrations):

```
cA*(x,z,t) = e^{-t}    sin(πx) cos(πz) + 1.5
cB*(x,z,t) = e^{-t/2}  cos(πx) sin(πz) + 1.5
```

**Substitution proof** (shown for `cA*`; `cB*` is identical in structure). Write `a(x,z) = sin(πx)cos(πz)` and
`E = e^{-t}`, so `cA* = E·a + 1.5`.

- Time: `∂t cA* = −E·a`.
- Advection (`v·∇ = vz ∂z`): `∂z cA* = E · sin(πx)·(−π sin(πz)) = −π E sin(πx) sin(πz)`, so
  `v·∇cA* = vz · (−π E sin(πx) sin(πz))`.
- Diffusion: `∂xx a = −π² a`, `∂zz a = −π² a`, so `∇²cA* = −2π² E·a` (the constant `1.5` drops out).
- Reaction: `−kf cA* cB*` (with the offset products), evaluated at the same `(x,z,t)`.

Define the source as **exactly** the operator applied to `c*`:

```
fA := ∂t cA* + vz ∂z cA* − D ∇²cA* + kf cA* cB*
fB := ∂t cB* + vz ∂z cB* − D ∇²cB* + kf cA* cB*
```

Then the residual `R_A := cA*_t + vz cA*_z − D ∇²cA* + kf cA* cB* − fA ≡ 0` for **every** `(x,z,t)` — by
construction `fA` cancels every term. The same holds for `R_B ≡ 0`. The Dirichlet BC (`= c*` on `∂Ω`) and the IC
(`= c*` at `t=0`) are likewise satisfied by `c*` itself. Hence `c*` is the exact solution of the posed forward
problem for all `t`, so the per-snapshot relative-L2 of the PINN vs `c*` is the **true** PINN error. (The current
`fA`, `fB` in the case already encode exactly these derivatives — `dt`, `adv = vz·dcA_z`, `lap = (∂xx+∂zz)`-mode,
`+ kf cA cB` — so the source is unchanged; we only re-axis the case.)

This proof holds independently of the slider value because the swept axis is *time*, a true network input the MMS
already covers — no per-`t` re-derivation is needed (contrast a physical-knob sweep, which would).

## Exact case `.py` edits (orchestrator applies; method/engine unchanged)

Keep the existing engine (DeepXDE `TimePDE`, `GeometryXTime`, soft Dirichlet/IC `= c*`, MMS source, FNN `[3]+[40]*4+[2]`,
Adam→L-BFGS, loss-weighted) and the existing `fA`/`fB`/`pde`/`build`. Only change the **axis bookkeeping** so the
case bakes as a time-scrubber, and make the cB metric pick up the per-variant `t`.

### 1) Imports — add `ParamSpec`, `Variant`

```python
from .base import CaseSpec, ParamSpec, Variant
```

### 2) `CaseSpec` — set `field_axes=(x,z)`, `grid` over the field only, add the `t` `param_spec`

```python
CASE = CaseSpec(
    id="mine-heap-leach-rt",
    category="mining-mineral-processing",
    title="Heap-leach reactive transport — advection-diffusion-reaction PINN (2 species, time-scrubber)",
    governing_equations=(
        r"\partial_t c_i + \mathbf{v}\cdot\nabla c_i = D\nabla^2 c_i - k_f c_A c_B + f_i,\ "
        r"\mathbf{v}=(0,1),\ D=0.05,\ k_f=1\ \text{on}\ (0,1)^2\times(0,1]"
    ),
    method="advection-diffusion-reaction",
    engine="deepxde",
    real_or_synthetic="synthetic-illustrative",
    inputs=("x", "z", "t"),
    outputs=("cA", "cB"),
    domain={"x": (0.0, 1.0), "z": (0.0, 1.0), "t": (0.0, 1.0)},
    grid={"x": 41, "z": 41},                       # FIELD grid only (was {x,z,t})
    field_axes=("x", "z"),                          # NEW: heatmap axes
    param_specs=(ParamSpec("t", "Time t", "Tiempo t", 0.5, 0.0, 1.0, 0.02),),  # NEW: t is the slider
    expected_band="downward-advecting reacting fronts; relative-L2 vs MMS analytic < 2e-2 per species",
    validation_anchor="analytic",
    train={
        "lr": 1e-3,
        "adam": 20000,
        "lbfgs": True,
        "num_domain": 4000,
        "num_boundary": 400,
        "num_initial": 400,
        "num_test": 8000,
        "loss_weights": [1, 1, 10, 10, 10, 10],    # [eqA, eqB, bcA, bcB, icA, icB]
    },
    notes="Bimolecular reaction + downward advection; soft Dirichlet/IC = c*; MMS source; time is the swept parameter. cA is the primary output; cB error in extra_metrics at the same snapshot.",
)
```

### 3) `analytic` — stash the snapshot `t` for the cB metric (so cB validates at the SAME `t`)

`evaluate.run` calls `analytic(XY)` (with the variant's `t` filled into the grid) **before** `extra_metrics(sf)`,
in the same single-threaded per-variant loop. Stash `t` there; `extra_metrics` reads it. This is the existing
`_STATE` idiom (cf. the FNO/operator and soil-heat cases) and needs **no** orchestrator change.

```python
_STATE: dict[str, float] = {}  # analytic() stashes the current snapshot t for extra_metrics (per-variant, sequential)


def analytic(xzt: np.ndarray) -> np.ndarray:
    """Primary output cA* (cB* checked in extra_metrics at the same snapshot t)."""
    xzt = np.asarray(xzt, dtype=np.float64)
    _STATE["t"] = float(xzt[0, 2])  # the variant's snapshot time (grid is param-filled in inputs order x,z,t)
    return _cA_np(xzt)
```

### 4) `extra_metrics` — cB relative-L2 on the FIELD grid at the stashed `t`

The old body used `linspace_grid(CASE.domain, CASE.grid)` (a 3-D grid). With `field_axes=(x,z)` and a 2-D `grid`,
rebuild the (x,z) field grid at the snapshot `t` and compare to `cB*`.

```python
def extra_metrics(sf) -> dict:
    from ..model.analytic import l2_relative

    t_snap = float(_STATE.get("t", 0.5))
    x = np.asarray(sf.axes["x"], dtype=np.float64)
    z = np.asarray(sf.axes["z"], dtype=np.float64)
    xx, zz = np.meshgrid(x, z, indexing="ij")
    tt = np.full(xx.size, t_snap)
    XZT = np.stack([xx.ravel(), zz.ravel(), tt], axis=1)
    cB_truth = _cB_np(XZT)
    cB_pred = sf.fields["cB"].reshape(-1, 1)
    return {"cB_l2_relative": round(l2_relative(cB_pred, cB_truth), 6)}
```

Everything else (`_cA_np`, `_cB_np`, `fA`, `fB`, `pde`, the BC/IC, `build`) is **unchanged**.

### 5) Add `variants()` — six time snapshots

```python
def variants() -> list[Variant]:
    presets = [
        ("t00", 0.0, "Lixiviant front entering (t=0) — both species at the manufactured initial state, sharpest contrast.",
                     "Frente de lixiviante entrando (t=0) — ambas especies en el estado inicial manufacturado, máximo contraste."),
        ("t02", 0.2, "t=0.2 — fronts advecting downward; cA has decayed faster than cB.",
                     "t=0.2 — frentes advectando hacia abajo; cA decae más rápido que cB."),
        ("t04", 0.4, "t=0.4 — clear split in the two species' amplitudes (e^{-t} vs e^{-t/2}).",
                     "t=0.4 — separación clara entre las amplitudes de las especies (e^{-t} vs e^{-t/2})."),
        ("t06", 0.6, "t=0.6 — cA noticeably weaker; reaction sink most uneven across the bed.",
                     "t=0.6 — cA notablemente más débil; el sumidero de reacción más desigual en el lecho."),
        ("t08", 0.8, "t=0.8 — both relaxing toward the +1.5 baseline; cA nearly flat.",
                     "t=0.8 — ambas relajando hacia la línea base +1.5; cA casi plana."),
        ("t10", 1.0, "t=1.0 — late percolation; cA almost uniform, cB still modulated.",
                     "t=1.0 — percolación tardía; cA casi uniforme, cB aún modulada."),
    ]
    return [Variant(vid, f"t={tv:g}", f"t={tv:g}", {"t": tv}, en, es) for vid, tv, en, es in presets]
```

## Bake recipe

- **Net:** `dde.nn.FNN([3] + [40]*4 + [2], "tanh", "Glorot normal")` (unchanged — 2 outputs cA, cB; 3 inputs x,z,t).
- **Engine:** DeepXDE `TimePDE` on `GeometryXTime(Rectangle, TimeDomain)`, soft Dirichlet BC (`=c*`) + IC (`=c*`),
  MMS source `fA`/`fB`. **Method unchanged** (advection-diffusion-reaction, soft IC/BC, loss-weighted multi-output).
- **Optimizer:** Adam `lr=1e-3`, `adam=20000`, then **one** L-BFGS polish. **No RAR** (the MMS fronts are smooth,
  width ~1, not a razor layer — RAR would only cost time).
- **Sampling:** `num_domain=4000`, `num_boundary=400`, `num_initial=400`, `num_test=8000`.
- **Loss weights:** `[1, 1, 10, 10, 10, 10]` = `[eqA, eqB, bcA, bcB, icA, icB]` (up-weight the soft IC/BC, as now).
- **`--quick` smoke first** (300 Adam, no L-BFGS) to catch the re-axis wiring before the ~10-15 min real bake.
- **Expected band:** per-species relative-L2 vs the MMS analytic **< 2e-2** (cA reported as `l2_relative`, cB as
  `cB_l2_relative`); this two-species advection-diffusion-reaction with smooth fronts and exact soft BC/IC should land
  comfortably in `1e-3…1e-2`. ONNX parity ~1e-6 (pure FNN, no output transform).

## Registry line (frontend `registry.tsx` — orchestrator applies)

```ts
import { HeapLeachContext } from "./HeapLeachContext";
// …
"mine-heap-leach-rt": HeapLeachContext,
```

## Risks / fallback

- **cB-`t` plumbing.** The cB metric reads `t` from the module-global `_STATE` set in `analytic()`. This is correct
  because the pipeline loops variants **sequentially** and `evaluate.run` calls `analytic` (param-filled grid) right
  before `extra_metrics` in the same iteration. If a future parallel/out-of-order evaluator is introduced this
  coupling breaks; the clean long-term fix is to thread `params` into `extra_metrics`. *Fallback if uneasy:* drop the
  per-variant cB number and report only the primary cA L2 (still a fully validated scrubber) — but the `_STATE`
  idiom is already used elsewhere in the codebase, so keep it.
- **Live multi-output ONNX.** The exported graph has 2 outputs (cA, cB) while the generic exporter names a single
  `"u"`. The trained net outputs `[N,2]`; the field bake reshapes each column (infer already does this). Confirm the
  Live lane reads `outputs[0]` (cA) for the heatmap, or offers a cA/cB toggle if the shell supports multi-output
  selection; default to **cA** (primary) to match the field trace. *Fallback:* if the web shell only renders the
  primary output, that is acceptable — cA is the headline and cB is covered in Context + the cB metric.
- **No physical-knob sweep.** We deliberately do not expose `Pe`/`Da`/`D` sliders — they are not honest for an MMS
  (the source would absorb them). The scrubber over `t` is the genuine, validated dynamic. This is documented in the
  Context "Scope & assumptions".
- **DOI framing.** Method precedent cited as reactive-transport PIML for critical minerals (arXiv:2506.15960);
  labeled **synthetic-illustrative** (literature-relevant parameter ranges, NOT fit to a column-test or plant
  dataset). No real-data claim is made.
```
