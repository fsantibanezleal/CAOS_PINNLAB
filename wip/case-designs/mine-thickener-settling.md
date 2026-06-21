# Design note — `mine-thickener-settling` → workbench

**Component:** `ThickenerContext` · **Context file:** `frontend/src/content/cases/ThickenerContext.tsx`
**Case file:** `data-pipeline/pinnlab/cases/mine_thickener_settling.py` · **Engine:** DeepXDE (unchanged)

---

## Decision: PARAMETRIC (6 variants) in the front descent rate `R`

The case already ships a **Method-of-Manufactured-Solutions (MMS)** descending-front field driven through the
*genuine* nonlinear Bürger–Concha operator (Richardson–Zaki hindered-settling flux `f_bk` + tanh-regularized
degenerate diffusion `D`). The manufactured solution is an exact `tanh` front:

```
phi*(z,t) = PHI_LO + (PHI_HI - PHI_LO) * 0.5 * (1 - tanh(u)),   u = (z - s)/W,   s = Z0 - R*t
```

and the source `f = L[phi*]` is computed analytically so `phi*` solves the modified PDE **exactly**. The physical
knob that controls the field is the **front descent rate** `R` (a settling speed): a faster-settling suspension drops
its mud-line faster. Because the MMS source is recomputed analytically *as a function of `R`*, the anchor stays exact
for every `R` — this is the same structure that makes `bench-burgers1d` honestly parametric (an exact `tanh` family in
a physical knob). So we expose `R` as a **network input** (in `inputs` + `param_specs`, NOT in `field_axes`): one
trained net covers the whole settling-rate family and the `Live` tab sweeps `R` continuously via the one shared ONNX.

**Why not single-variant?** Unlike Allen-Cahn (whose symmetric front is *stationary*, no closed-form family), this
front genuinely *moves*, and the descent rate is a clean physical 1-parameter family with an exact, substitution-proven
anchor. Faking regimes is the toy failure to avoid (ADR-0016 §9.A) — here the family is real, so parametric is the
honest choice. `R∈[0.3,0.9]` keeps the front interior to `z∈[0,1]` for `t∈[0,1]` (`s = Z0 - R t = 0.9 - R t`, so
`s∈[0.0,0.9]` at the fast end and `s∈[0.6,0.9]` at the slow end — the mud-line always stays in the column).

---

## Analytic anchor + substitution proof (exact for ALL `R`)

The reference is `phi*(z,t;R) = PHI_LO + amp·½(1 − tanh u)`, `u = (z − s)/W`, `s = Z0 − R t`, `amp = PHI_HI − PHI_LO`.
With `sech²u = 1 − tanh²u`:

```
phi_z  = −amp/(2W) · sech²u
phi_t  = −amp·R/(2W) · sech²u          (∂u/∂t = R/W)
phi_zz =  amp/W²     · sech²u · tanh u
```

The MMS source is **defined** as the operator applied to `phi*`:

```
f(z,t;R) := phi_t + f_bk'(phi*)·phi_z − [ D'(phi*)·phi_z² + D(phi*)·phi_zz ]
```

(with the case's existing `f_bk'`, `D`, `D'`). The PINN residual is

```
r := phi_t + f_bk'(phi)·phi_z − [ D'(phi)·phi_z² + D(phi)·phi_zz ] − f(z,t;R).
```

Substituting `phi = phi*`: every term cancels its counterpart in `f` **identically, for every `R`**, because `R`
enters only through `s` (and hence `u`, `phi_t`) and is carried analytically into `f`. Therefore `r ≡ 0` at `phi*` for
all `(z,t,R)` → the relative-L2 of the PINN vs `phi*` is the true forward error. QED. (The only `R`-free quantity is the
IC `phi*(z,0;R) = PHI_LO + amp·½(1 − tanh((z−Z0)/W))`, since `s(0)=Z0` independent of `R` — a clean shared IC.)

This preserves the case's existing SOTA ingredients verbatim: genuine `f_bk` (Richardson–Zaki), genuine degenerate
`D` with the tanh gel-switch at `phi_c`, and the analytic MMS source. We only lift the descent rate to an input.

---

## Exact case `.py` edits (orchestrator applies)

Add `R` constants + range, import `ParamSpec`/`Variant`, make every `phi*`/source/PDE consume `R` from the input
column, and set `field_axes`/`param_specs`/`variants()`. The geometry becomes a `Hypercube` over `(z, t, R)`.

```python
from .base import CaseSpec, ParamSpec, Variant   # add ParamSpec, Variant

# ... keep PHI_MAX, PHI_C, C_RZ, V0, DELTA, K_SIG, EPS_REG, PHI_LO/HI, Z0, W_FRONT ...
R_MIN, R_MAX = 0.30, 0.90          # NEW: front descent-rate (settling-speed) range
R_DEF = 0.80                       # the legacy single-case value, now the slider default
# REMOVE the old module-level R_FRONT = 0.8 constant (R is now an input)

CASE = CaseSpec(
    id="mine-thickener-settling",
    category="mining-mineral-processing",
    title="Thickener / tailings settling — Bürger-Concha degenerate conservation law (parametric descent rate)",
    governing_equations=(
        r"\phi_t + \partial_z f_{bk}(\phi) = \partial_z(D(\phi)\phi_z),\ "
        r"f_{bk}=v_0\phi(1-\phi/\phi_{max})^C,\ D\ \text{degenerate above}\ \phi_c"
    ),
    method="nonlinear-settling",
    engine="deepxde",
    real_or_synthetic="synthetic-illustrative",
    inputs=("z", "t", "R"),                          # CHANGED: R is now a network input
    outputs=("phi",),
    domain={"z": (0.0, 1.0), "t": (0.0, 1.0), "R": (R_MIN, R_MAX)},   # CHANGED
    grid={"z": 101, "t": 51},                        # field grid unchanged (R is not a field axis)
    field_axes=("z", "t"),                           # NEW
    param_specs=(ParamSpec("R", "Front descent rate R", "Tasa de descenso del frente R", R_DEF, R_MIN, R_MAX, 0.02),),  # NEW
    expected_band="descending settling front (faster for larger R) + rising consolidated bed; relative-L2 vs MMS analytic < 2e-2",
    validation_anchor="analytic",
    train={"lr": 1e-3, "adam": 18000, "lbfgs": True, "num_domain": 6000, "num_boundary": 300,
           "num_initial": 300, "num_test": 9000, "loss_weights": [1, 10, 10],
           "layers": [3, 64, 64, 64, 64, 1],
           "rar_rounds": 4, "rar_addk": 400, "rar_adam": 3000, "rar_pool": 60000},
    notes="Richardson-Zaki hindered settling + tanh-regularized degenerate diffusion; soft Dirichlet/IC = phi*; "
          "MMS source. Parametric in the front descent rate R (a network input). RAR sharpens the moving front.",
)


def _phi_star_np(ztR: np.ndarray) -> np.ndarray:
    ztR = np.asarray(ztR, dtype=np.float64)
    z = ztR[:, 0:1]
    t = ztR[:, 1:2]
    R = ztR[:, 2:3]                                   # CHANGED: R from the 3rd column
    s = Z0 - R * t
    return PHI_LO + (PHI_HI - PHI_LO) * 0.5 * (1.0 - np.tanh((z - s) / W_FRONT))


def analytic(ztR: np.ndarray) -> np.ndarray:
    return _phi_star_np(ztR)


def variants() -> list[Variant]:
    presets = [
        ("r030", 0.30, "Slow settle (R=0.30) — a gently descending mud-line; the bed barely forms.",
                        "Sedimentación lenta (R=0.30) — la interfase desciende suavemente; el lecho apenas se forma."),
        ("r042", 0.42, "R=0.42 — moderate descent.", "R=0.42 — descenso moderado."),
        ("r054", 0.54, "R=0.54 — clear settling, bed building.", "R=0.54 — sedimentación clara, el lecho crece."),
        ("r066", 0.66, "R=0.66 — fast front, well-defined bed.", "R=0.66 — frente rápido, lecho bien definido."),
        ("r078", 0.78, "R=0.78 — rapid settle (near the legacy case).", "R=0.78 — sedimentación rápida (cerca del caso original)."),
        ("r090", 0.90, "Fast settle (R=0.90) — the mud-line drops to the column bottom by t=1.",
                        "Sedimentación rápida (R=0.90) — la interfase llega al fondo de la columna en t=1."),
    ]
    return [Variant(vid, f"R={r:g}", f"R={r:g}", {"R": r}, en, es) for vid, r, en, es in presets]


def build(seed: int) -> dict:
    import deepxde as dde
    import torch

    dde.config.set_random_seed(int(seed))
    geom = dde.geometry.Hypercube([0.0, 0.0, R_MIN], [1.0, 1.0, R_MAX])   # CHANGED: (z,t,R) cube

    def f_bk_prime(phi):                       # UNCHANGED
        s = torch.clamp(1.0 - phi / PHI_MAX, min=0.0)
        return V0 * (s ** C_RZ - (C_RZ / PHI_MAX) * phi * s ** (C_RZ - 1.0))

    def _switch(phi):                          # UNCHANGED
        return 0.5 * (1.0 + torch.tanh((phi - PHI_C) / EPS_REG))

    def D_of(phi):                             # UNCHANGED
        shape = torch.clamp(phi, min=1e-4) / PHI_C
        return DELTA * shape ** (K_SIG - 1.0) * _switch(phi)

    def D_prime(phi):                          # UNCHANGED
        p = torch.clamp(phi, min=1e-4)
        shape = (p / PHI_C) ** (K_SIG - 1.0)
        dshape = (K_SIG - 1.0) / PHI_C * (p / PHI_C) ** (K_SIG - 2.0)
        sw = _switch(phi)
        dsw = 0.5 * (1.0 - torch.tanh((phi - PHI_C) / EPS_REG) ** 2) / EPS_REG
        return DELTA * (dshape * sw + shape * dsw)

    def mms_source(x):
        z = x[:, 0:1]
        t = x[:, 1:2]
        R = x[:, 2:3]                          # CHANGED: R from the input
        s = Z0 - R * t
        u = (z - s) / W_FRONT
        th = torch.tanh(u)
        sech2 = 1.0 - th ** 2
        amp = (PHI_HI - PHI_LO)
        phi = PHI_LO + amp * 0.5 * (1.0 - th)
        phi_z = -amp / (2.0 * W_FRONT) * sech2
        phi_t = -amp * R / (2.0 * W_FRONT) * sech2   # CHANGED: du/dt = R/W
        phi_zz = amp / (W_FRONT ** 2) * sech2 * th
        return phi_t + f_bk_prime(phi) * phi_z - (D_prime(phi) * phi_z ** 2 + D_of(phi) * phi_zz)

    def pde(x, y):                             # UNCHANGED (residual structure)
        phi = y
        phi_z = dde.grad.jacobian(y, x, i=0, j=0)
        phi_t = dde.grad.jacobian(y, x, i=0, j=1)
        phi_zz = dde.grad.hessian(y, x, i=0, j=0)
        conv = f_bk_prime(phi) * phi_z
        diff = D_prime(phi) * phi_z ** 2 + D_of(phi) * phi_zz
        return phi_t + conv - diff - mms_source(x)

    # Soft Dirichlet/IC = phi* (R carried through the [N,3] points) — the case's existing method, kept.
    bc = dde.icbc.DirichletBC(geom, _phi_star_np, lambda _, on_boundary: on_boundary)
    ic = dde.icbc.IC(geom, _phi_star_np, lambda x, _: np.isclose(x[1], 0.0))

    t = CASE.train
    data = dde.data.PDE(
        geom, pde, [bc, ic],
        num_domain=t["num_domain"], num_boundary=t["num_boundary"],
        solution=analytic, num_test=t["num_test"],
    )
    net = dde.nn.FNN(t["layers"], "tanh", "Glorot normal")   # [3,64,64,64,64,1]
    model = dde.Model(data, net)
    model.compile("adam", lr=t["lr"], loss_weights=t["loss_weights"], metrics=["l2 relative error"])
    return {"model": model, "input_dim": 3}


def refine(model, case, seed: int) -> None:
    """RAR-G: add the highest-residual points (they land on the moving settling front), then one L-BFGS polish."""
    geom = model.data.geom
    t = case.train
    pde = model.data.pde
    for _ in range(int(t.get("rar_rounds", 4))):
        X = geom.random_points(int(t.get("rar_pool", 60000)))
        err = np.abs(np.asarray(model.predict(X, operator=pde)))[:, 0]
        idx = np.argsort(err)[-int(t.get("rar_addk", 400)):]
        model.data.add_anchors(X[idx])
        model.compile("adam", lr=t["lr"], loss_weights=t["loss_weights"], metrics=["l2 relative error"])
        model.train(iterations=int(t.get("rar_adam", 3000)), disregard_previous_best=True)
    if t.get("lbfgs", True):
        model.compile("L-BFGS")
        model.train()
```

> **Geometry note.** The legacy `build` used `GeometryXTime(Interval, TimeDomain)` + `dde.data.TimePDE`. With `R` as a
> third coordinate we switch to a `Hypercube([0,0,R_MIN],[1,1,R_MAX])` + `dde.data.PDE` (the burgers/ocean parametric
> pattern), so the boundary/initial predicates select on the raw coordinate (`x[1]==0` for the IC). If the orchestrator
> prefers to keep `TimePDE`, an equivalent route is `GeometryXTime(Rectangle over (z,R), TimeDomain)` — but the
> `Hypercube`+`PDE` form above is the validated parametric path in this repo and is recommended.

> **`refine` is OPTIONAL.** If the orchestrator's runner does not call `refine`, drop the RAR keys and the function;
> the soft-BC Adam+L-BFGS bake alone already hit `< 2e-2` at the legacy single `R`. RAR is the front-sharpening upgrade.

---

## Bake recipe

- **Net:** `[3, 64, 64, 64, 64, 1]`, `tanh`, Glorot normal (one extra hidden layer vs the legacy `[2,64×4,1]` to
  absorb the new `R` dimension — same width as burgers' working net but lighter, per the bake-speed lessons).
- **Sampling:** `num_domain=6000`, `num_boundary=300`, `num_initial=300`, `num_test=9000`.
- **Optimizer:** Adam `lr=1e-3`, `18000` iters, `loss_weights=[pde, bc, ic]=[1,10,10]` (the front is the IC/BC-anchored
  feature, weight them up).
- **RAR:** 4 rounds, add 400 top-residual points from a 60k pool, 3000 Adam iters/round (concentrates collocation on
  the moving front across the `R` family).
- **Final polish:** ONE L-BFGS call after the RAR loop (not per round — bake-speed lesson).
- **Quick smoke first:** `--quick` (~30–60 s) to catch a structural bug before the ~10–15 min real bake. Keep ONE heavy
  bake at a time.
- **Expected relative-L2 band:** `< 2e-2` vs the MMS analytic across the `R` family (the sharp `W=0.06` front is the
  hard part; RAR brings it under 2%). Honest band, not a claim of SOTA.

## Registry line

Add to `frontend/src/content/cases/registry.tsx`:

```
"mine-thickener-settling": ThickenerContext,
```

(import `import { ThickenerContext } from "./ThickenerContext";`)

## Risks / fallback

- **Sharp front × new dimension.** `W=0.06` is razor-thin; lifting `R` to an input could blur the front at the
  extremes. *Mitigation:* RAR + `loss_weights` up; if it still misses `2e-2`, narrow to `R∈[0.4,0.8]` (default 0.6) —
  the front then never reaches the boundary, easing the soft BC.
- **Front hitting `z=0` at `R=0.9,t=1`.** `s→0.0`, so half the front sits at the bottom boundary. The soft Dirichlet
  `=phi*` still pins it correctly (it is the exact value), so this is fine, but it is the worst-resolved corner —
  watch its per-variant L2. *Fallback:* cap `R_MAX=0.85`.
- **Conservative fallback = SINGLE variant.** If the parametric bake will not validate under `2e-2`, ship the legacy
  fixed `R=0.8` as a single honest benchmark (drop `inputs` back to `(z,t)`, remove `param_specs`/`field_axes`/`R`),
  exactly the Allen-Cahn pattern. The anchor proof above still holds for the fixed `R`. Honesty over chip count.
