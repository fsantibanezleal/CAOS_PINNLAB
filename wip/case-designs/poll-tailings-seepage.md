# Case design — poll-tailings-seepage (Tailings-dam unsaturated seepage, Richards/Gardner)

## Decision: **PARAMETRIC** (sweep the Gardner sorptive number α), 6 variants.

The field axes stay **(z, t)** (already the natural 2-D heatmap). The swept physical knob is the
**Gardner sorptive number α** — a *network input* (in `inputs` + `param_specs`, NOT in `field_axes`). One
trained net + one ONNX → the Live tab sweeps α∈[1.0, 2.5] continuously and re-evaluates the field.

### Why parametric here is honest (the crux)
The case shipped as a **manufactured-source (MMS)** anchor whose head profile
`ψ* = −A + (A−ψ_top)·e^{−t/τ}·(1−z)` is **independent of α** — the source `f=L[ψ*]` absorbed all the
α-dependence. Making α a network input on top of that MMS would produce an **identical field for every α**
(a slider that changes nothing visible) → a *fake* parametric sweep, which we reject (ADR-0016 §9.A).

Instead we replace the anchor with a **genuinely α-dependent, provably-exact, SOURCE-FREE** family obtained
via the **Kirchhoff transform** `m = e^{αψ}`. Under this transform the nonlinear Gardner–Richards operator
becomes **linear with constant coefficients in m**, which admits a clean exact separable solution whose ψ
field genuinely changes with α (suction depth ≈ doubles across the sweep). This is strictly *more* honest than
the old MMS (no manufactured source at all) and gives a meaningful Live slider.

## Analytic anchor (with substitution proof)

Governing PDE (Gardner closure, z up, ψ<0 unsaturated):
```
C(ψ) ψ_t = ∂_z[ K(ψ)(ψ_z + 1) ],
K(ψ)=K_s e^{αψ},  K'(ψ)=αK,  θ(ψ)=θ_r+(θ_s−θ_r)e^{αψ},  C(ψ)=θ'(ψ)=(θ_s−θ_r)α e^{αψ}.
Expanded:  C ψ_t − K ψ_zz − K'(ψ_z² + ψ_z) = 0   (source-free).
```

**Kirchhoff variable** `m := e^{αψ}`  (so `ψ = (1/α) ln m`, and `0<m<1 ⇔ ψ<0` strictly unsaturated).
Substituting `ψ=(ln m)/α` into the operator and simplifying (verified symbolically, sympy):
```
C ψ_t − K ψ_zz − K'(ψ_z²+ψ_z)  =  (θ_s−θ_r) m_t − (K_s/α) m_zz − K_s m_z.
```
i.e. the operator is **exactly** the constant-coefficient linear advection–diffusion operator in m. A
source-free exact solution is the separable mode
```
m(z,t;α) = M0 + A·e^{−λ t}·e^{−κ z},
```
which satisfies `(θ_s−θ_r) m_t − (K_s/α) m_zz − K_s m_z = 0` **iff** the dispersion relation holds:
```
−(θ_s−θ_r)λ − (K_s/α)κ² + K_s κ = 0   ⇒   λ(α) = (K_s/(θ_s−θ_r))·κ·(α−κ)/α.
```
Therefore the head field
```
ψ*(z,t;α) = (1/α)·ln( M0 + A·e^{−λ(α) t}·e^{−κ z} ),   λ(α)=(K_s/(θ_s−θ_r))·κ·(α−κ)/α
```
is an **exact** solution of the homogeneous Gardner–Richards equation for **every** α (proof = the dispersion
relation makes the linear-in-m residual identically zero; the Kirchhoff identity maps it back to the ψ-form).
Constants: `θ_s=0.43, θ_r=0.078, K_s=0.25, κ=0.9, M0=0.30, A=0.62` and α∈[1.0, 2.5].

**Adversarial checks (all pass):**
- `κ=0.9 < α_min=1.0` ⇒ `λ(α)>0` for the whole range ⇒ a genuine **drying** transient (suction deepens in time).
- `M0+A=0.92<1` and `M0=0.30>0` ⇒ `m∈(0,1)` over the entire (z,t,α) slider grid ⇒ `ψ<0` **strictly** (always
  unsaturated — the honesty invariant). Verified on a 31-point α grid.
- Finite-difference residual of the **original ψ-form** PDE is `≤1×10⁻⁶` for α∈{1.0,1.5,2.0,2.5} (machine
  precision; confirms the Kirchhoff algebra maps back correctly — it is a true exact solution, not an MMS).
- **Genuinely α-dependent field**: `ψ_min` runs −0.623 (α=1.0) → −0.304 (α=2.5); spatial depth-gradient
  `ψ(z=0)−ψ(z=1)|_{t=0}` runs 0.511 → 0.204. Smaller α (broader pore-size distribution) → deeper, more
  sharply-stratified suction. A real, meaningful sweep — not a cosmetic slider.

## Exact case .py edits (orchestrator applies; keep DeepXDE engine + soft IC/BC method)

Replace the constants block, `CASE`, the analytic, add `variants()`, and switch the PDE to source-free:

```python
THETA_S, THETA_R = 0.43, 0.078
KS = 0.25
KAPPA, M0, A_AMP = 0.9, 0.30, 0.62   # m = M0 + A_AMP e^{-lam t} e^{-KAPPA z}; M0+A_AMP<1 => psi<0 strictly
ALPHA_MIN, ALPHA_MAX = 1.0, 2.5      # Gardner sorptive number (swept network input)


def _lam(alpha):
    # dispersion: (theta_s-theta_r) m_t = (Ks/alpha) m_zz + Ks m_z  =>  lam = (Ks/dth) kappa (alpha-kappa)/alpha
    return (KS / (THETA_S - THETA_R)) * KAPPA * (alpha - KAPPA) / alpha


CASE = CaseSpec(
    id="poll-tailings-seepage",
    category="pollution-environmental",
    title="Tailings-dam unsaturated seepage — Richards (Gardner), parametric sorptive number α",
    governing_equations=(
        r"C(\psi)\psi_t=\partial_z[K(\psi)(\psi_z+1)],\ K=K_s e^{\alpha\psi},\ "
        r"\psi^*=\tfrac1\alpha\ln\!\big(M_0+A\,e^{-\lambda(\alpha)t}e^{-\kappa z}\big),\ "
        r"\lambda=\tfrac{K_s}{\theta_s-\theta_r}\tfrac{\kappa(\alpha-\kappa)}{\alpha}"
    ),
    method="richards-seepage",
    engine="deepxde",
    real_or_synthetic="synthetic-illustrative",
    inputs=("z", "t", "alpha"),
    outputs=("psi",),
    domain={"z": (0.0, 1.0), "t": (0.0, 1.0), "alpha": (ALPHA_MIN, ALPHA_MAX)},
    grid={"z": 101, "t": 51},
    field_axes=("z", "t"),
    param_specs=(ParamSpec("alpha", "Sorptive number α", "Número sortivo α", 1.6, ALPHA_MIN, ALPHA_MAX, 0.05),),
    expected_band="drying unsaturated head profile (psi<0); deeper suction at small alpha; relative-L2 vs exact < 1e-2",
    validation_anchor="analytic",
    train={"lr": 1e-3, "adam": 18000, "lbfgs": True, "num_domain": 6000, "num_boundary": 300,
           "num_initial": 300, "num_test": 8000, "loss_weights": [1, 10, 10]},
    notes="Gardner exponential K(psi); Kirchhoff transform m=e^{alpha psi} linearises the operator => exact SOURCE-FREE family; alpha (sorptive number) is the swept network input; soft Dirichlet/IC = psi*.",
)


def _psi_star_np(zta: np.ndarray) -> np.ndarray:
    zta = np.asarray(zta, dtype=np.float64)
    z, t, alpha = zta[:, 0:1], zta[:, 1:2], zta[:, 2:3]
    lam = (KS / (THETA_S - THETA_R)) * KAPPA * (alpha - KAPPA) / alpha
    m = M0 + A_AMP * np.exp(-lam * t) * np.exp(-KAPPA * z)
    return np.log(m) / alpha


def analytic(zta: np.ndarray) -> np.ndarray:
    return _psi_star_np(zta)


def variants() -> list[Variant]:
    presets = [
        ("a100", 1.0, "Broad pore-size (α=1.0) — deepest, most stratified suction.", "Poros amplios (α=1.0) — succión más profunda y estratificada."),
        ("a130", 1.3, "α=1.3 — strong suction gradient.", "α=1.3 — fuerte gradiente de succión."),
        ("a160", 1.6, "α=1.6 — moderate Gardner sorptivity.", "α=1.6 — sortividad Gardner moderada."),
        ("a190", 1.9, "α=1.9 — shallower profile, faster drying.", "α=1.9 — perfil menos profundo, secado más rápido."),
        ("a220", 2.2, "α=2.2 — weak suction, near-saturated top.", "α=2.2 — succión débil, tope casi saturado."),
        ("a250", 2.5, "Fine pore-size (α=2.5) — shallowest suction.", "Poros finos (α=2.5) — succión más somera."),
    ]
    return [Variant(vid, f"α={a:g}", f"α={a:g}", {"alpha": a}, en, es) for vid, a, en, es in presets]


def build(seed: int) -> dict:
    import deepxde as dde
    import torch

    dde.config.set_random_seed(int(seed))
    geom = dde.geometry.Hypercube([0.0, 0.0, ALPHA_MIN], [1.0, 1.0, ALPHA_MAX])

    def pde(x, y):
        psi = y
        psi_t = dde.grad.jacobian(y, x, i=0, j=1)
        psi_z = dde.grad.jacobian(y, x, i=0, j=0)
        psi_zz = dde.grad.hessian(y, x, i=0, j=0)
        alpha = x[:, 2:3]
        ek = torch.exp(alpha * psi)
        cap = (THETA_S - THETA_R) * alpha * ek
        k = KS * ek
        kp = alpha * k
        return cap * psi_t - k * psi_zz - kp * (psi_z ** 2 + psi_z)   # source-free (exact Kirchhoff family)

    # Soft Dirichlet on z=0,1 and IC at t=0, both equal to the exact psi* (boundary in (z,t,alpha) space).
    def on_zbc(X, on_boundary):
        return on_boundary and (np.isclose(X[0], 0.0) or np.isclose(X[0], 1.0))

    def on_ic(X, on_boundary):
        return on_boundary and np.isclose(X[1], 0.0)

    bc = dde.icbc.DirichletBC(geom, _psi_star_np, on_zbc)
    ic = dde.icbc.DirichletBC(geom, _psi_star_np, on_ic)

    t = CASE.train
    data = dde.data.PDE(
        geom, pde, [bc, ic],
        num_domain=t["num_domain"], num_boundary=t["num_boundary"],
        solution=analytic, num_test=t["num_test"],
    )
    net = dde.nn.FNN([3] + [48] * 4 + [1], "tanh", "Glorot normal")
    model = dde.Model(data, net)
    model.compile("adam", lr=t["lr"], loss_weights=t["loss_weights"], metrics=["l2 relative error"])
    return {"model": model, "input_dim": 3}
```

Notes for the orchestrator:
- The OLD file used `GeometryXTime` + `dde.icbc.IC`. Because α is now a third input, switch to a 3-D
  `Hypercube([z,t,alpha])` + `dde.data.PDE` (matches the parametric exemplar `bench_burgers1d`). The IC and
  z-boundary are imposed **softly** as `DirichletBC` selecting the `t=0` and `z∈{0,1}` faces (the case keeps
  its soft-BC method — no hard output transform needed since ψ* is smooth and well-scaled).
- Drop the old `mms_source` (no longer needed — the family is exact and source-free).
- `mms_source` removal + the source-free `pde` preserve the case's **Richards/Gardner method and DeepXDE engine**.

## Bake recipe
- Net `[3, 48, 48, 48, 48, 1]`, tanh, Glorot normal (small net — the field is smooth, the slope is gentle).
- Adam `lr=1e-3`, `18000` iters; then **one** final L-BFGS polish. **No RAR** (no sharp front; the profile is
  smooth in z and t over the whole α-range).
- `num_domain=6000, num_boundary=300, num_test=8000`; loss weights `[pde, bc, ic] = [1, 10, 10]`.
- `--quick` smoke first (~30 s) to catch a structural bug; then the real bake (~10–15 min, one net).
- **Expected relative-L2 band: 0.3 %–1 % vs the exact analytic** (`< 1e-2`). Smooth parametric family with a
  forgiving conditioning — should land comfortably under 1 %.

## Registry line
```
"poll-tailings-seepage": TailingsSeepageContext,
```
(+ `import { TailingsSeepageContext } from "./TailingsSeepageContext";`) — orchestrator adds to `registry.tsx`.

## Risks / fallback
- **Risk:** α near `κ=0.9` makes `λ→0` (a nearly stationary profile). Mitigated: α_min=1.0 keeps `λ>0` and the
  field non-trivial; the band [1.0,2.5] is safe (verified, m∈(0,1) throughout).
- **Risk:** the parametric net under-resolves the small-α end (deepest suction, steepest z-gradient). If the
  L2 at α=1.0 exceeds 1 %, bump Adam to 24k or widen the net to `[3,64,64,64,64,1]` (still no RAR needed).
- **Fallback:** if the parametric bake is unexpectedly poor, ship the **single honest benchmark** at the
  default α=1.6 (one variant, same exact anchor) — the Context's parametric framing collapses cleanly to a
  fixed-parameter benchmark (like Allen-Cahn). But the family is provably exact and smooth, so parametric is
  expected to succeed.
```
