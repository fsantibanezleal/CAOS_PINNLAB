# env-soil-heat-real — subsurface heat conduction from REAL soil temperatures

**The flagship real-data case.** Every other case validates against a closed-form or reduced-model truth; this one is
trained on, and validated against, **real measured temperatures**.

## The data (real, vendored, reproducible)

NOAA's **U.S. Climate Reference Network** (USCRN) reports research-grade daily-mean soil temperature at five depths —
**5, 10, 20, 50, 100 cm** — at station **IL_Champaign_9_SW** (central Illinois), here over **2019–2021** (three full
seasonal cycles, 0 % missing in 2021). The fetcher `data-pipeline/pinnlab/datasets/uscrn_soil.py` pulls the open
archive once and vendors `data/reference/uscrn/soil_temp_il_champaign.json` (schema `pinnlab.dataset.uscrn/v1`), so
training is offline + reproducible and CI needs no network. This is the ingestion
[data contract](../architecture/data-contracts.md) exercised on a real dataset.

The measured signal is textbook diffusion: the seasonal wave is **damped and phase-lagged with depth** —

| depth | min … max (°C) | seasonal amplitude |
|-------|----------------|--------------------|
| 5 cm  | −2.6 … 27.0 | ~30 °C (full swing) |
| 50 cm | 1.0 … 24.2 | ~23 °C (damped) |
| 100 cm | 3.0 … 21.1 | ~18 °C (more damped + lagged) |

## The inverse problem

Subsurface heat conduction obeys the 1D heat equation
$$ T_t = \alpha\,T_{zz}, \qquad z \in [5,100]\ \text{cm}. $$
We take the **5 cm and 100 cm sensors as real, time-varying Dirichlet boundaries** (`PointSetBC` on the boundary
time-grid), seed the initial profile from the real 5-sensor depth column at $t=0$, and make the **thermal diffusivity
a single trainable scalar** $\alpha=\exp(\log\kappa)\cdot L^2/T_{\text{span}}$ (`dde.Variable`, recovered jointly with
the network). The case is **Adam-only** so the engine's generic L-BFGS recompile never drops the external trainable
variable.

The **10 / 20 / 50 cm sensors are held out** — never shown to the optimizer — and used only in `evaluate` to score the
recovered model out-of-sample against real interior temperatures (`extra_metrics` interpolates the baked field at each
held-out depth and compares to the measured series).

## Result (measured, seed 42)

| metric | value |
|--------|-------|
| recovered thermal diffusivity **α** | **0.30 mm²/s** (moist mineral soils: ~0.2–0.8 mm²/s ✓) |
| held-out relative-L2 vs REAL temps | **6.9 %** |
| held-out RMSE (10/20/50 cm, 243 pts) | **1.05 °C** (10 cm 1.26 · 20 cm 1.06 · 50 cm 0.75) |
| ONNX parity | 1.4e-6 · lane **live** (40 KB, 0.85 ms) |

A **single** recovered diffusivity reproduces unseen interior soil temperatures to ~1 °C across three depths and three
years. The error is largest at 10 cm (nearest the noisy near-surface boundary) and smallest at 50 cm (deepest, smoothest)
— physically exactly what you expect. The recovered α landing in the textbook range for moist mineral soil is the
independent sanity check that the inverse found physics, not a curve fit.

## Honesty

`real_or_synthetic = validated-real` (the green "real data" tag in the App). Boundaries and the validation anchor are
real measurements; nothing is manufactured. The near-surface (5 cm) boundary carries synoptic weather noise that a
single-α diffusion cannot fully resolve — that residual is visible in the 5 cm boundary loss and is reported, not
hidden; it does not contaminate the held-out interior score, which is what the case is judged on.

## Reproduce

```bash
python -m pinnlab.datasets.uscrn_soil            # re-vendor the real data (network, once)
python -m pinnlab.pipeline env-soil-heat-real --seed 42   # train + validate + bake (deterministic)
```


## Validation (the app's Diagnostics view: held-out REAL sensors)

The out-of-sample real-data test: the reconstruction vs the REAL measured USCRN temperatures at the **held-out**
10/20/50 cm depths (interior sensors never shown to the optimizer), over 2019-2021. Measured points vs the PINN curve,
RMSE **1.24 / 1.05 / 0.75 degC** - matching the case's own held-out metrics. See [the method ladder](../architecture/method-ladder-comparison.md).
