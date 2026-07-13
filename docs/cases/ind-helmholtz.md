# ind-helmholtz — high-wavenumber 2D Helmholtz with a Fourier-feature PINN

The **spectral-bias showcase** of the catalogue. A plain tanh MLP cannot represent a rapidly oscillating field — it
is biased toward low frequencies — so this case exercises the **random Fourier-feature** input embedding that injects
the right frequencies into layer 1 and lifts that plateau.

## Problem

The 2D Helmholtz (frequency-domain) equation on the unit square, with homogeneous Dirichlet walls:

$$ \nabla^2 u + k_0^2 u = -f,\quad k_0=2\pi n,\ n=3,\quad \text{on}\ (0,1)^2,\quad u|_{\partial\Omega}=0. $$

The forcing is **manufactured** (MMS) so the exact solution is known in closed form:

$$ f = k_0^2\sin(k_0 x)\sin(k_0 y),\qquad u^*(x,y)=\sin(k_0 x)\sin(k_0 y). $$

With $n=3$ the wavenumber is $k_0=6\pi$, i.e. three full standing-wave periods per axis. The domain is sampled on a
$121\times121$ grid; collocation uses ~12 points per wavelength per axis. This is high enough that a vanilla MLP
stalls, low enough to converge on the CPU lane.

## Method

**Random Fourier-feature embedding** (Tancik 2020; Wang–Wang–Perdikaris multi-scale, 2021). A frozen Gaussian matrix
$B$ maps the input $(x,y)$ through $[\sin(2\pi B^\top x),\cos(2\pi B^\top x)]$ before the first tanh layer, so the
network sees the high frequencies directly instead of having to synthesize them against its spectral bias. Two scales
are stacked ($\sigma=1$ and $\sigma=n$) to cover both slow and fast content, giving a 256-dim embedding feeding a
$4\times128$ tanh FNN.

Key choices, all per the source:
- **$B$ is frozen and seeded** (`torch.manual_seed(0)`) so the *same* feature map is used for training, parity check,
  and ONNX export — the map is a pure-tensor `apply_feature_transform` that traces cleanly into the graph.
- **Soft Dirichlet BC with loss weighting** (`loss_weights=[1, 100]`), not a hard multiplicative constraint: for an
  oscillatory solution a hard constraint fights the oscillation near the boundary, so the robust recipe is a strongly
  weighted soft BC.
- Optimization is **Adam (25 000) → L-BFGS**.

## Result

Validation anchor is the **analytic MMS field** $u^*$ (`validation_anchor = analytic`); measured at seed 42:

| metric | value |
|--------|-------|
| relative-L2 vs analytic $u^*$ | **0.1026** (~10 %) |
| max abs error | 0.158 |
| ONNX parity (max abs) | 4.77e-07 |
| lane | **live** (340 KB ONNX, infer 18.5 ms) |

The ~10 % relative-L2 is **honestly CPU-limited**. The Fourier map lifts the spectral-bias plateau that would
otherwise leave a vanilla MLP an order of magnitude worse, but at $k_0=6\pi$ on the CPU lane the standing pattern is
resolved to ~10 %, not to 1e-3. The `expected_band` says so plainly: GPU training plus frequency annealing would
tighten it further; this is not dressed up.

## Honesty

`real_or_synthetic = synthetic`. The truth here is **closed-form (MMS)**: the forcing $f$ is constructed precisely so
that $u^*=\sin(k_0 x)\sin(k_0 y)$ satisfies the PDE exactly, and the network is scored against that analytic field.
Nothing is fit to measured data and nothing is claimed to be. This is a method demonstrator — it shows that the
Fourier-feature embedding makes a high-wavenumber Helmholtz problem learnable at all — not a real-world Helmholtz
scattering dataset.

## Reproduce

```bash
python -m pinnlab.pipeline ind-helmholtz --seed 42
```


## Comparison & diagnostics (the app's Compare + Diagnostics views)

The pipeline bakes the full **method ladder** for this case (see [the method ladder](../architecture/method-ladder-comparison.md)): a classical
**finite-difference** standard solve, the **naive** plain-tanh PINN, and the **Fourier-feature** PINN, all on one grid.
- **Compare view**: standard | naive | adapted + the error maps. The naive lane reaches **120.8 %** relative-L2 vs the
  standard (spectral bias blurs the high-wavenumber pattern); the Fourier-feature fix reaches **9.3 %**.
- **Diagnostics view**: the **wavenumber sweep** (naive 3 % at n=1 rising to ~100 % at n>=2, while Fourier stays low)
  and the **radial spectral energy** (the high-|k| band the naive lane cannot reach). Real numbers from real training runs.

## Training view (watch it learn)

The app's **Training** view replays the field at REAL training checkpoints (0 to 12k iterations), naive vs Fourier side by side with the live L2 and the L2-vs-iteration curve. The naive tanh lane never leaves ~100 % relative-L2 at ANY checkpoint - spectral bias is a training-time pathology, not a capacity limit - while the Fourier lane converges to ~9 %. Baked by `build_training_dynamics.py`.
