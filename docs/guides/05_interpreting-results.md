# 05 · Interpreting results

What the numbers on the Benchmark page (and in each manifest) actually mean, and how to read them honestly.

## Relative-L2 (the error)

The headline accuracy number is the **relative L2** of the PINN against the case's validation anchor, over the field
grid:

$$
\varepsilon_{\text{rel}}=\frac{\lVert u_\theta - u^*\rVert_2}{\lVert u^*\rVert_2}
$$

- `u_θ` is the trained network; `u*` is the reference (analytic / dataset / FEM).
- It is reported **per variant** (each parameter regime is validated independently).
- Rough reading: `< 1e-2` is a solid PINN solve; `~1e-1` flags a genuinely hard case (high-wavenumber spectral bias,
  a coefficient-jump kink) — reported honestly, not hidden, usually with a note that a GPU / annealing tightens it.

A relative-L2 is **not** an accuracy claim about reality — it is the distance to the *reference*. If the reference is
an MMS or a reduced model, the number says "the network solved the stated equation well", not "this matches a real
plant".

## ONNX parity (the train→web bridge)

$$
\max_{\mathbf{x}}\big|u_{\text{ONNX}}(\mathbf{x}) - u_{\text{torch}}(\mathbf{x})\big| < 10^{-4}
$$

This proves the exported ONNX the **browser** runs is the same function that was **validated** offline — including the
hard-constraint output transforms and Fourier/SIREN encodings, which are captured in the graph. A passing parity is
what makes the Live lane trustworthy. Typical values are `~1e-7`.

## Lanes (`live` vs `precompute`)

A gate classifies each case from measurements — ONNX size, onnxruntime-web inference time over the full grid, and
replay-artifact bytes:

- **live** — the ONNX is small/fast enough to re-evaluate in the browser; the Live tab sweeps the parameter.
- **precompute** — too heavy for the browser; the app replays the baked field only (the Live tab explains this).

The lane is **derived from measurements**, never hand-set.

## Honesty labels (the data status)

Each case declares one:

- **`synthetic`** — a closed-form or MMS truth (most canonical benchmarks). The L2 is the true solver error.
- **`synthetic-illustrative`** — a faithful reduced model of a real process (mining/pollution closures), with
  engineering-scale values but **not** fit to a measured dataset. No open field data exists for these.
- **`validated-real`** — trained and validated against a **real measured dataset** (e.g. `env-soil-heat-real`, soil
  thermal diffusivity from NOAA USCRN), tested out-of-sample.

Nothing is presented as a calibrated digital twin if it is not one. That distinction is the point of the labels.

## Putting it together

A trustworthy case shows: a relative-L2 inside its `expected_band`, a parity `< 1e-4`, a derived lane, and an explicit
honesty label. The **Benchmark** page lays all four side by side so you can judge each case on its own terms — no
dressed-up numbers.

---

**Back to:** [Guides index](README.md) · [docs home](../README.md)
