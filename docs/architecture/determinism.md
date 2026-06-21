# Determinism

A PINN-Lab run is a **pure function of `(case, seed)`**. Given the same case module and the same seed, the pipeline
produces the same trained network, the same field trace, and the same manifest — byte-for-byte. This is what makes
the committed artifacts a *replay* rather than a snapshot, and what lets CI re-derive and re-check them.

## Where the seed reaches

`build(seed)` calls `dde.config.set_random_seed(seed)` before constructing the geometry, the collocation sampling,
and the network initialization. That single call seeds:

- **NumPy** — domain/boundary/initial point sampling, the reference grids;
- **PyTorch** — weight initialization (Glorot) and any stochastic training op.

Because the recipe (Adam steps, L-BFGS, the optional RAR refinement) is itself deterministic given those seeds, the
trained weights are fixed, the ONNX export is fixed, and the parity number is stable (~1e-7).

## Why determinism is load-bearing here

1. **Replay integrity.** The web ships the baked `field.json`; if a re-run produced a different field, the trace
   would silently disagree with what a "live" re-derivation shows. Determinism guarantees they match.
2. **Auditability of the gate.** The [lane verdict](the-gate.md) is computed from measured `onnx_bytes` / `infer_ms` /
   `trace_bytes`. CI re-derives the lane from the committed manifest; that is only meaningful if the run is
   reproducible.
3. **Honest benchmarks.** Cross-case accuracy tables (Experiments / Benchmark pages) compare numbers that anyone can
   regenerate from `(case, seed)`. No hidden run-to-run variance to cherry-pick.

## What would break it (and is therefore forbidden)

- **Wall-clock / RNG outside the seeded path.** No `Date.now()`-style entropy in the engine; timing is *measured and
  recorded* (`infer_ms`) but never *fed back* into the computation.
- **Non-deterministic kernels.** CUDA non-deterministic reductions are avoided for the committed artifacts; the
  reference runs are CPU/seeded so the manifest numbers are portable.
- **Unpinned dependencies.** The engine versions (DeepXDE, PyTorch, onnxscript, onnxruntime) are pinned in
  `requirements`; a float in a core numeric library could shift the last digits. Pinning keeps `(case, seed)` →
  artifact stable across machines.

## The practical contract

> Delete `data/derived/<case>/` and `models/<case>.onnx`, re-run `pipeline.py <case> --seed <seed>`, and you get the
> same files back.

That property is the reason every number in the app and the docs is traceable to a command, not to a one-off run that
can never be reproduced.
