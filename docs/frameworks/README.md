# Frameworks

The engines PINN-Lab trains with. The **deep research is binding** (ADR): every engine selected lives here *and* in
the pinned requirements, and the pipeline actually uses it — no hand-rolled substitute for a prescribed SOTA engine.
Each guide covers what the engine is, how to install/configure it deterministically, the API surface PINN-Lab leans
on, a runnable example, and the train→ONNX→web bridge.

| Engine | Role | Guide |
|--------|------|-------|
| **DeepXDE** (PyTorch backend) | the **primary** offline engine for most cases (forward + inverse PINNs, RAR, hard constraints) | [deepxde/](deepxde/README.md) · [runnable example](deepxde/example.py) |
| **PhysicsNeMo** | the GPU / geometry-heavy lane (3-D, steady Navier–Stokes) + an ONNX-deploy path | [physicsnemo/](physicsnemo/README.md) |
| **neuraloperator** | operator learning — the Fourier Neural Operator (FNO) for the parametric Darcy case | [neuraloperator/](neuraloperator/README.md) |
| **jax-pi (jaxpi)** | the SOTA-ceiling reference (causal weighting, curriculum) — cited as the accuracy ceiling, not the default | [jaxpi/](jaxpi/README.md) |
| **PINA** | a PyTorch-Lightning PINN framework — surveyed for the variational/operator lanes | [pina/](pina/README.md) |
| **NeuralPDE.jl** | the Julia symbolic-PDE PINN stack — surveyed for the strong-/weak-form discretizers | [neuralpde-jl/](neuralpde-jl/README.md) |

## The common bridge

Whatever the engine, the offline lane ends the same way: the trained network is exported to **ONNX** (opset 18,
dynamic batch axis) and **parity-checked** against the engine's own predict (`< 1e-4`). Because the hard-constraint
output transforms and Fourier/SIREN input encodings are pure tensor ops, they are captured in the ONNX graph — so the
browser's **Live** lane evaluates exactly the function that was validated offline. See
[../architecture/train-export-onnx.md](../architecture/train-export-onnx.md).

## Choosing an engine for a new case

Default to **DeepXDE/CPU**. Reach for **PhysicsNeMo** only when the case is 3-D or geometry-heavy beyond CPU fidelity,
or for **neuraloperator** when the case is an operator (a whole input→solution map) rather than a single BVP. The
choice is recorded in the case's manifest (`engine.framework`) and on the Experiments page.
