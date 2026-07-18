# Methods

The state-of-the-art PINN method families PINN-Lab uses. Each is **exercised** in at least one case (not merely
named) and reproduced on the web app's Methodology page with its formulation and a primary peer-reviewed reference.
The base recipe **Adam → L-BFGS** underlies every case.

| Method | The idea | Exercised in | Page |
|--------|----------|--------------|------|
| **Adaptive sampling (RAR)** | add collocation where the residual is largest → resolve shocks/interfaces | burgers1d, allencahn | [adaptive-sampling.md](adaptive-sampling.md) |
| **Architectures & spectral bias** | Fourier features / SIREN inject high frequencies; hard-constraint output transforms bake BC/IC exactly | wave1d (SIREN), poisson2d (hard BC), helmholtz (Fourier) | [architectures.md](architectures.md) |
| **Causal & curriculum training** | weight the temporal residual so early dynamics converge first | time-dependent cases | [causal-curriculum.md](causal-curriculum.md) |
| **Loss / gradient weighting** | rebalance PDE vs BC vs data terms (NTK / gradient-norm / self-adaptive) | navier-cavity | [loss-weighting.md](loss-weighting.md) |
| **Domain decomposition (FBPINN)** | a sub-net per subdomain blended by a partition of unity → handle kinks/jumps | soil-barrier | [domain-decomposition.md](domain-decomposition.md) |
| **Operator learning (FNO / DeepONet)** | learn the solution *map* over a family, one forward pass per new instance | darcy-operator (FNO) | [operator-learning.md](operator-learning.md) |
| **Inverse problems & UQ** | recover a field/parameter from sparse data + a deep-ensemble error bar | heat2d-inverse, soil-heat-real (real), source-uq-bpinn | [inverse-uq.md](inverse-uq.md) |
| **Variational & scalable** | weak-form (hp-VPINN) and separable (SPINN) routes to lower regularity / huge collocation | roadmap | [variational-scalable.md](variational-scalable.md) |
| **Optimization (Adam → L-BFGS)** | first-order into the basin, quasi-Newton to polish | every DeepXDE case | [optimization.md](optimization.md) |

## How to read a method page

Each page gives the idea, the math (term-by-term), the cases that exercise it, and the primary reference with a DOI.
The on-web **Methodology** page mirrors these with KaTeX. A method that is documented but not yet exercised by a
shipped case is labeled *roadmap*, honestly.
