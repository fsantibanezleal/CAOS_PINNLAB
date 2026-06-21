"""The CaseSpec — light, Pyodide-safe metadata for one PINN case (no torch/deepxde import).

Each `cases/<case>.py` exposes:
  - `CASE`            : a CaseSpec (this dataclass) — pure metadata, importable anywhere (registry, live, web export).
  - `analytic(xy)`    : the closed-form reference solution on [N, d] points (or returns None if no closed form).
  - `build(seed)`     : constructs + compiles the heavy DeepXDE/PhysicsNeMo model (lazy-imports the engine).

`build()` is the ONLY part that pulls the heavy SOTA engine; keeping it behind a function means importing a case
module (for its metadata) never drags torch into the live lane.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# The category taxonomy (dossier §5). Experiments/Benchmark group cross-case summaries by these.
CATEGORIES = (
    "canonical-benchmark",
    "mining-mineral-processing",
    "pollution-environmental",
    "industrial-fluids-heat",
    "control",
)


@dataclass(frozen=True)
class CaseSpec:
    id: str
    category: str
    title: str
    governing_equations: str            # LaTeX-ish; rendered in docs/ and the web Methodology/Benchmark pages
    method: str                         # the SOTA method exercised (dossier §4 key, e.g. "hard-constraints", "RAR")
    engine: str                         # "deepxde" | "physicsnemo" | "neuraloperator"
    real_or_synthetic: str              # "synthetic" | "synthetic-illustrative" | "validated"
    inputs: tuple[str, ...]             # input-axis order, e.g. ("x", "y") or ("x", "t")
    outputs: tuple[str, ...]            # output names, e.g. ("u",) or ("u", "v", "p")
    domain: dict[str, tuple[float, float]]   # eval-grid bounds per input axis
    grid: dict[str, int]                # eval resolution per input axis (the baked field; web re-evals via ONNX)
    expected_band: str                  # what a domain expert should see (honesty band)
    validation_anchor: str              # "analytic" | "fem-ref" | "none"
    train: dict[str, Any] = field(default_factory=dict)   # net + sampling + optimizer plan
    inverse_truth: dict[str, float] = field(default_factory=dict)  # ground-truth coeffs for inverse cases
    notes: str = ""

    def __post_init__(self):
        if self.category not in CATEGORIES:
            raise ValueError(f"{self.id}: unknown category {self.category!r}; valid: {CATEGORIES}")
        if set(self.domain) != set(self.inputs) or set(self.grid) != set(self.inputs):
            raise ValueError(f"{self.id}: domain/grid axes must match inputs {self.inputs}")
