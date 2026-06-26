"""The CaseSpec — light, Pyodide-safe metadata for one PINN case (no torch/deepxde import).

Each `cases/<case>.py` exposes:
  - `CASE`            : a CaseSpec (this dataclass) — pure metadata, importable anywhere (registry, live, web export).
  - `analytic(X)`     : the closed-form reference solution on [N, d] points (or returns None if no closed form).
  - `build(seed)`     : constructs + compiles the heavy DeepXDE/PhysicsNeMo model (lazy-imports the engine).
  - `variants()`      : (optional) the family of parameter regimes (≥ several), SimLab-style. If omitted, the case
                        ships a single "default" variant. For a PARAMETRIC case the tunable parameter is a network
                        INPUT (in `inputs` but not in `field_axes`); a variant fixes that parameter, the baked field
                        is the 2-D heatmap over `field_axes`, and the web `Live` tab sweeps the parameter via the ONE
                        shared ONNX (live re-evaluation) — superior to pre-simulating each regime.

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

# ADR-0063: an ORTHOGONAL "what kind of system is this" axis that SELECTS the web render kit. `category`
# stays the physics-domain bucket (Experiments/Benchmark grouping); `system_type` drives `view_kit`, so a
# time-evolution case animates, a dynamical system shows a trajectory, a vector field shows streamlines, etc.
SYSTEM_TYPES = (
    "ode-dynamical",      # t -> state vector; an animated trajectory + phase portrait (no spatial field)
    "time-evol-1d",       # (x,t) -> u; an animated 1-D profile over time + the x-t carpet as a seek-bar
    "time-evol-2d",       # (x,y,t) -> u; an animated 2-D field over time
    "steady-elliptic",    # (x,y) -> u; a static heatmap (the one honest still-field class)
    "vector-flow",        # (x,y) -> (u,v,p); streamlines / quiver
    "inverse-assim",      # reconstructed field + sparse observation overlay + parameter-convergence
    "eigen-modal",        # eigenvalues + animated mode shapes
    "uq-bayesian",        # mean ± uncertainty band
    "operator-surrogate", # input-function -> solution-field map
)
# system_type -> default render kit (a case may override via `view_kit`).
SYSTEM_TYPE_KIT = {
    "ode-dynamical": "TrajectoryAnimationKit",
    "time-evol-1d": "TimeEvolutionKit",
    "time-evol-2d": "SpatioTemporalKit",
    "steady-elliptic": "HeatmapKit",
    "vector-flow": "VectorFieldKit",
    "inverse-assim": "InverseOverlayKit",
    "eigen-modal": "ModeShapeKit",
    "uq-bayesian": "UQBandKit",
    "operator-surrogate": "HeatmapKit",
}
DEFAULT_KIT = "HeatmapKit"


@dataclass(frozen=True)
class ParamSpec:
    """One tunable physical parameter, surfaced as a slider in the web `Live` tab. For a parametric case the
    parameter is a network input; the slider drives a live ONNX re-evaluation of the field."""
    key: str
    label_en: str
    label_es: str
    default: float
    min: float
    max: float
    step: float = 0.1
    unit: str = ""


@dataclass(frozen=True)
class Variant:
    """A named parameter regime (a preset of the case's parameters), bilingual, with a one-line "what it shows".
    Each parametric case ships ≥ several so the app offers a chip selector and side-by-side comparison."""
    id: str
    label_en: str
    label_es: str
    params: dict
    note_en: str = ""
    note_es: str = ""


@dataclass(frozen=True)
class CaseSpec:
    id: str
    category: str
    title: str
    governing_equations: str            # LaTeX-ish; rendered in docs/ and the web Context/Methodology pages
    method: str                         # the SOTA method exercised (dossier §4 key, e.g. "hard-constraints", "RAR")
    engine: str                         # "deepxde" | "physicsnemo" | "neuraloperator" | "fno-torch"
    real_or_synthetic: str              # "synthetic" | "synthetic-illustrative" | "validated-real"
    inputs: tuple[str, ...]             # ALL network input-axis order, e.g. ("x","y") or ("x","y","k") for parametric
    outputs: tuple[str, ...]            # output names, e.g. ("u",) or ("u","v","p")
    domain: dict[str, tuple[float, float]]   # bounds per input axis (ALL inputs, incl. parameter axes)
    grid: dict[str, int]                # eval resolution per FIELD axis (the baked 2-D heatmap)
    expected_band: str                  # what a domain expert should see (honesty band)
    validation_anchor: str              # "analytic" | "numerical-ref" | "dataset" | "operator-test-l2" | "benchmark-ghia" | "real-data-holdout" | "integrator-ref" | "none"
    train: dict[str, Any] = field(default_factory=dict)   # net + sampling + optimizer plan
    field_axes: tuple[str, ...] = ()    # the 2 input axes that form the baked heatmap; defaults to `inputs`
    param_specs: tuple[ParamSpec, ...] = ()   # the tunable parameter axes (inputs NOT in field_axes)
    inverse_truth: dict[str, float] = field(default_factory=dict)  # ground-truth coeffs for inverse cases
    system_type: str = ""               # ADR-0063: kind-of-system axis that selects the render kit ("" -> HeatmapKit)
    view_kit: str = ""                  # explicit kit override ("" -> derived from system_type)
    notes: str = ""

    @property
    def axes(self) -> tuple[str, ...]:
        """The field (heatmap) axes — `field_axes` if set, else all `inputs` (non-parametric default)."""
        return self.field_axes or self.inputs

    @property
    def kit(self) -> str:
        """The web render kit: explicit `view_kit`, else the default for `system_type`, else HeatmapKit."""
        return self.view_kit or SYSTEM_TYPE_KIT.get(self.system_type, DEFAULT_KIT)

    def __post_init__(self):
        if self.category not in CATEGORIES:
            raise ValueError(f"{self.id}: unknown category {self.category!r}; valid: {CATEGORIES}")
        if self.system_type and self.system_type not in SYSTEM_TYPES:
            raise ValueError(f"{self.id}: unknown system_type {self.system_type!r}; valid: {SYSTEM_TYPES}")
        if set(self.domain) != set(self.inputs):
            raise ValueError(f"{self.id}: domain axes {set(self.domain)} must match inputs {self.inputs}")
        if set(self.grid) != set(self.axes):
            raise ValueError(f"{self.id}: grid axes {set(self.grid)} must match field axes {self.axes}")
        # parameter axes = inputs not in the field; they must each have a ParamSpec, and vice-versa
        param_axes = set(self.inputs) - set(self.axes)
        spec_keys = {p.key for p in self.param_specs}
        if param_axes != spec_keys:
            raise ValueError(f"{self.id}: parameter axes {param_axes} must match param_specs {spec_keys}")
