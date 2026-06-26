"""pinnlab — the offline+live engine for PINN-Lab (CAOS product-repo, ADR-0057).

A catalogue of Physics-Informed Neural Network cases (canonical benchmarks + mining/mineral-processing +
pollution/environmental + industrial fluids/heat). Each case trains a PINN OFFLINE with the deep-research-chosen
SOTA engine (DeepXDE primary; PhysicsNeMo per case; the operator case uses a self-contained FNO), validates it against an analytic or numerical reference,
exports the trained network to ONNX, and bakes a compact replay artifact + manifest. The web app runs the ONNX in
the browser (onnxruntime-web) for live field evaluation and replays the baked artifact as a fallback.

The two data contracts, the staged pipeline, the lane gate (extended to ONNX/ort-web), and the manifest/trace are
the FROZEN base; the per-case PINN engines + analytic references + viz are the rework surface.
"""

__version__ = "0.12.000"  # display X.XX.XXX; PEP 440 form in pyproject.toml
