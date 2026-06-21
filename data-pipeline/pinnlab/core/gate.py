"""The measured live-vs-precompute GATE (ADR-0054, extended for PINN-Lab per the agreed dossier).

The template's gate decided live-vs-precompute from a Pyodide-wheel check. PINN-Lab's live lane is NOT Pyodide
running DeepXDE (far too heavy) — it is **onnxruntime-web inference of the exported, trained PINN**. So the gate
here measures what actually governs an in-browser run:

  a case is LIVE  iff  onnx_bytes  <= ONNX_BYTES_GATE
                  AND  infer_ms    <= INFER_MS_GATE     (measured with onnxruntime CPU as an ort-web proxy)
                  AND  trace_bytes <= TRACE_BYTES_GATE  (the committed replay artifact stays small)

A LIVE case ships its `.onnx` and the SPA evaluates the field at arbitrary resolution / cursor probes in real time;
a PRECOMPUTE case ships only the baked replay artifact. The verdict + the measured numbers go into the manifest,
and CI fails on mislabeling. This is a MEASUREMENT, never a hand-wave.
"""
from __future__ import annotations

ONNX_BYTES_GATE = 4 * 1024 * 1024   # a tiny PINN MLP is tens of KB; the cap keeps the browser download cheap
INFER_MS_GATE = 120.0               # a full-grid ort-web evaluation must fit an interaction budget
TRACE_BYTES_GATE = 1024 * 1024      # the committed replay artifact must stay small


def classify_lane(*, onnx_bytes: int, infer_ms: float, trace_bytes: int) -> dict:
    reasons: list[str] = []
    live = True
    if onnx_bytes > ONNX_BYTES_GATE:
        live = False
        reasons.append(f"onnx_bytes {onnx_bytes} > {ONNX_BYTES_GATE}")
    if infer_ms > INFER_MS_GATE:
        live = False
        reasons.append(f"infer_ms {infer_ms:.1f} > {INFER_MS_GATE:.0f}")
    if trace_bytes > TRACE_BYTES_GATE:
        live = False
        reasons.append(f"trace_bytes {trace_bytes} > {TRACE_BYTES_GATE}")
    return {
        "lane": "live" if live else "precompute",
        "onnx_bytes": int(onnx_bytes),
        "infer_ms": round(float(infer_ms), 2),
        "trace_bytes": int(trace_bytes),
        "reasons": reasons,
    }
