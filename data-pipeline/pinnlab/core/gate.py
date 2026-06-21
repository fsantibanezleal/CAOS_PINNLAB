"""The measured live-vs-precompute GATE (ADR-0054). A case runs LIVE in the browser (Pyodide) iff it is
pure-Python AND its wheels are a subset of the Pyodide-safe set AND it is small+fast enough; otherwise it is
PRECOMPUTE and the SPA replays the committed artifact. The verdict + the measured numbers go into the manifest,
and CI fails on mislabeling. This is a MEASUREMENT, never a hand-wave."""
from __future__ import annotations

LIVE_WHEELS: set[str] = {"numpy"}   # the Pyodide-safe wheel set the live lane is allowed to import
RUN_MS_GATE = 1500.0                 # a live run must complete well within an interaction budget
TRACE_BYTES_GATE = 256 * 1024        # a live/replay artifact must stay small


def classify_lane(*, pure_python: bool, wheels: set[str], run_ms: float, trace_bytes: int) -> dict:
    reasons: list[str] = []
    live = True
    if not pure_python:
        live = False
        reasons.append("not pure-python")
    extra = set(wheels) - LIVE_WHEELS
    if extra:
        live = False
        reasons.append(f"wheels not Pyodide-safe: {sorted(extra)}")
    if run_ms > RUN_MS_GATE:
        live = False
        reasons.append(f"run_ms {run_ms:.0f} > {RUN_MS_GATE:.0f}")
    if trace_bytes > TRACE_BYTES_GATE:
        live = False
        reasons.append(f"trace_bytes {trace_bytes} > {TRACE_BYTES_GATE}")
    return {
        "lane": "live" if live else "precompute",
        "pure_python": pure_python,
        "wheels": sorted(wheels),
        "run_ms": round(run_ms, 2),
        "trace_bytes": trace_bytes,
        "reasons": reasons,
    }
