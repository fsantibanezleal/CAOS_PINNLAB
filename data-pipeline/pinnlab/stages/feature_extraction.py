"""Stage 2 — feature_extraction: derive the collocation/sampling plan for a case (deterministic, recorded for
provenance). DeepXDE draws the actual collocation points inside build(); this stage fixes + documents the plan
(domain/boundary/test counts, sampler) so the manifest records HOW the field was sampled."""
from __future__ import annotations

from ..registry import get_case


def run(case_id: str) -> dict:
    t = get_case(case_id).train
    return {
        "num_domain": int(t.get("num_domain", 2000)),
        "num_boundary": int(t.get("num_boundary", 0)),
        "num_test": int(t.get("num_test", 2000)),
        "sampler": t.get("sampler", "pseudo (uniform)"),
    }
