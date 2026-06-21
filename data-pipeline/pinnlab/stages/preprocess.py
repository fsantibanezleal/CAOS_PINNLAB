"""Stage 1 — preprocess: confirm the case is well-posed and (for inverse cases) apply CONTRACT 1 to an observation
table. Forward cases have no external data — the PDE + BC/IC determine the solution — so this stage validates the
config and, if a bring-your-own-data CSV is supplied, ingests + screens the observations."""
from __future__ import annotations

from ..io.contract import ContractReport, validate_observations
from ..io.formats import read_csv_rows
from ..registry import get_case


def run(case_id: str, obs_csv_path: str | None = None) -> dict:
    case = get_case(case_id)
    info: dict = {"case_id": case_id, "well_posed": True, "flags": [], "n_obs": 0}
    for ax, (lo, hi) in case.domain.items():
        if not (hi > lo):
            info["well_posed"] = False
            info["flags"].append({"flag": f"domain axis {ax} not increasing: [{lo},{hi}]"})
    if obs_csv_path:
        rep: ContractReport = validate_observations(read_csv_rows(obs_csv_path), n_coords=len(case.inputs))
        info["n_obs"] = len(rep.accepted)
        info["flags"].extend(rep.flagged)
        info["rejected"] = rep.rejected
        info["observations"] = rep.accepted
    return info
