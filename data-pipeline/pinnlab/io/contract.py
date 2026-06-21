"""CONTRACT 1 — ingestion (raw -> pipeline): the *bring-your-own-data* gate for INVERSE PINN cases.

Forward cases need no external data (the PDE + BC/IC fully determine the solution). Inverse cases assimilate sparse,
possibly noisy MEASUREMENTS of the field to recover unknown PDE coefficients; this contract declares the required
schema of an observation table and an EXPLICIT outlier policy. A row is ACCEPTED iff it parses and lies in range;
bad rows are REJECTED with a reason (never silently coerced); finite-but-extreme values are FLAGGED (accepted, but
the manifest records the flag). Documented in data/README.md.

Required columns: case_id, x0..x{d-1} (the d input coordinates, in the case's input order), value [, weight].
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .schema import ObservationRow


@dataclass
class ContractReport:
    accepted: list[ObservationRow]
    rejected: list[dict[str, Any]]
    flagged: list[dict[str, Any]]

    @property
    def ok(self) -> bool:
        return len(self.accepted) > 0

    def summary(self) -> str:
        return f"{len(self.accepted)} accepted, {len(self.rejected)} rejected, {len(self.flagged)} flagged"


def validate_observations(
    rows: list[dict[str, Any]],
    n_coords: int,
    *,
    value_range: tuple[float, float] = (-1e6, 1e6),
    coord_range: tuple[float, float] = (-1e6, 1e6),
) -> ContractReport:
    """Apply CONTRACT 1 to raw observation rows. Pure; deterministic; no I/O."""
    accepted: list[ObservationRow] = []
    rejected: list[dict[str, Any]] = []
    flagged: list[dict[str, Any]] = []
    coord_cols = [f"x{i}" for i in range(n_coords)]
    required = ["case_id", *coord_cols, "value"]

    for i, row in enumerate(rows):
        cid = str(row.get("case_id", f"row{i}"))
        missing = [c for c in required if c not in row or row[c] in (None, "")]
        if missing:
            rejected.append({"row": i, "case_id": cid, "reason": f"missing/empty columns: {missing}"})
            continue
        try:
            coords = tuple(float(row[c]) for c in coord_cols)
            value = float(row["value"])
            weight = float(row.get("weight") or 1.0)
        except (TypeError, ValueError):
            rejected.append({"row": i, "case_id": cid, "reason": "non-numeric coordinate/value"})
            continue
        if any(math.isnan(v) or math.isinf(v) for v in (*coords, value, weight)):
            rejected.append({"row": i, "case_id": cid, "reason": "NaN/Inf value"})
            continue
        bad = [
            f"x{j}={c:g} out of [{coord_range[0]:g},{coord_range[1]:g}]"
            for j, c in enumerate(coords)
            if not (coord_range[0] <= c <= coord_range[1])
        ]
        if bad:
            rejected.append({"row": i, "case_id": cid, "reason": "; ".join(bad)})
            continue
        if not (value_range[0] <= value <= value_range[1]):
            flagged.append({"case_id": cid, "flag": f"value={value:g} outside typical [{value_range[0]:g},{value_range[1]:g}]"})
        accepted.append(ObservationRow(case_id=cid, coords=coords, value=value, weight=max(0.0, weight)))
    return ContractReport(accepted=accepted, rejected=rejected, flagged=flagged)
