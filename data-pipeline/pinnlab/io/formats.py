"""Standard-format readers/writers.

In: CSV (observation tables for inverse cases) + the case registry (PDE configs). Out: compact JSON (the committed
web-replay artifact) + ONNX (the trained PINN, written by stages/export). Heavy intermediate fields go to npz under
data/raw/ (git-ignored). Never invent a bespoke ad-hoc format — keep everything standard so data is portable.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_json(path: str | Path, obj: Any) -> int:
    """Write compact JSON; return the byte size (used by the gate + manifest). UTF-8, no BOM."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    p.write_bytes(encoded)
    return len(encoded)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_npz(path: str | Path, **arrays: np.ndarray) -> int:
    """Write a heavy intermediate field bundle (data/raw, git-ignored). Returns byte size."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, **arrays)
    return p.stat().st_size
